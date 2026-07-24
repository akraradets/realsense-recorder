"""
Wires the pipeline exactly as specified:

    camera_handler --sub--> stream_viewer
    camera_handler --sub--> processor (compression) --> recorder

Supports preview-only mode (live GUI) and recording mode (processor+recorder
armed). Frame drops on the record path are counted; preview may drop-oldest.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import cv2

from poc1.bag_recorder import start_bag_recording, stop_bag_recording
from poc1.camera_handler import CameraHandler, FrameEnvelope
from poc1.frame_source import FrameSource
from poc1.monitor import SystemMonitor
from poc1.processor import Processor
from poc1.recorder import Recorder
from poc1.stream_viewer import StreamViewer

logger = logging.getLogger("poc1.pipeline")


class CvCaptureSource:
    """Wraps cv2.VideoCapture (webcam / virtual cam / capture card)."""

    def __init__(
        self,
        device_index: int,
        width: int,
        height: int,
        fps: int,
        backend=None,
    ):
        self.device_index = device_index
        self.width = width
        self.height = height
        self.target_fps = fps
        self._backend = cv2.CAP_DSHOW if backend is None else backend
        self._cap: Optional[cv2.VideoCapture] = None
        self.actual_fps: float = 0.0
        self.actual_width: int = width
        self.actual_height: int = height
        self.allow_fps_remux: bool = True  # webcam: fix playback when HW lies about fps

    def start(self) -> None:
        from poc1.device_enum import quiet_opencv
        with quiet_opencv():
            self._cap = cv2.VideoCapture(self.device_index, self._backend)
            if not self._cap.isOpened() and self._backend == cv2.CAP_DSHOW:
                self._cap.release()
                self._cap = cv2.VideoCapture(self.device_index, cv2.CAP_MSMF)
                self._backend = cv2.CAP_MSMF
            if not self._cap.isOpened():
                raise RuntimeError(f"Could not open capture device index={self.device_index}")
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            try:
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:  # noqa: BLE001
                pass
            self.actual_fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 0)
            self.actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH) or self.width)
            self.actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or self.height)
            if self.actual_width > 0:
                self.width = self.actual_width
            if self.actual_height > 0:
                self.height = self.actual_height
        logger.info(
            "CvCaptureSource: %dx%d@%d (device reports %.1ffps)",
            self.width, self.height, self.target_fps, self.actual_fps,
        )

    def stop(self) -> None:
        if self._cap:
            self._cap.release()
            self._cap = None

    def read(self):
        if self._cap is None:
            return None
        ok, frame = self._cap.read()
        return frame if ok else None


@dataclass
class Pipeline:
    source: FrameSource
    on_preview_frame: Callable[[FrameEnvelope], None]
    output_path: Optional[Path] = None
    monitor_csv: Optional[Path] = None

    def __post_init__(self) -> None:
        self.camera_handler = CameraHandler(source=self.source)
        self.processor = Processor(in_queue=self.camera_handler.processor_queue)
        self.viewer = StreamViewer(
            in_queue=self.camera_handler.viewer_queue, on_frame=self.on_preview_frame
        )
        self.recorder: Optional[Recorder] = None
        self.monitor: Optional[SystemMonitor] = None
        self._preview_started = False
        self._last_report: dict = {}
        self._bag_path: Optional[Path] = None
        self._bag_written: Optional[Path] = None

    def start_preview(self) -> None:
        if self._preview_started:
            return
        self.viewer.start()
        self.camera_handler.start()
        self._preview_started = True
        logger.info("Preview started")

    def start_recording(
        self,
        output_path: Path,
        monitor_csv: Path,
        bag_path: Optional[Path] = None,
    ) -> None:
        if not self._preview_started:
            self.start_preview()
        self.output_path = output_path
        self.monitor_csv = monitor_csv
        self._bag_path = bag_path
        self._bag_written = None

        # Fresh queues so a previous take cannot leak frames into this one.
        self.camera_handler.processor_queue.clear()
        self.processor.out_queue.clear()

        # Optional RealSense .bag (hardware only). Restart SDK before arming MP4 path.
        if bag_path is not None:
            start_bag_recording(self.source, bag_path)

        # Compression lives in the processor; recorder only accounts encoded tokens.
        self.processor.configure_output(
            output_path=output_path,
            width=self.source.width,
            height=self.source.height,
            fps=self.source.target_fps,
        )
        # Remux only for real capture devices that lie about FPS (webcam).
        allow_remux = bool(getattr(self.source, "allow_fps_remux", True))
        self.recorder = Recorder(
            out_queue=self.processor.out_queue,
            output_path=output_path,
            width=self.source.width,
            height=self.source.height,
            fps=self.source.target_fps,
            fourcc=self.processor.chosen_fourcc,
            codec_label=self.processor.codec_label,
            correct_container_fps=allow_remux,
        )
        self.monitor = SystemMonitor(output_csv=monitor_csv)

        self.processor.start()
        self.recorder.start()
        self.monitor.start()
        self.camera_handler.enable_recording()
        logger.info(
            "Recording started -> %s (compression=%s)",
            output_path, self.processor.codec_label,
        )

    def stop_recording(self) -> dict:
        """Stop record path, keep preview alive if it was running."""
        self.camera_handler.disable_recording()
        self._bag_written = stop_bag_recording(self.source)
        if self.processor:
            self.processor.stop()
        if self.recorder:
            self.recorder.stop()
        if self.monitor:
            self.monitor.stop()
        report = self.report()
        self._last_report = report
        if self.output_path:
            report_path = self.output_path.with_suffix(".report.json")
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            report["report_path"] = str(report_path)
        logger.info("Recording stopped: %s", report)
        return report

    def stop(self) -> dict:
        """Full shutdown (recording + preview + camera)."""
        report: dict = {}
        if self.camera_handler.is_recording:
            report = self.stop_recording()
        self.camera_handler.stop()
        self.viewer.stop()
        self._preview_started = False
        return report or self._last_report

    def report(self) -> dict:
        recorder_summary = self.recorder.summary() if self.recorder else {
            "frames_written": 0,
            "sequence_gaps": [],
            "dropped_before_recorder": 0,
            "codec": "",
            "fourcc": "",
            "output_path": "",
        }
        read = self.camera_handler.frames_read
        written = recorder_summary["frames_written"]
        gaps = recorder_summary["sequence_gaps"]
        dropped_proc = self.camera_handler.processor_queue.dropped_count
        dropped_rec = recorder_summary["dropped_before_recorder"]
        proc_summary = self.processor.summary()
        return {
            "frames_read_by_camera": read,
            "frames_processed": self.processor.frames_processed,
            "frames_written": written,
            "sequence_gaps": gaps,
            "dropped_viewer_queue": self.camera_handler.viewer_queue.dropped_count,
            "dropped_processor_queue": dropped_proc,
            "dropped_recorder_queue": dropped_rec,
            "codec": proc_summary.get("codec") or recorder_summary.get("codec", ""),
            "fourcc": proc_summary.get("fourcc") or recorder_summary.get("fourcc", ""),
            "output_path": recorder_summary.get("output_path", ""),
            "width": self.source.width,
            "height": self.source.height,
            "target_fps": self.source.target_fps,
            "measured_fps": recorder_summary.get("measured_fps", 0.0),
            "container_fps": recorder_summary.get("container_fps", self.source.target_fps),
            "fps_corrected": recorder_summary.get("fps_corrected", False),
            "slow_writes": recorder_summary.get("slow_writes", 0),
            "slow_encodes": proc_summary.get("slow_encodes", 0),
            "compression_stage": "processor",
            "bag_path": str(self._bag_written) if self._bag_written else "",
            "bag_recorded": bool(self._bag_written),
            "no_frame_drops": (
                read > 0
                and read == written
                and not gaps
                and dropped_proc == 0
                and dropped_rec == 0
            ),
        }
