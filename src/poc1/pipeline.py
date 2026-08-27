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
        self._uvc_bag = None

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
        self._bag_path = None
        self._bag_written = None
        self._uvc_bag = None
        self.processor.sidecar_bag = None

        # Fresh queues so a previous take cannot leak frames into this one.
        self.camera_handler.processor_queue.clear()
        self.processor.out_queue.clear()

        if bag_path is not None:
            tag = str(getattr(self.source, "device_tag", "") or "")
            if tag == "elgato":
                from poc1.uvc_rosbag import UvcRos2Bag

                self._uvc_bag = UvcRos2Bag(Path(bag_path))
                self._uvc_bag.start()
                self._bag_path = Path(bag_path)
                self.processor.sidecar_bag = self._uvc_bag
            else:
                self.camera_handler.pause_reads()
                try:
                    active = start_bag_recording(self.source, bag_path)
                    self._bag_path = Path(active)
                finally:
                    self.camera_handler.resume_reads()

        # Capture cards / UVC often advertise 120 while HDMI only sends ~60.
        self._requested_fps = int(
            getattr(self.source, "requested_fps", None)
            or getattr(self.source, "target_fps", 30)
            or 30
        )
        self.source.requested_fps = self._requested_fps
        self._align_capture_fps()

        # Compression lives in the processor; recorder only accounts encoded tokens.
        self.processor.configure_output(
            output_path=output_path,
            width=self.source.width,
            height=self.source.height,
            fps=self.source.target_fps,
        )
        # Remux safety net for devices that still drift after stamping.
        allow_remux = bool(getattr(self.source, "allow_fps_remux", True))
        if getattr(self.source, "device_tag", "") == "elgato":
            allow_remux = True
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

        # Arm capture BEFORE starting encode workers so Elgato frames during
        # worker startup are counted (fixes 0-frame Record with live preview).
        self.camera_handler.enable_recording()
        self.processor.start()
        self.recorder.start()
        self.monitor.start()
        logger.info(
            "Recording started -> %s (compression=%s bag=%s fps=%s requested=%s)",
            output_path,
            self.processor.codec_label,
            bool(self._bag_path),
            self.source.target_fps,
            getattr(self, "_requested_fps", self.source.target_fps),
        )

    def _align_capture_fps(self) -> None:
        """Stamp MP4 at the real delivery rate. Never steal the live UVC handle."""
        from poc1.deliverable1.devices import honest_container_fps

        tag = str(getattr(self.source, "device_tag", "") or "")
        if tag not in {"elgato", "uvc"}:
            return
        # Prefer live camera-handler delivery rate over open-time sample / UI paint.
        # Do NOT sleep here — blocking Record start raced Elgato reads and
        # contributed to "0 frames captured" on both stations.
        rate = float(getattr(self.source, "actual_fps", 0) or 0)
        requested = int(getattr(self, "_requested_fps", self.source.target_fps) or 30)
        try:
            live = float(self.camera_handler.live_delivery_fps() or 0)
            if live >= 5:
                rate = max(rate, live)
        except Exception:  # noqa: BLE001
            pass
        hint = float(getattr(self, "_preview_fps_hint", 0) or 0)
        # Preview paint is capped (~15Hz) — only use it if we have no better signal.
        if rate < 5 and hint >= 5:
            rate = hint
        if rate < 5:
            return
        self.source.actual_fps = rate
        stamped = honest_container_fps(rate, requested)
        if stamped != int(self.source.target_fps):
            logger.warning(
                "%s Record stamp %dfps → %dfps (measured ~%.1f). "
                "Quit OBS; keep 1920x1080@120 for true 120.",
                tag,
                requested,
                stamped,
                rate,
            )
        else:
            logger.info(
                "%s Record stamp %dfps (live ~%.1f)",
                tag,
                stamped,
                rate,
            )
        self.source.target_fps = stamped

    def _align_elgato_fps(self) -> None:
        self._align_capture_fps()

    def stop_recording(self) -> dict:
        """Stop record path, keep preview alive if it was running."""
        expected_bag = None
        if self._bag_path is not None:
            expected_bag = (
                getattr(self.source, "_bag_final_path", None)
                or getattr(self.source, "_bag_path", None)
                or self._bag_path
            )
        try:
            self.camera_handler.disable_recording()
        except Exception:  # noqa: BLE001
            logger.exception("disable_recording failed")
        if self._uvc_bag is None:
            try:
                self.camera_handler.pause_reads()
                try:
                    self._bag_written = stop_bag_recording(self.source)
                finally:
                    self.camera_handler.resume_reads()
            except Exception:  # noqa: BLE001
                logger.exception("stop_bag_recording failed")
                self._bag_written = None
                try:
                    self.camera_handler.resume_reads()
                except Exception:  # noqa: BLE001
                    pass
        if self.processor:
            try:
                self.processor.stop()
            except Exception:  # noqa: BLE001
                logger.exception("processor.stop failed")
        bag_dropped = 0
        bag_frames = 0
        if self._uvc_bag is not None:
            try:
                bag_dropped = int(getattr(self._uvc_bag, "dropped", 0) or 0)
                bag_frames = int(getattr(self._uvc_bag, "frames_written", 0) or 0)
                self._bag_written = self._uvc_bag.stop() or self._bag_path
                bag_dropped = int(getattr(self._uvc_bag, "dropped", bag_dropped) or 0)
                bag_frames = int(getattr(self._uvc_bag, "frames_written", bag_frames) or 0)
            except Exception:  # noqa: BLE001
                logger.exception("UVC ROS2 bag stop failed")
            self._uvc_bag = None
            self.processor.sidecar_bag = None
        if self.recorder:
            try:
                self.recorder.stop()
            except Exception:  # noqa: BLE001
                logger.exception("recorder.stop failed")
        if self.monitor:
            try:
                self.monitor.stop()
            except Exception:  # noqa: BLE001
                logger.exception("monitor.stop failed")
        report = self.report()
        report["bag_dropped"] = bag_dropped
        report["bag_frames_written"] = bag_frames
        if bag_dropped > 0:
            report["bag_queue_overflow"] = True
            logger.warning(
                "Elgato ROS2 bag dropped %d frame(s) (queue overflow) — MP4 path separate",
                bag_dropped,
            )
        if hasattr(self, "_requested_fps"):
            report["requested_fps"] = int(self._requested_fps)
        self._last_report = report
        if self.output_path:
            try:
                if self.monitor_csv is not None:
                    report_path = Path(self.monitor_csv).with_name(
                        f"{self.output_path.stem}.report.json"
                    )
                else:
                    report_path = self.output_path.with_suffix(".report.json")
                report_path.parent.mkdir(parents=True, exist_ok=True)
                report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
                report["report_path"] = str(report_path)
            except Exception:  # noqa: BLE001
                logger.exception("could not write report json")
        logger.info("Recording stopped: %s", report)
        is_uvc_bag = str(getattr(self.source, "device_tag", "") or "") == "elgato"
        if expected_bag is not None and not self._bag_written and not is_uvc_bag:
            # Last chance: bag file may exist even if stop helper returned None.
            candidate = Path(expected_bag)
            if candidate.exists() and candidate.stat().st_size > 0:
                self._bag_written = candidate
                report["bag_path"] = str(candidate)
                report["bag_recorded"] = True
                self._last_report = report
            else:
                raise RuntimeError(
                    f"MP4 was saved, but RealSense .bag was not "
                    f"(expected {Path(expected_bag).name}). "
                    "Close RealSense Viewer, use USB 3, check Also save RealSense .bag, "
                    "then Record again."
                )
        # Auto-fix Elgato/webcam playback when stamp still disagrees with delivery.
        if (
            self.recorder is not None
            and report.get("fps_mismatch")
            and (
                getattr(self.source, "device_tag", "") == "elgato"
                or bool(getattr(self.source, "allow_fps_remux", False))
            )
        ):
            try:
                if self.convert_container_fps():
                    report = self._last_report or self.report()
            except Exception:  # noqa: BLE001
                logger.exception("auto FPS convert failed")
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
        no_drops = (
            read > 0
            and read == written
            and not gaps
            and dropped_proc == 0
            and dropped_rec == 0
        )
        no_capture = read == 0
        requested = int(getattr(self, "_requested_fps", self.source.target_fps) or 30)
        measured = float(recorder_summary.get("measured_fps") or 0)
        if measured < 5:
            measured = float(getattr(self.source, "actual_fps", 0) or 0)
        container = recorder_summary.get("container_fps", self.source.target_fps)
        tag = str(getattr(self.source, "device_tag", "") or "")
        r7_120_ok = bool(
            tag == "elgato"
            and no_drops
            and measured >= 110.0
        )
        hdmi_not_120 = bool(tag == "elgato" and requested >= 90 and measured < requested * 0.85)
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
            "requested_fps": requested,
            "measured_fps": measured,
            "container_fps": container,
            "fps_corrected": recorder_summary.get("fps_corrected", False),
            "fps_mismatch": recorder_summary.get("fps_mismatch", False),
            "suggested_container_fps": recorder_summary.get(
                "suggested_container_fps", self.source.target_fps
            ),
            "slow_writes": recorder_summary.get("slow_writes", 0),
            "slow_encodes": proc_summary.get("slow_encodes", 0),
            "compression_stage": "processor",
            "bag_path": str(self._bag_written) if self._bag_written else "",
            "bag_recorded": bool(self._bag_written),
            "bag_dropped": 0,
            "bag_frames_written": 0,
            "no_frame_drops": no_drops,
            "no_capture": no_capture,
            "r7_120_ok": r7_120_ok,
            "hdmi_not_120hz": hdmi_not_120,
            "device_tag": tag,
        }

    def convert_container_fps(self) -> bool:
        """Optional remux to measured FPS. Safe to call from a worker thread."""
        if self.recorder is None:
            return False
        ok = self.recorder.convert_container_fps()
        if ok and self.output_path:
            report = self.report()
            if self.monitor_csv is not None:
                report_path = Path(self.monitor_csv).with_name(
                    f"{self.output_path.stem}.report.json"
                )
            else:
                report_path = self.output_path.with_suffix(".report.json")
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report["report_path"] = str(report_path)
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            self._last_report = report
        return ok
