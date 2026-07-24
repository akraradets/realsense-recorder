"""
Recorder: last stage. Consumes EncodedEnvelope tokens from the processor
(compression already done) and tracks drop/gap/FPS accounting.

The MP4 file itself is written by the processor's VideoWriter. On stop,
this stage may remux the file if measured capture fps differs from the
container stamp (webcam hardware lie).
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import cv2

from poc1.camera_handler import DropCountingQueue
from poc1.processor import EncodedEnvelope

logger = logging.getLogger("poc1.recorder")


def remux_with_fps(
    src: Path,
    dst: Path,
    fps: float,
    fourcc_str: str,
    width: int,
    height: int,
) -> bool:
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        return False
    fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
    writer = cv2.VideoWriter(str(dst), fourcc, float(fps), (width, height))
    if not writer.isOpened():
        cap.release()
        return False
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height))
        writer.write(frame)
    cap.release()
    writer.release()
    return dst.exists() and dst.stat().st_size > 0


@dataclass
class Recorder:
    out_queue: DropCountingQueue
    output_path: Path
    width: int
    height: int
    fps: int
    fourcc: str = "mp4v"
    codec_label: str = "MPEG-4 (mp4v)"
    # Webcam hardware often lies about FPS; remux fixes playback speed.
    # Synthetic sources must keep the stamped target FPS (R7 claim).
    correct_container_fps: bool = True

    def __post_init__(self) -> None:
        self._thread: Optional[threading.Thread] = None
        self._running = threading.Event()
        self.frames_written = 0
        self._last_seq: Optional[int] = None
        self.gaps: list[tuple[int, int]] = []
        self._first_capture_ts: Optional[float] = None
        self._last_capture_ts: Optional[float] = None
        self.measured_fps: float = 0.0
        self.container_fps: float = float(self.fps)
        self.fps_corrected: bool = False
        self.bytes_from_processor = 0

    def start(self) -> None:
        self.frames_written = 0
        self.gaps = []
        self._last_seq = None
        self._first_capture_ts = None
        self._last_capture_ts = None
        self.measured_fps = 0.0
        self.fps_corrected = False
        self.container_fps = float(self.fps)
        self.bytes_from_processor = 0
        self._running.set()
        self._thread = threading.Thread(target=self._loop, name="recorder", daemon=True)
        self._thread.start()
        logger.info(
            "Recorder accounting for %s (encode owned by processor, codec=%s)",
            self.output_path, self.codec_label,
        )

    def stop(self) -> None:
        self._running.clear()
        if self._thread:
            self._thread.join(timeout=30.0)
        self._correct_container_fps_if_needed()

    def _correct_container_fps_if_needed(self) -> None:
        if not self.correct_container_fps:
            self.container_fps = float(self.fps)
            if (
                self.frames_written >= 2
                and self._first_capture_ts is not None
                and self._last_capture_ts is not None
            ):
                elapsed = self._last_capture_ts - self._first_capture_ts
                if elapsed > 0:
                    self.measured_fps = (self.frames_written - 1) / elapsed
            return
        if (
            self.frames_written < 2
            or self._first_capture_ts is None
            or self._last_capture_ts is None
        ):
            return
        elapsed = self._last_capture_ts - self._first_capture_ts
        if elapsed <= 0:
            return
        measured = (self.frames_written - 1) / elapsed
        self.measured_fps = measured
        if abs(measured - self.fps) / max(self.fps, 1) < 0.10:
            self.container_fps = float(self.fps)
            return

        corrected = max(1.0, round(measured, 3))
        logger.info(
            "Webcam delivered ~%.1ffps (not %d) — fixing playback speed to realtime",
            measured, self.fps,
        )
        tmp = self.output_path.with_suffix(".fpsfix.mp4")
        ok = remux_with_fps(
            self.output_path, tmp, corrected, self.fourcc, self.width, self.height,
        )
        if ok:
            self.output_path.unlink(missing_ok=True)
            tmp.replace(self.output_path)
            self.container_fps = corrected
            self.fps_corrected = True
        else:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass

    def _loop(self) -> None:
        while self._running.is_set() or self.out_queue.qsize() > 0:
            item: Optional[Union[EncodedEnvelope, object]] = self.out_queue.get(timeout=0.2)
            if item is None:
                if not self._running.is_set() and self.out_queue.qsize() == 0:
                    break
                continue
            if not isinstance(item, EncodedEnvelope):
                continue

            if self._last_seq is not None and item.seq != self._last_seq + 1:
                self.gaps.append((self._last_seq + 1, item.seq))
                logger.warning(
                    "recorder: sequence gap, expected %d got %d",
                    self._last_seq + 1, item.seq,
                )
            self._last_seq = item.seq
            if self._first_capture_ts is None:
                self._first_capture_ts = item.capture_ts
            self._last_capture_ts = item.capture_ts
            self.bytes_from_processor += item.encoded_bytes_est
            self.frames_written += 1

    def summary(self) -> dict:
        return {
            "frames_written": self.frames_written,
            "sequence_gaps": self.gaps,
            "dropped_before_recorder": self.out_queue.dropped_count,
            "codec": self.codec_label,
            "fourcc": self.fourcc,
            "output_path": str(self.output_path),
            "measured_fps": round(self.measured_fps, 3) if self.measured_fps else 0.0,
            "container_fps": self.container_fps,
            "fps_corrected": self.fps_corrected,
            "slow_writes": 0,
            "bytes_from_processor": self.bytes_from_processor,
            "compression_stage": "processor",
        }
