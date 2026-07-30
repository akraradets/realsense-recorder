"""
Recorder: last stage. Consumes EncodedEnvelope tokens from the processor
(compression already done) and tracks drop/gap/FPS accounting.

The MP4 file itself is written by the processor's VideoWriter. On stop,
this stage measures actual capture FPS. If it differs from the configured
target, it flags a mismatch — conversion is optional and must be requested
explicitly (GUI warns the user; remux runs on a worker thread).
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
    # Webcam hardware often lies about FPS; mismatch is detected on stop.
    # Synthetic / capture-card sources keep stamped target FPS (R7 claim).
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
        self.fps_mismatch: bool = False
        self.suggested_fps: float = float(self.fps)
        self.bytes_from_processor = 0
        self._convert_lock = threading.Lock()

    def start(self) -> None:
        self.frames_written = 0
        self.gaps = []
        self._last_seq = None
        self._first_capture_ts = None
        self._last_capture_ts = None
        self.measured_fps = 0.0
        self.fps_corrected = False
        self.fps_mismatch = False
        self.suggested_fps = float(self.fps)
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
        # Measure only — do not remux here (keeps Stop responsive; GUI may ask).
        self._measure_fps()

    def _measure_fps(self) -> None:
        """Compute measured FPS and flag mismatch; never remux on the stop path."""
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
        self.container_fps = float(self.fps)

        if not self.correct_container_fps:
            return
        if abs(measured - self.fps) / max(self.fps, 1) < 0.10:
            return

        suggested = max(1.0, round(measured, 3))
        self.fps_mismatch = True
        self.suggested_fps = suggested
        logger.info(
            "FPS mismatch: configured %dfps but measured ~%.1ffps — "
            "conversion available (not applied automatically)",
            self.fps, measured,
        )

    def convert_container_fps(self) -> bool:
        """
        Remux the MP4 so playback uses measured FPS (realtime).

        Safe to call from a worker thread. Returns True if the file was rewritten.
        """
        with self._convert_lock:
            if not self.fps_mismatch or self.fps_corrected:
                return self.fps_corrected
            if not self.output_path.exists():
                logger.warning("convert_container_fps: missing file %s", self.output_path)
                return False

            corrected = self.suggested_fps
            logger.info(
                "Converting container FPS %.1f → %.1f on worker thread…",
                float(self.fps), corrected,
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
                self.fps_mismatch = False
                logger.info("FPS conversion done → %s @ %.1ffps", self.output_path, corrected)
                return True

            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            logger.warning("FPS conversion failed; original file kept")
            return False

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
            "fps_mismatch": self.fps_mismatch,
            "suggested_container_fps": self.suggested_fps,
            "slow_writes": 0,
            "bytes_from_processor": self.bytes_from_processor,
            "compression_stage": "processor",
        }
