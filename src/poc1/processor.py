"""
Processor: compression stage between camera_handler and recorder.

Spec shape:
    camera_handler → processor (compression) → recorder

This stage owns the VideoWriter encode (H.264 when available, else mp4v).
Raw ~0.75GB/s frames are compressed here; the recorder only accounts
encoded units and may remux FPS metadata after stop.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2

from poc1.camera_handler import DropCountingQueue, FrameEnvelope
from poc1.codec import choose_best_fourcc

logger = logging.getLogger("poc1.processor")


@dataclass
class EncodedEnvelope:
    """Lightweight token after compression — recorder does not re-encode."""
    seq: int
    capture_ts: float
    encoded_bytes_est: int = 0


@dataclass
class Processor:
    in_queue: DropCountingQueue
    out_queue_size: int = 512

    def __post_init__(self) -> None:
        self.out_queue = DropCountingQueue(
            self.out_queue_size, drop_oldest=False, name="recorder_in"
        )
        self._thread: Optional[threading.Thread] = None
        self._running = threading.Event()
        self._encode_lock = threading.Lock()
        self.frames_processed = 0
        self.bytes_encoded_est = 0
        self._writer: Optional[cv2.VideoWriter] = None
        self.output_path: Optional[Path] = None
        self.width = 0
        self.height = 0
        self.fps = 30
        self.chosen_fourcc = "mp4v"
        self.codec_label = "MPEG-4 (mp4v)"
        self._slow_encodes = 0
        self._slow_encode_ms_max = 0.0

    def configure_output(
        self,
        output_path: Path,
        width: int,
        height: int,
        fps: int,
        fourcc: Optional[str] = None,
    ) -> None:
        self.output_path = output_path
        self.width = width
        self.height = height
        self.fps = fps
        if fourcc:
            self.chosen_fourcc = fourcc
            self.codec_label = fourcc
        else:
            self.chosen_fourcc, self.codec_label = choose_best_fourcc()

    def start(self) -> None:
        if self.output_path is None:
            raise RuntimeError("Processor.configure_output() required before start()")
        if self._thread is not None and self._thread.is_alive():
            self.stop()

        self.frames_processed = 0
        self.bytes_encoded_est = 0
        self._slow_encodes = 0
        self._slow_encode_ms_max = 0.0
        self.out_queue.reset_dropped()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*self.chosen_fourcc)
        self._writer = cv2.VideoWriter(
            str(self.output_path), fourcc, float(self.fps), (self.width, self.height)
        )
        if not self._writer.isOpened() and self.chosen_fourcc != "mp4v":
            self.chosen_fourcc, self.codec_label = "mp4v", "MPEG-4 (mp4v)"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(
                str(self.output_path), fourcc, float(self.fps), (self.width, self.height)
            )
        if not self._writer.isOpened():
            raise RuntimeError(
                f"Processor VideoWriter failed for {self.output_path} "
                f"fourcc={self.chosen_fourcc}"
            )
        logger.info(
            "Processor encoding -> %s (%dx%d@%d) codec=%s",
            self.output_path, self.width, self.height, self.fps, self.codec_label,
        )
        self._running.set()
        self._thread = threading.Thread(target=self._loop, name="processor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running.clear()
        if self._thread:
            # FHD@120 drain can take well over 30s on mp4v — never race _drain.
            self._thread.join(timeout=300.0)
            if self._thread.is_alive():
                logger.error("processor thread still alive after join timeout")
            self._thread = None
        else:
            self._drain()
        with self._encode_lock:
            if self._writer is not None:
                self._writer.release()
                self._writer = None
        if self._slow_encodes:
            logger.info(
                "processor: %d/%d encodes exceeded realtime budget (max %.1fms)",
                self._slow_encodes, self.frames_processed, self._slow_encode_ms_max,
            )

    def _encode(self, env: FrameEnvelope) -> None:
        with self._encode_lock:
            if self._writer is None:
                return
            frame = env.frame
            h, w = frame.shape[:2]
            if w != self.width or h != self.height:
                frame = cv2.resize(frame, (self.width, self.height))

            t0 = time.perf_counter()
            self._writer.write(frame)
            encode_ms = (time.perf_counter() - t0) * 1000.0
            budget = (1000.0 / max(self.fps, 1)) * 2
            if encode_ms > budget:
                self._slow_encodes += 1
                self._slow_encode_ms_max = max(self._slow_encode_ms_max, encode_ms)

            raw = self.width * self.height * 3
            est = max(1, raw // 40)
            self.bytes_encoded_est += est
            self.frames_processed += 1
            self.out_queue.put(
                EncodedEnvelope(
                    seq=env.seq, capture_ts=env.capture_ts, encoded_bytes_est=est
                )
            )

    def _drain(self) -> None:
        while True:
            env = self.in_queue.get(timeout=0.05)
            if env is None:
                break
            self._encode(env)

    def _loop(self) -> None:
        while self._running.is_set():
            env: Optional[FrameEnvelope] = self.in_queue.get(timeout=0.1)
            if env is None:
                continue
            self._encode(env)
        self._drain()

    def summary(self) -> dict:
        return {
            "frames_processed": self.frames_processed,
            "bytes_encoded_est": self.bytes_encoded_est,
            "codec": self.codec_label,
            "fourcc": self.chosen_fourcc,
            "output_path": str(self.output_path) if self.output_path else "",
            "slow_encodes": self._slow_encodes,
            "compression_stage": "processor",
        }
