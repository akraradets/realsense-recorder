"""
Capture card / Elgato frame source (UVC via DirectShow / MSMF).

Elgato Game Capture and similar cards appear as standard video capture devices
on Windows. This module auto-detects a non-webcam, non-virtualcam device and
opens it at FHD@120 when the hardware + input signal support it.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from poc1.device_enum import find_capture_card_index, list_capture_devices
from poc1.pipeline import CvCaptureSource

logger = logging.getLogger("poc1.capturecard")


@dataclass
class CaptureCardSource:
    """
    Elgato / UVC capture card via OpenCV VideoCapture.

    Defaults target FHD@120 (R7). Set allow_fps_remux=False so the recorder keeps
    the stamped container FPS when the card delivers real 120fps.
    """

    width: int = 1920
    height: int = 1080
    target_fps: int = 120
    preferred_index: Optional[int] = None
    allow_fps_remux: bool = False
    capture_mode: str = field(default="capturecard", init=False)

    device_index: int = field(default=-1, init=False)
    backend_used: int = field(default=cv2.CAP_DSHOW, init=False)
    actual_fps: float = field(default=0.0, init=False)
    _inner: Optional[CvCaptureSource] = field(default=None, init=False, repr=False)

    def start(self) -> None:
        idx, backend = find_capture_card_index(
            preferred=self.preferred_index,
            width=self.width,
            height=self.height,
            fps=self.target_fps,
        )
        self.device_index = idx
        self.backend_used = backend
        self._inner = CvCaptureSource(
            idx, self.width, self.height, self.target_fps, backend=backend
        )
        self._inner.start()
        self.width = self._inner.width
        self.height = self._inner.height
        self.target_fps = self._inner.target_fps
        self.actual_fps = self._inner.actual_fps
        self.capture_mode = "elgato" if idx >= 0 else "capturecard"
        logger.info(
            "CaptureCardSource: index=%d %dx%d@%d (device reports %.1ffps)",
            self.device_index, self.width, self.height, self.target_fps, self.actual_fps,
        )

    def stop(self) -> None:
        if self._inner is not None:
            self._inner.stop()
            self._inner = None

    def read(self) -> Optional[np.ndarray]:
        if self._inner is None:
            return None
        return self._inner.read()


def create_capture_card_source(
    width: int = 1920,
    height: int = 1080,
    fps: int = 120,
    preferred_index: Optional[int] = None,
) -> CaptureCardSource:
    return CaptureCardSource(
        width=width, height=height, target_fps=fps, preferred_index=preferred_index
    )


def list_capture_card_candidates() -> list[dict]:
    """Human-readable list for GUI / CLI (excludes virtual-cam barcode devices)."""
    out = []
    for info in list_capture_devices(max_index=10):
        if info.barcode_increasing:
            continue
        out.append({
            "index": info.index,
            "backend": info.backend_name,
            "width": info.width,
            "height": info.height,
        })
    return out
