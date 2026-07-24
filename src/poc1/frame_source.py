"""
Frame source abstraction.

Any camera in the pipeline (real webcam, virtual cam, RealSense, or a
synthetic generator) implements this same interface, so CameraHandler never
needs to know which one it's talking to.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Protocol

import cv2
import numpy as np


class FrameSource(Protocol):
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def read(self) -> Optional[np.ndarray]: ...
    @property
    def width(self) -> int: ...
    @property
    def height(self) -> int: ...
    @property
    def target_fps(self) -> int: ...


# Wide blocks so the barcode survives lossy H.264/MPEG-4 compression.
_BAR_BITS = 32
_BAR_BIT_W = 10
_BAR_H = 20


def embed_seq_barcode(frame: np.ndarray, seq: int) -> None:
    """
    Encode `seq` as a 32-bit binary barcode along the top edge.
    Each bit is a {_BAR_BIT_W}x{_BAR_H} white (1) or black (0) block so
    verifiers can recover the index after MP4 compression without OCR.
    """
    need_w = _BAR_BITS * _BAR_BIT_W
    if frame.shape[1] < need_w or frame.shape[0] < _BAR_H:
        return
    for bit in range(_BAR_BITS):
        on = bool((seq >> (31 - bit)) & 1)
        x0 = bit * _BAR_BIT_W
        frame[0:_BAR_H, x0:x0 + _BAR_BIT_W, :] = 255 if on else 0


def read_seq_barcode(frame: np.ndarray) -> Optional[int]:
    need_w = _BAR_BITS * _BAR_BIT_W
    if frame.shape[1] < need_w or frame.shape[0] < _BAR_H:
        return None
    seq = 0
    for bit in range(_BAR_BITS):
        x0 = bit * _BAR_BIT_W
        block = frame[2:_BAR_H - 2, x0 + 2:x0 + _BAR_BIT_W - 2]
        if block.size == 0:
            return None
        if float(block.mean()) > 127.0:
            seq |= 1 << (31 - bit)
    return seq

def embed_status_banner(frame: np.ndarray, text: str) -> None:
    """Large on-frame banner so simulation/loopback is obvious in the GUI."""
    h, w = frame.shape[:2]
    y1 = max(0, h // 2 - 28)
    y2 = min(h, h // 2 + 28)
    frame[y1:y2, :] = (30, 30, 30)
    cv2.putText(
        frame, text, (24, h // 2 + 8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2, cv2.LINE_AA,
    )


@dataclass
class FakeFrameSource:
    """
    Generates synthetic FHD/120fps-capable frames purely in software.

    Each frame has:
      - a burned-in human-readable counter
      - a machine-readable 32-bit barcode (see embed_seq_barcode)

    Pacing uses a monotonic clock and tracks *cumulative* target time rather
    than sleeping (1/fps) each iteration, so small scheduling jitter doesn't
    accumulate into drift over a long recording.
    """

    width: int = 1920
    height: int = 1080
    target_fps: int = 120
    pattern: str = "bars"  # "bars" | "noise"
    allow_fps_remux: bool = False  # keep stamped FPS for R7 synthetic proofs

    def __post_init__(self) -> None:
        self._frame_idx = 0
        self._t0 = None
        self._running = False
        self._base_frame = self._make_base_frame()

    def _make_base_frame(self) -> Optional[np.ndarray]:
        if self.pattern == "noise":
            return None
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        n_bars = 8
        bar_w = self.width // n_bars
        colors = [
            (255, 255, 255), (0, 255, 255), (255, 255, 0), (0, 255, 0),
            (255, 0, 255), (0, 0, 255), (255, 0, 0), (0, 0, 0),
        ]
        for i, color in enumerate(colors):
            frame[:, i * bar_w:(i + 1) * bar_w] = color
        return frame

    def start(self) -> None:
        self._t0 = time.perf_counter()
        self._frame_idx = 0
        self._running = True

    def reset_sequence(self) -> None:
        """Restart frame index + pacing clock (called when recording arms)."""
        self._t0 = time.perf_counter()
        self._frame_idx = 0

    def stop(self) -> None:
        self._running = False

    def read(self) -> Optional[np.ndarray]:
        if not self._running:
            return None

        target_t = self._t0 + (self._frame_idx / self.target_fps)
        now = time.perf_counter()
        if target_t > now:
            time.sleep(target_t - now)

        frame = (
            np.random.randint(0, 256, (self.height, self.width, 3), dtype=np.uint8)
            if self.pattern == "noise"
            else self._base_frame.copy()
        )

        seq = self._frame_idx
        embed_seq_barcode(frame, seq)

        ts = time.time()
        cv2.putText(
            frame, f"frame={seq:08d}", (24, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 6, cv2.LINE_AA,
        )
        cv2.putText(
            frame, f"frame={seq:08d}", (24, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2, cv2.LINE_AA,
        )
        cv2.putText(
            frame, f"t={ts:.6f}", (24, 100),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 6, cv2.LINE_AA,
        )
        cv2.putText(
            frame, f"t={ts:.6f}", (24, 100),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA,
        )

        self._frame_idx += 1
        return frame
