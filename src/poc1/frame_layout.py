"""Raw frame layouts from ffmpeg pipe / capture (BGR vs NV12)."""
from __future__ import annotations

import cv2
import numpy as np

PIPE_BGR24 = "bgr24"
PIPE_NV12 = "nv12"


def pipe_pix_fmt_for_fps(target_fps: int) -> str:
    """High-rate Elgato: NV12 halves pipe bandwidth vs BGR24."""
    return PIPE_NV12 if int(target_fps) >= 90 else PIPE_BGR24


def frame_bytes(width: int, height: int, pipe_pix_fmt: str) -> int:
    w, h = int(width), int(height)
    if pipe_pix_fmt == PIPE_NV12:
        return w * h * 3 // 2
    return w * h * 3


def reshape_raw_frame(
    buf: bytes | memoryview, width: int, height: int, pipe_pix_fmt: str
) -> np.ndarray:
    w, h = int(width), int(height)
    if pipe_pix_fmt == PIPE_NV12:
        return np.frombuffer(buf, dtype=np.uint8).reshape(h * 3 // 2, w)
    return np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)


def is_nv12_frame(frame: np.ndarray, height: int, width: int) -> bool:
    if frame is None or frame.ndim != 2:
        return False
    h, w = int(height), int(width)
    return frame.shape == (h * 3 // 2, w)


def ensure_bgr(frame: np.ndarray, height: int = 0, width: int = 0) -> np.ndarray:
    """Convert NV12 pipe frames to BGR; pass BGR through unchanged."""
    if frame is None:
        raise ValueError("frame is None")
    if frame.ndim == 3 and frame.shape[2] == 3:
        return frame
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_NV12)
    raise ValueError(f"unsupported frame shape {frame.shape}")
