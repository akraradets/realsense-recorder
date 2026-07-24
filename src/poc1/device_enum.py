"""
Enumerate OpenCV capture devices and find the OBS / pyvirtualcam device.

Designed to be FAST: DSHOW-first, few indices, barcode check only when needed.
"""
from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Optional

import cv2

from poc1.frame_source import read_seq_barcode

logger = logging.getLogger("poc1.device_enum")


@contextmanager
def quiet_opencv() -> Iterator[None]:
    """Silence OpenCV's noisy 'can't open index' spam during probing."""
    prev_env = os.environ.get("OPENCV_LOG_LEVEL")
    os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
    try:
        if hasattr(cv2, "setLogLevel") and hasattr(cv2, "LOG_LEVEL_ERROR"):
            old = cv2.getLogLevel() if hasattr(cv2, "getLogLevel") else None
            cv2.setLogLevel(cv2.LOG_LEVEL_ERROR)
            try:
                yield
            finally:
                if old is not None:
                    cv2.setLogLevel(old)
        else:
            yield
    finally:
        if prev_env is None:
            os.environ.pop("OPENCV_LOG_LEVEL", None)
        else:
            os.environ["OPENCV_LOG_LEVEL"] = prev_env


@dataclass
class CameraInfo:
    index: int
    backend: int
    backend_name: str
    width: int
    height: int
    has_barcode: bool
    barcode_increasing: bool
    negotiated_width: int = 0
    negotiated_height: int = 0
    negotiated_fps: float = 0.0


def _probe_index(index: int, backend: int, backend_name: str) -> Optional[CameraInfo]:
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        cap.release()
        return None
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    seqs: list[Optional[int]] = []
    for _ in range(3):
        ok, frame = cap.read()
        if not ok or frame is None:
            seqs.append(None)
            continue
        seqs.append(read_seq_barcode(frame))
    cap.release()
    valid = [s for s in seqs if isinstance(s, int)]
    has_barcode = len(valid) >= 2
    increasing = has_barcode and all(valid[j] < valid[j + 1] for j in range(len(valid) - 1))
    return CameraInfo(
        index=index,
        backend=backend,
        backend_name=backend_name,
        width=w,
        height=h,
        has_barcode=has_barcode,
        barcode_increasing=increasing,
    )


def _try_negotiate(
    index: int,
    backend: int,
    width: int,
    height: int,
    fps: int,
) -> tuple[bool, int, int, float]:
    """Open device, request W/H/FPS, optionally read one frame. Returns ok, w, h, fps."""
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        cap.release()
        return False, 0, 0, 0.0
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:  # noqa: BLE001
        pass
    aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    afps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        # No HDMI signal yet — still accept if driver reports HD-capable dimensions.
        return aw >= min(width, 1280) and ah >= min(height, 720), aw, ah, afps
    return True, aw, ah, afps


def list_capture_devices(max_index: int = 10) -> list[CameraInfo]:
    """Fast scan: DirectShow first (Windows), then MSMF only if needed."""
    found: list[CameraInfo] = []
    with quiet_opencv():
        for i in range(max_index):
            info = _probe_index(i, cv2.CAP_DSHOW, "DSHOW")
            if info:
                found.append(info)
        if not found:
            for i in range(max_index):
                info = _probe_index(i, cv2.CAP_MSMF, "MSMF")
                if info:
                    found.append(info)
    return found


def find_capture_card_index(
    preferred: Optional[int] = None,
    width: int = 1920,
    height: int = 1080,
    fps: int = 120,
    timeout_s: float = 8.0,
) -> tuple[int, int]:
    """
    Return (device_index, backend) for an Elgato / UVC capture card.

    Heuristics:
    - Skip pyvirtualcam/OBS devices (monotonic barcode).
    - Prefer devices that negotiate >=1280x720 (capture cards vs laptop webcam).
    - Prefer index > 0 when scores tie (built-in webcam is usually index 0).
    """
    if preferred is not None:
        with quiet_opencv():
            for backend, name in ((cv2.CAP_DSHOW, "DSHOW"), (cv2.CAP_MSMF, "MSMF")):
                ok, aw, ah, afps = _try_negotiate(preferred, backend, width, height, fps)
                if ok:
                    logger.info(
                        "Capture card at index=%d backend=%s (%dx%d@%.1f)",
                        preferred, name, aw, ah, afps,
                    )
                    return preferred, backend
        raise RuntimeError(
            f"Could not open capture card at device index={preferred}. "
            "Check Elgato is plugged in, Game Capture / 4K Capture utility is closed, "
            "and an HDMI signal is connected."
        )

    deadline = time.time() + timeout_s
    last_err = "no devices"
    while time.time() < deadline:
        best: Optional[tuple[int, int, int]] = None  # score, index, backend
        with quiet_opencv():
            for info in list_capture_devices(max_index=10):
                if info.barcode_increasing:
                    continue
                for backend, name in ((cv2.CAP_DSHOW, "DSHOW"), (cv2.CAP_MSMF, "MSMF")):
                    ok, aw, ah, afps = _try_negotiate(
                        info.index, backend, width, height, fps
                    )
                    if not ok:
                        continue
                    # Auto-detect: never treat a typical laptop webcam (index 0,
                    # sub-FHD) as an Elgato. Require FHD or a non-zero index with HD.
                    if preferred is None:
                        if info.index == 0 and (aw < 1920 or ah < 1080):
                            continue
                        if aw < 1280 or ah < 720:
                            continue
                    score = aw * ah + int(afps * 100)
                    # Prefer true FHD when requested.
                    if aw >= width and ah >= height:
                        score += 2_000_000
                    elif aw >= 1920 and ah >= 1080:
                        score += 1_500_000
                    if info.index > 0:
                        score += 300_000
                    if best is None or score > best[0]:
                        best = (score, info.index, backend)
                        logger.debug(
                            "capture card candidate index=%d %s %dx%d@%.1f score=%d",
                            info.index, name, aw, ah, afps, score,
                        )
        if best is not None:
            _, idx, backend = best
            logger.info("Capture card detected index=%d", idx)
            return idx, backend
        last_err = (
            "no Elgato/capture card found "
            "(laptop webcam at index 0 is ignored — plug in the card or pass --capture-index)"
        )
        time.sleep(0.4)

    devices = list_capture_devices(max_index=10)
    non_virtual = [d for d in devices if not d.barcode_increasing]
    hint = ", ".join(
        f"{d.index}/{d.backend_name}({d.width}x{d.height})" for d in non_virtual
    ) or "none"
    raise RuntimeError(
        f"Could not find capture card (Elgato / UVC). {last_err}. "
        f"Devices seen: {hint}. "
        "Try --device-index N in the GUI, or set POC1_CAPTURE_INDEX."
    )


def find_virtualcam_index(
    preferred: Optional[int] = None,
    timeout_s: float = 8.0,
) -> tuple[int, int]:
    """
    Return (device_index, backend) for a live pyvirtualcam/OBS feed.
    Prefers a device whose burned-in barcode is monotonically increasing.
    """
    deadline = time.time() + timeout_s
    last: list[CameraInfo] = []
    while time.time() < deadline:
        # Fast path: check index 1 then 0 on DSHOW only (typical OBS layout).
        with quiet_opencv():
            for i in (1, 0, 2, 3):
                info = _probe_index(i, cv2.CAP_DSHOW, "DSHOW")
                if info and info.barcode_increasing:
                    logger.info(
                        "Virtual cam detected index=%d backend=%s (%dx%d)",
                        info.index, info.backend_name, info.width, info.height,
                    )
                    return info.index, info.backend
                if info:
                    last = [c for c in last if c.index != info.index] + [info]

        if preferred is not None:
            for cam in last:
                if cam.index == preferred:
                    return cam.index, cam.backend
        time.sleep(0.25)

    for cam in last:
        if cam.index > 0:
            logger.warning(
                "Using capture index=%d backend=%s without barcode confirm",
                cam.index, cam.backend_name,
            )
            return cam.index, cam.backend
    raise RuntimeError(
        "Could not find virtual camera device. "
        "Keep `python -m poc1.virtual_cam_sender` running, or use "
        "--source virtualcam which auto-starts it. Devices seen: "
        + (", ".join(f"{c.index}/{c.backend_name}" for c in last) if last else "none")
    )
