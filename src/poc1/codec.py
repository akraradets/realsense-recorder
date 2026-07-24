"""
Pick a working OpenCV VideoWriter fourcc. Result is cached so Record starts fast
and we never probe broken H.264 paths that spam the console.
"""
from __future__ import annotations

import logging
import os
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from poc1.device_enum import quiet_opencv

logger = logging.getLogger("poc1.codec")


def _openh264_present() -> bool:
    names = ("openh264-2.5.0-win64.dll", "openh264-2.4.1-win64.dll", "openh264-1.8.0-win64.dll")
    search = [Path.cwd()]
    for base in (Path(sys.prefix), Path(sys.prefix) / "Scripts"):
        if base.is_dir():
            search.append(base)
    for p in os.environ.get("PATH", "").split(os.pathsep):
        if not p:
            continue
        base = Path(p)
        try:
            if base.is_dir():
                search.append(base)
        except OSError:
            continue
    for base in search:
        for name in names:
            try:
                if (base / name).is_file():
                    return True
            except OSError:
                continue
    return False


def _probe_fourcc(fourcc_str: str, width: int = 64, height: int = 64, fps: int = 30) -> bool:
    tmp = Path(tempfile.gettempdir()) / f"poc1_codec_probe_{fourcc_str}_{os.getpid()}.mp4"
    with quiet_opencv():
        try:
            fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
            writer = cv2.VideoWriter(str(tmp), fourcc, fps, (width, height))
            if not writer.isOpened():
                writer.release()
                return False
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            writer.write(frame)
            writer.release()
            if not tmp.exists() or tmp.stat().st_size < 32:
                return False
            cap = cv2.VideoCapture(str(tmp))
            ok = cap.isOpened()
            if ok:
                ok, _ = cap.read()
            cap.release()
            return bool(ok)
        except Exception:
            return False
        finally:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass


@lru_cache(maxsize=1)
def choose_best_fourcc() -> tuple[str, str]:
    """
    Prefer mp4v (always works, no OpenH264 DLL spam). Only try H.264 if the
    Cisco DLL is actually present.
    """
    if _openh264_present():
        for fourcc_str, label in (("avc1", "H.264 (avc1)"), ("H264", "H.264 (H264)")):
            if _probe_fourcc(fourcc_str):
                logger.info("Using codec %s", label)
                return fourcc_str, label
    # Default: skip failed H.264 probes entirely.
    if _probe_fourcc("mp4v"):
        logger.info("Using codec MPEG-4 (mp4v)")
        return "mp4v", "MPEG-4 (mp4v)"
    return "mp4v", "MPEG-4 (mp4v)"
