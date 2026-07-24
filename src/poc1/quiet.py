"""Silence OpenCV / FFmpeg console noise as early as possible."""
from __future__ import annotations

import logging
import os
import warnings


def silence_opencv() -> None:
    os.environ["OPENCV_LOG_LEVEL"] = "FATAL"
    os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "quiet")
    warnings.filterwarnings("ignore", message=".*OpenCV.*")
    try:
        import cv2

        if hasattr(cv2, "utils") and hasattr(cv2.utils, "logging"):
            cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
        elif hasattr(cv2, "setLogLevel") and hasattr(cv2, "LOG_LEVEL_SILENT"):
            cv2.setLogLevel(cv2.LOG_LEVEL_SILENT)
        elif hasattr(cv2, "setLogLevel") and hasattr(cv2, "LOG_LEVEL_ERROR"):
            cv2.setLogLevel(cv2.LOG_LEVEL_ERROR)
    except Exception:
        pass


def configure_app_logging(level: int = logging.INFO) -> None:
    silence_opencv()
    # Keep our app logs, but don't let third-party spam WARN.
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    logging.getLogger("poc1.recorder").setLevel(logging.INFO)
