import os
import platform
import re
import subprocess
import cv2
import logging

logger = logging.getLogger(__name__)
# Suppress OpenCV driver warning logs during probing
# os.environ["OPENCV_LOG_LEVEL"] = "ERROR"


def get_native_backend():
    """Detects OS and returns the optimal OpenCV capture backend."""
    sys_os = platform.system()
    logger.debug(f"Detected OS: {sys_os}")
    if sys_os == "Windows":
        return cv2.CAP_DSHOW
    elif sys_os == "Darwin":  # macOS
        return cv2.CAP_AVFOUNDATION
    else:  # Linux / Unix
        return cv2.CAP_V4L2