from realsense_recorder.utils import get_native_backend
from .os_helper import get_v4l2_capabilities
import cv2
import platform
import os

def list_camera_configurations(max_tested_indices=5):
    """Scans for cameras across Windows, macOS, and Linux."""
    current_os = platform.system()
    devices = []

    for idx in range(max_tested_indices):
        # Linux fast path: query v4l2 hardware directly
        if current_os == "Linux":
            dev_path = f"/dev/video{idx}"
            v4l2_caps = get_v4l2_capabilities(dev_path) if os.path.exists(dev_path) else None
            if v4l2_caps:
                devices.append(
                    {
                        "index": idx,
                        "method": "V4L2 Hardware Query",
                        "capabilities": v4l2_caps,
                    }
                )
                continue

        # Standard Cross-Platform probing (Windows, macOS, Linux fallback)
        cv_caps = probe_opencv_capabilities(idx)
        if cv_caps:
            backend_names = {"Windows": "DirectShow", "Darwin": "AVFoundation", "Linux": "V4L2"}
            backend_str = backend_names.get(current_os, "OpenCV")

            devices.append(
                {
                    "index": idx,
                    "method": f"OpenCV Probing ({backend_str})",
                    "capabilities": cv_caps,
                }
            )

    return devices

def probe_opencv_capabilities(camera_index):
    """Cross-platform probing logic using OS-native backend."""
    backend = get_native_backend()
    cap = cv2.VideoCapture(camera_index, backend)

    if not cap.isOpened():
        return None

    # Request MJPEG format to unlock higher resolutions and framerates
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    test_resolutions = [
        (320, 240),
        (640, 360),
        (640, 480),
        (800, 600),
        (1024, 768),
        (1280, 720),
        (1280, 800),
        (1600, 1200),
        (1920, 1080),
        (2560, 1440),
        (3840, 2160),
    ]

    test_fps = [15.0, 24.0, 30.0, 60.0, 120.0]
    supported_configs = []

    for width, height in test_resolutions:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Check if resolution request was accepted
        if (actual_w, actual_h) == (width, height):
            supported_fps = []
            for target_fps in test_fps:
                cap.set(cv2.CAP_PROP_FPS, target_fps)
                actual_fps = round(cap.get(cv2.CAP_PROP_FPS), 2)
                if actual_fps > 0 and actual_fps not in supported_fps:
                    supported_fps.append(actual_fps)

            supported_configs.append(
                {"resolution": f"{width}x{height}", "framerates": sorted(supported_fps)}
            )

    cap.release()
    return supported_configs if supported_configs else None
