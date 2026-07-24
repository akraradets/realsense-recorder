"""
Virtual-camera FrameSource.

- Default (GUI): instant loopback — same synthetic frames as the sender, no wait.
- require_os=True (suite): auto-start sender + open the real OBS/DirectShow device.
"""
from __future__ import annotations

import logging
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from poc1.device_enum import find_virtualcam_index, quiet_opencv
from poc1.frame_source import FakeFrameSource, embed_status_banner

logger = logging.getLogger("poc1.virtualcam_source")


@dataclass
class VirtualCamCaptureSource:
    width: int = 1920
    height: int = 1080
    target_fps: int = 120
    preferred_index: Optional[int] = None
    auto_start_sender: bool = True
    pattern: str = "bars"
    # False = instant preview (GUI). True = real OS capture (suite / proof).
    require_os: bool = False

    _cap: Optional[cv2.VideoCapture] = field(default=None, init=False, repr=False)
    _sender: Optional[subprocess.Popen] = field(default=None, init=False, repr=False)
    _loopback: Optional[FakeFrameSource] = field(default=None, init=False, repr=False)
    device_index: int = field(default=-1, init=False)
    backend_used: int = field(default=cv2.CAP_DSHOW, init=False)
    capture_mode: str = field(default="loopback", init=False)

    def start(self) -> None:
        if not self.require_os:
            # Instant path for GUI — no blank screen, no 3–10s wait.
            self._loopback = FakeFrameSource(
                width=self.width, height=self.height, target_fps=self.target_fps,
                pattern=self.pattern,
            )
            self._loopback.start()
            self.capture_mode = "loopback"
            logger.info(
                "VirtualCamCaptureSource: instant loopback %dx%d@%d "
                "(use require_os=True / poc1.run_poc1_tests for real OBS capture)",
                self.width, self.height, self.target_fps,
            )
            return

        if self.auto_start_sender:
            self._start_sender()

        try:
            with quiet_opencv():
                idx, backend = find_virtualcam_index(
                    preferred=self.preferred_index, timeout_s=8.0
                )
                self.device_index = idx
                self.backend_used = backend
                self._cap = cv2.VideoCapture(idx, backend)
                if not self._cap.isOpened():
                    raise RuntimeError(f"VideoCapture failed for index={idx}")
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self._cap.set(cv2.CAP_PROP_FPS, self.target_fps)
                try:
                    self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:  # noqa: BLE001
                    pass
                aw = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH) or self.width)
                ah = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or self.height)
                if aw > 0:
                    self.width = aw
                if ah > 0:
                    self.height = ah
                ok, frame = self._cap.read()
                if not ok or frame is None:
                    raise RuntimeError(f"Virtual cam index={idx} read() failed")
        except Exception:
            self._stop_sender()
            if self._cap is not None:
                self._cap.release()
                self._cap = None
            raise

        self.capture_mode = "os"
        logger.info(
            "VirtualCamCaptureSource: OS index=%d %dx%d@%d",
            self.device_index, self.width, self.height, self.target_fps,
        )

    def _start_sender(self) -> None:
        cmd = [
            sys.executable, "-m", "poc1.virtual_cam_sender",
            "--width", str(self.width),
            "--height", str(self.height),
            "--fps", str(min(self.target_fps, 60)),  # sender: keep OBS happy
            "--pattern", self.pattern,
        ]
        self._sender = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        # Give OBS/pyvirtualcam a moment to register the device.
        time.sleep(1.5)

    def _stop_sender(self) -> None:
        if self._sender is None:
            return
        try:
            self._sender.terminate()
            self._sender.wait(timeout=3)
        except Exception:  # noqa: BLE001
            try:
                self._sender.kill()
            except Exception:  # noqa: BLE001
                pass
        self._sender = None

    def stop(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self._loopback is not None:
            self._loopback.stop()
            self._loopback = None
        self._stop_sender()

    def reset_sequence(self) -> None:
        if self._loopback is not None:
            self._loopback.reset_sequence()

    def read(self) -> Optional[np.ndarray]:
        if self._loopback is not None:
            frame = self._loopback.read()
            if frame is not None:
                embed_status_banner(
                    frame, "LOOPBACK — not OS camera (start sender for real OBS)"
                )
            return frame
        if self._cap is None:
            return None
        with quiet_opencv():
            ok, frame = self._cap.read()
        return frame if ok else None
