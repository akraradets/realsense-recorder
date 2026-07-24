"""
Intel RealSense frame source (optional dependency: pyrealsense2).

When no physical device is connected, SimulatedRealSenseSource keeps the
POC-1 RealSense *code path* testable (same pipeline, barcode proof) and
switches to real hardware automatically when a device appears.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Union

import numpy as np

from poc1.frame_source import FakeFrameSource, embed_status_banner

logger = logging.getLogger("poc1.realsense")


def realsense_available() -> bool:
    try:
        import pyrealsense2 as rs  # noqa: F401
        return True
    except ImportError:
        return False


def list_realsense_devices() -> list[dict[str, str]]:
    if not realsense_available():
        return []
    import pyrealsense2 as rs

    ctx = rs.context()
    devices = []
    for dev in ctx.query_devices():
        devices.append({
            "name": dev.get_info(rs.camera_info.name),
            "serial": dev.get_info(rs.camera_info.serial_number),
            "product_line": (
                dev.get_info(rs.camera_info.product_line)
                if dev.supports(rs.camera_info.product_line) else ""
            ),
        })
    return devices


@dataclass
class SimulatedRealSenseSource:
    """
    Stand-in used when pyrealsense2 is installed but no camera is plugged in.
    Uses the same FakeFrameSource pacing/barcode so no-drop proofs still work.
    """

    width: int = 1280
    height: int = 720
    target_fps: int = 30
    serial: Optional[str] = "SIMULATED"
    mode: str = field(default="simulated", init=False)
    allow_fps_remux: bool = False
    _inner: FakeFrameSource = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._inner = FakeFrameSource(
            width=self.width, height=self.height, target_fps=self.target_fps
        )

    def start(self) -> None:
        logger.info(
            "No RealSense device connected — using SimulatedRealSenseSource "
            "(%dx%d@%d). Plug in hardware for the real SDK path.",
            self.width, self.height, self.target_fps,
        )
        self._inner.start()

    def stop(self) -> None:
        self._inner.stop()

    def read(self) -> Optional[np.ndarray]:
        frame = self._inner.read()
        if frame is not None:
            embed_status_banner(frame, "SIMULATED — no RealSense connected")
        return frame

    def reset_sequence(self) -> None:
        self._inner.reset_sequence()


@dataclass
class RealSenseFrameSource:
    """Color stream from a RealSense device via pyrealsense2."""

    width: int = 1920
    height: int = 1080
    target_fps: int = 30
    serial: Optional[str] = None
    enable_depth: bool = False
    bag_path: Optional[Any] = None
    mode: str = field(default="hardware", init=False)

    _pipeline: Any = field(default=None, init=False, repr=False)
    _bag_path: Optional[Any] = field(default=None, init=False, repr=False)

    def start(self) -> None:
        if not realsense_available():
            raise RuntimeError(
                "pyrealsense2 is not installed. Install with: "
                "uv sync --extra realsense   (or pip install pyrealsense2)"
            )
        import pyrealsense2 as rs

        self._pipeline = rs.pipeline()
        config = rs.config()
        if self.serial:
            config.enable_device(self.serial)

        config.enable_stream(
            rs.stream.color, self.width, self.height, rs.format.bgr8, self.target_fps
        )
        if self.enable_depth:
            config.enable_stream(
                rs.stream.depth, self.width, self.height, rs.format.z16, self.target_fps
            )
        if self.bag_path:
            config.enable_record_to_file(str(self.bag_path))

        try:
            profile = self._pipeline.start(config)
        except RuntimeError as exc:
            logger.warning(
                "RealSense rejected %dx%d@%d (%s); trying 1280x720@30",
                self.width, self.height, self.target_fps, exc,
            )
            config = rs.config()
            if self.serial:
                config.enable_device(self.serial)
            self.width, self.height, self.target_fps = 1280, 720, 30
            config.enable_stream(
                rs.stream.color, self.width, self.height, rs.format.bgr8, self.target_fps
            )
            if self.bag_path:
                config.enable_record_to_file(str(self.bag_path))
            profile = self._pipeline.start(config)

        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        self.width = color_profile.width()
        self.height = color_profile.height()
        self.target_fps = int(color_profile.fps())
        self.mode = "hardware"
        logger.info(
            "RealSenseFrameSource started: %dx%d@%d serial=%s bag=%s",
            self.width, self.height, self.target_fps, self.serial or "auto",
            bool(self.bag_path),
        )

    def stop(self) -> None:
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:  # noqa: BLE001
                pass
            self._pipeline = None

    def read(self) -> Optional[np.ndarray]:
        if self._pipeline is None:
            return None
        try:
            frames = self._pipeline.wait_for_frames(timeout_ms=1000)
        except Exception:  # noqa: BLE001
            return None
        color = frames.get_color_frame()
        if not color:
            return None
        return np.ascontiguousarray(color.get_data())


def create_realsense_source(
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    serial: Optional[str] = None,
    allow_simulate: bool = True,
) -> Union[RealSenseFrameSource, SimulatedRealSenseSource]:
    """Prefer real hardware; optionally fall back to simulation for POC-1."""
    if not realsense_available():
        if allow_simulate:
            return SimulatedRealSenseSource(width=width, height=height, target_fps=fps)
        raise RuntimeError("pyrealsense2 not installed")

    devices = list_realsense_devices()
    if devices:
        ser = serial or devices[0]["serial"]
        return RealSenseFrameSource(
            width=width, height=height, target_fps=fps, serial=ser
        )
    if allow_simulate:
        return SimulatedRealSenseSource(width=width, height=height, target_fps=fps)
    raise RuntimeError(
        "No RealSense device connected. Plug in the camera, or pass allow_simulate=True."
    )
