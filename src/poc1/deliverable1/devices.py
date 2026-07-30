"""
R1 / R2: list connected cameras (UVC + RealSense) and their stream modes.

Uses existing helpers where possible; adds cross-platform UVC probing and
RealSense profile enumeration via pyrealsense2 without changing those modules.
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from typing import Any, Optional

import cv2
import numpy as np

from poc1.device_enum import quiet_opencv
from poc1.frame_source import FakeFrameSource
from poc1.pipeline import CvCaptureSource
from poc1.realsense_source import (
    create_realsense_source,
    list_realsense_devices,
    realsense_available,
)

logger = logging.getLogger("poc1.d1.devices")

# Sensible presets when the driver does not expose a full mode list.
_UVC_PRESET_MODES: list[tuple[int, int, int, str]] = [
    (1920, 1080, 120, "bgr8"),
    (1920, 1080, 60, "bgr8"),
    (1920, 1080, 30, "bgr8"),
    (1280, 720, 60, "bgr8"),
    (1280, 720, 30, "bgr8"),
    (640, 480, 30, "bgr8"),
    (1920, 1080, 30, "mjpg"),
    (1280, 720, 30, "mjpg"),
    (1280, 720, 30, "yuyv"),
]


@dataclass(frozen=True)
class ConnectedCamera:
    """One physical / logical camera entry for R1."""

    cam_id: str
    kind: str  # "uvc" | "realsense" | "fake"
    name: str
    index: Optional[int] = None
    serial: Optional[str] = None
    backend: Optional[int] = None
    backend_name: str = ""

    def label(self) -> str:
        extra = []
        if self.serial:
            extra.append(f"sn={self.serial}")
        if self.index is not None:
            extra.append(f"#{self.index}")
        if self.backend_name:
            extra.append(self.backend_name)
        suffix = f" ({', '.join(extra)})" if extra else ""
        return f"[{self.kind}] {self.name}{suffix}"


@dataclass(frozen=True)
class StreamMode:
    """One selectable configuration for R2."""

    width: int
    height: int
    fps: int
    pixel_format: str  # bgr8 | mjpg | yuyv | z16 | …

    def __post_init__(self) -> None:
        # Some Windows UVC drivers report -1/0 for unknown FPS. Never pass that
        # sentinel to VideoWriter; 30fps is the safe recording fallback.
        if self.width <= 0:
            object.__setattr__(self, "width", 640)
        if self.height <= 0:
            object.__setattr__(self, "height", 480)
        if self.fps <= 0:
            object.__setattr__(self, "fps", 30)
        if not self.pixel_format:
            object.__setattr__(self, "pixel_format", "bgr8")

    def label(self) -> str:
        return f"{self.width}x{self.height}@{self.fps} {self.pixel_format}"


def _opencv_backends() -> list[tuple[int, str]]:
    backends: list[tuple[int, str]] = []
    if sys.platform == "win32":
        backends = [(cv2.CAP_DSHOW, "DSHOW"), (cv2.CAP_MSMF, "MSMF")]
    elif sys.platform == "darwin":
        if hasattr(cv2, "CAP_AVFOUNDATION"):
            backends.append((cv2.CAP_AVFOUNDATION, "AVFOUNDATION"))
        backends.append((cv2.CAP_ANY, "ANY"))
    else:
        if hasattr(cv2, "CAP_V4L2"):
            backends.append((cv2.CAP_V4L2, "V4L2"))
        backends.append((cv2.CAP_ANY, "ANY"))
    return backends


def _probe_uvc_index(index: int, backend: int, backend_name: str) -> Optional[ConnectedCamera]:
    with quiet_opencv():
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            cap.release()
            return None
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        cap.release()
    name = f"UVC camera {index}"
    if w > 0 and h > 0:
        name = f"UVC #{index} ({w}x{h}" + (f"@{fps:.0f}" if fps > 0 else "") + ")"
    # Heuristic: high-res non-zero index is often a capture card (Elgato).
    if index > 0 and w >= 1280 and h >= 720:
        name = f"UVC/capture #{index} ({w}x{h})"
    return ConnectedCamera(
        cam_id=f"uvc:{index}:{backend_name}",
        kind="uvc",
        name=name,
        index=index,
        backend=backend,
        backend_name=backend_name,
    )


def _probe_uvc_index_safe(
    index: int,
    backend: int,
    backend_name: str,
    timeout_s: float = 2.0,
) -> Optional[ConnectedCamera]:
    """OpenCV can hang when another process owns the camera; bound the wait."""
    from concurrent.futures import ThreadPoolExecutor
    from concurrent.futures import TimeoutError as FuturesTimeout

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_probe_uvc_index, index, backend, backend_name)
        try:
            return future.result(timeout=timeout_s)
        except FuturesTimeout:
            logger.warning(
                "UVC probe timed out index=%d backend=%s (device busy or locked)",
                index, backend_name,
            )
            return None
        except Exception as exc:  # noqa: BLE001
            logger.debug("UVC probe failed index=%d: %s", index, exc)
            return None


def list_uvc_cameras(max_index: int = 6) -> list[ConnectedCamera]:
    """R1 — enumerate OpenCV/UVC devices (webcam, Elgato, virtual cam, …)."""
    found: list[ConnectedCamera] = []
    seen_indices: set[int] = set()
    for backend, bname in _opencv_backends():
        for i in range(max_index):
            if i in seen_indices:
                continue
            cam = _probe_uvc_index_safe(i, backend, bname)
            if cam is None:
                continue
            seen_indices.add(i)
            found.append(cam)
        if found and sys.platform == "win32":
            # Prefer first successful backend on Windows (matches POC-1 DSHOW-first).
            break
    return found


def list_realsense_cameras() -> list[ConnectedCamera]:
    """R1 — enumerate Intel RealSense devices via pyrealsense2."""
    out: list[ConnectedCamera] = []
    if not realsense_available():
        return out
    for dev in list_realsense_devices():
        serial = dev.get("serial") or ""
        name = dev.get("name") or "RealSense"
        pl = dev.get("product_line") or ""
        label = f"{name}" + (f" [{pl}]" if pl else "")
        out.append(
            ConnectedCamera(
                cam_id=f"realsense:{serial}",
                kind="realsense",
                name=label,
                serial=serial,
            )
        )
    return out


def list_all_cameras(
    *,
    include_fake: bool = False,
    probe_uvc: bool = True,
    probe_realsense: bool = True,
) -> list[ConnectedCamera]:
    """R1 — unified UVC + RealSense inventory."""
    cams: list[ConnectedCamera] = []
    if probe_uvc:
        cams.extend(list_uvc_cameras())
    if probe_realsense:
        cams.extend(list_realsense_cameras())
    if include_fake:
        cams.append(
            ConnectedCamera(
                cam_id="fake:0",
                kind="fake",
                name="Synthetic fake A (color bars)",
                index=0,
            )
        )
        cams.append(
            ConnectedCamera(
                cam_id="fake:1",
                kind="fake",
                name="Synthetic fake B (color bars)",
                index=1,
            )
        )
    return cams


def _rs_format_name(fmt: Any) -> str:
    try:
        import pyrealsense2 as rs

        mapping = {
            rs.format.bgr8: "bgr8",
            rs.format.rgb8: "rgb8",
            rs.format.yuyv: "yuyv",
            rs.format.z16: "z16",
            rs.format.y8: "y8",
        }
        return mapping.get(fmt, str(fmt).split(".")[-1])
    except Exception:  # noqa: BLE001
        return "unknown"


def list_realsense_modes(serial: Optional[str] = None) -> list[StreamMode]:
    """R2 — stream profiles advertised by a RealSense device (color preferred)."""
    if not realsense_available():
        return [
            StreamMode(1280, 720, 30, "bgr8"),
            StreamMode(640, 480, 30, "bgr8"),
        ]
    import pyrealsense2 as rs

    ctx = rs.context()
    device = None
    for dev in ctx.query_devices():
        sn = dev.get_info(rs.camera_info.serial_number)
        if serial is None or sn == serial:
            device = dev
            break
    if device is None:
        return [StreamMode(1280, 720, 30, "bgr8")]

    modes: list[StreamMode] = []
    seen: set[tuple[int, int, int, str]] = set()
    try:
        sensors = list(device.query_sensors())
    except Exception:  # noqa: BLE001
        sensors = []
    for sensor in sensors:
        try:
            profiles = sensor.get_stream_profiles()
        except Exception:  # noqa: BLE001
            continue
        for p in profiles:
            try:
                vp = p.as_video_stream_profile()
            except Exception:  # noqa: BLE001
                continue
            try:
                if vp.stream_type() != rs.stream.color:
                    continue
            except Exception:  # noqa: BLE001
                continue
            w, h, fps = vp.width(), vp.height(), int(vp.fps())
            fmt = _rs_format_name(vp.format())
            # Deliverable 1 can convert these color formats to the BGR ndarray
            # expected by the shared processor/preview pipeline.
            if fmt not in {"bgr8", "rgb8", "yuyv", "y8"}:
                continue
            key = (w, h, fps, fmt)
            if key in seen:
                continue
            seen.add(key)
            modes.append(StreamMode(w, h, fps, fmt))

    if not modes:
        modes = [
            StreamMode(1280, 720, 30, "bgr8"),
            StreamMode(640, 480, 30, "bgr8"),
        ]
    modes.sort(key=lambda m: (m.width * m.height, m.fps), reverse=True)
    return modes


def list_uvc_modes(camera: ConnectedCamera) -> list[StreamMode]:
    """R2 — preset + lightly probed modes for a UVC device."""
    # Prefer modest defaults first. Many UVC webcams cannot sustain FHD@120,
    # and selecting that as the first option caused confusing recordings.
    preferred = [
        StreamMode(1280, 720, 30, "bgr8"),
        StreamMode(640, 480, 30, "bgr8"),
        StreamMode(1280, 720, 60, "bgr8"),
        StreamMode(1920, 1080, 30, "bgr8"),
        StreamMode(1920, 1080, 60, "bgr8"),
        StreamMode(1920, 1080, 120, "bgr8"),
        StreamMode(1280, 720, 30, "mjpg"),
        StreamMode(1920, 1080, 30, "mjpg"),
        StreamMode(1280, 720, 30, "yuyv"),
    ]
    modes = list(preferred)
    for w, h, fps, fmt in _UVC_PRESET_MODES:
        candidate = StreamMode(w, h, fps, fmt)
        if candidate not in modes:
            modes.append(candidate)
    if camera.index is None or camera.backend is None:
        return modes
    # Prepend the currently negotiated mode if we can open briefly.
    with quiet_opencv():
        cap = cv2.VideoCapture(camera.index, camera.backend)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            reported_fps = int(cap.get(cv2.CAP_PROP_FPS) or 0)
            fps = reported_fps if reported_fps > 0 else 30
            cap.release()
            if w > 0 and h > 0:
                current = StreamMode(w, h, fps, "bgr8")
                if current not in modes:
                    modes.insert(0, current)
    return modes


def list_stream_modes(camera: ConnectedCamera) -> list[StreamMode]:
    """R2 — modes for any connected camera kind."""
    if camera.kind == "realsense":
        return list_realsense_modes(camera.serial)
    if camera.kind == "fake":
        return [
            StreamMode(1920, 1080, 120, "bgr8"),
            StreamMode(1280, 720, 120, "bgr8"),
            StreamMode(1280, 720, 60, "bgr8"),
            StreamMode(640, 480, 30, "bgr8"),
        ]
    return list_uvc_modes(camera)


def _apply_uvc_fourcc(cap: cv2.VideoCapture, pixel_format: str) -> None:
    fmt = pixel_format.lower()
    fourcc_map = {
        "mjpg": "MJPG",
        "mjpeg": "MJPG",
        "yuyv": "YUY2",
        "yuy2": "YUY2",
        "bgr8": None,
        "rgb8": None,
    }
    code = fourcc_map.get(fmt)
    if code:
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*code))
        except Exception:  # noqa: BLE001
            pass


class FormattedUvcSource(CvCaptureSource):
    """UVC/OpenCV source that applies a requested pixel format (R2)."""

    def __init__(
        self,
        device_index: int,
        width: int,
        height: int,
        fps: int,
        backend,
        pixel_format: str = "bgr8",
    ):
        safe_fps = fps if fps > 0 else 30
        super().__init__(device_index, width, height, safe_fps, backend=backend)
        self.pixel_format = pixel_format

    def start(self) -> None:
        with quiet_opencv():
            self._cap = cv2.VideoCapture(self.device_index, self._backend)
            if not self._cap.isOpened() and self._backend == cv2.CAP_DSHOW:
                self._cap.release()
                self._cap = cv2.VideoCapture(self.device_index, cv2.CAP_MSMF)
                self._backend = cv2.CAP_MSMF
            if not self._cap.isOpened() and self._backend != cv2.CAP_ANY:
                self._cap.release()
                self._cap = cv2.VideoCapture(self.device_index, cv2.CAP_ANY)
                self._backend = cv2.CAP_ANY
            if not self._cap.isOpened():
                raise RuntimeError(
                    f"Could not open UVC device index={self.device_index}"
                )
            _apply_uvc_fourcc(self._cap, self.pixel_format)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            try:
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:  # noqa: BLE001
                pass
            reported_fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 0)
            self.actual_fps = reported_fps if reported_fps > 0 else 0.0
            self.actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH) or self.width)
            self.actual_height = int(
                self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or self.height
            )
            if self.actual_width > 0:
                self.width = self.actual_width
            if self.actual_height > 0:
                self.height = self.actual_height
        logger.info(
            "D1 UVC source: idx=%d %dx%d@%d fmt=%s (reports %.1ffps)",
            self.device_index, self.width, self.height, self.target_fps,
            self.pixel_format, self.actual_fps,
        )


class ConfiguredRealSenseSource:
    """RealSense color source honoring the selected resolution/FPS/format."""

    mode = "hardware"
    allow_fps_remux = False

    def __init__(
        self,
        serial: str,
        width: int,
        height: int,
        fps: int,
        pixel_format: str,
    ) -> None:
        self.serial = serial
        self.width = width
        self.height = height
        self.target_fps = fps if fps > 0 else 30
        self.pixel_format = pixel_format.lower()
        self.bag_path: Optional[Any] = None
        self._bag_path: Optional[Any] = None
        self._pipeline: Optional[Any] = None

    @staticmethod
    def _rs_format(rs, name: str):
        formats = {
            "bgr8": rs.format.bgr8,
            "rgb8": rs.format.rgb8,
            "yuyv": rs.format.yuyv,
            "y8": rs.format.y8,
        }
        if name not in formats:
            raise ValueError(f"Unsupported RealSense color format: {name}")
        return formats[name]

    def start(self) -> None:
        if not realsense_available():
            raise RuntimeError(
                "pyrealsense2 is not installed. Run: "
                "uv sync --extra dev --extra realsense"
            )
        import pyrealsense2 as rs

        pipeline = rs.pipeline()
        config = rs.config()
        if self.serial:
            config.enable_device(self.serial)
        rs_format = self._rs_format(rs, self.pixel_format)
        config.enable_stream(
            rs.stream.color,
            self.width,
            self.height,
            rs_format,
            self.target_fps,
        )
        if self.bag_path:
            config.enable_record_to_file(str(self.bag_path))
        try:
            profile = pipeline.start(config)
        except Exception as exc:
            try:
                pipeline.stop()
            except Exception:  # noqa: BLE001
                pass
            raise RuntimeError(
                "RealSense rejected "
                f"{self.width}x{self.height}@{self.target_fps} "
                f"{self.pixel_format}: {exc}"
            ) from exc

        color = profile.get_stream(rs.stream.color).as_video_stream_profile()
        self.width = color.width()
        self.height = color.height()
        self.target_fps = int(color.fps())
        self._pipeline = pipeline
        logger.info(
            "D1 RealSense: serial=%s %dx%d@%d fmt=%s bag=%s",
            self.serial,
            self.width,
            self.height,
            self.target_fps,
            self.pixel_format,
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
            color = frames.get_color_frame()
        except Exception:  # noqa: BLE001
            return None
        if not color:
            return None
        frame = np.asanyarray(color.get_data())
        if self.pixel_format == "rgb8":
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        elif self.pixel_format == "yuyv":
            frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_YUY2)
        elif self.pixel_format == "y8":
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        return np.ascontiguousarray(frame)


def build_frame_source(
    camera: ConnectedCamera,
    mode: StreamMode,
    *,
    allow_simulate_realsense: bool = True,
):
    """
    Build a POC-1-compatible frame source for the given camera + mode.

    RealSense color streams that are not bgr8 are still opened as bgr8 when
    possible (OpenCV/pipeline expect BGR ndarrays); depth-only formats fall back.
    """
    if camera.kind == "fake":
        return FakeFrameSource(
            width=mode.width, height=mode.height, target_fps=mode.fps
        )

    if camera.kind == "realsense":
        if camera.serial and realsense_available():
            return ConfiguredRealSenseSource(
                serial=camera.serial,
                width=mode.width,
                height=mode.height,
                fps=mode.fps,
                pixel_format=mode.pixel_format,
            )
        # Kept for API/test callers that explicitly allow simulation.
        return create_realsense_source(
            width=mode.width,
            height=mode.height,
            fps=mode.fps,
            serial=camera.serial,
            allow_simulate=allow_simulate_realsense,
        )

    backend = camera.backend
    if backend is None:
        backend = _opencv_backends()[0][0]
    src = FormattedUvcSource(
        device_index=int(camera.index or 0),
        width=mode.width,
        height=mode.height,
        fps=mode.fps,
        backend=backend,
        pixel_format=mode.pixel_format,
    )
    # Keep mismatch detection enabled for every UVC device. A non-zero index
    # may be Elgato, another webcam, or a virtual camera; index alone is not
    # reliable enough to suppress actual-vs-configured FPS warnings.
    src.allow_fps_remux = True
    return src
