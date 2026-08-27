"""
R1 / R2: list connected cameras (UVC + RealSense) and their stream modes.

Uses existing helpers where possible; adds cross-platform UVC probing and
RealSense profile enumeration via pyrealsense2 without changing those modules.
"""
from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from poc1.deliverable1.win_names import (
    classify_capture_name,
    clear_name_cache,
    dshow_open_path,
    dshow_open_paths_for_tag,
    elgato_open_name_paths,
    ffmpeg_available,
    friendly_name_for_index,
    list_windows_capture_names,
    names_are_index_aligned,
)
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
    (1280, 720, 30, "mjpg"),
    (640, 480, 30, "mjpg"),
    (1280, 720, 30, "yuyv"),
    (640, 480, 30, "yuyv"),
    (1280, 720, 30, "bgr8"),
    (640, 480, 30, "bgr8"),
    (1920, 1080, 30, "mjpg"),
    (1920, 1080, 30, "bgr8"),
    (1280, 720, 60, "mjpg"),
    (1920, 1080, 60, "mjpg"),
    (1920, 1080, 120, "mjpg"),
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
    # Windows DirectShow path, e.g. "video=Elgato HD60 S+" (more reliable than index).
    open_path: Optional[str] = None
    device_tag: str = "uvc"  # elgato | realsense-uvc | virtual | uvc | realsense | fake

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
        # Make depth/high-FPS options obvious in the Setup dropdown.
        kind = ""
        if self.pixel_format == "z16":
            kind = " depth"
        elif self.pixel_format == "y8":
            kind = " ir"
        return f"{self.width}x{self.height}@{self.fps} {self.pixel_format}{kind}"


def is_fhd_high_rate(mode: Optional[StreamMode]) -> bool:
    """True for 1920x1080 (or larger) at >=90 fps — Elgato/fake 120, not RealSense @30."""
    if mode is None:
        return False
    return int(mode.width) >= 1920 and int(mode.height) >= 1080 and int(mode.fps) >= 90


def too_many_1080p120(modes: list[Optional[StreamMode]]) -> bool:
    """Two or more 1080p@≥90 encodes in one take — refuse. One 120 + @30/@60 is OK."""
    return sum(1 for m in modes if is_fhd_high_rate(m)) >= 2


def honest_container_fps(rate: float, requested: int) -> int:
    """Map measured delivery rate to an honest MP4 container FPS.

    Never invent 120 from ~60 HDMI. Snap common bands so a short under-count
    (e.g. 55–57 → 60, or 90–99 while settling toward 120 → 120) does not
    stamp a bogus mid value like 55 or 95.
    """
    req = int(requested) if int(requested or 0) > 0 else 30
    r = float(rate or 0.0)
    if r < 5.0:
        return 30 if req >= 90 else max(15, min(req, 60))

    # True / settling high-rate (OBS-like 120). Open-time samples often land
    # ~95 briefly while live delivery is already climbing to ~120 — stamp req.
    if req >= 90 and r >= 90.0:
        return req
    if 100.0 <= r <= 130.0:
        return 120 if req >= 90 else min(req, 120)

    # HDMI / NTSC ~60 — includes short-window undercounts that used to become 55.
    if 48.0 <= r <= 63.0:
        return 60

    if 28.0 <= r < 48.0:
        return 30

    if 20.0 <= r < 28.0:
        return 24

    stamped = int(round(r / 5.0) * 5) or 30
    stamped = max(15, stamped)
    if req >= 90:
        return min(stamped, req)
    return stamped


def elgato_open_profiles(
    wanted_w: int, wanted_h: int, wanted_fps: int, wanted_fmt: str = "mjpg"
) -> list[tuple[int, int, int, str]]:
    """Ordered Elgato open attempts — **same resolution only**.

    Never silently fall back 1920x1080 → 1280x720 (Station A/B regression).
    When the operator picks 1080p120, we only retry that size (FPS variants).
    """
    fmt = "mjpg" if (wanted_fmt or "mjpg").lower() != "mjpg" else "mjpg"
    w, h = int(wanted_w), int(wanted_h)
    fps = int(wanted_fps) if int(wanted_fps) > 0 else 30
    primary = [
        (w, h, fps, fmt),
        (w, h, fps, "mjpg"),
    ]
    # Same WxH only — no cross-resolution fallback.
    same_res_high = [(w, h, 120, "mjpg")] if fps != 120 else []
    same_res_mid = [
        (w, h, 60, "mjpg"),
        (w, h, 50, "mjpg"),
        (w, h, 30, "mjpg"),
    ]
    if fps >= 90:
        profiles = primary + same_res_high + same_res_mid
    else:
        profiles = primary + same_res_mid + same_res_high
    seen: set[tuple[int, int, int, str]] = set()
    out: list[tuple[int, int, int, str]] = []
    for prof in profiles:
        if prof not in seen:
            seen.add(prof)
            out.append(prof)
    return out


def prefix_for_camera(camera: ConnectedCamera) -> str:
    """SL naming: Elgato/mirrorless = m, RealSense = r."""
    if camera.kind == "realsense":
        return "r"
    if camera.device_tag == "elgato":
        return "m"
    if camera.kind == "fake":
        return "f"
    return "c"


def _opencv_backends(*, include_msmf: bool = False) -> list[tuple[int, str]]:
    backends: list[tuple[int, str]] = []
    if sys.platform == "win32":
        backends = [(cv2.CAP_DSHOW, "DSHOW")]
        if include_msmf:
            backends.append((cv2.CAP_MSMF, "MSMF"))
    elif sys.platform == "darwin":
        if hasattr(cv2, "CAP_AVFOUNDATION"):
            backends.append((cv2.CAP_AVFOUNDATION, "AVFOUNDATION"))
        backends.append((cv2.CAP_ANY, "ANY"))
    else:
        if hasattr(cv2, "CAP_V4L2"):
            backends.append((cv2.CAP_V4L2, "V4L2"))
        backends.append((cv2.CAP_ANY, "ANY"))
    return backends


def _uvc_display_name(
    index: int,
    w: int,
    h: int,
    fps: float,
    *,
    busy: bool = False,
) -> tuple[str, str, Optional[str]]:
    """Return (display_name, device_tag, open_path)."""
    fallback = f"UVC #{index} ({w}x{h}" + (f"@{fps:.0f}" if fps > 0 else "") + ")"
    friendly, tag = friendly_name_for_index(index, fallback)
    if tag == "elgato":
        name = f"Elgato / capture card — {friendly}"
    elif tag == "realsense-uvc":
        name = f"RealSense (OpenCV/UVC twin) — {friendly} [use SDK entry instead]"
    elif tag == "virtual":
        name = f"Virtual camera — {friendly}"
    elif friendly != fallback:
        name = f"{friendly} ({w}x{h})" if w > 0 and h > 0 else friendly
    elif index > 0 and w >= 1280 and h >= 720:
        name = f"UVC/capture #{index} ({w}x{h})"
    else:
        name = fallback
    if busy:
        name = f"{name} [busy at scan — close Zoom/Teams/Camera app, then Start preview]"

    open_path = None
    if sys.platform == "win32":
        open_path = dshow_open_path(index)
        if open_path is None:
            # Prefer opening by the Windows friendly name (works even when the
            # OpenCV index probe hangs because another app briefly locks it).
            names = list_windows_capture_names()
            if 0 <= index < len(names):
                open_path = f"video={names[index]}"
            elif tag == "elgato":
                elgato_paths = dshow_open_paths_for_tag("elgato")
                open_path = elgato_paths[0] if elgato_paths else None
            elif tag == "uvc" and names and (index == 0 or busy):
                # Only map the built-in/first webcam name onto index 0 (or a
                # timed-out busy slot). Never attach it to UVC #1+ (often OBS).
                for candidate in names:
                    if classify_capture_name(candidate) in {"uvc", "elgato"}:
                        open_path = f"video={candidate}"
                        if tag == "uvc" and friendly == fallback:
                            name = (
                                f"{candidate} #{index}"
                                + (
                                    " [busy at scan — close other apps, then Start preview]"
                                    if busy
                                    else ""
                                )
                            )
                        break
            elif index > 0 and w <= 640 and h <= 480 and not busy:
                # Common Windows layout: #0 = laptop webcam, #1 = OBS Virtual Camera.
                name = f"UVC/virtual? #{index} ({w}x{h}) — often OBS Virtual Camera"
    return name, tag, open_path


def _make_uvc_camera(
    index: int,
    backend: int,
    backend_name: str,
    *,
    w: int = 640,
    h: int = 480,
    fps: float = 0.0,
    busy: bool = False,
    open_path: Optional[str] = None,
    name: Optional[str] = None,
    device_tag: Optional[str] = None,
) -> ConnectedCamera:
    disp, tag, path = _uvc_display_name(index, w, h, fps, busy=busy)
    return ConnectedCamera(
        cam_id=f"uvc:{index}:{backend_name}",
        kind="uvc",
        name=name or disp,
        index=index,
        backend=backend,
        backend_name=backend_name,
        open_path=open_path if open_path is not None else path,
        device_tag=device_tag or tag,
    )


def _probe_uvc_index(index: int, backend: int, backend_name: str) -> Optional[ConnectedCamera]:
    with quiet_opencv():
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            cap.release()
            return None
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        # Do not require a frame here — some Windows drivers hang on the first
        # read while another process briefly holds the device, which made the
        # real webcam disappear from Refresh. Frame delivery is verified in
        # FormattedUvcSource.start() when preview begins.
        cap.release()
        if w <= 0 or h <= 0:
            w, h = 640, 480
    return _make_uvc_camera(index, backend, backend_name, w=w, h=h, fps=fps)


def _probe_uvc_index_safe(
    index: int,
    backend: int,
    backend_name: str,
    timeout_s: float = 2.0,
) -> tuple[Optional[ConnectedCamera], str]:
    """
    OpenCV can hang when another process owns the camera; bound the wait.

    Returns (camera_or_none, status) where status is ok | missing | timeout | error.
    """
    from concurrent.futures import ThreadPoolExecutor
    from concurrent.futures import TimeoutError as FuturesTimeout

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_probe_uvc_index, index, backend, backend_name)
        try:
            cam = future.result(timeout=timeout_s)
            return cam, ("ok" if cam is not None else "missing")
        except FuturesTimeout:
            logger.warning(
                "UVC probe timed out index=%d backend=%s (device busy or locked)",
                index, backend_name,
            )
            return None, "timeout"
        except Exception as exc:  # noqa: BLE001
            logger.debug("UVC probe failed index=%d: %s", index, exc)
            return None, "error"


def _merge_windows_named_cameras(
    found_by_index: dict[int, ConnectedCamera],
    default_backend: int,
    default_backend_name: str,
) -> None:
    """
    Keep physical cameras visible even when OpenCV index probe fails.

    Windows still knows 'USB2.0 HD UVC WebCam'; we expose it with a DirectShow
    name path so Start preview can open it without a successful Refresh probe.
    """
    if sys.platform != "win32":
        return
    names = list_windows_capture_names()
    if not names:
        return

    claimed_paths = {
        c.open_path for c in found_by_index.values() if c.open_path
    }
    claimed_names = {c.name.lower() for c in found_by_index.values()}

    for i, name in enumerate(names):
        tag = classify_capture_name(name)
        if tag == "virtual":
            continue
        path = f"video={name}"
        if path in claimed_paths:
            continue
        if any(name.lower() in existing for existing in claimed_names):
            continue
        # Prefer the natural index when free; otherwise use a high synthetic index
        # that still carries the open_path (FormattedUvcSource opens by name first).
        index = i if i not in found_by_index else (100 + i)
        if index in found_by_index:
            continue
        found_by_index[index] = ConnectedCamera(
            cam_id=f"uvc:name:{name}:{default_backend_name}",
            kind="uvc",
            name=(
                f"Elgato / capture card — {name}"
                if tag == "elgato"
                else f"{name} (Windows name — try if index probe missed it)"
            ),
            index=index if index < 100 else i,
            backend=default_backend,
            backend_name=default_backend_name,
            open_path=path,
            device_tag=tag,
        )
        claimed_paths.add(path)
        claimed_names.add(name.lower())
        logger.info("Added Windows-named UVC device missed by OpenCV probe: %s", name)


def _ensure_elgato_entries(
    found_by_index: dict[int, ConnectedCamera],
    default_backend: int,
    default_backend_name: str,
) -> None:
    """Always expose Elgato capture cards from Windows names even when probe times out."""
    if sys.platform != "win32":
        return
    names = [
        n for n in list_windows_capture_names() if classify_capture_name(n) == "elgato"
    ]
    if not names:
        return
    claimed_paths = {c.open_path for c in found_by_index.values() if c.open_path}
    for i, name in enumerate(names):
        path = f"video={name}"
        if path in claimed_paths:
            continue
        index = 200 + i
        while index in found_by_index:
            index += 1
        found_by_index[index] = ConnectedCamera(
            cam_id=f"uvc:elgato:{name}:{default_backend_name}",
            kind="uvc",
            name=f"Elgato / capture card — {name}",
            index=i,
            backend=default_backend,
            backend_name=default_backend_name,
            open_path=path,
            device_tag="elgato",
        )
        claimed_paths.add(path)
        logger.info("Ensured Elgato entry from Windows name: %s", name)


def _windows_has_elgato_name() -> bool:
    if sys.platform != "win32":
        return False
    return any(
        classify_capture_name(n) == "elgato" for n in list_windows_capture_names()
    )


def list_uvc_cameras(
    max_index: int = 6,
    *,
    include_msmf: bool = False,
    refresh_name_cache: bool = False,
) -> list[ConnectedCamera]:
    """R1 — enumerate OpenCV/UVC devices (webcam, Elgato, virtual cam, …)."""
    if refresh_name_cache:
        clear_name_cache()
    found_by_index: dict[int, ConnectedCamera] = {}
    timed_out: set[int] = set()
    backends = _opencv_backends(include_msmf=include_msmf)
    elgato_expected = _windows_has_elgato_name()
    for backend, bname in backends:
        for i in range(max_index):
            if i in found_by_index:
                continue
            # Built-in webcams / capture cards on index 0 are often slow or locked.
            if i == 0 and elgato_expected:
                timeout_s = 3.5
            else:
                timeout_s = 2.0 if i == 0 else 1.0
            cam, status = _probe_uvc_index_safe(i, backend, bname, timeout_s=timeout_s)
            if cam is not None:
                found_by_index[i] = cam
                continue
            if status == "timeout":
                timed_out.add(i)

    # Timed-out indices still belong in the dropdown (usually the real webcam
    # locked by Zoom/Teams/Camera while OBS Virtual Cam remains easy to open).
    if timed_out and backends:
        backend, bname = backends[0]
        for i in sorted(timed_out):
            if i in found_by_index:
                continue
            found_by_index[i] = _make_uvc_camera(
                i, backend, bname, w=640, h=480, busy=True
            )
            logger.info(
                "Keeping busy UVC index=%d in device list for manual Start preview",
                i,
            )

    if backends:
        _merge_windows_named_cameras(found_by_index, backends[0][0], backends[0][1])
        _ensure_elgato_entries(found_by_index, backends[0][0], backends[0][1])

    return [found_by_index[i] for i in sorted(found_by_index)]


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
                device_tag="realsense",
            )
        )
    return out


def list_all_cameras(
    *,
    include_fake: bool = False,
    probe_uvc: bool = True,
    probe_realsense: bool = True,
    refresh_name_cache: bool = False,
) -> list[ConnectedCamera]:
    """R1 — unified UVC + RealSense inventory."""
    cams: list[ConnectedCamera] = []
    if probe_uvc:
        cams.extend(
            list_uvc_cameras(refresh_name_cache=refresh_name_cache)
        )
    rs_cams: list[ConnectedCamera] = []
    if probe_realsense:
        rs_cams = list_realsense_cameras()
        cams.extend(rs_cams)
    # When the RealSense SDK sees hardware, hide the OpenCV "UVC twin" entry so
    # users pick the working [realsense] path instead of a black UVC stream.
    if rs_cams:
        cams = [c for c in cams if c.device_tag != "realsense-uvc"]
    if include_fake:
        cams.append(
            ConnectedCamera(
                cam_id="fake:0",
                kind="fake",
                name="Synthetic fake A (color bars)",
                index=0,
                device_tag="fake",
            )
        )
        cams.append(
            ConnectedCamera(
                cam_id="fake:1",
                kind="fake",
                name="Synthetic fake B (color bars)",
                index=1,
                device_tag="fake",
            )
        )
    return cams


def pick_auto_camera_for_slot(
    slot_id: int,
    devices: list[ConnectedCamera],
    used_cam_ids: set[str],
) -> Optional[ConnectedCamera]:
    """Slot 0 → RealSense, slot 1 → Elgato when both are in the device list."""
    slot_order: dict[int, tuple[str, ...]] = {
        0: ("realsense", "uvc", "fake"),
        1: ("elgato", "uvc", "fake"),
    }
    prefer = slot_order.get(slot_id, ("realsense", "elgato", "uvc", "fake"))
    for kind in prefer:
        for d in devices:
            if d.cam_id in used_cam_ids:
                continue
            if "busy at scan" in d.name.lower() and kind in {"uvc", "elgato"}:
                continue
            if kind == "realsense" and d.kind == "realsense":
                return d
            if kind == "elgato" and d.device_tag == "elgato":
                return d
            if (
                kind == "uvc"
                and d.kind == "uvc"
                and d.device_tag not in {"virtual", "realsense-uvc"}
            ):
                return d
            if kind == "fake" and d.kind == "fake":
                return d
    return None


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
    """
    R2 — all video profiles the RealSense SDK advertises for this device.

    Includes color (preferred for MP4) plus depth/IR so higher FPS options the
    sensor supports (often on depth) appear in the configuration dropdown.
    """
    if not realsense_available():
        return [
            StreamMode(1280, 720, 30, "bgr8"),
            StreamMode(1920, 1080, 30, "bgr8"),
            StreamMode(640, 480, 30, "bgr8"),
            StreamMode(848, 480, 90, "z16"),
            StreamMode(640, 480, 90, "z16"),
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
        logger.warning(
            "list_realsense_modes: serial=%s not found — returning common presets",
            serial,
        )
        return [
            StreamMode(1280, 720, 30, "bgr8"),
            StreamMode(1920, 1080, 30, "bgr8"),
            StreamMode(640, 480, 30, "bgr8"),
            StreamMode(848, 480, 90, "z16"),
        ]

    modes: list[StreamMode] = []
    seen: set[tuple[int, int, int, str]] = set()
    try:
        sensors = list(device.query_sensors())
    except Exception:  # noqa: BLE001
        sensors = []

    allowed_streams = {
        rs.stream.color,
        rs.stream.depth,
        rs.stream.infrared,
    }
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
                st = vp.stream_type()
            except Exception:  # noqa: BLE001
                continue
            if st not in allowed_streams:
                continue
            try:
                w, h = int(vp.width()), int(vp.height())
                fps = int(round(float(vp.fps())))
                fmt = _rs_format_name(vp.format())
            except Exception:  # noqa: BLE001
                continue
            if fps <= 0 or w <= 0 or h <= 0:
                continue

            # Map stream+format to a pipeline pixel_format the GUI can open.
            if st == rs.stream.color:
                if fmt not in {"bgr8", "rgb8", "yuyv", "y8"}:
                    continue
                # D400 color is 30 (sometimes 60). Never invent 90/120 for RGB.
                if fps > 60:
                    continue
            elif st == rs.stream.depth:
                # Depth is recorded/previewed via colorizer → BGR for MP4.
                fmt = "z16"
            elif st == rs.stream.infrared:
                if fmt not in {"y8", "y16"}:
                    fmt = "y8"
                else:
                    fmt = "y8"
            else:
                continue

            key = (w, h, fps, fmt)
            if key in seen:
                continue
            seen.add(key)
            modes.append(StreamMode(w, h, fps, fmt))

    # Extra color presets at 30fps only. Do not advertise 120 for D400 color —
    # the SDK list above already includes 60 if the device supports it.
    for preset in (
        StreamMode(1920, 1080, 30, "bgr8"),
        StreamMode(1280, 720, 30, "bgr8"),
        StreamMode(848, 480, 30, "bgr8"),
        StreamMode(640, 480, 30, "bgr8"),
        StreamMode(640, 480, 90, "z16"),
        StreamMode(848, 480, 90, "z16"),
        StreamMode(1280, 720, 30, "z16"),
    ):
        key = (preset.width, preset.height, preset.fps, preset.pixel_format)
        if key not in seen:
            seen.add(key)
            modes.append(preset)

    if not modes:
        modes = [
            StreamMode(1280, 720, 30, "bgr8"),
            StreamMode(640, 480, 30, "bgr8"),
        ]

    # Sort: color first, 30fps color preferred, bgr8 before rgb8, then resolution.
    def mode_priority(mode: StreamMode) -> tuple:
        is_color = 0 if mode.pixel_format in {"bgr8", "rgb8", "yuyv"} else 1
        pixels = mode.width * mode.height
        fps_bias = 0 if mode.fps == 30 else (1 if mode.fps < 90 else 2)
        fmt_pref = (
            0
            if mode.pixel_format == "bgr8"
            else (1 if mode.pixel_format == "rgb8" else 2)
        )
        return (is_color, fps_bias, fmt_pref, -pixels, -mode.fps)

    modes.sort(key=mode_priority)
    logger.info(
        "RealSense modes for sn=%s: %d profiles (fps set=%s)",
        serial,
        len(modes),
        sorted({m.fps for m in modes}),
    )
    return modes


def list_uvc_modes(camera: ConnectedCamera) -> list[StreamMode]:
    """R2 — preset + lightly probed modes for a UVC device."""
    # Capture cards (Elgato) usually need MJPG for high-res; webcams vary.
    if camera.device_tag == "elgato":
        # Prefer OBS-equivalent 1080p120 MJPG first so stations default to the
        # mode that hardware already proves in OBS (not silent 1080p60).
        preferred = [
            StreamMode(1920, 1080, 120, "mjpg"),
            StreamMode(1920, 1080, 60, "mjpg"),
            StreamMode(1280, 720, 120, "mjpg"),
            StreamMode(1920, 1080, 50, "mjpg"),
            StreamMode(1920, 1080, 30, "mjpg"),
            StreamMode(1280, 720, 60, "mjpg"),
            StreamMode(1280, 720, 30, "mjpg"),
            StreamMode(1920, 1080, 25, "mjpg"),
            StreamMode(640, 480, 30, "mjpg"),
        ]
    else:
        # Prefer modest defaults first. Many UVC webcams cannot sustain FHD@120,
        # and selecting that as the first option caused confusing recordings.
        preferred = [
            StreamMode(1280, 720, 30, "mjpg"),
            StreamMode(640, 480, 30, "mjpg"),
            StreamMode(1280, 720, 30, "yuyv"),
            StreamMode(640, 480, 30, "bgr8"),
            StreamMode(1280, 720, 30, "bgr8"),
            StreamMode(1920, 1080, 30, "mjpg"),
            StreamMode(1280, 720, 60, "mjpg"),
        ]
    modes = list(preferred)
    for w, h, fps, fmt in _UVC_PRESET_MODES:
        candidate = StreamMode(w, h, fps, fmt)
        if candidate not in modes:
            modes.append(candidate)
    if camera.index is None or camera.backend is None:
        return modes
    # Do not prepend a probed 720p bgr8 mode for Elgato — that made HDMI look
    # worse than RealSense. Keep 1080p mjpg first; hide bgr8/yuyv (green screen).
    if camera.device_tag == "elgato":
        modes = [m for m in modes if m.pixel_format == "mjpg"]
        if not modes:
            modes = [StreamMode(1920, 1080, 60, "mjpg")]
        return modes
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
            StreamMode(1280, 720, 30, "bgr8"),
            StreamMode(1280, 720, 60, "bgr8"),
            StreamMode(640, 480, 30, "bgr8"),
            StreamMode(1920, 1080, 30, "bgr8"),
            StreamMode(1280, 720, 120, "bgr8"),
            StreamMode(1920, 1080, 120, "bgr8"),
        ]
    return list_uvc_modes(camera)


def _looks_like_packed_yuyv(bgr: np.ndarray) -> bool:
    """True when YUY2 was likely decoded as BGR (fine zebra / chroma stripes)."""
    if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3:
        return False
    h, w = bgr.shape[:2]
    if w < 16 or h < 8:
        return False
    sample = bgr[h // 4 : 3 * h // 4 : 4, : min(w, 320)]
    even = sample[:, 0::2].astype(np.int16)
    odd = sample[:, 1::2].astype(np.int16)
    cols = min(even.shape[1], odd.shape[1])
    if cols < 4:
        return False
    delta = float(np.mean(np.abs(even[:, :cols] - odd[:, :cols])))
    return delta > 48.0


def _looks_like_solid_green(bgr: np.ndarray) -> bool:
    """True only for near-uniform green (wrong FOURCC / empty buffer).

    Textured chroma-key / green-screen studios must NOT match — rejecting those
    made OpenCV fall through from @120 to @60 while OBS (no green reject) held 120.
    """
    if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3:
        return False
    sample = bgr[:: max(1, bgr.shape[0] // 24), :: max(1, bgr.shape[1] // 24)]
    # Spatial variance per channel (not all-BGR std — flat [0,180,0] has high
    # cross-channel std even when every pixel is identical).
    spatial = float(
        max(float(np.std(sample[:, :, c])) for c in range(3))
    )
    if spatial >= 12.0:
        return False
    b = sample[:, :, 0].astype(np.float32)
    g = sample[:, :, 1].astype(np.float32)
    r = sample[:, :, 2].astype(np.float32)
    return float(g.mean()) > 140.0 and float(b.mean()) < 70.0 and float(r.mean()) < 70.0


def _frame_is_unusable_elgato(bgr: np.ndarray) -> bool:
    """Reject empty / flat wrong-format buffers — not real green-screen scenes."""
    if bgr is None or bgr.size == 0:
        return True
    mean = float(np.mean(bgr))
    if mean < 1.5:
        return True
    # Flat near-uniform green only (wrong FOURCC). Textured studio green is OK.
    spatial = float(max(float(np.std(bgr[:, :, c])) for c in range(3)))
    if spatial < 8.0 and float(np.mean(bgr[:, :, 1])) > 120.0:
        return True
    if _looks_like_solid_green(bgr):
        return True
    return False


def _open_uvc_timeout(target: Any, backend: int, timeout_s: float = 3.0) -> cv2.VideoCapture:
    """Open VideoCapture with a deadline so a locked webcam cannot freeze Tk."""
    from concurrent.futures import ThreadPoolExecutor
    from concurrent.futures import TimeoutError as FuturesTimeout

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(cv2.VideoCapture, target, backend)
        try:
            cap = future.result(timeout=timeout_s)
        except FuturesTimeout as exc:
            raise RuntimeError(
                f"Camera open timed out ({target!r}). Close Zoom/Teams/Camera "
                "and any other POC1 window, then Refresh."
            ) from exc
    if cap is None or not cap.isOpened():
        if cap is not None:
            try:
                cap.release()
            except Exception:  # noqa: BLE001
                pass
        raise RuntimeError(f"Could not open {target!r} backend={backend}")
    return cap


def _elgato_open_targets(
    open_path: Optional[str],
    device_index: int,
    *,
    max_index: int = 8,
    dshow_only: bool = False,
) -> list[tuple[Any, int]]:
    """
    Ordered OpenCV open attempts for Elgato 4K X / HD60.

    Prefer ffmpeg DirectShow names, then scan indices (PnP index is unreliable).
    For high-rate lock passes, use dshow_only=True (MSMF often caps Elgato at 60).
    """
    targets: list[tuple[Any, int]] = []
    name_paths = list(elgato_open_name_paths())
    if open_path and open_path not in name_paths:
        name_paths.insert(0, open_path)

    # 1) DirectShow by friendly name (ffmpeg-aligned preferred).
    for path in name_paths:
        targets.append((path, cv2.CAP_DSHOW))

    # 2) Scan OpenCV indices — do not trust PnP/synthetic index alone.
    indices = list(range(max_index))
    if device_index not in indices and 0 <= device_index < 100:
        indices.insert(0, device_index)
    for idx in indices:
        targets.append((idx, cv2.CAP_DSHOW))

    if dshow_only:
        return targets

    # 3) MSMF last-resort for Elgato 4K X when DSHOW fails on Station A.
    for path in name_paths:
        targets.append((path, cv2.CAP_MSMF))
    for idx in indices[:4]:
        targets.append((idx, cv2.CAP_MSMF))

    # 4) CAP_ANY
    for idx in indices[:4]:
        targets.append((idx, cv2.CAP_ANY))

    return targets


def _uvc_open_failure_message(
    *,
    is_elgato: bool,
    opened_once: bool,
    last_exc: Optional[BaseException],
) -> str:
    """Distinguish could-not-open vs opened-but-no-frames for Station A dialogs."""
    detail = f" ({last_exc})" if last_exc else ""
    if is_elgato:
        ff_hint = ""
        if not ffmpeg_available():
            ff_hint = (
                " ffmpeg is missing — install ffmpeg on PATH so Refresh uses "
                "DirectShow device names (PnP labels often fail to open)."
            )
        if opened_once:
            return (
                f"Elgato opened but delivered no frames{detail}."
                " Fully quit OBS (not minimize), Elgato 4K Capture Utility, and"
                " RealSense Viewer. HDMI ON at 1080p120 for true 120, then Start preview."
                + ff_hint
            )
        return (
            f"Could not open Elgato capture card{detail}."
            " Fully quit OBS, Elgato 4K Capture Utility, Zoom, Teams, and Windows Camera."
            " Confirm HDMI is on, click Refresh, then Start preview on the Elgato card alone."
            + ff_hint
        )
    if opened_once:
        return (
            f"UVC device opened but delivered no frames{detail}."
            " Close Zoom/Teams/Camera and other POC1 windows, then Refresh."
            " Use 1280x720@30 mjpg for the laptop webcam."
        )
    return (
        f"Could not open UVC device{detail}."
        " Close other camera apps and click Refresh."
    )


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
        open_path: Optional[str] = None,
        device_tag: str = "uvc",
    ):
        safe_fps = fps if fps > 0 else 30
        super().__init__(device_index, width, height, safe_fps, backend=backend)
        self.pixel_format = pixel_format
        self.open_path = open_path
        self.device_tag = device_tag
        self._pending_frame: Optional[np.ndarray] = None

    def _open_capture(self) -> cv2.VideoCapture:
        """Try DirectShow-by-name first on Windows, then index + backend fallbacks."""
        attempts: list[tuple[Any, int]] = []
        if self.open_path and sys.platform == "win32":
            attempts.append((self.open_path, cv2.CAP_DSHOW))
            if self.device_tag != "elgato":
                attempts.append((self.open_path, cv2.CAP_MSMF))
        # Extra named Elgato / webcam paths help when index mapping is ambiguous.
        if sys.platform == "win32":
            if self.device_tag == "elgato":
                for path in dshow_open_paths_for_tag("elgato"):
                    if path != self.open_path:
                        attempts.append((path, cv2.CAP_DSHOW))
            elif self.device_tag == "uvc":
                for path in dshow_open_paths_for_tag("uvc"):
                    if path != self.open_path:
                        attempts.append((path, cv2.CAP_DSHOW))
        attempts.append((self.device_index, self._backend if self.device_tag != "elgato" else cv2.CAP_DSHOW))
        if self.device_tag != "elgato" and self._backend == cv2.CAP_DSHOW:
            attempts.append((self.device_index, cv2.CAP_MSMF))
        if self._backend != cv2.CAP_ANY:
            attempts.append((self.device_index, cv2.CAP_ANY))

        last_error = "unknown"
        for target, backend in attempts:
            try:
                cap = _open_uvc_timeout(target, backend, timeout_s=3.0)
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                continue
            if cap.isOpened():
                self._backend = backend
                return cap
            cap.release()
            last_error = f"open failed target={target!r} backend={backend}"
        raise RuntimeError(
            f"Could not open UVC device index={self.device_index}"
            + (f" path={self.open_path!r}" if self.open_path else "")
            + f" ({last_error}). Close other camera apps and click Refresh."
        )

    def _configure_and_grab(
        self,
        cap: cv2.VideoCapture,
        width: int,
        height: int,
        fps: int,
        pixel_format: str,
    ) -> Optional[np.ndarray]:
        # OBS-like DirectShow order for Elgato: MJPG → size → FPS (repeated).
        # High-rate needs a deeper driver buffer so dual-cam Record does not starve.
        is_elgato = getattr(self, "device_tag", "") == "elgato"
        if is_elgato and fps >= 90:
            return self._elgato_configure_obs_pin(cap, width, height, fps)
        fmt = pixel_format
        _apply_uvc_fourcc(cap, fmt)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
        _apply_uvc_fourcc(cap, fmt)
        for _ in range(3):
            cap.set(cv2.CAP_PROP_FPS, float(fps))
        buf = 4 if fps >= 60 else 1
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, float(buf))
        except Exception:  # noqa: BLE001
            pass
        warm = 20 if is_elgato else 4
        frame = None
        for _ in range(warm):
            ok, candidate = cap.read()
            if ok and candidate is not None:
                frame = candidate
        return frame

    def _elgato_configure_obs_pin(
        self,
        cap: cv2.VideoCapture,
        width: int,
        height: int,
        fps: int,
    ) -> Optional[np.ndarray]:
        """Force OBS-equivalent Custom WxH @ fps MJPEG pin (multiple property orders)."""
        mjpg = cv2.VideoWriter_fourcc(*"MJPG")
        sequences: list[list[tuple[str, float | int]]] = [
            # Closest to OBS: format → size → fps, then re-assert.
            [
                ("fourcc", mjpg),
                ("w", width),
                ("h", height),
                ("fps", fps),
                ("fourcc", mjpg),
                ("fps", fps),
                ("fps", fps),
            ],
            [
                ("w", width),
                ("h", height),
                ("fourcc", mjpg),
                ("fps", fps),
                ("fps", fps),
            ],
            [
                ("fps", fps),
                ("w", width),
                ("h", height),
                ("fourcc", mjpg),
                ("fps", fps),
            ],
        ]
        best: Optional[np.ndarray] = None
        for seq_i, seq in enumerate(sequences):
            for key, val in seq:
                try:
                    if key == "fourcc":
                        cap.set(cv2.CAP_PROP_FOURCC, float(int(val)))
                    elif key == "w":
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(val))
                    elif key == "h":
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(val))
                    elif key == "fps":
                        cap.set(cv2.CAP_PROP_FPS, float(val))
                except Exception:  # noqa: BLE001
                    pass
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 8.0)
            except Exception:  # noqa: BLE001
                pass
            frame = None
            for _ in range(50):
                ok, candidate = cap.read()
                if ok and candidate is not None:
                    frame = candidate
            if frame is None:
                continue
            best = frame
            got_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            got_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            got_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
            logger.info(
                "Elgato OBS-pin try#%d asked %dx%d@%d → driver %dx%d@%.1f",
                seq_i + 1,
                width,
                height,
                fps,
                got_w,
                got_h,
                got_fps,
            )
            # Driver advertises high-rate (or 0/unknown) at the right size → use it.
            size_ok = got_w >= int(width * 0.9) and got_h >= int(height * 0.9)
            fps_ok = got_fps <= 0.5 or got_fps >= fps * 0.85
            if size_ok and fps_ok:
                return frame
            # Driver still stuck at ~60 — try next property order before measuring.
            if got_fps > 0 and got_fps < fps * 0.85:
                continue
            if size_ok:
                return frame
        return best

    def _measure_delivery_fps(self, seconds: float = 0.7) -> float:
        """Count frames for a short window to learn the real HDMI/driver rate."""
        if self._cap is None:
            return 0.0
        n = 0
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < seconds:
            ok, frame = self._cap.read()
            if ok and frame is not None:
                n += 1
                self._pending_frame = np.ascontiguousarray(frame)
        elapsed = time.perf_counter() - t0
        if elapsed <= 0.05 or n < 2:
            return 0.0
        return (n - 1) / elapsed

    def _elgato_measure_high_rate(
        self, cap: cv2.VideoCapture, fps: int
    ) -> float:
        """Longer settle + retries so DirectShow can lock 120 like OBS."""
        self._cap = cap
        for _ in range(40):
            ok, frame = cap.read()
            if ok and frame is not None:
                self._pending_frame = np.ascontiguousarray(frame)
        measured = self._measure_delivery_fps(1.5)
        if measured >= fps * 0.85:
            return measured
        # Re-apply OBS pin and sample again.
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or self.width or 1920)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or self.height or 1080)
        self._elgato_configure_obs_pin(cap, w, h, fps)
        time.sleep(0.3)
        for _ in range(40):
            ok, frame = cap.read()
            if ok and frame is not None:
                self._pending_frame = np.ascontiguousarray(frame)
        measured2 = self._measure_delivery_fps(1.5)
        return max(measured, measured2)

    def start(self) -> None:
        with quiet_opencv():
            wanted_w, wanted_h = self.width, self.height
            wanted_fps = self.target_fps if self.target_fps > 0 else 30
            self.requested_fps = wanted_fps
            wanted_fmt = (self.pixel_format or "bgr8").lower()
            is_elgato = self.device_tag == "elgato"

            if is_elgato:
                if not ffmpeg_available():
                    logger.warning(
                        "ffmpeg not available — Elgato open may fail with PnP-only names. "
                        "Install ffmpeg on PATH, then Refresh."
                    )
                # Prefer the Setup dropdown mode first, then common HDMI fallbacks.
                profiles = elgato_open_profiles(
                    wanted_w, wanted_h, wanted_fps, wanted_fmt
                )
            else:
                # Laptop UVC almost never delivers real bgr8. MJPG first avoids
                # YUY2-as-BGR zebra stripes. Keep the list tiny so start() cannot
                # freeze Tk for tens of seconds.
                cam_fps = 30 if wanted_fmt == "bgr8" else min(wanted_fps, 30)
                profiles = [
                    (wanted_w, wanted_h, cam_fps, "mjpg"),
                    (wanted_w, wanted_h, cam_fps, wanted_fmt),
                    (1280, 720, 30, "mjpg"),
                    (640, 480, 30, "yuyv"),
                    (640, 480, 30, "bgr8"),
                ]

            seen_p: set[tuple[int, int, int, str]] = set()
            unique_profiles: list[tuple[int, int, int, str]] = []
            for prof in profiles:
                if prof not in seen_p:
                    seen_p.add(prof)
                    unique_profiles.append(prof)

            last_exc: Optional[Exception] = None
            opened_once = False
            chosen: Optional[
                tuple[float, int, int, int, str, cv2.VideoCapture, np.ndarray]
            ] = None
            best_effort: Optional[
                tuple[float, int, int, int, str, cv2.VideoCapture, np.ndarray]
            ] = None
            want_high = is_elgato and wanted_fps >= 90
            # Same resolution only — never silently accept 720 when operator picked 1080.
            high_only = (
                [
                    p
                    for p in unique_profiles
                    if p[2] >= 90 and p[0] == wanted_w and p[1] == wanted_h
                ]
                if want_high
                else unique_profiles
            )
            if want_high and not high_only:
                high_only = [(wanted_w, wanted_h, wanted_fps, "mjpg")]
            # Best-effort same-res list (may include @60 at the same WxH only).
            same_res_profiles = [
                p for p in unique_profiles if p[0] == wanted_w and p[1] == wanted_h
            ] or [(wanted_w, wanted_h, wanted_fps, "mjpg")]

            if is_elgato:
                # High-rate: DSHOW-only first (MSMF often caps Elgato at 60). Two
                # passes so we reopen after OBS-style pin retries fail.
                if want_high:
                    open_targets = _elgato_open_targets(
                        self.open_path,
                        int(self.device_index or 0),
                        dshow_only=True,
                    )
                    open_targets = open_targets + open_targets  # second exclusive pass
                else:
                    open_targets = _elgato_open_targets(
                        self.open_path, int(self.device_index or 0)
                    )
                open_timeout = 6.0
            else:
                open_targets = []
                if self.open_path and sys.platform == "win32":
                    open_targets.append((self.open_path, cv2.CAP_DSHOW))
                if sys.platform == "win32" and self.device_tag == "uvc" and not self.open_path:
                    for path in dshow_open_paths_for_tag("uvc"):
                        open_targets.append((path, cv2.CAP_DSHOW))
                open_targets.append((self.device_index, self._backend))
                if self.device_tag == "uvc" and self._backend == cv2.CAP_DSHOW:
                    open_targets.append((self.device_index, cv2.CAP_MSMF))
                open_timeout = 3.0

            profile_list = high_only if want_high else unique_profiles
            seen_open: set[tuple[str, int]] = set()
            for target, backend in open_targets:
                ok_key = (str(target), int(backend))
                # Allow a second DSHOW pass for high-rate by tagging pass index.
                pass_key = ok_key
                if want_high and pass_key in seen_open:
                    # Second identical target: reopen for exclusive pin after OBS.
                    pass
                elif pass_key in seen_open:
                    continue
                seen_open.add(pass_key)
                try:
                    cap = _open_uvc_timeout(target, backend, timeout_s=open_timeout)
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc if isinstance(exc, Exception) else Exception(str(exc))
                    logger.warning("UVC open skipped %s: %s", target, exc)
                    continue

                opened_once = True
                got_any = False
                accepted_here = False
                for width, height, fps, fmt in profile_list:
                    try:
                        frame = self._configure_and_grab(cap, width, height, fps, fmt)
                    except Exception as exc:  # noqa: BLE001
                        last_exc = exc
                        continue
                    if frame is None:
                        continue
                    if is_elgato and _frame_is_unusable_elgato(frame):
                        logger.warning(
                            "Rejecting %dx%d %s — unusable Elgato frame (green/empty)",
                            width,
                            height,
                            fmt,
                        )
                        continue
                    if not is_elgato and _looks_like_solid_green(frame):
                        logger.warning(
                            "Rejecting %dx%d %s — solid green (wrong format / 10-bit HDMI)",
                            width,
                            height,
                            fmt,
                        )
                        continue
                    if fmt == "bgr8" and _looks_like_packed_yuyv(frame):
                        logger.warning(
                            "Rejecting %dx%d bgr8 — looks like packed YUY2 stripes",
                            width,
                            height,
                        )
                        continue
                    got_any = True
                    self._cap = cap
                    self._backend = backend
                    if isinstance(target, str) and target.startswith("video="):
                        self.open_path = target
                    elif isinstance(target, int):
                        self.device_index = target
                    owned = np.ascontiguousarray(frame)
                    self._pending_frame = owned
                    if is_elgato and fps >= 90:
                        measured = self._elgato_measure_high_rate(cap, fps)
                    else:
                        measured = 0.0
                    reported = float(cap.get(cv2.CAP_PROP_FPS) or 0)
                    rate = measured if measured > 1 else (
                        reported if reported > 0 else float(fps)
                    )
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or width)
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or height)
                    # Reject silent resolution fallback (1080 request → 720 open).
                    if want_high and (
                        w < int(wanted_w * 0.9) or h < int(wanted_h * 0.9)
                    ):
                        logger.warning(
                            "Rejecting size fallback %dx%d (operator selected %dx%d@%d)",
                            w,
                            h,
                            wanted_w,
                            wanted_h,
                            wanted_fps,
                        )
                        continue
                    candidate = (rate, w, h, fps, fmt, cap, owned)
                    logger.info(
                        "UVC opened via target=%r backend=%s %dx%d@%d %s (measured ~%.1f)",
                        target,
                        backend,
                        w,
                        h,
                        fps,
                        fmt,
                        rate,
                    )
                    # Real / settling 120 lock: accept when delivery is already high-rate
                    # (>=90). Do not require a perfect 102 sample (that caused 95 stamps).
                    if fps >= 90 and rate >= 90.0:
                        chosen = candidate
                        accepted_here = True
                        logger.info(
                            "Elgato 120 LOCKED like OBS (~%.1ffps) at %dx%d via %r",
                            rate,
                            w,
                            h,
                            target,
                        )
                        break
                    # Soft ~60 while asking 120: keep searching; remember best effort.
                    if want_high and fps >= 90 and rate < 90.0:
                        if best_effort is None or rate > best_effort[0]:
                            if (
                                best_effort is not None
                                and best_effort[5] is not cap
                            ):
                                try:
                                    best_effort[5].release()
                                except Exception:  # noqa: BLE001
                                    pass
                            best_effort = candidate
                        logger.warning(
                            "Elgato asked @%d but delivery ~%.1f — not accepting yet; "
                            "retrying OBS-style pin / next target (quit OBS if open)",
                            fps,
                            rate,
                        )
                        continue
                    chosen = candidate
                    accepted_here = True
                    break

                if chosen is not None:
                    break
                keep_best = best_effort is not None and best_effort[5] is cap
                if not accepted_here and not keep_best:
                    try:
                        cap.release()
                    except Exception:  # noqa: BLE001
                        pass
                if not got_any:
                    last_exc = last_exc or RuntimeError("no frames after open")

            # Fallbacks when 120 never locked: same resolution only (no 1080→720).
            if chosen is None and want_high:
                logger.warning(
                    "Elgato could not lock ~%dfps at %dx%d after OBS-style DSHOW retries. "
                    "Fully quit OBS, then Start preview again. "
                    "Best effort stays at %dx%d (no resolution fallback).",
                    wanted_fps,
                    wanted_w,
                    wanted_h,
                    wanted_w,
                    wanted_h,
                )
                if best_effort is not None:
                    chosen = best_effort
                else:
                    open_targets = _elgato_open_targets(
                        self.open_path, int(self.device_index or 0)
                    )
                    seen_open = set()
                    for target, backend in open_targets:
                        ok_key = (str(target), int(backend))
                        if ok_key in seen_open:
                            continue
                        seen_open.add(ok_key)
                        try:
                            cap = _open_uvc_timeout(
                                target, backend, timeout_s=open_timeout
                            )
                        except Exception as exc:  # noqa: BLE001
                            last_exc = (
                                exc
                                if isinstance(exc, Exception)
                                else Exception(str(exc))
                            )
                            continue
                        opened_once = True
                        for width, height, fps, fmt in same_res_profiles:
                            try:
                                frame = self._configure_and_grab(
                                    cap, width, height, fps, fmt
                                )
                            except Exception as exc:  # noqa: BLE001
                                last_exc = exc
                                continue
                            if frame is None:
                                continue
                            if is_elgato and _frame_is_unusable_elgato(frame):
                                continue
                            owned = np.ascontiguousarray(frame)
                            self._cap = cap
                            self._backend = backend
                            if isinstance(target, str) and target.startswith("video="):
                                self.open_path = target
                            measured = (
                                self._elgato_measure_high_rate(cap, fps)
                                if fps >= 90
                                else 0.0
                            )
                            reported = float(cap.get(cv2.CAP_PROP_FPS) or 0)
                            rate = (
                                measured
                                if measured > 1
                                else (reported if reported > 0 else float(fps))
                            )
                            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or width)
                            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or height)
                            if w < int(wanted_w * 0.9) or h < int(wanted_h * 0.9):
                                logger.warning(
                                    "Last-resort rejecting size %dx%d (want %dx%d)",
                                    w,
                                    h,
                                    wanted_w,
                                    wanted_h,
                                )
                                continue
                            chosen = (rate, w, h, fps, fmt, cap, owned)
                            break
                        if chosen is not None:
                            break
                        try:
                            cap.release()
                        except Exception:  # noqa: BLE001
                            pass

            if chosen is None:
                if want_high:
                    raise RuntimeError(
                        f"Could not open Elgato at {wanted_w}x{wanted_h}@{wanted_fps} "
                        f"(no silent fallback to 720p). Fully quit OBS / Elgato Utility / "
                        f"Viewer, confirm HDMI is {wanted_w}x{wanted_h}@120, install ffmpeg, "
                        f"Refresh, then Start preview on the Elgato alone."
                        + (f" Last error: {last_exc}" if last_exc else "")
                    )
                raise RuntimeError(
                    _uvc_open_failure_message(
                        is_elgato=is_elgato,
                        opened_once=opened_once,
                        last_exc=last_exc,
                    )
                )

            rate, w, h, req_fps, fmt, cap, frame = chosen
            self._cap = cap
            self.width = w
            self.height = h
            self.pixel_format = fmt
            self.actual_fps = rate
            self.actual_width = w
            self.actual_height = h
            self._pending_frame = frame

            if is_elgato and (req_fps >= 90 or want_high):
                stamp_req = wanted_fps if want_high else req_fps
                stamped = honest_container_fps(rate, stamp_req)
                if stamped < stamp_req * 0.85:
                    logger.warning(
                        "Elgato requested %dfps but measured ~%.1ffps — stamping @%dfps. "
                        "Fully quit OBS (Exit), pick 1920x1080@120 mjpg, HDMI 1080p120. "
                        "Hardware can do 120 in OBS; this process must own the pin alone.",
                        stamp_req,
                        rate,
                        stamped,
                    )
                else:
                    logger.info(
                        "Elgato high-rate locked: requested %d measured ~%.1f → stamp %d",
                        stamp_req,
                        rate,
                        stamped,
                    )
                self.target_fps = stamped
                self.requested_fps = stamp_req
            else:
                self.target_fps = (
                    honest_container_fps(rate, req_fps)
                    if is_elgato and rate > 5
                    else (req_fps if req_fps > 0 else 30)
                )

        mean = float(np.mean(self._pending_frame)) if self._pending_frame is not None else 0.0
        if mean < 2.0:
            logger.warning(
                "UVC preview frames are nearly black (mean=%.2f). "
                "If this is a capture card, check HDMI signal / input source.",
                mean,
            )
        logger.info(
            "D1 UVC source: idx=%d path=%s %dx%d@%d fmt=%s tag=%s (measured ~%.1ffps)",
            self.device_index,
            self.open_path or "-",
            self.width,
            self.height,
            self.target_fps,
            self.pixel_format,
            self.device_tag,
            self.actual_fps,
        )

    def read(self) -> Optional[np.ndarray]:
        if self._pending_frame is not None:
            frame = self._pending_frame
            self._pending_frame = None
            return frame
        return super().read()


class ConfiguredRealSenseSource:
    """RealSense color/depth source honoring selected resolution/FPS/format."""

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
        self.requested_fps = self.target_fps
        self.pixel_format = pixel_format.lower()
        self.device_tag = "realsense"
        self.actual_fps = float(self.target_fps)
        self.actual_width = width
        self.actual_height = height
        self.bag_path: Optional[Any] = None
        self._bag_path: Optional[Any] = None
        self._bag_final_path: Optional[Any] = None
        self._pipeline: Optional[Any] = None
        self._profile: Optional[Any] = None
        self._rs_recorder: Optional[Any] = None
        self._pending_frame: Optional[np.ndarray] = None
        self._colorizer: Optional[Any] = None
        self._wanted = (self.width, self.height, self.target_fps, self.pixel_format)
        # When True, pause SDK bag writer immediately after start (preview pre-arm).
        self.bag_start_paused: bool = False

    @staticmethod
    def _rs_format(rs, name: str):
        formats = {
            "bgr8": rs.format.bgr8,
            "rgb8": rs.format.rgb8,
            "yuyv": rs.format.yuyv,
            "y8": rs.format.y8,
            "z16": rs.format.z16,
        }
        if name not in formats:
            raise ValueError(f"Unsupported RealSense format: {name}")
        return formats[name]

    def _require_device(self, rs) -> None:
        devices = list(rs.context().query_devices())
        if not devices:
            raise RuntimeError(
                "No RealSense device found by the Intel SDK (pyrealsense2). "
                "Close Intel RealSense Viewer and any other app using the camera, "
                "use a USB 3 port, then click Refresh. "
                "Install the SDK extra with: uv sync --extra realsense"
            )
        if not self.serial:
            self.serial = devices[0].get_info(rs.camera_info.serial_number)
            return
        serials = [d.get_info(rs.camera_info.serial_number) for d in devices]
        if self.serial not in serials:
            raise RuntimeError(
                f"RealSense serial {self.serial} is not connected. "
                f"Connected: {', '.join(serials) or '(none)'}. Click Refresh."
            )

    def _start_profile(
        self,
        rs,
        width: int,
        height: int,
        fps: int,
        pixel_format: str,
    ):
        pipeline = rs.pipeline()
        config = rs.config()
        if self.serial:
            config.enable_device(self.serial)
        fmt = pixel_format.lower()
        if fmt == "z16":
            config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
            self._colorizer = rs.colorizer()
        elif fmt == "y8":
            try:
                config.enable_stream(rs.stream.infrared, width, height, rs.format.y8, fps)
            except Exception:  # noqa: BLE001
                config.enable_stream(rs.stream.color, width, height, rs.format.y8, fps)
            self._colorizer = None
        else:
            config.enable_stream(
                rs.stream.color, width, height, self._rs_format(rs, fmt), fps
            )
            self._colorizer = None
        if self.bag_path:
            # Absolute path — relative paths can fail silently on Windows SDK builds.
            # Newer librealsense requires .db3 (not legacy .bag); coerce here so even
            # a stale caller that passes .pending_*.bag cannot crash Start preview.
            from poc1.bag_recorder import coerce_record_path, set_recording_suffix

            bag = coerce_record_path(Path(self.bag_path)).resolve()
            bag.parent.mkdir(parents=True, exist_ok=True)
            if bag.exists() and bag.is_file() and bag.stat().st_size == 0:
                bag.unlink(missing_ok=True)
            try:
                config.enable_record_to_file(str(bag))
            except Exception as exc:  # noqa: BLE001
                msg = str(exc).lower()
                if "db3" in msg and bag.suffix.lower() != ".db3":
                    set_recording_suffix(".db3")
                    bag = bag.with_suffix(".db3")
                    config.enable_record_to_file(str(bag))
                elif "bag" in msg and "extension" in msg and bag.suffix.lower() != ".bag":
                    set_recording_suffix(".bag")
                    bag = bag.with_suffix(".bag")
                    config.enable_record_to_file(str(bag))
                else:
                    raise
            self.bag_path = bag
            self._bag_path = bag
            set_recording_suffix(bag.suffix)
        profile = pipeline.start(config)
        return pipeline, profile

    def start(self, *, allow_fallback: bool = True) -> None:
        if not realsense_available():
            raise RuntimeError(
                "pyrealsense2 is not installed. Run: "
                "uv sync --extra dev --extra realsense"
            )
        import pyrealsense2 as rs

        self._require_device(rs)
        wanted = (
            self.width,
            self.height,
            self.target_fps if self.target_fps > 0 else 30,
            self.pixel_format,
        )
        self._wanted = wanted
        w, h, fps, fmt = wanted
        # When arming .bag (or restoring after a failed bag), never silently drop
        # to 640x480 — that mismatch is what users saw in Record error dialogs.
        strict = bool(self.bag_path) or not allow_fallback
        if strict:
            attempts = [
                (w, h, fps, fmt),
                (w, h, fps, "bgr8") if fmt not in {"z16", "y8"} else (w, h, fps, fmt),
                (w, h, fps, "yuyv") if fmt not in {"z16", "y8"} else (w, h, fps, fmt),
            ]
        else:
            attempts = [
                (w, h, fps, fmt),
                (w, h, fps, "bgr8") if fmt != "z16" else (w, h, fps, "z16"),
                (w, h, fps, "yuyv") if fmt not in {"z16", "y8"} else (w, h, fps, fmt),
                (1280, 720, fps, "bgr8") if fmt != "z16" else (848, 480, fps, "z16"),
                (640, 480, fps, "bgr8") if fmt != "z16" else (640, 480, fps, "z16"),
                (1280, 720, 30, "bgr8"),
                (640, 480, 30, "bgr8"),
                (848, 480, 30, "bgr8"),
            ]
        seen: set[tuple[int, int, int, str]] = set()
        errors: list[str] = []
        pipeline = None
        profile = None
        used = attempts[0]
        for attempt in attempts:
            if attempt in seen:
                continue
            seen.add(attempt)
            width, height, try_fps, try_fmt = attempt
            try:
                if pipeline is not None:
                    try:
                        pipeline.stop()
                    except Exception:  # noqa: BLE001
                        pass
                    pipeline = None
                pipeline, profile = self._start_profile(rs, width, height, try_fps, try_fmt)
                used = attempt
                self.pixel_format = try_fmt
                break
            except Exception as exc:  # noqa: BLE001
                msg = str(exc)
                errors.append(f"{width}x{height}@{try_fps} {try_fmt}: {exc}")
                pipeline = None
                profile = None
                # Auto-fix wrong record extension mid-profile loop (SDK demands .db3).
                if self.bag_path and "db3" in msg.lower():
                    from poc1.bag_recorder import coerce_record_path, set_recording_suffix

                    set_recording_suffix(".db3")
                    self.bag_path = coerce_record_path(Path(self.bag_path))
                    self._bag_path = self.bag_path
                    try:
                        pipeline, profile = self._start_profile(
                            rs, width, height, try_fps, try_fmt
                        )
                        used = attempt
                        self.pixel_format = try_fmt
                        break
                    except Exception as exc2:  # noqa: BLE001
                        errors.append(
                            f"{width}x{height}@{try_fps} {try_fmt} (.db3 retry): {exc2}"
                        )
                        pipeline = None
                        profile = None

        if pipeline is None or profile is None:
            raise RuntimeError(
                "RealSense rejected every profile tried. "
                "Close Intel RealSense Viewer, reconnect USB3, then Refresh. "
                + " | ".join(errors[-3:])
            )

        try:
            if self.pixel_format == "z16":
                stream = profile.get_stream(rs.stream.depth).as_video_stream_profile()
            elif self.pixel_format == "y8":
                try:
                    stream = profile.get_stream(rs.stream.infrared).as_video_stream_profile()
                except Exception:  # noqa: BLE001
                    stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
            else:
                stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
            self.width = stream.width()
            self.height = stream.height()
            self.target_fps = int(stream.fps())
            self.actual_width = self.width
            self.actual_height = self.height
            self.actual_fps = float(self.target_fps)
        except Exception as exc:  # noqa: BLE001
            self.stop()
            raise RuntimeError(f"Could not read RealSense stream profile: {exc}") from exc

        if used != wanted:
            logger.warning(
                "RealSense opened %dx%d@%d %s instead of requested %dx%d@%d %s",
                self.width,
                self.height,
                self.target_fps,
                self.pixel_format,
                wanted[0],
                wanted[1],
                wanted[2],
                wanted[3],
            )

        self._pipeline = pipeline
        self._profile = profile
        self._rs_recorder = None
        if self.bag_path:
            # enable_record_to_file already writes the .bag. as_recorder() is only
            # needed for pause/resume — must NOT abort if the cast fails (common
            # on some Windows SDK builds) or we lose bag writes entirely.
            try:
                self._rs_recorder = profile.get_device().as_recorder()
                if self.bag_start_paused and self._rs_recorder is not None:
                    self._rs_recorder.pause()
                    logger.info("RealSense .bag pre-armed (paused) -> %s", self.bag_path)
                else:
                    logger.info("RealSense .bag recording active -> %s", self.bag_path)
            except Exception as exc:  # noqa: BLE001
                self._rs_recorder = None
                logger.warning(
                    "as_recorder() unavailable (%s) — continuing with "
                    "enable_record_to_file only",
                    exc,
                )

        try:
            first: Optional[np.ndarray] = None
            for _ in range(8):
                frames = pipeline.wait_for_frames(timeout_ms=3000)
                first = self._frames_to_bgr(frames)
                if first is not None and first.size:
                    break
            if first is None:
                raise RuntimeError("SDK returned framesets without a usable frame")
            self._pending_frame = first
            if float(np.mean(first)) < 1.5:
                logger.warning(
                    "RealSense first frames are nearly black — lens cap / AE settling?"
                )
        except Exception as exc:
            self.stop()
            raise RuntimeError(
                "RealSense stream opened but no frame arrived. "
                "Close Intel RealSense Viewer and other camera apps, use USB 3, "
                f"then try again (wanted {wanted[0]}x{wanted[1]}@{wanted[2]} {wanted[3]}). "
                f"SDK error: {exc}"
            ) from exc

        logger.info(
            "D1 RealSense: serial=%s %dx%d@%d fmt=%s bag=%s paused=%s",
            self.serial,
            self.width,
            self.height,
            self.target_fps,
            self.pixel_format,
            bool(self.bag_path),
            bool(self.bag_start_paused and self._rs_recorder is not None),
        )

    def pause_bag(self) -> None:
        if self._rs_recorder is not None:
            try:
                self._rs_recorder.pause()
            except Exception as exc:  # noqa: BLE001
                logger.warning("pause_bag: %s", exc)

    def resume_bag(self) -> None:
        if self._rs_recorder is None:
            raise RuntimeError("No RealSense .bag recorder is armed")
        self._rs_recorder.resume()
        self.bag_start_paused = False

    def stop(self) -> None:
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:  # noqa: BLE001
                pass
            self._pipeline = None
        self._profile = None
        self._rs_recorder = None
        self._pending_frame = None
        self._colorizer = None

    def _frames_to_bgr(self, frames: Any) -> Optional[np.ndarray]:
        if self.pixel_format == "z16":
            depth = frames.get_depth_frame()
            if not depth:
                return None
            if self._colorizer is None:
                import pyrealsense2 as rs

                self._colorizer = rs.colorizer()
            colorized = self._colorizer.colorize(depth)
            frame = np.asanyarray(colorized.get_data())
            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            return np.ascontiguousarray(frame)

        if self.pixel_format == "y8":
            ir = frames.get_infrared_frame()
            if ir:
                frame = np.asanyarray(ir.get_data())
                if frame.ndim == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                return np.ascontiguousarray(frame)
            color = frames.get_color_frame()
            if not color:
                return None
            return self._convert_color(color)

        color = frames.get_color_frame()
        if not color:
            return None
        return self._convert_color(color)

    def _convert_color(self, color: Any) -> np.ndarray:
        frame = np.asanyarray(color.get_data())
        if self.pixel_format == "rgb8":
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        elif self.pixel_format == "yuyv":
            frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_YUY2)
        elif self.pixel_format == "y8":
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        return np.ascontiguousarray(frame)

    def read(self) -> Optional[np.ndarray]:
        if self._pending_frame is not None:
            frame = self._pending_frame
            self._pending_frame = None
            return frame
        if self._pipeline is None:
            return None
        try:
            frames = self._pipeline.wait_for_frames(timeout_ms=1000)
        except Exception:  # noqa: BLE001
            return None
        return self._frames_to_bgr(frames)



def build_frame_source(
    camera: ConnectedCamera,
    mode: StreamMode,
    *,
    allow_simulate_realsense: bool = False,
):
    """
    Build a POC-1-compatible frame source for the given camera + mode.

    RealSense color streams that are not bgr8 are still opened as bgr8 when
    possible (OpenCV/pipeline expect BGR ndarrays); depth-only formats fall back.
    Simulation is off by default so a listed RealSense never silently becomes
    a fake source in the GUI.
    """
    if camera.kind == "fake":
        return FakeFrameSource(
            width=mode.width, height=mode.height, target_fps=mode.fps
        )

    if camera.kind == "realsense":
        if not realsense_available():
            raise RuntimeError(
                "RealSense selected but pyrealsense2 is not installed. "
                "Run: uv sync --extra realsense"
            )
        if camera.serial:
            return ConfiguredRealSenseSource(
                serial=camera.serial,
                width=mode.width,
                height=mode.height,
                fps=mode.fps,
                pixel_format=mode.pixel_format,
            )
        return create_realsense_source(
            width=mode.width,
            height=mode.height,
            fps=mode.fps,
            serial=camera.serial,
            allow_simulate=allow_simulate_realsense,
        )

    backend = camera.backend
    if camera.device_tag == "elgato" and sys.platform == "win32":
        backend = cv2.CAP_DSHOW
    elif backend is None:
        backend = _opencv_backends()[0][0]
    src = FormattedUvcSource(
        device_index=int(camera.index or 0),
        width=mode.width,
        height=mode.height,
        fps=mode.fps,
        backend=backend,
        pixel_format=mode.pixel_format,
        open_path=camera.open_path,
        device_tag=camera.device_tag,
    )
    # Webcam/virtual: remux + preview-based stamp (drivers often lie about FPS).
    # Elgato: stamp from measured delivery at Record; remux remains a safety net.
    src.allow_fps_remux = True
    return src
