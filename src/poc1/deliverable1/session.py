"""
R3–R6: multi-camera session — preview ≥2 streams, per-slot arming,
shared save folder / prefixes, single Record for all armed slots.

Each slot owns its own POC-1 Pipeline instance (no shared camera handles).
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from poc1.preview_draw import PREVIEW_HZ, downscale_for_preview
from poc1.bag_recorder import clear_bag_on_source, with_sdk_record_suffix
from poc1.camera_handler import FrameEnvelope
from poc1.deliverable1.devices import (
    ConnectedCamera,
    StreamMode,
    build_frame_source,
    list_all_cameras,
    list_stream_modes,
    prefix_for_camera,
)
from poc1.pipeline import Pipeline

logger = logging.getLogger("poc1.d1.session")

DEFAULT_PREFIXES = ("m", "r")


def _default_prefix(index: int) -> str:
    if index < len(DEFAULT_PREFIXES):
        return DEFAULT_PREFIXES[index]
    return f"cam{index + 1}"


@dataclass
class CameraSlot:
    """One camera lane in the multi-cam UI / session."""

    slot_id: int
    prefix: str = "cam1"
    armed: bool = True
    camera: Optional[ConnectedCamera] = None
    mode: Optional[StreamMode] = None
    available_modes: list[StreamMode] = field(default_factory=list)
    pipeline: Optional[Pipeline] = None
    last_frame: Optional[object] = None  # numpy array
    last_report: dict = field(default_factory=dict)
    status: str = "idle"
    record_bag: bool = False
    _frame_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _preview_ts: list[float] = field(default_factory=list, repr=False)
    _preview_draw_t: float = field(default=0.0, repr=False)

    def on_preview(self, env: FrameEnvelope) -> None:
        # Preview is display-only. Do not copy full 1080p@120 onto the UI path.
        now = time.monotonic()
        min_dt = 1.0 / PREVIEW_HZ
        if now - self._preview_draw_t < min_dt:
            return
        if env.frame is None:
            return
        small = downscale_for_preview(env.frame)
        self._preview_draw_t = now
        with self._frame_lock:
            self.last_frame = small
            self._preview_ts.append(time.time())
            if len(self._preview_ts) > 90:
                self._preview_ts = self._preview_ts[-90:]

    def get_preview_frame(self):
        with self._frame_lock:
            if self.last_frame is None:
                return None
            return self.last_frame

    def estimate_preview_fps(self) -> Optional[float]:
        """Recent live preview rate (informational only — never changes record stamp)."""
        with self._frame_lock:
            stamps = list(self._preview_ts)
        if len(stamps) < 8:
            return None
        elapsed = stamps[-1] - stamps[0]
        if elapsed <= 0.05:
            return None
        return (len(stamps) - 1) / elapsed


class MultiCamSession:
    """
    Manages N camera slots (default 2).

    R3: start_previews() streams all slots that have a camera assigned.
    R4: out_dir + per-slot prefix.
    R5: slot.armed
    R6: start_recording_armed() / stop_recording_armed()
    """

    def __init__(
        self,
        n_slots: int = 2,
        out_dir: Path | str = Path("./recordings/deliverable1"),
    ) -> None:
        if n_slots < 2:
            raise ValueError("Deliverable 1 requires at least 2 camera slots (R3)")
        self.out_dir = Path(out_dir)
        self.devices: list[ConnectedCamera] = []
        self.slots: list[CameraSlot] = [
            CameraSlot(slot_id=i, prefix=_default_prefix(i), armed=True)
            for i in range(n_slots)
        ]
        self._recording = False
        self._lock = threading.Lock()
        self.last_start_warnings: list[str] = []
        # Authoritative .bag intent for the next/current Record (slot_id -> bool).
        # UI checkboxes can race with busy/lock; Record uses this map only.
        self.bag_intent: dict[int, bool] = {}

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def previews_running(self) -> bool:
        return any(slot.pipeline is not None for slot in self.slots)

    def add_slot(self) -> CameraSlot:
        """Add another independently configurable camera lane (R3)."""
        if self._recording:
            raise RuntimeError("Stop recording before adding a camera slot")
        index = len(self.slots)
        slot = CameraSlot(
            slot_id=index,
            prefix=_default_prefix(index),
            armed=True,
        )
        self.slots.append(slot)
        logger.info("D1 camera slot %d added", index + 1)
        return slot

    def remove_slot(self, slot_id: int) -> None:
        """Remove a camera lane while retaining the R3 minimum of two."""
        if self._recording:
            raise RuntimeError("Stop recording before removing a camera slot")
        if len(self.slots) <= 2:
            raise RuntimeError("At least 2 camera slots are required (R3)")
        if slot_id < 0 or slot_id >= len(self.slots):
            raise IndexError(f"Invalid camera slot: {slot_id}")
        self.stop_slot_preview(slot_id)
        self.slots.pop(slot_id)
        # Public methods use list position as slot_id; keep them aligned.
        for index, slot in enumerate(self.slots):
            slot.slot_id = index
        logger.info("D1 camera slot %d removed", slot_id + 1)

    def refresh_devices(
        self,
        *,
        include_fake: bool = True,
        probe_uvc: bool = True,
        probe_realsense: bool = True,
        refresh_name_cache: bool = True,
    ) -> list[ConnectedCamera]:
        """R1 — rescan UVC + RealSense (+ optional fake for demos/tests)."""
        if self.previews_running:
            raise RuntimeError("Stop previews before refreshing the camera list")
        self.devices = list_all_cameras(
            include_fake=include_fake,
            probe_uvc=probe_uvc,
            probe_realsense=probe_realsense,
            refresh_name_cache=refresh_name_cache,
        )
        logger.info("D1 devices: %d found", len(self.devices))
        return list(self.devices)

    def find_device(self, cam_id: str) -> Optional[ConnectedCamera]:
        for d in self.devices:
            if d.cam_id == cam_id:
                return d
        return None

    def assign_camera(self, slot_id: int, cam_id: str) -> CameraSlot:
        """Bind a listed camera to a slot and load its R2 mode list."""
        slot = self.slots[slot_id]
        if slot.pipeline is not None:
            self.stop_slot_preview(slot_id)
        cam = self.find_device(cam_id)
        if cam is None:
            raise ValueError(f"Unknown camera id: {cam_id}")
        # Avoid assigning the same physical device to two slots.
        for other in self.slots:
            if other.slot_id == slot_id:
                continue
            if other.camera and other.camera.cam_id == cam_id:
                raise ValueError(
                    f"Camera {cam_id} already assigned to slot {other.slot_id}"
                )
        slot.camera = cam
        slot.available_modes = list_stream_modes(cam)
        slot.mode = slot.available_modes[0] if slot.available_modes else StreamMode(
            1280, 720, 30, "bgr8"
        )
        wanted = prefix_for_camera(cam)
        used = {
            s.prefix.casefold()
            for s in self.slots
            if s.slot_id != slot_id and s.camera is not None
        }
        prefix = wanted
        n = 2
        while prefix.casefold() in used:
            prefix = f"{wanted}{n}"
            n += 1
        slot.prefix = prefix
        # RealSense SDK file + Elgato ROS2 color bag default ON.
        slot.record_bag = cam.kind == "realsense" or cam.device_tag == "elgato"
        slot.status = f"assigned {cam.label()}"
        return slot

    def set_mode(self, slot_id: int, mode: StreamMode) -> None:
        """R2 — change resolution / fps / format (applies on next preview start)."""
        slot = self.slots[slot_id]
        if slot.pipeline is not None:
            self.stop_slot_preview(slot_id)
        slot.mode = mode
        slot.status = f"mode {mode.label()}"

    def set_prefix(self, slot_id: int, prefix: str) -> None:
        """R4 — naming prefix for this slot's files."""
        p = (prefix or "").strip() or _default_prefix(slot_id)
        self.slots[slot_id].prefix = p

    def set_armed(self, slot_id: int, armed: bool) -> None:
        """R5 — include/exclude this slot from the next Record."""
        self.slots[slot_id].armed = bool(armed)

    def set_out_dir(self, path: Path | str) -> None:
        """R4 — shared save folder."""
        value = str(path).strip()
        if not value:
            raise ValueError("Save folder cannot be empty")
        self.out_dir = Path(value)

    def start_slot_preview(self, slot_id: int) -> None:
        slot = self.slots[slot_id]
        if slot.camera is None or slot.mode is None:
            raise RuntimeError(f"Slot {slot_id}: assign camera + mode first")
        if slot.pipeline is not None:
            return
        # Never silently substitute a simulated RealSense for a hardware pick.
        source = build_frame_source(
            slot.camera, slot.mode, allow_simulate_realsense=False
        )
        # Preview must never enable SDK record_to_file (some SDKs reject .bag
        # and would fail Start preview when the checkbox is on).
        clear_bag_on_source(source)
        pipe = Pipeline(source=source, on_preview_frame=slot.on_preview)
        pipe.start_preview()
        slot.pipeline = pipe
        slot.status = f"preview {slot.camera.kind} {slot.mode.label()}"
        if slot.record_bag and slot.camera.kind == "realsense":
            slot.status += " (.bag/.db3 on Record)"
        logger.info("Slot %d preview started (%s)", slot_id, slot.status)

    def stop_slot_preview(self, slot_id: int) -> None:
        slot = self.slots[slot_id]
        if slot.pipeline is None:
            return
        try:
            if slot.pipeline.camera_handler.is_recording:
                slot.last_report = slot.pipeline.stop_recording()
            slot.pipeline.stop()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Slot %d stop preview: %s", slot_id, exc)
        slot.pipeline = None
        slot.status = "idle"

    def start_previews(self) -> None:
        """R3 — start preview on every assigned slot.

        When any Elgato is @≥90, open Elgato first (exclusive DirectShow pin like
        OBS) then RealSense. Otherwise RealSense-first (USB settle for Station A).
        """
        errors: list[str] = []
        want_elgato_first = any(
            s.camera is not None
            and s.camera.device_tag == "elgato"
            and s.mode is not None
            and int(s.mode.fps) >= 90
            for s in self.slots
        )

        def _order(slot: CameraSlot) -> tuple:
            cam = slot.camera
            if cam is None:
                return (9, slot.slot_id)
            if want_elgato_first:
                if cam.device_tag == "elgato":
                    return (0, slot.slot_id)
                if cam.kind == "realsense":
                    return (1, slot.slot_id)
                return (2, slot.slot_id)
            if cam.kind == "realsense":
                return (0, slot.slot_id)
            if cam.device_tag == "elgato":
                return (1, slot.slot_id)
            return (2, slot.slot_id)

        ordered = sorted(
            [s for s in self.slots if s.camera is not None],
            key=_order,
        )
        for slot in ordered:
            try:
                self.start_slot_preview(slot.slot_id)
                # Brief pause so USB / exclusive access settles before the next open.
                if slot.camera and (
                    slot.camera.kind == "realsense"
                    or (want_elgato_first and slot.camera.device_tag == "elgato")
                ):
                    time.sleep(0.35)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"slot{slot.slot_id}: {exc}")
                slot.status = f"error: {exc}"
                logger.exception("start_slot_preview failed for slot %d", slot.slot_id)
        if errors:
            raise RuntimeError("; ".join(errors))

    def stop_previews(self) -> None:
        for slot in self.slots:
            self.stop_slot_preview(slot.slot_id)

    def start_recording_armed(
        self,
        stamp: Optional[str] = None,
        bag_intent: Optional[dict[int, bool]] = None,
    ) -> list[Path]:
        """
        R6 — one Record action: start encode on every armed slot.

        ``bag_intent`` is the authoritative map of slot_id -> want .bag. When
        provided it overrides checkbox/slot.record_bag (avoids UI lock races).
        """
        with self._lock:
            if self._recording:
                raise RuntimeError("Already recording")
            self.last_start_warnings = []
            if bag_intent is not None:
                self.bag_intent = {int(k): bool(v) for k, v in bag_intent.items()}
            intent = dict(self.bag_intent)

            armed = [s for s in self.slots if s.armed]
            if not armed:
                raise RuntimeError("No cameras armed — enable Armed on at least one slot")

            # Elgato/UVC first so DirectShow is recording before RealSense bag
            # loads the USB bus (dual-cam left Elgato at frames_read=0 on v26).
            def _record_start_priority(slot: CameraSlot) -> int:
                cam = slot.camera
                if cam is not None and cam.device_tag == "elgato":
                    return 0
                if cam is not None and cam.kind == "uvc":
                    return 1
                return 2

            armed = sorted(armed, key=_record_start_priority)

            for s in armed:
                if s.camera is None or s.mode is None:
                    raise RuntimeError(
                        f"Camera {s.slot_id + 1} ({s.prefix}) is armed but has no "
                        "device/configuration. Assign a camera and mode, or uncheck Armed."
                    )
                # Keep slot.record_bag aligned with authoritative intent for UI restore.
                want_bag = bool(intent.get(s.slot_id, s.record_bag))
                if s.camera.kind == "realsense" or s.camera.device_tag == "elgato":
                    s.record_bag = want_bag
                else:
                    s.record_bag = False
                if s.pipeline is None or not s.pipeline._preview_started:
                    try:
                        # Drop any half-open pipeline, then start fresh.
                        if s.pipeline is not None:
                            try:
                                s.pipeline.stop()
                            except Exception:  # noqa: BLE001
                                pass
                            s.pipeline = None
                        self.start_slot_preview(s.slot_id)
                    except Exception as exc:
                        raise RuntimeError(
                            f"Camera {s.slot_id + 1} ({s.prefix}) could not start "
                            f"preview for Record: {exc}\n\n"
                            "Check HDMI/USB, close other camera apps, then try again "
                            "— or uncheck Armed on this camera."
                        ) from exc
                if s.pipeline is not None and s.pipeline.source.target_fps <= 0:
                    s.pipeline.source.target_fps = 30

            prefixes = [s.prefix.casefold() for s in armed]
            if len(prefixes) != len(set(prefixes)):
                raise RuntimeError(
                    "Armed cameras must use different naming prefixes "
                    "(for example m and r)"
                )

            self.out_dir.mkdir(parents=True, exist_ok=True)
            if not self.out_dir.is_dir():
                raise RuntimeError(f"Save folder is not a directory: {self.out_dir}")
            meta_dir = self.out_dir / "meta"
            meta_dir.mkdir(parents=True, exist_ok=True)
            stamp = stamp or datetime.now().strftime("%Y%m%d_%H%M%S")
            paths: list[Path] = []
            started: list[CameraSlot] = []
            try:
                for s in armed:
                    if s.pipeline is not None:
                        try:
                            live = float(
                                s.pipeline.camera_handler.live_delivery_fps() or 0
                            )
                        except Exception:  # noqa: BLE001
                            live = 0.0
                        if live >= 5:
                            s.pipeline._preview_fps_hint = live
                    mp4 = self.out_dir / f"{s.prefix}_{stamp}.mp4"
                    csv = meta_dir / f"{s.prefix}_{stamp}_sysmon.csv"
                    bag = None
                    want_bag = bool(intent.get(s.slot_id, s.record_bag))
                    if want_bag:
                        if not s.camera:
                            raise RuntimeError(
                                f"Camera {s.slot_id + 1} ({s.prefix}): "
                                ".bag is checked but no device is assigned."
                            )
                        if s.camera.kind == "realsense":
                            src = s.pipeline.source if s.pipeline else None
                            if getattr(src, "mode", "") != "hardware":
                                raise RuntimeError(
                                    f"Camera {s.slot_id + 1} ({s.prefix}): "
                                    "RealSense .bag needs a live hardware stream."
                                )
                            bag = with_sdk_record_suffix(
                                self.out_dir / f"{s.prefix}_{stamp}.bag"
                            )
                        elif s.camera.device_tag == "elgato":
                            # ROS2 color bag (JPEG), not Intel SDK .bag.
                            bag = self.out_dir / f"{s.prefix}_{stamp}_color"
                        else:
                            raise RuntimeError(
                                f"Camera {s.slot_id + 1} ({s.prefix}): "
                                "SDK/ROS bag is only for RealSense or Elgato."
                            )
                        s.record_bag = True
                    assert s.pipeline is not None
                    logger.info(
                        "Record slot %d prefix=%s bag_intent=%s bag_path=%s",
                        s.slot_id,
                        s.prefix,
                        want_bag,
                        bag,
                    )
                    try:
                        s.pipeline.start_recording(mp4, csv, bag_path=bag)
                    except Exception as exc:
                        # Leave preview alive and clean up any zero-byte/partial
                        # writer created by a failed start.
                        try:
                            s.pipeline.processor.stop()
                        except Exception:  # noqa: BLE001
                            pass
                        if mp4.exists() and mp4.stat().st_size == 0:
                            mp4.unlink(missing_ok=True)
                        raise RuntimeError(
                            f"Camera {s.slot_id + 1} ({s.prefix}) could not start "
                            f"recording at {s.pipeline.source.width}x"
                            f"{s.pipeline.source.height}@"
                            f"{s.pipeline.source.target_fps}: {exc}"
                        ) from exc
                    if bag is not None and s.camera and s.camera.kind == "realsense" and (
                        s.pipeline._bag_path is None
                        or not Path(bag).exists()
                        or Path(bag).stat().st_size <= 0
                    ):
                        try:
                            s.pipeline.stop_recording()
                        except Exception:  # noqa: BLE001
                            pass
                        raise RuntimeError(
                            f"Camera {s.slot_id + 1} ({s.prefix}): .bag was checked but "
                            f"no bag file was created at {bag.name}. "
                            "Close RealSense Viewer, use USB 3, then try again."
                        )
                    s.status = f"recording → {mp4.name}"
                    if bag is not None:
                        s.status += f" + {bag.name}"
                    paths.append(mp4)
                    started.append(s)
            except Exception:
                for s in started:
                    try:
                        if s.pipeline:
                            s.pipeline.stop_recording()
                    except Exception:  # noqa: BLE001
                        pass
                raise
            self._recording = True
            logger.info("D1 recording started on %d armed slot(s)", len(paths))
            return paths

    def stop_recording_armed(self) -> dict[str, dict]:
        """
        Stop all active recordings; return per-prefix reports.

        Disables capture into the encode path first (so queues stop growing),
        then finalizes each pipeline. The session lock is not held during
        encoder shutdown so Stop cannot deadlock the UI / other session calls.
        """
        with self._lock:
            if not self._recording:
                return {}
            targets = [
                s
                for s in self.slots
                if s.pipeline is not None and s.pipeline.camera_handler.is_recording
            ]
            # Flip the flag early so the GUI can treat recording as ending.
            self._recording = False

        # Phase 1: stop feeding the encoder immediately (all cameras).
        for s in targets:
            try:
                assert s.pipeline is not None
                s.pipeline.camera_handler.disable_recording()
                s.status = "stopping…"
            except Exception as exc:  # noqa: BLE001
                logger.warning("disable_recording failed for %s: %s", s.prefix, exc)

        reports: dict[str, dict] = {}

        def _stop_one(slot: CameraSlot) -> tuple[str, dict]:
            try:
                assert slot.pipeline is not None
                report = slot.pipeline.stop_recording()
                slot.last_report = report
                drops = "OK" if report.get("no_frame_drops") else "DROPS"
                slot.status = (
                    f"stopped [{drops}] written={report.get('frames_written')}"
                )
                return slot.prefix, report
            except Exception as exc:  # noqa: BLE001
                logger.exception("stop_recording failed for %s", slot.prefix)
                slot.status = f"stop error: {exc}"
                return slot.prefix, {"error": str(exc)}

        # Phase 2: finalize writers (independent pipelines — parallel is fine).
        if len(targets) <= 1:
            for s in targets:
                prefix, report = _stop_one(s)
                reports[prefix] = report
        else:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            with ThreadPoolExecutor(
                max_workers=min(4, len(targets)), thread_name_prefix="d1-stop"
            ) as pool:
                futures = [pool.submit(_stop_one, s) for s in targets]
                for fut in as_completed(futures):
                    prefix, report = fut.result()
                    reports[prefix] = report

        return reports

    def shutdown(self) -> None:
        try:
            if self._recording:
                self.stop_recording_armed()
        finally:
            self.stop_previews()
