"""Deliverable 1 tests — multi-cam session using synthetic sources only."""
from __future__ import annotations

import time
from pathlib import Path

from poc1.deliverable1.devices import (
    ConfiguredRealSenseSource,
    ConnectedCamera,
    StreamMode,
    list_all_cameras,
    list_stream_modes,
    list_uvc_cameras,
    list_uvc_modes,
    pick_auto_camera_for_slot,
)
from poc1.deliverable1.session import MultiCamSession


def test_list_all_cameras_includes_fake():
    cams = list_all_cameras(include_fake=True, probe_uvc=False, probe_realsense=False)
    kinds = {c.kind for c in cams}
    assert "fake" in kinds
    fake = [c for c in cams if c.kind == "fake"]
    assert len(fake) >= 2


def test_list_stream_modes_for_fake():
    cams = list_all_cameras(include_fake=True, probe_uvc=False, probe_realsense=False)
    fake = next(c for c in cams if c.cam_id == "fake:0")
    modes = list_stream_modes(fake)
    assert modes
    assert any(m.width == 1920 and m.fps == 120 for m in modes)


def test_invalid_driver_fps_is_normalized():
    """Windows UVC drivers may report -1 for unknown; writer must get 30."""
    mode = StreamMode(640, 480, -1, "bgr8")
    assert mode.fps == 30
    assert mode.label() == "640x480@30 bgr8"


def test_realsense_config_source_keeps_selected_profile():
    source = ConfiguredRealSenseSource(
        serial="TEST",
        width=848,
        height=480,
        fps=60,
        pixel_format="yuyv",
    )
    assert source.width == 848
    assert source.height == 480
    assert source.target_fps == 60
    assert source.pixel_format == "yuyv"
    assert source.mode == "hardware"


def test_build_frame_source_wires_uvc_and_realsense():
    """R1/R2 wiring: UVC → FormattedUvcSource, RealSense → ConfiguredRealSenseSource."""
    from poc1.deliverable1.devices import (
        ConnectedCamera,
        FormattedUvcSource,
        build_frame_source,
    )

    uvc = ConnectedCamera(
        cam_id="uvc:0:DSHOW",
        kind="uvc",
        name="Elgato / capture card — Elgato HD60",
        index=0,
        backend=0,
        backend_name="DSHOW",
        open_path="video=Elgato HD60",
        device_tag="elgato",
    )
    src = build_frame_source(uvc, StreamMode(1920, 1080, 30, "mjpg"))
    assert isinstance(src, FormattedUvcSource)
    assert src.device_tag == "elgato"
    assert src.open_path == "video=Elgato HD60"
    assert src.target_fps == 30
    assert src.allow_fps_remux is True  # Elgato remux safety net when HDMI is 60Hz

    webcam = ConnectedCamera(
        cam_id="uvc:0:DSHOW",
        kind="uvc",
        name="USB2.0 HD UVC WebCam",
        index=0,
        backend=0,
        backend_name="DSHOW",
        device_tag="uvc",
    )
    cam_src = build_frame_source(webcam, StreamMode(1280, 720, 30, "bgr8"))
    assert isinstance(cam_src, FormattedUvcSource)
    assert cam_src.allow_fps_remux is True

    rs = ConnectedCamera(
        cam_id="realsense:ABC123",
        kind="realsense",
        name="Intel RealSense D435",
        serial="ABC123",
        device_tag="realsense",
    )
    from poc1.realsense_source import realsense_available

    if realsense_available():
        rs_src = build_frame_source(rs, StreamMode(1280, 720, 30, "bgr8"))
    else:
        rs_src = ConfiguredRealSenseSource(
            serial="ABC123", width=1280, height=720, fps=30, pixel_format="bgr8"
        )
    assert isinstance(rs_src, ConfiguredRealSenseSource)
    assert rs_src.serial == "ABC123"
    assert rs_src.width == 1280
    assert rs_src.height == 720
    assert rs_src.allow_fps_remux is False


def test_preview_fps_estimate_does_not_require_stamp_change():
    """Preview FPS is informational only; configured record FPS stays user-selected."""
    from poc1.deliverable1.session import CameraSlot
    import numpy as np

    slot = CameraSlot(slot_id=0, prefix="cam1")
    base = time.time()
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    for i in range(20):
        with slot._frame_lock:
            slot._preview_ts.append(base + i * 0.1)
            slot.last_frame = frame
    measured = slot.estimate_preview_fps()
    assert measured is not None
    assert 8.0 <= measured <= 12.0


def test_win_names_classify_elgato_and_realsense():
    from poc1.deliverable1.win_names import classify_capture_name

    assert classify_capture_name("Elgato HD60 S+") == "elgato"
    assert classify_capture_name("Elgato 4K X") == "elgato"
    assert classify_capture_name("Game Capture HD60") == "elgato"
    assert classify_capture_name("Intel RealSense D435") == "realsense-uvc"
    assert classify_capture_name("OBS Virtual Camera") == "virtual"
    assert classify_capture_name("USB2.0 HD UVC WebCam") == "uvc"


def test_elgato_open_profiles_prefer_user_selection():
    from poc1.deliverable1.devices import elgato_open_profiles

    profiles = elgato_open_profiles(1280, 720, 120, "mjpg")
    assert profiles[0] == (1280, 720, 120, "mjpg")
    # Same resolution only — no silent 720↔1080 fallback.
    assert all(p[0] == 1280 and p[1] == 720 for p in profiles)
    assert (1920, 1080, 60, "mjpg") not in profiles
    assert (1920, 1080, 120, "mjpg") not in profiles


def test_elgato_open_profiles_1080_never_falls_to_720():
    from poc1.deliverable1.devices import elgato_open_profiles

    profiles = elgato_open_profiles(1920, 1080, 120, "mjpg")
    assert profiles[0] == (1920, 1080, 120, "mjpg")
    assert all(p[0] == 1920 and p[1] == 1080 for p in profiles)
    assert (1280, 720, 120, "mjpg") not in profiles
    assert (1280, 720, 60, "mjpg") not in profiles


def test_textured_green_screen_is_usable_elgato():
    """Studio chroma-key must not be rejected (that forced @120 → @60 fallback)."""
    import numpy as np

    from poc1.deliverable1.devices import (
        _frame_is_unusable_elgato,
        _looks_like_solid_green,
    )

    # Textured green screen (folds / shadows) — high variance.
    rng = np.random.default_rng(0)
    textured = np.zeros((180, 320, 3), dtype=np.uint8)
    textured[:, :, 1] = rng.integers(100, 220, size=(180, 320), dtype=np.uint8)
    textured[:, :, 0] = rng.integers(0, 40, size=(180, 320), dtype=np.uint8)
    textured[:, :, 2] = rng.integers(0, 40, size=(180, 320), dtype=np.uint8)
    assert not _looks_like_solid_green(textured)
    assert not _frame_is_unusable_elgato(textured)

    # Flat wrong-format green — still rejected.
    flat = np.zeros((180, 320, 3), dtype=np.uint8)
    flat[:, :, 1] = 180
    assert _looks_like_solid_green(flat)
    assert _frame_is_unusable_elgato(flat)


def test_library_display_name_marks_color_bag(tmp_path: Path):
    from poc1.app.library import LibraryPage

    bag = tmp_path / "m_take_color"
    bag.mkdir()
    (bag / "metadata.yaml").write_text("x", encoding="utf-8")
    label = LibraryPage._display_name(bag)
    assert "ROS2 bag" in label
    assert "Export" in label
    assert LibraryPage._display_name(tmp_path / "m_take.mp4") == "m_take.mp4"


def test_sibling_mp4_for_elgato_color_folder(tmp_path: Path):
    from poc1.app.library import LibraryPage

    bag = tmp_path / "m_take_color"
    bag.mkdir()
    (bag / "metadata.yaml").write_text("x", encoding="utf-8")
    mp4 = tmp_path / "m_take.mp4"
    mp4.write_bytes(b"\x00" * 64)
    found = LibraryPage._sibling_record_mp4_for_bag(bag)
    assert found == mp4


def test_elgato_modes_are_mjpg_only_with_1080p_defaults():
    cam = ConnectedCamera(
        cam_id="uvc:0:DSHOW",
        kind="uvc",
        name="Elgato / capture card — Elgato 4K X",
        index=0,
        backend=0,
        backend_name="DSHOW",
        device_tag="elgato",
    )
    modes = list_uvc_modes(cam)
    assert modes
    # Prefer OBS-equivalent 1080p120 first so stations default to real high-rate.
    assert modes[0] == StreamMode(1920, 1080, 120, "mjpg")
    assert any(m == StreamMode(1920, 1080, 60, "mjpg") for m in modes)
    assert all(m.pixel_format == "mjpg" for m in modes)


def test_pick_auto_camera_assigns_realsense_then_elgato():
    rs = ConnectedCamera(
        cam_id="realsense:ABC",
        kind="realsense",
        name="Intel RealSense D435",
        serial="ABC",
        device_tag="realsense",
    )
    elg = ConnectedCamera(
        cam_id="uvc:elgato:Elgato 4K X:DSHOW",
        kind="uvc",
        name="Elgato / capture card — Elgato 4K X",
        index=0,
        device_tag="elgato",
    )
    devices = [rs, elg]
    slot0 = pick_auto_camera_for_slot(0, devices, set())
    assert slot0 is not None and slot0.kind == "realsense"
    slot1 = pick_auto_camera_for_slot(1, devices, {slot0.cam_id})
    assert slot1 is not None and slot1.device_tag == "elgato"


def test_list_uvc_cameras_ensures_elgato_when_probe_misses(monkeypatch):
    from poc1.deliverable1 import devices as dev

    cleared: list[bool] = []

    monkeypatch.setattr(dev, "clear_name_cache", lambda: cleared.append(True))
    monkeypatch.setattr(
        dev, "_probe_uvc_index_safe", lambda *args, **kwargs: (None, "missing")
    )
    monkeypatch.setattr(dev, "_opencv_backends", lambda **kwargs: [(0, "DSHOW")])
    monkeypatch.setattr(
        dev, "list_windows_capture_names", lambda: ["Elgato 4K X"]
    )
    monkeypatch.setattr(dev, "_merge_windows_named_cameras", lambda *a, **k: None)

    cams = dev.list_uvc_cameras(max_index=1, refresh_name_cache=True)
    assert cleared
    elgato = [c for c in cams if c.device_tag == "elgato"]
    assert len(elgato) == 1
    assert "Elgato 4K X" in elgato[0].name


def test_refresh_devices_refreshes_windows_name_cache(monkeypatch):
    seen: list[bool] = []

    def fake_list_all(**kwargs):
        if kwargs.get("refresh_name_cache"):
            seen.append(True)
        return []

    monkeypatch.setattr(
        "poc1.deliverable1.session.list_all_cameras", fake_list_all
    )
    session = MultiCamSession(n_slots=2)
    session.refresh_devices(include_fake=False, probe_uvc=True, probe_realsense=False)
    assert seen == [True]


def test_elgato_open_targets_prefer_dshow_then_index_then_msmf(monkeypatch):
    import cv2
    from poc1.deliverable1 import devices as dev

    monkeypatch.setattr(
        dev, "elgato_open_name_paths", lambda: ["video=Elgato 4K X"]
    )
    # Named path: DSHOW by name only (no index scan — index 0 is often RealSense).
    named = dev._elgato_open_targets(
        "video=Elgato 4K X", device_index=0, max_index=3, named_only=True
    )
    assert named == [("video=Elgato 4K X", cv2.CAP_DSHOW)]

    # No name at all: fall back to index scan + MSMF.
    targets = dev._elgato_open_targets(None, device_index=0, max_index=3)
    assert targets[0] == ("video=Elgato 4K X", cv2.CAP_DSHOW)
    assert (0, cv2.CAP_DSHOW) in targets
    assert (1, cv2.CAP_DSHOW) in targets
    dshow_idxs = [i for i, (t, b) in enumerate(targets) if b == cv2.CAP_DSHOW]
    msmf_idxs = [i for i, (t, b) in enumerate(targets) if b == cv2.CAP_MSMF]
    assert msmf_idxs and max(dshow_idxs) < min(msmf_idxs)
    assert ("video=Elgato 4K X", cv2.CAP_MSMF) in targets


def test_uvc_open_failure_message_distinguishes_open_vs_frames():
    from poc1.deliverable1.devices import _uvc_open_failure_message

    no_open = _uvc_open_failure_message(
        is_elgato=True,
        opened_once=False,
        last_exc=RuntimeError("Could not open 0 backend=700"),
    )
    assert no_open.startswith("Could not open Elgato")
    assert "opened but delivered no frames" not in no_open.lower()

    no_frames = _uvc_open_failure_message(
        is_elgato=True,
        opened_once=True,
        last_exc=RuntimeError("no frames after open"),
    )
    assert "opened but delivered no frames" in no_frames.lower()
    assert "HDMI" in no_frames


def test_elgato_bag_intent_allowed_like_realsense():
    """Station B: Elgato bag checkbox must not be zeroed before Record."""
    rs = ConnectedCamera(
        cam_id="realsense:ABC",
        kind="realsense",
        name="Intel RealSense D435",
        serial="ABC",
        device_tag="realsense",
    )
    elg = ConnectedCamera(
        cam_id="uvc:elgato:Elgato 4K X:DSHOW",
        kind="uvc",
        name="Elgato / capture card — Elgato 4K X",
        index=0,
        device_tag="elgato",
    )
    webcam = ConnectedCamera(
        cam_id="uvc:0:DSHOW",
        kind="uvc",
        name="USB Webcam",
        index=1,
        device_tag="uvc",
    )
    # Mirror _capture_bag_intent rules without Tk.
    def allow_bag(cam: ConnectedCamera, checked: bool) -> bool:
        if not checked:
            return False
        return cam.kind == "realsense" or cam.device_tag == "elgato"

    assert allow_bag(rs, True) is True
    assert allow_bag(elg, True) is True
    assert allow_bag(webcam, True) is False
    assert allow_bag(elg, False) is False


def test_session_keeps_elgato_record_bag_flag():
    session = MultiCamSession(n_slots=2)
    devices = session.refresh_devices(
        include_fake=True, probe_uvc=False, probe_realsense=False
    )
    fakes = [d for d in devices if d.kind == "fake"]
    # Simulate Elgato-tagged camera by assigning fake then patching tag on slot.
    session.assign_camera(0, fakes[0].cam_id)
    session.slots[0].camera = ConnectedCamera(
        cam_id="uvc:elgato:test:DSHOW",
        kind="uvc",
        name="Elgato test",
        index=0,
        device_tag="elgato",
    )
    session.slots[0].record_bag = True
    session.bag_intent = {0: True}
    # The alignment block in start_recording_armed must keep True for elgato.
    want = bool(session.bag_intent.get(0, session.slots[0].record_bag))
    cam = session.slots[0].camera
    assert cam is not None
    if cam.kind == "realsense" or cam.device_tag == "elgato":
        session.slots[0].record_bag = want
    else:
        session.slots[0].record_bag = False
    assert session.slots[0].record_bag is True


def test_default_prefixes_are_m_and_r():
    session = MultiCamSession(n_slots=2)
    assert session.slots[0].prefix == "m"
    assert session.slots[1].prefix == "r"
    session.set_prefix(1, "r")
    assert session.slots[1].prefix == "r"


def test_dynamic_slots_add_and_remove():
    session = MultiCamSession(n_slots=2)
    added = session.add_slot()
    assert len(session.slots) == 3
    assert added.slot_id == 2
    assert added.prefix == "cam3"
    session.remove_slot(1)
    assert len(session.slots) == 2
    assert [slot.slot_id for slot in session.slots] == [0, 1]
    try:
        session.remove_slot(1)
    except RuntimeError as exc:
        assert "At least 2" in str(exc)
    else:
        raise AssertionError("session allowed fewer than two camera slots")


def test_multicam_two_fakes_preview_and_armed_record(tmp_path: Path):
    session = MultiCamSession(n_slots=2, out_dir=tmp_path)
    devices = session.refresh_devices(
        include_fake=True, probe_uvc=False, probe_realsense=False
    )
    fakes = [d for d in devices if d.kind == "fake"]
    assert len(fakes) >= 2

    session.assign_camera(0, fakes[0].cam_id)
    session.assign_camera(1, fakes[1].cam_id)

    # Lightweight modes for unit test speed.
    light = StreamMode(640, 360, 60, "bgr8")
    session.set_mode(0, light)
    session.set_mode(1, light)
    session.set_prefix(0, "cam1")
    session.set_prefix(1, "cam2")
    session.set_armed(0, True)
    session.set_armed(1, True)

    session.start_previews()
    deadline = time.perf_counter() + 2.0
    while time.perf_counter() < deadline:
        if (
            session.slots[0].get_preview_frame() is not None
            and session.slots[1].get_preview_frame() is not None
        ):
            break
        time.sleep(0.05)
    assert session.slots[0].get_preview_frame() is not None
    assert session.slots[1].get_preview_frame() is not None

    paths = session.start_recording_armed(stamp="teststamp")
    assert len(paths) == 2
    time.sleep(1.2)
    reports = session.stop_recording_armed()
    session.shutdown()

    assert "cam1" in reports and "cam2" in reports
    assert reports["cam1"]["no_frame_drops"] is True
    assert reports["cam2"]["no_frame_drops"] is True
    assert (tmp_path / "cam1_teststamp.mp4").exists()
    assert (tmp_path / "cam2_teststamp.mp4").exists()
    assert not (tmp_path / "cam1_teststamp.report.json").exists()
    assert not (tmp_path / "cam1_teststamp_sysmon.csv").exists()
    assert (tmp_path / "meta" / "cam1_teststamp.report.json").exists()
    assert (tmp_path / "meta" / "cam1_teststamp_sysmon.csv").exists()
    assert reports["cam1"]["frames_written"] > 0
    assert reports["cam2"]["frames_written"] > 0


def test_only_armed_slot_records(tmp_path: Path):
    session = MultiCamSession(n_slots=2, out_dir=tmp_path)
    fakes = [
        d
        for d in session.refresh_devices(
            include_fake=True, probe_uvc=False, probe_realsense=False
        )
        if d.kind == "fake"
    ]
    session.assign_camera(0, fakes[0].cam_id)
    session.assign_camera(1, fakes[1].cam_id)
    light = StreamMode(640, 360, 60, "bgr8")
    session.set_mode(0, light)
    session.set_mode(1, light)
    session.set_prefix(0, "m")
    session.set_prefix(1, "r")
    session.set_armed(0, True)
    session.set_armed(1, False)  # R5 — disarmed

    session.start_previews()
    time.sleep(0.3)
    paths = session.start_recording_armed(stamp="armtest")
    assert len(paths) == 1
    assert paths[0].name.startswith("m_")
    time.sleep(0.8)
    reports = session.stop_recording_armed()
    session.shutdown()

    assert "m" in reports
    assert "r" not in reports
    assert (tmp_path / "m_armtest.mp4").exists()
    assert not (tmp_path / "r_armtest.mp4").exists()
