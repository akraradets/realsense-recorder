"""Deliverable 1 tests — multi-cam session using synthetic sources only."""
from __future__ import annotations

import time
from pathlib import Path

from poc1.deliverable1.devices import (
    ConfiguredRealSenseSource,
    StreamMode,
    list_all_cameras,
    list_stream_modes,
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


def test_dynamic_slots_add_and_remove():
    session = MultiCamSession(n_slots=2)
    added = session.add_slot()
    assert len(session.slots) == 3
    assert added.slot_id == 2
    assert added.prefix == "m"
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
    time.sleep(0.4)
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
