"""Honest FPS listing, report flags, and overflow-only backpressure."""
from __future__ import annotations

from poc1.camera_handler import DropCountingQueue
from poc1.deliverable1.devices import (
    ConnectedCamera,
    StreamMode,
    is_fhd_high_rate,
    list_realsense_modes,
    prefix_for_camera,
    too_many_1080p120,
)


def test_realsense_modes_default_30_no_invented_color_120() -> None:
    modes = list_realsense_modes(serial="no-such-serial-poc1")
    color = [m for m in modes if m.pixel_format in {"bgr8", "rgb8", "yuyv"}]
    assert color, modes
    assert all(m.fps <= 60 for m in color)
    assert all(m.fps < 90 for m in color)
    assert any(m.fps == 30 for m in color)
    assert not any(m.width >= 1920 and m.fps >= 90 for m in color)


def test_fhd_high_rate_helper() -> None:
    assert is_fhd_high_rate(StreamMode(1920, 1080, 120, "mjpg"))
    assert not is_fhd_high_rate(StreamMode(1920, 1080, 30, "bgr8"))
    assert not is_fhd_high_rate(StreamMode(1280, 720, 120, "bgr8"))


def test_one_elgato_120_plus_30_companions_allowed() -> None:
    elgato120 = StreamMode(1920, 1080, 120, "mjpg")
    rs30 = StreamMode(1280, 720, 30, "bgr8")
    cam30 = StreamMode(1280, 720, 30, "mjpg")
    assert not too_many_1080p120([elgato120, rs30, cam30])
    assert too_many_1080p120(
        [elgato120, StreamMode(1920, 1080, 120, "bgr8")]
    )


def test_honest_container_fps_never_55_from_near_60() -> None:
    from poc1.deliverable1.devices import honest_container_fps

    # Station A/B regression: short window ~55–57 with request 120 must stamp 60.
    assert honest_container_fps(55.0, 120) == 60
    assert honest_container_fps(56.5, 120) == 60
    assert honest_container_fps(59.995, 120) == 60
    assert honest_container_fps(62.0, 120) == 60
    # Never invent 120 from ~60 HDMI.
    assert honest_container_fps(60.0, 120) == 60
    assert honest_container_fps(70.0, 120) != 120
    # Real OBS-like 120 lock.
    assert honest_container_fps(118.0, 120) == 120
    assert honest_container_fps(120.0, 120) == 120
    assert honest_container_fps(30.1, 30) == 30


def test_elgato_open_targets_dshow_only_skips_msmf() -> None:
    import cv2

    from poc1.deliverable1.devices import _elgato_open_targets

    targets = _elgato_open_targets(None, 0, dshow_only=True)
    assert targets
    assert all(backend == cv2.CAP_DSHOW for _, backend in targets)
    full = _elgato_open_targets(None, 0, dshow_only=False)
    assert any(backend == cv2.CAP_MSMF for _, backend in full)


def test_prefix_elgato_m_realsense_r() -> None:
    rs = ConnectedCamera("rs", "realsense", "D435", serial="1")
    elgato = ConnectedCamera("e", "uvc", "Elgato", device_tag="elgato")
    assert prefix_for_camera(rs) == "r"
    assert prefix_for_camera(elgato) == "m"


def test_put_live_blocks_near_full_instead_of_drop() -> None:
    q = DropCountingQueue(10, drop_oldest=False, name="t")
    for i in range(9):
        q.put_live(i)
    assert q.dropped_count == 0
    q.put_live(9)
    assert q.qsize() == 10
    assert q.dropped_count == 0


def test_uvc_ros2_bag_writes_jpeg_frames(tmp_path) -> None:
    import time

    import numpy as np

    from poc1.uvc_rosbag import UvcRos2Bag

    bag = UvcRos2Bag(tmp_path / "m_x_color")
    bag.start()
    for i in range(3):
        frame = np.full((32, 48, 3), i * 40, dtype=np.uint8)
        bag.submit(frame, time.time())
    written = bag.stop()
    assert written is not None and written.is_file()
    assert bag.frames_written >= 3
