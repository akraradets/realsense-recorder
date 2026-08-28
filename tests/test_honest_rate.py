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
    # Settling high-rate (~95) must stamp 120 — not nearest-5 → 95.
    assert honest_container_fps(95.0, 120) == 120
    assert honest_container_fps(90.0, 120) == 120
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


def test_capture_releases_record_lock_before_processor_put() -> None:
    """Regression: holding _record_lock across put stalled Elgato → 0 Record frames."""
    import time

    from poc1.camera_handler import CameraHandler
    from poc1.frame_source import FakeFrameSource

    source = FakeFrameSource(width=64, height=48, target_fps=60)
    handler = CameraHandler(source)

    lock_free_during_put: list[bool] = []
    orig_put = handler.processor_queue.put

    def checking_put(item):
        acquired = handler._record_lock.acquire(blocking=False)
        lock_free_during_put.append(bool(acquired))
        if acquired:
            handler._record_lock.release()
        return orig_put(item)

    handler.processor_queue.put = checking_put  # type: ignore[method-assign]
    handler.start()
    try:
        handler.enable_recording()
        deadline = time.time() + 2.0
        while time.time() < deadline and handler.frames_read < 5:
            time.sleep(0.02)
        assert handler.frames_read >= 5, handler.frames_read
        assert lock_free_during_put, "processor.put never called"
        assert all(lock_free_during_put), lock_free_during_put
    finally:
        handler.disable_recording()
        handler.stop()


def test_hud_lines_realsense_no_name_error() -> None:
    from types import SimpleNamespace

    from poc1.preview_draw import hud_lines_for_source

    slot = SimpleNamespace(
        mode=SimpleNamespace(fps=30),
        pipeline=None,
        camera=SimpleNamespace(kind="realsense"),
    )
    src = SimpleNamespace(
        device_tag="realsense",
        width=1920,
        height=1080,
        actual_width=1920,
        actual_height=1080,
        requested_fps=30,
        target_fps=30,
        actual_fps=30.0,
        _ffmpeg=None,
    )
    lines = hud_lines_for_source(slot, src)
    assert lines
    assert "realsense" in lines[0]


def test_lock_threshold_120() -> None:
    from poc1.ffmpeg_dshow_source import _lock_threshold

    assert _lock_threshold(120) == 90.0
    assert _lock_threshold(30) == 25.5


def test_dshow_input_names_strips_video_prefix() -> None:
    from poc1.ffmpeg_dshow_source import dshow_input_names

    names = dshow_input_names("video=Elgato 4K S")
    assert names
    assert names[0] == "Elgato 4K S"
    assert not names[0].startswith("video=")


def test_input_modes_mjpg_uses_vcodec() -> None:
    from poc1.ffmpeg_dshow_source import (
        _input_modes_to_try,
        _pin_args_for_mode,
    )

    order = _input_modes_to_try("mjpg")
    assert order[0] == "vcodec_mjpeg"
    assert _pin_args_for_mode("vcodec_mjpeg") == ["-vcodec", "mjpeg"]
    assert "-pixel_format" not in _pin_args_for_mode("vcodec_mjpeg")


def test_pixel_formats_to_try_mjpg_first() -> None:
    from poc1.ffmpeg_dshow_source import _pixel_formats_to_try

    order = _pixel_formats_to_try("mjpg")
    assert order[0] == "vcodec_mjpeg"


def test_pipe_pix_fmt_nv12_for_high_rate() -> None:
    from poc1.frame_layout import PIPE_BGR24, PIPE_NV12, frame_bytes, pipe_pix_fmt_for_fps
    from poc1.ffmpeg_dshow_source import FfmpegDshowCaptureSource

    assert pipe_pix_fmt_for_fps(120) == PIPE_NV12
    assert pipe_pix_fmt_for_fps(30) == PIPE_BGR24
    assert frame_bytes(1920, 1080, PIPE_NV12) == 1920 * 1080 * 3 // 2
    ff = FfmpegDshowCaptureSource(1920, 1080, 120, device_names=["Elgato 4K X"])
    assert ff.pipe_pix_fmt == PIPE_NV12
    cmd = ff._build_cmd("Elgato 4K X", "vcodec_mjpeg")
    assert cmd[cmd.index("-pix_fmt") + 1] == "nv12"
    assert "-vcodec" in cmd and "mjpeg" in cmd


def test_ensure_bgr_nv12_and_bgr() -> None:
    import numpy as np

    from poc1.frame_layout import ensure_bgr

    bgr = np.zeros((48, 64, 3), dtype=np.uint8)
    assert ensure_bgr(bgr, 48, 64) is bgr
    nv12 = np.zeros((72, 64), dtype=np.uint8)
    out = ensure_bgr(nv12, 48, 64)
    assert out.shape == (48, 64, 3)


def test_ffmpeg_dshow_low_latency_settings() -> None:
    from poc1.ffmpeg_dshow_source import DSHOW_RTBUFSIZE, FfmpegDshowCaptureSource

    ff = FfmpegDshowCaptureSource(1920, 1080, 120, device_names=["Elgato 4K X"])
    cmd = ff._build_cmd("Elgato 4K X", "vcodec_mjpeg")
    assert cmd[cmd.index("-rtbufsize") + 1] == DSHOW_RTBUFSIZE
    assert DSHOW_RTBUFSIZE != "150M"
    assert "-fflags" in cmd and "nobuffer" in cmd


def test_ffmpeg_dshow_read_prefers_newest_queued() -> None:
    import numpy as np

    from poc1.ffmpeg_dshow_source import FfmpegDshowCaptureSource

    ff = FfmpegDshowCaptureSource(4, 4, 120, device_names=["x"])
    ff._running = True
    old = np.zeros((6, 4), dtype=np.uint8)
    old[0, 0] = 1
    new = np.zeros((6, 4), dtype=np.uint8)
    new[0, 0] = 9
    ff._queue.put_nowait(old)
    ff._queue.put_nowait(new)
    got = ff.read()
    assert got is not None
    assert got[0, 0] == 9


def test_ffmpeg_dshow_seek_live_edge() -> None:
    import numpy as np

    from poc1.ffmpeg_dshow_source import FfmpegDshowCaptureSource

    ff = FfmpegDshowCaptureSource(4, 4, 120, device_names=["x"])
    stale = np.zeros((6, 4), dtype=np.uint8)
    stale[0, 0] = 1
    fresh = np.zeros((6, 4), dtype=np.uint8)
    fresh[0, 0] = 7
    ff._queue.put_nowait(stale)
    ff._queue.put_nowait(fresh)
    ff._seek_live_edge()
    got = ff.read()
    assert got is not None
    assert got[0, 0] == 7


def test_wait_recorded_frames_true_when_armed() -> None:
    import time

    from poc1.camera_handler import CameraHandler
    from poc1.frame_source import FakeFrameSource

    handler = CameraHandler(FakeFrameSource(width=64, height=48, target_fps=60))
    handler.start()
    try:
        handler.enable_recording()
        assert handler.wait_recorded_frames(2.0)
        assert handler.frames_read > 0
    finally:
        handler.disable_recording()
        handler.stop()


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
