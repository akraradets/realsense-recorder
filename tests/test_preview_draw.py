from poc1.preview_draw import bgr_to_rgb_fill, downscale_for_preview, fill_bgr, hud_lines_for_source, overlay_hud
from poc1.deliverable1.devices import _looks_like_packed_yuyv
from poc1.deliverable1.session import CameraSlot
import numpy as np


def test_fill_bgr_matches_destination_size() -> None:
    src = np.zeros((1080, 1920, 3), dtype=np.uint8)
    src[:, :, 2] = 200
    out = fill_bgr(src, 640, 400)
    assert out.shape == (400, 640, 3)


def test_downscale_for_preview_caps_width() -> None:
    src = np.zeros((1080, 1920, 3), dtype=np.uint8)
    small = downscale_for_preview(src, max_width=960)
    assert small.shape[1] == 960
    assert small.shape[0] == 540


def test_bgr_to_rgb_fill_channels() -> None:
    src = np.zeros((100, 200, 3), dtype=np.uint8)
    src[:, :] = (255, 0, 0)  # BGR blue
    rgb = bgr_to_rgb_fill(src, 80, 80)
    assert rgb.shape == (80, 80, 3)
    assert int(rgb[:, :, 2].mean()) > 200  # still blue in RGB


def test_packed_yuyv_heuristic() -> None:
    zebra = np.zeros((120, 160, 3), dtype=np.uint8)
    zebra[:, 0::2] = (255, 0, 0)
    zebra[:, 1::2] = (0, 255, 0)
    assert _looks_like_packed_yuyv(zebra)
    smooth = np.full((120, 160, 3), 80, dtype=np.uint8)
    assert not _looks_like_packed_yuyv(smooth)


def test_overlay_hud_keeps_shape() -> None:
    rgb = np.zeros((180, 320, 3), dtype=np.uint8)
    out = overlay_hud(rgb, ["realsense  1280x720@30", "requested 30  camera ~30 fps"])
    assert out.shape == rgb.shape
    assert out.dtype == rgb.dtype


def test_hud_separates_camera_and_preview_fps() -> None:
    slot = CameraSlot(slot_id=0, prefix="m")
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    import time

    base = time.time()
    for i in range(12):
        with slot._frame_lock:
            slot._preview_ts.append(base + i * 0.08)
            slot.last_frame = frame

    class Src:
        device_tag = "elgato"
        actual_width = 1920
        actual_height = 1080
        target_fps = 120
        requested_fps = 120
        actual_fps = 118.0

    lines = hud_lines_for_source(slot, Src())
    assert any("camera ~118" in line for line in lines)
    assert any("UI paint" in line and "not file FPS" in line for line in lines)
    assert not any("delivering" in line for line in lines)
    assert not any("preview redraw" in line for line in lines)


def test_hud_hdmi_not_120_message() -> None:
    slot = CameraSlot(slot_id=0, prefix="m")

    class Src:
        device_tag = "elgato"
        actual_width = 1920
        actual_height = 1080
        target_fps = 60
        requested_fps = 120
        actual_fps = 58.0

    lines = hud_lines_for_source(slot, Src())
    assert any("1080p120" in line for line in lines)
    assert any("Station HDMI is ~58" in line for line in lines)
    assert any("file stamped" in line for line in lines)
