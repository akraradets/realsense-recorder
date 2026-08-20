"""Live-preview scaling: fill the widget (cover-crop) without blocking Record."""
from __future__ import annotations

import cv2
import numpy as np

# Preview copies stay small so Tk stays responsive at 1080p/120 capture.
PREVIEW_MAX_WIDTH = 960
PREVIEW_HZ = 15.0


def downscale_for_preview(bgr: np.ndarray, max_width: int = PREVIEW_MAX_WIDTH) -> np.ndarray:
    h, w = bgr.shape[:2]
    if w <= max_width:
        return bgr.copy()
    scale = max_width / float(w)
    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))
    return cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)


def fill_bgr(bgr: np.ndarray, dest_w: int, dest_h: int) -> np.ndarray:
    """Scale to cover dest_w x dest_h, then center-crop (no black bars)."""
    dest_w = max(int(dest_w), 1)
    dest_h = max(int(dest_h), 1)
    h, w = bgr.shape[:2]
    if w < 1 or h < 1:
        return np.zeros((dest_h, dest_w, 3), dtype=np.uint8)
    scale = max(dest_w / float(w), dest_h / float(h))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    x0 = max(0, (nw - dest_w) // 2)
    y0 = max(0, (nh - dest_h) // 2)
    crop = resized[y0 : y0 + dest_h, x0 : x0 + dest_w]
    if crop.shape[0] != dest_h or crop.shape[1] != dest_w:
        return cv2.resize(crop, (dest_w, dest_h), interpolation=cv2.INTER_LINEAR)
    return crop


def bgr_to_rgb_fill(bgr: np.ndarray, dest_w: int, dest_h: int) -> np.ndarray:
    return cv2.cvtColor(fill_bgr(bgr, dest_w, dest_h), cv2.COLOR_BGR2RGB)


def overlay_hud(rgb: np.ndarray, lines: list[str]) -> np.ndarray:
    """Viewer-style status on a preview tile (display only)."""
    if rgb is None or not lines:
        return rgb
    img = np.ascontiguousarray(rgb)
    y = 20
    for line in lines:
        text = str(line)[:80]
        cv2.putText(
            img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 3, cv2.LINE_AA
        )
        cv2.putText(
            img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (40, 255, 90), 1, cv2.LINE_AA
        )
        y += 18
    return img


def hud_lines_for_source(slot, src) -> list[str]:
    """Build overlay lines from a live slot + frame source."""
    cam = getattr(slot, "camera", None)
    kind = getattr(cam, "kind", "") if cam is not None else ""
    tag = str(getattr(src, "device_tag", "") or kind or "cam")
    w = int(getattr(src, "actual_width", 0) or getattr(src, "width", 0) or 0)
    h = int(getattr(src, "actual_height", 0) or getattr(src, "height", 0) or 0)
    opened_fps = int(getattr(src, "target_fps", 0) or 0)
    requested = int(
        getattr(src, "requested_fps", 0) or getattr(src, "target_fps", 0) or 0
    )
    camera_fps = float(getattr(src, "actual_fps", 0) or 0)
    preview_fps = 0.0
    est = getattr(slot, "estimate_preview_fps", None)
    if callable(est):
        live = est()
        if live and live > 1:
            preview_fps = float(live)
    rec = ""
    try:
        if slot.pipeline and slot.pipeline.camera_handler.is_recording:
            rec = "REC"
    except Exception:
        rec = ""
    line1 = f"{tag}  {w}x{h}@{opened_fps}"
    if rec:
        line1 = f"{rec}  {line1}"
    lines = [line1, f"requested {requested}  camera ~{camera_fps:.0f} fps"]
    if preview_fps > 1:
        lines.append(f"preview redraw ~{preview_fps:.0f} fps (display only)")
    if requested >= 90 and camera_fps < requested * 0.85 and tag == "elgato":
        lines.append("HDMI source is not 120Hz")
    elif kind == "realsense" and requested >= 90:
        lines.append("D400 color is not 120fps")
    return lines
