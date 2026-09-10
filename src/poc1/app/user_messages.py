"""Operator-facing error text for the unified app (plain language, not dev jargon)."""
from __future__ import annotations

from typing import Optional


def elgato_preview_failed_message(raw_error: str) -> str:
    """Turn OpenCV/ffmpeg exceptions into actionable Elgato preview steps."""
    from poc1.ffmpeg_dshow_source import find_ffmpeg

    err = (raw_error or "").strip()
    low = err.lower()
    ffmpeg_ok = bool(find_ffmpeg())
    ffmpeg_line = (
        "ffmpeg is on PATH (DirectShow names + 1080p120 capture)."
        if ffmpeg_ok
        else "ffmpeg is NOT on PATH — install ffmpeg, then Refresh."
    )
    busy = any(
        k in low
        for k in (
            "timed out",
            "could not open",
            "backend=",
            "exited with no frames",
            "another app",
        )
    )
    busy_hint = (
        "Another app may still own the card (fully Exit OBS / Elgato Utility)."
        if busy
        else "If another app might own the card, fully Exit OBS / Elgato Utility."
    )
    return (
        "Elgato preview could not start at 1920x1080@120.\n\n"
        f"{ffmpeg_line}\n"
        "The app tries ffmpeg DirectShow first (OBS pin), then OpenCV.\n\n"
        "Try this:\n"
        "1) Setup mode: 1920x1080@120 mjpg (not @60, not bgr8).\n"
        f"2) {busy_hint}\n"
        "3) Unarm other cameras → Start preview on Elgato alone.\n"
        "4) In a terminal: ffmpeg -version (must work).\n"
        "5) If it still fails, paste log lines containing “ffmpeg dshow”.\n\n"
        f"Technical detail: {err}"
    )


def preview_failed_message(
    slot_id: int,
    raw_error: str,
    *,
    device_tag: str = "",
) -> str:
    if device_tag == "elgato":
        return elgato_preview_failed_message(raw_error)
    return (
        f"{raw_error}\n\n"
        "Close Zoom/Teams/Camera and other POC1 windows, then try again."
    )


def capture_not_120_status(
    requested: int,
    measured: float,
    stamped: float | int,
) -> str:
    """Status-bar text when Elgato asked for ≥90 but delivery was lower."""
    return (
        f"Capture delivered ~{measured:.0f}, not {requested} "
        f"(pin/driver — file stamped {stamped}; honest rate, not a drop)"
    )


def capture_not_120_hud_hint(measured: float, requested: int, stamped: int) -> str:
    return (
        f"Capture ~{measured:.0f} fps, not {requested} — pin/driver limit "
        f"(file stamped {stamped}; not a fake {requested} label)"
    )


def no_frames_captured_message(
    lines: list[str],
    save_dir: str,
    build_id: str,
) -> str:
    return (
        "Preview looked live but Record counted 0 frames on:\n"
        + "\n".join(f"• {line}" for line in lines)
        + f"\n\nSave folder: {save_dir}\n\n"
        "Record-path miss (capture stalled before encode).\n"
        "1) Stop all previews → Start preview on that camera alone.\n"
        "2) Elgato: wait for HUD “ffmpeg DirectShow” and camera ~110–120.\n"
        "3) Try with bag unchecked first; fully Exit OBS if it is running.\n"
        f"Build: {build_id}."
    )


def bag_file_missing(report: dict) -> bool:
    """True when .bag was expected but no non-empty file/folder exists."""
    from pathlib import Path

    raw = report.get("bag_path")
    if not raw:
        return True
    path = Path(str(raw))
    if path.is_file():
        return path.stat().st_size <= 0
    if path.is_dir():
        db3 = list(path.glob("*.db3"))
        if not db3:
            return True
        return db3[0].stat().st_size <= 0
    return True
