"""Operator-facing message helpers."""
from __future__ import annotations

from pathlib import Path

from poc1.app.user_messages import (
    bag_file_missing,
    capture_not_120_status,
    elgato_preview_failed_message,
    no_frames_captured_message,
)


def test_capture_not_120_status_wording() -> None:
    text = capture_not_120_status(120, 58.0, 60)
    assert "Capture delivered" in text
    assert "pin/driver" in text
    assert "HDMI source is not" not in text


def test_elgato_preview_message_mentions_ffmpeg() -> None:
    msg = elgato_preview_failed_message("Could not open 'video=Elgato' backend=700")
    assert "ffmpeg" in msg.lower()
    assert "1920x1080@120" in msg
    assert "backend=700" in msg  # kept as technical detail at end


def test_no_frames_message_v30() -> None:
    msg = no_frames_captured_message(["t6-m"], "/tmp/out", "sdk-record-v30-x")
    assert "ffmpeg DirectShow" in msg
    assert "queue lock" not in msg


def test_bag_file_missing_file_and_folder() -> None:
    assert bag_file_missing({}) is True
    assert bag_file_missing({"bag_path": ""}) is True
    assert bag_file_missing({"bag_recorded": True, "bag_path": "/nope"}) is True


def test_bag_file_present(tmp_path: Path) -> None:
    f = tmp_path / "r.db3"
    f.write_bytes(b"x" * 100)
    assert not bag_file_missing({"bag_recorded": True, "bag_path": str(f)})


def test_bag_ros2_folder_present(tmp_path: Path) -> None:
    folder = tmp_path / "m_color"
    folder.mkdir()
    (folder / "bag.db3").write_bytes(b"x" * 50)
    assert not bag_file_missing({"bag_recorded": True, "bag_path": str(folder)})
