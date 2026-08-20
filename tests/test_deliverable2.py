"""Deliverable 2 tests — R8 prompt, R9 listing, R10 export helpers."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from poc1.deliverable2.export import (
    export_to_mp4,
    list_media_files,
    resolve_export_fourcc,
)
from poc1.deliverable2.review import show_review_prompt


def _write_tiny_mp4(path: Path, frames: int = 8, fps: float = 10.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (64, 48),
    )
    assert writer.isOpened(), "VideoWriter failed in test setup"
    for i in range(frames):
        frame = np.full((48, 64, 3), (i * 20) % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    assert path.is_file() and path.stat().st_size > 32
    return path


def test_list_media_files(tmp_path: Path):
    _write_tiny_mp4(tmp_path / "a.mp4")
    (tmp_path / "notes.txt").write_text("x", encoding="utf-8")
    (tmp_path / "clip.bag").write_bytes(b"not-a-real-bag")
    (tmp_path / "ros.bd3").write_bytes(b"x")
    (tmp_path / "ros.db3").write_bytes(b"y")
    names = {p.name for p in list_media_files(tmp_path)}
    assert names == {"a.mp4", "clip.bag", "ros.bd3", "ros.db3"}
    assert list_media_files(tmp_path / "missing") == []


def test_resolve_export_fourcc_h264_and_h265():
    fourcc_h264, label_h264 = resolve_export_fourcc("h264")
    assert fourcc_h264
    assert "H.264" in label_h264 or "MPEG-4" in label_h264

    fourcc_h265, label_h265 = resolve_export_fourcc("h265")
    assert fourcc_h265
    assert "H.265" in label_h265 or "MPEG-4" in label_h265


def test_export_mp4_reencode(tmp_path: Path):
    src = _write_tiny_mp4(tmp_path / "src.mp4", frames=12)
    out = tmp_path / "out_h264.mp4"
    result = export_to_mp4(src, out, codec="h264")
    assert result.ok, result.message
    assert result.output_path == out
    assert out.is_file()
    assert result.frames >= 1
    assert result.codec_label


def test_export_missing_file(tmp_path: Path):
    result = export_to_mp4(tmp_path / "nope.bag", codec="h264")
    assert not result.ok
    assert "not found" in result.message.lower()


def test_export_bd3_unreadable_gives_clear_error(tmp_path: Path):
    junk = tmp_path / "junk.bd3"
    junk.write_bytes(b"not a video container")
    result = export_to_mp4(junk, codec="h264")
    assert not result.ok
    assert "Could not decode" in result.message or "Could not convert" in result.message or "failed" in result.message.lower()


def test_export_db3_does_not_copy_sibling_record_mp4(tmp_path: Path):
    """Junk Intel-like .db3 must not succeed by copying the Record MP4."""
    db3 = tmp_path / "cam1_take.db3"
    db3.write_bytes(b"not a real bag")
    sibling = _write_tiny_mp4(tmp_path / "cam1_take.mp4", frames=8)
    sibling_size = sibling.stat().st_size
    out = tmp_path / "cam1_take_h264.mp4"
    result = export_to_mp4(db3, out, codec="h264")
    assert not result.ok, result.message
    assert sibling.is_file() and sibling.stat().st_size == sibling_size
    assert "RealSense" in result.message or "SDK" in result.message or "decode" in result.message.lower()
    assert "no image topics" not in result.message.split("\n")[0].lower()


def test_export_elgato_color_folder_does_not_copy_sibling(tmp_path: Path):
    bag_dir = tmp_path / "m_take_color"
    bag_dir.mkdir()
    (bag_dir / "metadata.yaml").write_text("rosbag2_bagfile_information:\n", encoding="utf-8")
    sibling = _write_tiny_mp4(tmp_path / "m_take.mp4", frames=6)
    size = sibling.stat().st_size
    out = tmp_path / "m_take_h264.mp4"
    result = export_to_mp4(bag_dir, out, codec="h264")
    assert not result.ok, result.message
    assert sibling.stat().st_size == size

def test_export_rosbag2_db3_to_mp4(tmp_path: Path):
    """R10: ROS2 .db3 with Image topic exports to a playable MP4."""
    pytest.importorskip("rosbags")
    pytest.importorskip("rosbags.image")
    from rosbags.rosbag2 import Writer
    from rosbags.typesys import Stores, get_typestore
    from rosbags.typesys.stores.latest import builtin_interfaces__msg__Time as Time
    from rosbags.typesys.stores.latest import sensor_msgs__msg__Image as Image
    from rosbags.typesys.stores.latest import std_msgs__msg__Header as Header

    typestore = get_typestore(Stores.LATEST)
    bag_dir = tmp_path / "sample_bag"
    with Writer(bag_dir, version=9) as writer:
        conn = writer.add_connection(
            "/camera/color/image_raw", Image.__msgtype__, typestore=typestore
        )
        for i in range(10):
            img = np.zeros((48, 64, 3), dtype=np.uint8)
            img[:, :] = (i * 20 % 255, 40, 90)
            msg = Image(
                header=Header(
                    stamp=Time(sec=0, nanosec=i * 50_000_000),
                    frame_id="cam",
                ),
                height=48,
                width=64,
                encoding="bgr8",
                is_bigendian=0,
                step=64 * 3,
                data=np.frombuffer(img.tobytes(), dtype=np.uint8),
            )
            writer.write(
                conn,
                i * 50_000_000,
                typestore.serialize_cdr(msg, Image.__msgtype__),
            )

    db3 = next(bag_dir.glob("*.db3"))
    sibling = _write_tiny_mp4(tmp_path / f"{db3.stem}.mp4", frames=3)
    sibling_size = sibling.stat().st_size
    out = tmp_path / "from_db3.mp4"
    result = export_to_mp4(db3, out, codec="h264")
    assert result.ok, result.message
    assert out.is_file() and out.stat().st_size > 32
    assert result.frames >= 1
    assert sibling.is_file() and sibling.stat().st_size == sibling_size
    assert "matching" not in result.message.lower()
    assert "copied" not in result.message.lower()


def test_show_review_prompt_behavior():
    """R8: missing files → no popup; real files → popup that auto-dismisses."""
    tkinter = pytest.importorskip("tkinter")
    try:
        root = tkinter.Tk()
    except tkinter.TclError as exc:
        pytest.skip(f"Tk not usable in this environment: {exc}")
    root.withdraw()
    try:
        assert (
            show_review_prompt(
                root,
                [Path("definitely_missing_xyz.mp4")],
                on_review=lambda _p: None,
            )
            is None
        )

        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as td:
            mp4 = _write_tiny_mp4(Path(td) / "clip.mp4")
            reviewed: list[Path] = []
            popup = show_review_prompt(
                root,
                [mp4],
                on_review=reviewed.append,
                timeout_ms=200,
            )
            assert popup is not None
            root.update()
            root.after(250, root.quit)
            root.mainloop()
            assert not popup.winfo_exists()
            assert reviewed == []
    finally:
        try:
            root.destroy()
        except tkinter.TclError:
            pass
