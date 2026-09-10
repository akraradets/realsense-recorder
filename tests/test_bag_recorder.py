"""Unit tests for RealSense SDK record path coercion (.bag vs .db3)."""
from __future__ import annotations

from pathlib import Path

from poc1.bag_recorder import (
    BUILD_ID,
    coerce_record_path,
    paths_to_try,
    recording_suffix,
    set_recording_suffix,
)


def test_default_suffix_is_db3() -> None:
    set_recording_suffix(".db3")
    assert recording_suffix() == ".db3"
    assert BUILD_ID.startswith("sdk-record-")


def test_coerce_rewrites_bag_to_db3() -> None:
    set_recording_suffix(".db3")
    p = coerce_record_path(Path(r"C:\tmp\cam1_take.bag"))
    assert p.suffix == ".db3"
    assert p.name == "cam1_take.db3"


def test_coerce_strips_pending_hidden_name() -> None:
    set_recording_suffix(".db3")
    p = coerce_record_path(
        Path(r"C:\Users\Phue\realsense-recorder\recordings\deliverable1\.pending_cam1_slot0.bag")
    )
    assert not p.name.startswith(".pending_")
    assert p.suffix == ".db3"
    assert "cam1_slot0" in p.name or "pending_cam1" not in p.name


def test_paths_to_try_db3_first() -> None:
    set_recording_suffix(".db3")
    paths = paths_to_try(Path("out/cam1.bag"))
    assert paths[0].suffix == ".db3"
    assert paths[1].suffix == ".bag"
