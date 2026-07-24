"""Automated POC-1 proofs: zero drops at FHD/120 via FakeFrameSource."""
from __future__ import annotations

import time
from pathlib import Path

from poc1.camera_handler import DropCountingQueue, FrameEnvelope
from poc1.frame_source import FakeFrameSource, embed_seq_barcode, read_seq_barcode
from poc1.pipeline import Pipeline
from poc1.verifier import verify_from_report, verify_mp4
import numpy as np


def test_barcode_roundtrip():
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    embed_seq_barcode(frame, 123456)
    assert read_seq_barcode(frame) == 123456


def test_drop_counting_queue_records_drops():
    q = DropCountingQueue(maxsize=2, drop_oldest=False, name="t")
    f = np.zeros((4, 4, 3), dtype=np.uint8)
    q.put(FrameEnvelope(0, 0.0, f))
    q.put(FrameEnvelope(1, 0.0, f))
    q.put(FrameEnvelope(2, 0.0, f))  # should drop
    assert q.dropped_count == 1
    assert q.qsize() == 2


def test_fake_pipeline_no_drops(tmp_path: Path):
    mp4 = tmp_path / "t.mp4"
    csv = tmp_path / "t.csv"
    previews = {"n": 0}

    def on_preview(_env: FrameEnvelope) -> None:
        previews["n"] += 1

    # Short but non-trivial: ~2 seconds at 60fps to keep CI/local fast.
    # Separate test hits 120fps in proof module / longer run.
    src = FakeFrameSource(width=640, height=360, target_fps=60)
    pipe = Pipeline(source=src, on_preview_frame=on_preview)
    pipe.start_preview()
    pipe.start_recording(mp4, csv)
    time.sleep(2.0)
    report = pipe.stop()

    assert report["frames_read_by_camera"] > 0
    assert report["frames_written"] == report["frames_read_by_camera"]
    assert report["sequence_gaps"] == []
    assert report["dropped_processor_queue"] == 0
    assert report["dropped_recorder_queue"] == 0
    assert report["no_frame_drops"] is True
    assert report.get("compression_stage") == "processor"
    assert mp4.exists() and mp4.stat().st_size > 0
    assert csv.exists()
    assert previews["n"] > 0

    report_path = Path(report["report_path"])
    result = verify_from_report(report_path, require_barcode=True)
    assert result.ok, result.message


def test_fake_120_short_burst(tmp_path: Path):
    """R7-oriented: 120fps burst at 1280x720 (lighter than full FHD for unit test)."""
    mp4 = tmp_path / "fhd120.mp4"
    csv = tmp_path / "fhd120.csv"

    src = FakeFrameSource(width=1280, height=720, target_fps=120)
    pipe = Pipeline(source=src, on_preview_frame=lambda _e: None)
    pipe.start_preview()
    pipe.start_recording(mp4, csv)
    # Wall-clock sleep is noisy under load; wait for enough frames or timeout.
    deadline = time.perf_counter() + 4.0
    while time.perf_counter() < deadline:
        if pipe.camera_handler.frames_read >= 90:
            break
        time.sleep(0.05)
    report = pipe.stop()

    assert report["no_frame_drops"], report
    assert report["frames_written"] >= 90, report
    assert report.get("compression_stage") == "processor"
    result = verify_mp4(
        mp4,
        expected_frames=report["frames_written"],
        expected_width=1280,
        expected_height=720,
        require_barcode=True,
    )
    assert result.ok, result.message
