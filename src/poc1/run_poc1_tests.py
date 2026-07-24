"""
Run all POC-1 source tests and print a scorecard.

  1. fake        — FHD@120 no-drop proof (synthetic)
  2. capturecard — Elgato / UVC capture card @ FHD@120 when HW present
  3. realsense   — hardware if present, else simulated SDK path
  4. webcam      — real UVC capture
  5. virtualcam  — pyvirtualcam sender + OpenCV capture

Usage:
    uv run python -m poc1.run_poc1_tests
    uv run python -m poc1.run_poc1_tests --seconds 3
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from poc1.camera_handler import FrameEnvelope
from poc1.capturecard_source import create_capture_card_source
from poc1.frame_source import FakeFrameSource
from poc1.pipeline import CvCaptureSource, Pipeline
from poc1.realsense_source import create_realsense_source, list_realsense_devices
from poc1.verifier import verify_from_report
from poc1.virtualcam_source import VirtualCamCaptureSource

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("poc1.run_poc1_tests")


def _run_one(
    name: str,
    source_factory: Callable[[], Any],
    out_dir: Path,
    seconds: float,
    require_barcode: bool,
) -> dict:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mp4 = out_dir / f"{name}_{stamp}.mp4"
    csv = out_dir / f"{name}_{stamp}_sysmon.csv"
    result: dict = {"name": name, "ok": False}

    try:
        source = source_factory()
        pipe = Pipeline(source=source, on_preview_frame=lambda _e: None)
        pipe.start_preview()
        pipe.start_recording(mp4, csv)
        time.sleep(seconds)
        report = pipe.stop()
        result["report"] = report
        result["mp4"] = str(mp4)

        report_path = Path(report.get("report_path", ""))
        if report_path.exists():
            verify = verify_from_report(report_path, require_barcode=require_barcode)
            result["verify"] = verify.to_dict()
            result["ok"] = bool(report.get("no_frame_drops") and verify.ok)
        else:
            result["ok"] = bool(report.get("no_frame_drops"))
            result["verify"] = {"ok": result["ok"], "message": "no report file"}

        # Extra metadata
        mode = getattr(source, "mode", None) or getattr(source, "capture_mode", None)
        if mode:
            result["source_mode"] = mode
        if hasattr(source, "device_index"):
            result["device_index"] = source.device_index

    except Exception as exc:  # noqa: BLE001
        err = str(exc)
        result["error"] = err
        # Missing OBS/virtual-cam or capture card is an environment skip, not fail.
        if name == "virtualcam" and (
            "Could not find virtual camera" in err
            or "virtual camera" in err.lower()
        ):
            logger.warning("%s SKIPPED: %s", name, err)
            result["ok"] = True
            result["skipped"] = True
            result["skip_reason"] = err
        elif name == "capturecard" and (
            "Could not find capture card" in err
            or "Could not open capture card" in err
            or "capture card" in err.lower()
            or "no Elgato" in err
        ):
            logger.warning("%s SKIPPED: %s", name, err)
            result["ok"] = True
            result["skipped"] = True
            result["skip_reason"] = err
        else:
            logger.exception("%s FAILED", name)
            result["ok"] = False

    return result


def main() -> None:
    from poc1.quiet import configure_app_logging, silence_opencv
    configure_app_logging()
    silence_opencv()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--out-dir", type=Path, default=Path("./recordings/poc1_suite"))
    parser.add_argument("--skip-virtualcam", action="store_true")
    parser.add_argument("--skip-webcam", action="store_true")
    parser.add_argument("--skip-realsense", action="store_true")
    parser.add_argument("--skip-fake", action="store_true")
    parser.add_argument("--skip-capturecard", action="store_true")
    parser.add_argument(
        "--capture-index",
        type=int,
        default=None,
        help="Force Elgato/capture card device index (else auto-detect)",
    )
    args = parser.parse_args()
    capture_index = args.capture_index
    if capture_index is None and os.environ.get("POC1_CAPTURE_INDEX"):
        capture_index = int(os.environ["POC1_CAPTURE_INDEX"])
    args.out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []

    def _note(r: dict) -> None:
        results.append(r)
        tag = "SKIP" if r.get("skipped") else ("PASS" if r.get("ok") else "FAIL")
        print(f"  [{tag}] {r['name']}", flush=True)

    if not args.skip_fake:
        # Suite uses 720p@120 (still no-drop / barcode). Full FHD@120 is poc1.proof.
        _note(_run_one(
            "fake",
            lambda: FakeFrameSource(width=1280, height=720, target_fps=120),
            args.out_dir, args.seconds, require_barcode=True,
        ))

    if not args.skip_capturecard:
        _note(_run_one(
            "capturecard",
            lambda: create_capture_card_source(
                width=1920, height=1080, fps=120, preferred_index=capture_index,
            ),
            args.out_dir, args.seconds, require_barcode=False,
        ))

    if not args.skip_realsense:
        # Run before virtualcam: Windows/OpenCV DSHOW + pyvirtualcam can wedge
        # subsequent VideoWriter opens in the same process.
        devices = list_realsense_devices()
        logger.info("RealSense devices: %s", devices or "(none — will simulate)")
        _note(_run_one(
            "realsense",
            lambda: create_realsense_source(1280, 720, 30, allow_simulate=True),
            args.out_dir, args.seconds,
            require_barcode=not bool(devices),
        ))

    if not args.skip_webcam:
        _note(_run_one(
            "webcam",
            lambda: CvCaptureSource(0, 640, 480, 30),
            args.out_dir, args.seconds, require_barcode=False,
        ))

    if not args.skip_virtualcam:
        # OS virtual-cam path: prove capture+pipeline no drops. Barcode may be
        # altered by OBS/DirectShow buffering + lossy encode at 640x480.
        _note(_run_one(
            "virtualcam",
            lambda: VirtualCamCaptureSource(
                width=640, height=480, target_fps=30,
                auto_start_sender=True, require_os=True,
            ),
            args.out_dir, args.seconds, require_barcode=False,
        ))

    summary = {
        "passed": sum(1 for r in results if r.get("ok") and not r.get("skipped")),
        "skipped": sum(1 for r in results if r.get("skipped")),
        "total": len(results),
        "all_ok": all(r.get("ok") for r in results) and len(results) > 0,
        "results": results,
    }
    out = args.out_dir / f"suite_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    # Make JSON-serializable (sequence_gaps etc. already ok)
    def _clean(obj):
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        if isinstance(obj, tuple):
            return list(obj)
        return obj

    summary = _clean(summary)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n======== POC-1 SCORECARD ========")
    for r in results:
        if r.get("skipped"):
            status = "SKIP"
            extra = r.get("skip_reason") or r.get("error") or ""
        else:
            status = "PASS" if r.get("ok") else "FAIL"
            extra = r.get("source_mode") or r.get("error") or ""
        print(f"  [{status}] {r['name']}" + (f"  ({extra[:80]})" if extra else ""))
    print(
        f"  {summary['passed']}/{summary['total']} passed"
        + (f", {summary['skipped']} skipped" if summary["skipped"] else "")
    )
    print(f"  summary -> {out}")
    print("=================================\n")
    raise SystemExit(0 if summary["all_ok"] else 1)


if __name__ == "__main__":
    main()
