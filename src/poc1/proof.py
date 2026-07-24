"""
Headless POC-1 proof run: FakeFrameSource @ FHD/120 for N seconds, then verify.

Usage:
    uv run python -m poc1.proof --seconds 5 --fps 120
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path

from poc1.camera_handler import FrameEnvelope
from poc1.frame_source import FakeFrameSource
from poc1.pipeline import Pipeline
from poc1.verifier import verify_from_report

from poc1.quiet import silence_opencv

silence_opencv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("poc1.proof")


def run_proof(
    seconds: float = 5.0,
    width: int = 1920,
    height: int = 1080,
    fps: int = 120,
    out_dir: Path = Path("./recordings/proof"),
) -> dict:
    silence_opencv()
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mp4 = out_dir / f"proof_fake_{width}x{height}_{fps}fps_{stamp}.mp4"
    csv = out_dir / f"proof_fake_{width}x{height}_{fps}fps_{stamp}_sysmon.csv"

    preview_n = {"n": 0}

    def on_preview(_env: FrameEnvelope) -> None:
        preview_n["n"] += 1

    source = FakeFrameSource(width=width, height=height, target_fps=fps)
    pipe = Pipeline(source=source, on_preview_frame=on_preview)
    pipe.start_preview()
    pipe.start_recording(mp4, csv)
    logger.info("Recording for %.1fs ...", seconds)
    time.sleep(seconds)
    report = pipe.stop()

    report_path = Path(report.get("report_path", mp4.with_suffix(".report.json")))
    verify = verify_from_report(report_path, require_barcode=True)
    result = {
        "report": report,
        "verify": verify.to_dict(),
        "preview_callbacks": preview_n["n"],
        "mp4": str(mp4),
        "sysmon": str(csv),
        "passed": bool(report.get("no_frame_drops") and verify.ok),
    }
    summary_path = out_dir / f"proof_summary_{stamp}.json"
    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("Proof summary -> %s", summary_path)
    logger.info("PASSED" if result["passed"] else "FAILED")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=120)
    parser.add_argument("--out-dir", type=Path, default=Path("./recordings/proof"))
    args = parser.parse_args()
    result = run_proof(args.seconds, args.width, args.height, args.fps, args.out_dir)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
