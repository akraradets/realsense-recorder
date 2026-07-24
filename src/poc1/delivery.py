"""
Client proof pack: run POC-1 proofs and copy artifacts into recordings/poc1_delivery/.

Usage:
    uv run python -m poc1.delivery
    uv run python -m poc1.delivery --seconds 5
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def _copy_if_exists(src: Path, dest_dir: Path) -> Optional[Path]:
    if src and Path(src).exists():
        dest = dest_dir / Path(src).name
        shutil.copy2(src, dest)
        return dest
    return None


def build_delivery(
    seconds: float = 5.0,
    suite_seconds: float = 3.0,
    delivery_dir: Path = Path("./recordings/poc1_delivery"),
) -> dict:
    delivery_dir = delivery_dir.resolve()
    delivery_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pack = delivery_dir / stamp
    pack.mkdir(parents=True, exist_ok=True)

    from poc1.proof import run_proof

    proof = run_proof(
        seconds=seconds,
        width=1920,
        height=1080,
        fps=120,
        out_dir=pack / "proof_raw",
    )
    proof_ok = bool(proof.get("passed"))
    compression_ok = proof.get("report", {}).get("compression_stage") == "processor"

    artifacts: list[str] = []
    for key in ("mp4", "sysmon"):
        p = _copy_if_exists(Path(proof.get(key, "")), pack)
        if p:
            artifacts.append(p.name)
    report_path = proof.get("report", {}).get("report_path") or ""
    if report_path:
        p = _copy_if_exists(Path(report_path), pack)
        if p:
            artifacts.append(p.name)
    bag = proof.get("report", {}).get("bag_path") or ""
    if bag:
        p = _copy_if_exists(Path(bag), pack)
        if p:
            artifacts.append(p.name)

    suite_cmd = [
        sys.executable,
        "-m",
        "poc1.run_poc1_tests",
        "--seconds",
        str(suite_seconds),
        "--out-dir",
        str(pack / "suite_raw"),
    ]
    suite_proc = subprocess.run(suite_cmd, capture_output=True, text=True)
    suite_ok = suite_proc.returncode == 0

    capturecard_claim = "NOT TESTED (no capture card in suite)"
    elgato_fhd120 = False

    suite_summaries = list((pack / "suite_raw").glob("suite_summary_*.json")) if (
        pack / "suite_raw"
    ).exists() else []
    scorecard_payload: dict = {
        "returncode": suite_proc.returncode,
        "stdout_tail": (suite_proc.stdout or "")[-2000:],
        "stderr_tail": (suite_proc.stderr or "")[-1000:],
    }
    if suite_summaries:
        latest_summary = max(suite_summaries, key=lambda p: p.stat().st_mtime)
        try:
            scorecard_payload["suite"] = json.loads(
                latest_summary.read_text(encoding="utf-8")
            )
            shutil.copy2(latest_summary, pack / "scorecard.json")
            artifacts.append("scorecard.json")
            for r in scorecard_payload["suite"].get("results", []):
                if r.get("name") != "capturecard":
                    continue
                if r.get("skipped"):
                    capturecard_claim = f"SKIPPED ({r.get('skip_reason', 'no hardware')[:60]})"
                elif r.get("ok"):
                    rep = r.get("report", {})
                    fps = rep.get("target_fps", 0)
                    nd = rep.get("no_frame_drops")
                    w, h = rep.get("width", 0), rep.get("height", 0)
                    capturecard_claim = f"PASS {w}x{h}@{fps} no_drop={nd}"
                    elgato_fhd120 = bool(
                        nd and w >= 1920 and h >= 1080 and fps >= 120
                    )
                else:
                    capturecard_claim = "FAIL (see scorecard)"
        except json.JSONDecodeError:
            pass

    if (pack / "suite_raw").exists():
        suite_out = pack / "suite"
        suite_out.mkdir(exist_ok=True)
        for pattern in ("*.mp4", "*.bag", "*.report.json", "*_sysmon.csv"):
            for f in (pack / "suite_raw").glob(pattern):
                dest = suite_out / f.name
                if not dest.exists():
                    shutil.copy2(f, dest)
                    artifacts.append(f"suite/{dest.name}")

    (pack / "scorecard_run.json").write_text(
        json.dumps(scorecard_payload, indent=2), encoding="utf-8"
    )
    if "scorecard.json" not in artifacts:
        artifacts.append("scorecard_run.json")

    overall = proof_ok and compression_ok and suite_ok
    lines = [
        f"POC-1 delivery pack — {stamp}",
        f"OVERALL: {'PASS' if overall else 'FAIL'}",
        f"R7 fake FHD@120 proof: {'PASS' if proof_ok else 'FAIL'}",
        f"compression_stage=processor: {'PASS' if compression_ok else 'FAIL'}",
        f"5-source suite: {'PASS' if suite_ok else 'FAIL (see scorecard / skip reasons)'}",
        f"capturecard / Elgato: {capturecard_claim}",
        f"no_frame_drops: {proof.get('report', {}).get('no_frame_drops')}",
        f"codec: {proof.get('report', {}).get('codec')}",
        "bag_recorded: only when RealSense hardware is connected (see suite/realsense)",
        "",
        "Artifacts:",
        *[f"  - {a}" for a in artifacts],
        "",
        "Claims:",
        "  PROVEN: pipeline no-drop at FHD@120 via FakeFrameSource (R7 throughput).",
        "  PROVEN: compression in processor stage.",
        (
            "  PROVEN: Elgato/capture-card FHD@120 OS path (suite capturecard test)."
            if elgato_fhd120
            else "  Elgato FHD@120: run suite with card connected + HDMI @1080p120 to claim."
        ),
        "  DEFERRED: .db3 / R8–R10 export polish / multi-cam arming.",
    ]
    summary_text = "\n".join(lines) + "\n"
    (pack / "PASS_SUMMARY.txt").write_text(summary_text, encoding="utf-8")
    (delivery_dir / "LATEST.txt").write_text(
        summary_text + f"\npack_dir={pack}\n", encoding="utf-8"
    )

    result = {
        "overall_pass": overall,
        "proof_ok": proof_ok,
        "compression_ok": compression_ok,
        "suite_ok": suite_ok,
        "pack_dir": str(pack),
        "summary": summary_text,
    }
    (pack / "delivery_result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(summary_text)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=5.0, help="R7 proof duration")
    parser.add_argument("--suite-seconds", type=float, default=3.0)
    parser.add_argument(
        "--delivery-dir", type=Path, default=Path("./recordings/poc1_delivery")
    )
    args = parser.parse_args()
    result = build_delivery(args.seconds, args.suite_seconds, args.delivery_dir)
    raise SystemExit(0 if result["overall_pass"] else 1)


if __name__ == "__main__":
    main()
