"""
Independent verification that a recorded MP4 matches the JSON stop report.
Barcode checks are used for synthetic sources (fake / simulated RealSense).
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2

from poc1.frame_source import read_seq_barcode

logger = logging.getLogger("poc1.verifier")


@dataclass
class VerifyResult:
    ok: bool
    frames_checked: int
    first_seq: int | None
    last_seq: int | None
    gaps: list[tuple[int, int]]
    duplicates: int
    unreadable_barcodes: int
    message: str

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "frames_checked": self.frames_checked,
            "first_seq": self.first_seq,
            "last_seq": self.last_seq,
            "gaps": self.gaps,
            "duplicates": self.duplicates,
            "unreadable_barcodes": self.unreadable_barcodes,
            "message": self.message,
        }


def verify_mp4(
    path: Path,
    expected_frames: int | None = None,
    expected_width: int | None = None,
    expected_height: int | None = None,
    expected_fps: float | None = None,
    require_barcode: bool = True,
) -> VerifyResult:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return VerifyResult(
            ok=False, frames_checked=0, first_seq=None, last_seq=None,
            gaps=[], duplicates=0, unreadable_barcodes=0,
            message=f"Could not open {path}",
        )

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)

    gaps: list[tuple[int, int]] = []
    duplicates = 0
    unreadable = 0
    first_seq: int | None = None
    last_seq: int | None = None
    count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        count += 1
        if not require_barcode:
            continue
        seq = read_seq_barcode(frame)
        if seq is None:
            unreadable += 1
            continue
        if first_seq is None:
            first_seq = seq
        if last_seq is not None:
            if seq == last_seq:
                duplicates += 1
            elif seq != last_seq + 1:
                gaps.append((last_seq + 1, seq))
        last_seq = seq
    cap.release()

    problems: list[str] = []
    if count == 0:
        problems.append("decoded 0 frames")
    if expected_frames is not None and count != expected_frames:
        problems.append(f"frame count {count} != expected {expected_frames}")
    if expected_width is not None and width != expected_width:
        problems.append(f"width {width} != expected {expected_width}")
    if expected_height is not None and height != expected_height:
        problems.append(f"height {height} != expected {expected_height}")
    if expected_fps is not None and fps > 0:
        abs_ok = abs(fps - expected_fps) <= 1.0
        rel_ok = abs(fps - expected_fps) / max(expected_fps, 1.0) <= 0.15
        if not (abs_ok or rel_ok):
            problems.append(f"fps {fps:.1f} != expected {expected_fps}")
    if require_barcode:
        if gaps:
            problems.append(f"{len(gaps)} sequence gaps in barcode")
        if duplicates:
            problems.append(f"{duplicates} duplicate barcodes")
        if unreadable:
            problems.append(f"{unreadable} frames with unreadable barcode")
        if first_seq is not None and last_seq is not None:
            expected_span = last_seq - first_seq + 1
            readable = count - unreadable
            if expected_span != readable and not gaps and not duplicates:
                problems.append(
                    f"barcode span {expected_span} != readable frames {readable}"
                )
        if first_seq is not None and first_seq != 0 and expected_frames is None:
            problems.append(f"first barcode seq={first_seq}, expected 0")

    ok = len(problems) == 0
    return VerifyResult(
        ok=ok,
        frames_checked=count,
        first_seq=first_seq,
        last_seq=last_seq,
        gaps=gaps,
        duplicates=duplicates,
        unreadable_barcodes=unreadable,
        message="PASS" if ok else "; ".join(problems),
    )


def verify_from_report(report_path: Path, require_barcode: bool = True) -> VerifyResult:
    data = json.loads(report_path.read_text(encoding="utf-8"))
    mp4 = Path(data.get("output_path") or str(report_path).replace(".report.json", ".mp4"))
    # After webcam FPS remux, container_fps is authoritative.
    expected_fps = data.get("container_fps") or data.get("measured_fps") or data.get("target_fps")
    result = verify_mp4(
        mp4,
        expected_frames=data.get("frames_written"),
        expected_width=data.get("width"),
        expected_height=data.get("height"),
        expected_fps=float(expected_fps) if expected_fps else None,
        require_barcode=require_barcode,
    )
    if not data.get("no_frame_drops", False):
        return VerifyResult(
            ok=False,
            frames_checked=result.frames_checked,
            first_seq=result.first_seq,
            last_seq=result.last_seq,
            gaps=list(data.get("sequence_gaps") or []) + result.gaps,
            duplicates=result.duplicates,
            unreadable_barcodes=result.unreadable_barcodes,
            message=f"report no_frame_drops=false; file check: {result.message}",
        )
    return result


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="MP4 file or .report.json")
    parser.add_argument(
        "--no-barcode",
        action="store_true",
        help="Skip barcode checks (for real webcam / OS virtual-cam footage)",
    )
    args = parser.parse_args()
    require_barcode = not args.no_barcode
    if args.path.suffix.lower() == ".json" or args.path.name.endswith(".report.json"):
        result = verify_from_report(args.path, require_barcode=require_barcode)
    else:
        result = verify_mp4(args.path, require_barcode=require_barcode)
    print(json.dumps(result.to_dict(), indent=2))
    raise SystemExit(0 if result.ok else 1)


if __name__ == "__main__":
    main()
