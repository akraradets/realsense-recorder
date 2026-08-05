"""
R10 — export .bag / .bd3 / .db3 to MP4 (H.264 or H.265 when available).

.bag  → Intel RealSense SDK playback (color preferred, else colorized depth)
.bd3 / .db3 → OpenCV decode if the file is video-like; else ffmpeg remux/transcode
              when ffmpeg is on PATH; clear error otherwise
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from poc1.codec import _probe_fourcc
from poc1.device_enum import quiet_opencv

logger = logging.getLogger("poc1.d2.export")

ProgressCb = Optional[Callable[[str], None]]


@dataclass
class ExportResult:
    ok: bool
    output_path: Optional[Path]
    codec_label: str
    frames: int = 0
    message: str = ""


def resolve_export_fourcc(codec: str) -> tuple[str, str]:
    """
    Return (fourcc, label) for the requested codec.

    Falls back to mp4v when H.264/H.265 writers are unavailable on this machine
    (common without OpenH264 / HEVC encoder plugins).
    """
    name = (codec or "h264").strip().lower()
    if name in {"h265", "hevc", "h.265"}:
        candidates = [
            ("hev1", "H.265 (hev1)"),
            ("H265", "H.265 (H265)"),
            ("hvc1", "H.265 (hvc1)"),
            ("mp4v", "MPEG-4 (mp4v fallback)"),
        ]
    else:
        candidates = [
            ("avc1", "H.264 (avc1)"),
            ("H264", "H.264 (H264)"),
            ("X264", "H.264 (X264)"),
            ("mp4v", "MPEG-4 (mp4v fallback)"),
        ]
    for fourcc, label in candidates:
        if _probe_fourcc(fourcc):
            if "fallback" in label:
                logger.warning(
                    "Requested %s but OpenCV cannot write it here — using %s",
                    name, label,
                )
            return fourcc, label
    return "mp4v", "MPEG-4 (mp4v fallback)"


def list_media_files(folder: Path) -> list[Path]:
    """MP4 / bag / bd3 / db3 files in a folder (non-recursive)."""
    folder = Path(folder)
    if not folder.is_dir():
        return []
    exts = {".mp4", ".bag", ".bd3", ".db3", ".avi", ".mkv"}
    out = [
        p for p in sorted(folder.iterdir())
        if p.is_file() and p.suffix.lower() in exts
    ]
    return out


def export_to_mp4(
    source: Path,
    output: Optional[Path] = None,
    *,
    codec: str = "h264",
    on_progress: ProgressCb = None,
) -> ExportResult:
    """Convert .bag / .bd3 / .db3 (or re-encode video) to MP4."""
    source = Path(source)
    if not source.is_file():
        return ExportResult(False, None, "", message=f"File not found: {source}")

    ext = source.suffix.lower()
    fourcc, label = resolve_export_fourcc(codec)
    if output is None:
        output = source.with_name(f"{source.stem}_{codec.lower()}.mp4")
    else:
        output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    def progress(msg: str) -> None:
        logger.info("%s", msg)
        if on_progress:
            on_progress(msg)

    if ext == ".bag":
        return _export_realsense_bag(source, output, fourcc, label, progress)
    if ext in {".bd3", ".db3"}:
        return _export_bd3_or_db3(source, output, fourcc, label, progress)
    # Allow re-encoding existing MP4/AVI through the same path (handy in browser).
    return _export_via_opencv(source, output, fourcc, label, progress)


def _open_writer(
    path: Path, fourcc_str: str, fps: float, width: int, height: int
) -> tuple[Optional[cv2.VideoWriter], str]:
    with quiet_opencv():
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*fourcc_str),
            float(max(fps, 1.0)),
            (width, height),
        )
        if writer.isOpened():
            return writer, fourcc_str
        writer.release()
        if fourcc_str != "mp4v":
            writer = cv2.VideoWriter(
                str(path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                float(max(fps, 1.0)),
                (width, height),
            )
            if writer.isOpened():
                return writer, "mp4v"
            writer.release()
    return None, fourcc_str


def _export_realsense_bag(
    source: Path,
    output: Path,
    fourcc: str,
    label: str,
    progress: Callable[[str], None],
) -> ExportResult:
    try:
        import pyrealsense2 as rs
    except ImportError:
        return ExportResult(
            False,
            None,
            label,
            message="pyrealsense2 is required for .bag export. "
            "Run: uv sync --extra realsense",
        )

    progress(f"Opening RealSense bag: {source.name}")
    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, str(source), repeat_playback=False)
    # Prefer color; enable depth as fallback for older bags.
    try:
        config.enable_stream(rs.stream.color)
    except Exception:  # noqa: BLE001
        pass
    try:
        config.enable_stream(rs.stream.depth)
    except Exception:  # noqa: BLE001
        pass

    try:
        profile = pipeline.start(config)
    except Exception as exc:  # noqa: BLE001
        return ExportResult(
            False, None, label, message=f"Could not open bag: {exc}"
        )

    playback = profile.get_device().as_playback()
    playback.set_real_time(False)

    colorizer = rs.colorizer()
    writer: Optional[cv2.VideoWriter] = None
    used_fourcc = fourcc
    frames_written = 0
    width = height = 0
    fps = 30.0

    try:
        # Probe FPS from color profile when present.
        try:
            color_prof = profile.get_stream(rs.stream.color).as_video_stream_profile()
            fps = float(color_prof.fps() or 30)
        except Exception:  # noqa: BLE001
            try:
                depth_prof = profile.get_stream(rs.stream.depth).as_video_stream_profile()
                fps = float(depth_prof.fps() or 30)
            except Exception:  # noqa: BLE001
                fps = 30.0

        while True:
            try:
                frameset = pipeline.wait_for_frames(timeout_ms=2000)
            except RuntimeError:
                break
            color = frameset.get_color_frame()
            if color:
                frame = np.asanyarray(color.get_data())
                # RealSense color is often RGB; OpenCV writer expects BGR.
                if frame.ndim == 3 and frame.shape[2] == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                depth = frameset.get_depth_frame()
                if not depth:
                    continue
                frame = np.asanyarray(colorizer.colorize(depth).get_data())

            h, w = frame.shape[:2]
            if writer is None:
                width, height = w, h
                writer, used_fourcc = _open_writer(output, fourcc, fps, w, h)
                if writer is None:
                    return ExportResult(
                        False,
                        None,
                        label,
                        message=f"VideoWriter failed for {fourcc}/{w}x{h}",
                    )
                progress(f"Writing {output.name} ({used_fourcc}) …")

            if (w, h) != (width, height):
                frame = cv2.resize(frame, (width, height))
            writer.write(np.ascontiguousarray(frame))
            frames_written += 1
            if frames_written % 60 == 0:
                progress(f"Exported {frames_written} frames…")
    finally:
        try:
            pipeline.stop()
        except Exception:  # noqa: BLE001
            pass
        if writer is not None:
            writer.release()

    if frames_written <= 0:
        output.unlink(missing_ok=True)
        return ExportResult(
            False, None, label, message="Bag contained no exportable frames"
        )

    final_label = label if used_fourcc == fourcc else f"{label} → {used_fourcc}"
    return ExportResult(
        True,
        output,
        final_label,
        frames=frames_written,
        message=f"Wrote {frames_written} frames → {output.name}",
    )


def _export_via_opencv(
    source: Path,
    output: Path,
    fourcc: str,
    label: str,
    progress: Callable[[str], None],
) -> ExportResult:
    progress(f"Opening with OpenCV: {source.name}")
    with quiet_opencv():
        cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        cap.release()
        return ExportResult(
            False, None, label, message=f"OpenCV could not open {source.name}"
        )

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    writer, used_fourcc = _open_writer(output, fourcc, fps, max(width, 1), max(height, 1))
    if writer is None:
        cap.release()
        return ExportResult(False, None, label, message="VideoWriter failed to open")

    frames_written = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if frame.shape[1] != width or frame.shape[0] != height:
                if width > 0 and height > 0:
                    frame = cv2.resize(frame, (width, height))
                else:
                    height, width = frame.shape[:2]
            writer.write(frame)
            frames_written += 1
            if frames_written % 120 == 0:
                progress(f"Exported {frames_written} frames…")
    finally:
        cap.release()
        writer.release()

    if frames_written <= 0:
        output.unlink(missing_ok=True)
        return ExportResult(False, None, label, message="No frames decoded")

    return ExportResult(
        True,
        output,
        label if used_fourcc == fourcc else f"{label} → {used_fourcc}",
        frames=frames_written,
        message=f"Wrote {frames_written} frames → {output.name}",
    )


def _export_via_ffmpeg(
    source: Path,
    output: Path,
    codec: str,
    label: str,
    progress: Callable[[str], None],
) -> ExportResult:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return ExportResult(
            False,
            None,
            label,
            message="ffmpeg not found on PATH",
        )
    vcodec = "libx265" if codec.lower() in {"h265", "hevc", "h.265"} else "libx264"
    progress(f"ffmpeg {vcodec}: {source.name} → {output.name}")
    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-c:v",
        vcodec,
        "-pix_fmt",
        "yuv420p",
        "-an",
        str(output),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600, check=False)
    except Exception as exc:  # noqa: BLE001
        return ExportResult(False, None, label, message=str(exc))
    if proc.returncode != 0 or not output.is_file() or output.stat().st_size < 32:
        err = (proc.stderr or proc.stdout or "ffmpeg failed").strip()
        return ExportResult(False, None, label, message=err[:500])
    return ExportResult(
        True,
        output,
        f"{label} via ffmpeg/{vcodec}",
        message=f"ffmpeg wrote {output.name}",
    )


def _export_bd3_or_db3(
    source: Path,
    output: Path,
    fourcc: str,
    label: str,
    progress: Callable[[str], None],
) -> ExportResult:
    """
    .bd3 / .db3 may be ROS2 sqlite bags or misc containers.

    Try OpenCV → ffmpeg → explicit failure (ROS topic extraction needs rosbag2).
    """
    result = _export_via_opencv(source, output, fourcc, label, progress)
    if result.ok:
        return result

    codec_hint = "h265" if "H.265" in label or "265" in fourcc else "h264"
    ff = _export_via_ffmpeg(source, output, codec_hint, label, progress)
    if ff.ok:
        return ff

    # Misnamed RealSense bag: copy with .bag extension and try the SDK path.
    with tempfile.TemporaryDirectory() as tmp:
        renamed = Path(tmp) / f"{source.stem}.bag"
        try:
            shutil.copy2(source, renamed)
            bag_try = _export_realsense_bag(renamed, output, fourcc, label, progress)
            if bag_try.ok:
                return bag_try
            bag_msg = bag_try.message
        except Exception as exc:  # noqa: BLE001
            bag_msg = str(exc)

    return ExportResult(
        False,
        None,
        label,
        message=(
            f"Could not convert {source.name}.\n"
            "Supported: playable video, RealSense .bag (SDK), or ffmpeg on PATH.\n"
            "ROS2 topic .db3/.bd3 needs rosbag2 (not bundled).\n"
            f"OpenCV: {result.message}\nffmpeg: {ff.message}\nbag-try: {bag_msg}"
        ),
    )
