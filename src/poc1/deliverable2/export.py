"""
R10 — export .bag / .bd3 / .db3 to MP4 (H.264 or H.265 when available).

.bag  → Intel RealSense SDK playback (color preferred, else colorized depth)
.bd3 / .db3 → ROS 2 bags via pure-Python ``rosbags`` (+ ``rosbags-image``);
              fallbacks: OpenCV, ffmpeg, misnamed RealSense bag
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

_IMAGE_MSG_TYPES = {
    "sensor_msgs/msg/Image",
    "sensor_msgs/Image",
    "sensor_msgs/msg/CompressedImage",
    "sensor_msgs/CompressedImage",
}


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
    ffmpeg = _find_ffmpeg()
    if not ffmpeg:
        return ExportResult(
            False,
            None,
            label,
            message="ffmpeg not found on PATH (install ffmpeg or add it to PATH)",
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


def _find_ffmpeg() -> Optional[str]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg
    # Common Windows install locations when ffmpeg is not on PATH.
    candidates = [
        Path(r"C:\ffmpeg\bin\ffmpeg.exe"),
        Path(r"C:\Program Files\ffmpeg\bin\ffmpeg.exe"),
        Path.home() / "scoop" / "apps" / "ffmpeg" / "current" / "ffmpeg.exe",
        Path.home() / "AppData" / "Local" / "Microsoft" / "WinGet" / "Links" / "ffmpeg.exe",
    ]
    for path in candidates:
        if path.is_file():
            return str(path)
    return None


def _looks_like_ros2_db3(source: Path) -> bool:
    """Best-effort detect ROS2 sqlite3 bag without requiring rosbag2."""
    try:
        import sqlite3

        conn = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
        try:
            cur = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' LIMIT 20"
            )
            tables = {row[0] for row in cur.fetchall()}
        finally:
            conn.close()
        return bool({"topics", "messages", "schema"} & tables) or "topics" in tables
    except Exception:  # noqa: BLE001
        return False


def _resolve_rosbag_dir(source: Path) -> tuple[Path, Optional[tempfile.TemporaryDirectory]]:
    """
    Return a directory path AnyReader can open.

    ROS2 bags are usually a folder (metadata.yaml + *.db3). If the user selected
    a lone .db3, build a temporary bag folder with a minimal metadata.yaml.
    """
    source = Path(source)
    if source.is_dir():
        return source, None

    meta = source.parent / "metadata.yaml"
    if meta.is_file():
        return source.parent, None

    # Sibling folder with the same stem (common layout).
    sibling = source.parent / source.stem
    if sibling.is_dir() and (sibling / "metadata.yaml").is_file():
        return sibling, None

    tmp = tempfile.TemporaryDirectory(
        prefix="poc1_rosbag_", ignore_cleanup_errors=True
    )
    bag_dir = Path(tmp.name)
    db_name = source.name
    shutil.copy2(source, bag_dir / db_name)
    # Minimal metadata so rosbags can open sqlite3 storage; topic list is
    # discovered from the database by the reader.
    (bag_dir / "metadata.yaml").write_text(
        "\n".join(
            [
                "rosbag2_bagfile_information:",
                "  version: 5",
                "  storage_identifier: sqlite3",
                "  duration:",
                "    nanoseconds: 0",
                "  starting_time:",
                "    nanoseconds_since_epoch: 0",
                "  message_count: 0",
                "  topics_with_message_count: []",
                "  compression_format: ''",
                "  compression_mode: ''",
                "  relative_file_paths:",
                f"    - {db_name}",
                "  files:",
                f"    - path: {db_name}",
                "      starting_time:",
                "        nanoseconds_since_epoch: 0",
                "      duration:",
                "        nanoseconds: 0",
                "      message_count: 0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return bag_dir, tmp


def _pick_image_connections(connections) -> list:
    """Prefer color/image topics; fall back to any Image / CompressedImage."""
    image_conns = [
        c
        for c in connections
        if getattr(c, "msgtype", "") in _IMAGE_MSG_TYPES
        or str(getattr(c, "msgtype", "")).endswith("/Image")
        or str(getattr(c, "msgtype", "")).endswith("/CompressedImage")
    ]
    if not image_conns:
        return []

    def score(conn) -> tuple:
        topic = (conn.topic or "").lower()
        msg = (conn.msgtype or "").lower()
        prefer = 0
        if "compressed" in msg:
            prefer += 1
        if any(k in topic for k in ("color", "rgb", "image_raw", "bgr", "camera")):
            prefer += 3
        if "depth" in topic or "infra" in topic:
            prefer -= 2
        return (prefer, -len(topic))

    image_conns.sort(key=score, reverse=True)
    return image_conns


def _export_rosbag2_to_mp4(
    source: Path,
    output: Path,
    fourcc: str,
    label: str,
    progress: Callable[[str], None],
) -> ExportResult:
    """Convert ROS2 .db3/.bd3 (or bag folder) to MP4 using pure-Python rosbags."""
    try:
        from rosbags.highlevel import AnyReader
        from rosbags.image import message_to_cvimage
    except ImportError:
        return ExportResult(
            False,
            None,
            label,
            message=(
                "ROS bag support needs packages: rosbags + rosbags-image. "
                "Run: uv sync"
            ),
        )

    # Skip obvious non-sqlite junk before creating temp bag folders.
    if source.is_file() and not _looks_like_ros2_db3(source):
        return ExportResult(
            False,
            None,
            label,
            message=f"{source.name} is not a ROS 2 sqlite bag (.db3)",
        )

    bag_dir, tmp_holder = _resolve_rosbag_dir(source)
    try:
        progress(f"Opening ROS bag: {source.name}")
        with AnyReader([bag_dir]) as reader:
            image_conns = _pick_image_connections(reader.connections)
            if not image_conns:
                topics = ", ".join(
                    f"{c.topic} ({c.msgtype})" for c in list(reader.connections)[:12]
                ) or "(none)"
                return ExportResult(
                    False,
                    None,
                    label,
                    message=(
                        f"No image topics in {source.name}.\n"
                        f"Topics found: {topics}\n"
                        "Need sensor_msgs/Image or CompressedImage to make an MP4."
                    ),
                )

            # Use the best-scoring image topic.
            chosen = image_conns[0]
            progress(f"Exporting topic {chosen.topic} ({chosen.msgtype})")

            writer: Optional[cv2.VideoWriter] = None
            used_fourcc = fourcc
            frames_written = 0
            width = height = 0
            stamps: list[int] = []
            connections = [chosen]

            for connection, timestamp, rawdata in reader.messages(
                connections=connections
            ):
                try:
                    msg = reader.deserialize(rawdata, connection.msgtype)
                    frame = message_to_cvimage(msg, "bgr8")
                except Exception as exc:  # noqa: BLE001
                    logger.debug("skip frame at %s: %s", timestamp, exc)
                    continue
                if frame is None or not hasattr(frame, "shape"):
                    continue
                if frame.ndim == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                elif frame.shape[2] == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                h, w = frame.shape[:2]
                stamps.append(int(timestamp))
                if writer is None:
                    width, height = w, h
                    # Estimate FPS from first few stamps later; start with 30.
                    writer, used_fourcc = _open_writer(output, fourcc, 30.0, w, h)
                    if writer is None:
                        return ExportResult(
                            False,
                            None,
                            label,
                            message=f"VideoWriter failed for {fourcc}/{w}x{h}",
                        )
                if (w, h) != (width, height):
                    frame = cv2.resize(frame, (width, height))
                writer.write(np.ascontiguousarray(frame))
                frames_written += 1
                if frames_written % 60 == 0:
                    progress(f"Exported {frames_written} frames from ROS bag…")

            if writer is not None:
                writer.release()

            if frames_written <= 0:
                output.unlink(missing_ok=True)
                return ExportResult(
                    False,
                    None,
                    label,
                    message=(
                        f"Topic {chosen.topic} had no decodable image frames."
                    ),
                )

            # Re-stamp container FPS from message timestamps when possible.
            fps = 30.0
            if len(stamps) >= 2:
                elapsed_ns = stamps[-1] - stamps[0]
                if elapsed_ns > 0:
                    fps = (len(stamps) - 1) / (elapsed_ns / 1e9)
                    fps = float(min(max(fps, 1.0), 120.0))
            if abs(fps - 30.0) > 0.5:
                progress(f"Adjusting container FPS to ~{fps:.1f}")
                tmp_out = output.with_suffix(".retimed.mp4")
                if _rewrite_mp4_fps(output, tmp_out, fps, used_fourcc):
                    tmp_out.replace(output)
                else:
                    tmp_out.unlink(missing_ok=True)

            final_label = (
                label if used_fourcc == fourcc else f"{label} -> {used_fourcc}"
            )
            return ExportResult(
                True,
                output,
                f"{final_label} via rosbags/{chosen.topic}",
                frames=frames_written,
                message=(
                    f"Wrote {frames_written} frames from {chosen.topic} -> {output.name}"
                ),
            )
    except Exception as exc:  # noqa: BLE001
        logger.exception("rosbag export failed")
        output.unlink(missing_ok=True)
        return ExportResult(
            False,
            None,
            label,
            message=f"ROS bag export failed: {exc}",
        )
    finally:
        if tmp_holder is not None:
            try:
                tmp_holder.cleanup()
            except OSError:
                pass


def _rewrite_mp4_fps(
    source: Path, dest: Path, fps: float, fourcc: str
) -> bool:
    """Copy frames into a new MP4 stamped with the given FPS."""
    with quiet_opencv():
        cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        cap.release()
        return False
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    writer, _used = _open_writer(dest, fourcc, fps, max(width, 1), max(height, 1))
    if writer is None:
        cap.release()
        return False
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            writer.write(frame)
    finally:
        cap.release()
        writer.release()
    return dest.is_file() and dest.stat().st_size > 32


def _export_bd3_or_db3(
    source: Path,
    output: Path,
    fourcc: str,
    label: str,
    progress: Callable[[str], None],
) -> ExportResult:
    """
    .bd3 / .db3 may be ROS2 sqlite bags or misc containers.

    Prefer pure-Python rosbags image export, then OpenCV → ffmpeg →
    misnamed RealSense bag.
    """
    ros_try = _export_rosbag2_to_mp4(source, output, fourcc, label, progress)
    if ros_try.ok:
        return ros_try

    result = _export_via_opencv(source, output, fourcc, label, progress)
    if result.ok:
        return result

    codec_hint = "h265" if "H.265" in label or "265" in fourcc else "h264"
    ff = _export_via_ffmpeg(source, output, codec_hint, label, progress)
    if ff.ok:
        return ff

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

    ros2 = _looks_like_ros2_db3(source)
    guidance = (
        f"Could not convert {source.name} to MP4.\n\n"
        f"ROS bag path: {ros_try.message}\n\n"
        "Tips:\n"
        "  • Run: uv sync   (installs rosbags + rosbags-image)\n"
        "  • Bag must contain sensor_msgs/Image or CompressedImage\n"
        "  • If the bag is a folder, Export that folder’s .db3 or keep metadata.yaml beside it\n\n"
        f"OpenCV: {result.message}\n"
        f"ffmpeg: {ff.message}\n"
        f"bag-try: {bag_msg}"
    )
    if not ros2 and "No image topics" not in ros_try.message:
        guidance = (
            f"Could not convert {source.name} to MP4.\n\n"
            f"ROS bag path: {ros_try.message}\n"
            f"OpenCV: {result.message}\n"
            f"ffmpeg: {ff.message}\n"
            f"bag-try: {bag_msg}"
        )

    return ExportResult(False, None, label, message=guidance)
