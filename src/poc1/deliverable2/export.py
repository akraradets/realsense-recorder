"""
R10 — export .bag / .bd3 / .db3 to MP4 (H.264 or H.265 when available).

.bag / RealSense .db3 → Intel RealSense SDK playback (color preferred, else depth)
ROS2 .db3 / .bd3      → pure-Python ``rosbags`` (+ ``rosbags-image``) when Image topics exist
fallbacks             → OpenCV, ffmpeg
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

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
    for p in sorted(folder.iterdir()):
        if p.is_dir() and (
            p.name.endswith("_color") or (p / "metadata.yaml").is_file()
        ):
            out.append(p)
    return out


def export_to_mp4(
    source: Path,
    output: Optional[Path] = None,
    *,
    codec: str = "h264",
    on_progress: ProgressCb = None,
) -> ExportResult:
    """Decode .bag / .bd3 / .db3 (or ROS2 folder) to a NEW MP4. Never overwrite Record MP4."""
    source = Path(source)
    if not source.exists():
        return ExportResult(False, None, "", message=f"File not found: {source}")

    ext = source.suffix.lower()
    codec_key = (codec or "h264").strip().lower()
    fourcc, label = resolve_export_fourcc(codec_key)
    if output is None:
        output = source.with_name(f"{source.stem}_{codec_key}.mp4")
    else:
        output = Path(output)
    output = _avoid_overwriting_record_mp4(source, output, codec_key)
    output.parent.mkdir(parents=True, exist_ok=True)

    def progress(msg: str) -> None:
        logger.info("%s", msg)
        if on_progress:
            on_progress(msg)

    if source.is_dir():
        return _export_bd3_or_db3(source, output, fourcc, label, progress)
    if ext == ".bag":
        return _export_realsense_bag(source, output, fourcc, label, progress)
    if ext in {".bd3", ".db3"}:
        return _export_bd3_or_db3(source, output, fourcc, label, progress)
    # Allow re-encoding existing MP4/AVI through the same path (handy in browser).
    return _export_via_opencv(source, output, fourcc, label, progress)


def _avoid_overwriting_record_mp4(source: Path, output: Path, codec: str) -> Path:
    """Keep the original Record {stem}.mp4; write {stem}_{codec}.mp4 instead."""
    record = source.with_suffix(".mp4") if source.is_file() else None
    if source.is_dir():
        name = source.name
        record = source.parent / (f"{name[:-6]}.mp4" if name.endswith("_color") else f"{name}.mp4")
    if record is not None and output.resolve() == record.resolve():
        alt = source.with_name(f"{source.stem}_{codec}.mp4")
        if alt.resolve() == record.resolve():
            alt = source.with_name(f"{source.stem}_export_{codec}.mp4")
        return alt
    return output


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


def _sibling_record_mp4(source: Path) -> Optional[Path]:
    """MP4 written beside a RealSense .bag/.db3 or Elgato ROS2 folder on the same take."""
    source = Path(source)
    candidates: list[Path] = [source.with_suffix(".mp4")]
    if source.is_dir():
        name = source.name
        if name.endswith("_color"):
            candidates.append(source.parent / f"{name[:-6]}.mp4")
        candidates.append(source.parent / f"{source.name}.mp4")
    else:
        stem = source.stem
        if stem.endswith("_color"):
            candidates.append(source.parent / f"{stem[:-6]}.mp4")
            candidates.append(source.parent.parent / f"{stem[:-6]}.mp4")
        candidates.append(source.parent / f"{stem}.mp4")
    seen: set[Path] = set()
    for candidate in candidates:
        key = candidate.resolve() if candidate.exists() else candidate
        if key in seen:
            continue
        seen.add(key)
        if candidate.is_file() and candidate.stat().st_size > 32:
            return candidate
    return None


def _start_realsense_playback(rs: Any, source: Path):
    """
    Open a RealSense .bag/.db3 for playback.

    Important: do not force color+depth together — Record usually writes color
    only, and enabling a missing stream makes resolve() fail.
    """
    source_s = str(Path(source).resolve())
    last_exc: Optional[BaseException] = None

    strategies: list[tuple[str, Callable[[], Any]]] = []

    def _cfg_file_only():
        pipeline = rs.pipeline()
        config = rs.config()
        try:
            rs.config.enable_device_from_file(config, source_s, repeat_playback=False)
        except TypeError:
            rs.config.enable_device_from_file(config, source_s)
        return pipeline, pipeline.start(config)

    def _cfg_color():
        pipeline = rs.pipeline()
        config = rs.config()
        try:
            rs.config.enable_device_from_file(config, source_s, repeat_playback=False)
        except TypeError:
            rs.config.enable_device_from_file(config, source_s)
        config.enable_stream(rs.stream.color)
        return pipeline, pipeline.start(config)

    def _cfg_depth():
        pipeline = rs.pipeline()
        config = rs.config()
        try:
            rs.config.enable_device_from_file(config, source_s, repeat_playback=False)
        except TypeError:
            rs.config.enable_device_from_file(config, source_s)
        config.enable_stream(rs.stream.depth)
        return pipeline, pipeline.start(config)

    def _load_device():
        ctx = rs.context()
        device = ctx.load_device(source_s)
        pipeline = rs.pipeline()
        config = rs.config()
        # Bind to the loaded playback device by serial when available.
        try:
            serial = device.get_info(rs.camera_info.serial_number)
            config.enable_device(serial)
        except Exception:  # noqa: BLE001
            pass
        try:
            rs.config.enable_device_from_file(config, source_s, repeat_playback=False)
        except TypeError:
            try:
                rs.config.enable_device_from_file(config, source_s)
            except Exception:  # noqa: BLE001
                pass
        return pipeline, pipeline.start(config)

    for name, fn in (
        ("file-streams", _cfg_file_only),
        ("color", _cfg_color),
        ("depth", _cfg_depth),
        ("load_device", _load_device),
    ):
        try:
            pipeline, profile = fn()
            logger.info("RealSense playback opened via %s: %s", name, source.name)
            return pipeline, profile
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            logger.warning("RealSense playback %s failed for %s: %s", name, source.name, exc)

    raise RuntimeError(str(last_exc) if last_exc else "Could not open RealSense recording")


def _export_realsense_bag(
    source: Path,
    output: Path,
    fourcc: str,
    label: str,
    progress: Callable[[str], None],
) -> ExportResult:
    """Playback a RealSense SDK recording (.bag or .db3) and encode color/depth to MP4."""
    try:
        import pyrealsense2 as rs
    except ImportError:
        return ExportResult(
            False,
            None,
            label,
            message="pyrealsense2 is required for RealSense .bag/.db3 export. "
            "Run: uv sync --extra realsense",
        )

    source = Path(source).resolve()
    if not source.is_file():
        return ExportResult(False, None, label, message=f"File not found: {source}")
    if source.stat().st_size < 64:
        return ExportResult(
            False,
            None,
            label,
            message=f"{source.name} is empty/too small — re-Record with SDK file checked",
        )

    progress(f"Opening RealSense recording: {source.name}")
    try:
        pipeline, profile = _start_realsense_playback(rs, source)
    except Exception as exc:  # noqa: BLE001
        return ExportResult(
            False,
            None,
            label,
            message=f"Could not open RealSense recording: {exc}",
        )

    try:
        playback = profile.get_device().as_playback()
        playback.set_real_time(False)
    except Exception as exc:  # noqa: BLE001
        logger.warning("as_playback failed (continuing): %s", exc)

    colorizer = rs.colorizer()
    writer: Optional[cv2.VideoWriter] = None
    used_fourcc = fourcc
    frames_written = 0
    width = height = 0
    fps = 30.0

    def to_bgr(color_frame) -> Optional[np.ndarray]:
        arr = np.asanyarray(color_frame.get_data())
        fmt = None
        try:
            fmt = color_frame.get_profile().format()
        except Exception:  # noqa: BLE001
            pass
        if arr.ndim == 3 and arr.shape[2] == 3:
            if fmt == rs.format.bgr8:
                return arr
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        if arr.ndim == 3 and arr.shape[2] == 2:
            return cv2.cvtColor(arr, cv2.COLOR_YUV2BGR_YUY2)
        if arr.ndim == 2:
            if fmt == rs.format.yuyv:
                return cv2.cvtColor(arr, cv2.COLOR_YUV2BGR_YUY2)
            if arr.dtype == np.uint16:
                scaled = cv2.convertScaleAbs(arr, alpha=255.0 / 65535.0)
                return cv2.cvtColor(scaled, cv2.COLOR_GRAY2BGR)
            return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        return None

    try:
        try:
            color_prof = profile.get_stream(rs.stream.color).as_video_stream_profile()
            fps = float(color_prof.fps() or 30)
        except Exception:  # noqa: BLE001
            try:
                depth_prof = profile.get_stream(rs.stream.depth).as_video_stream_profile()
                fps = float(depth_prof.fps() or 30)
            except Exception:  # noqa: BLE001
                fps = 30.0

        idle = 0
        while True:
            try:
                frameset = pipeline.wait_for_frames(timeout_ms=5000)
            except RuntimeError:
                idle += 1
                if idle >= 4:
                    break
                continue
            idle = 0
            color = frameset.get_color_frame()
            frame = None
            if color:
                frame = to_bgr(color)
            if frame is None:
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
            False, None, label, message="Recording contained no exportable frames"
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
    Decode .bd3 / .db3 / ROS2 folder to a NEW MP4.

    RealSense SDK recordings → pyrealsense2 playback.
    Elgato/UVC ROS2 bags (Image / CompressedImage) → rosbags.
    Never copy the sibling Record MP4.
    """
    source = Path(source)
    # ROS2 bags are folders (or a .db3 beside metadata.yaml). Lone Intel .db3
    # files can also be sqlite and falsely match _looks_like_ros2_db3 — always
    # try RealSense SDK first for those.
    is_ros_folder = source.is_dir() or (
        source.is_file()
        and (source.parent / "metadata.yaml").is_file()
    )
    looks_ros_sqlite = source.is_file() and _looks_like_ros2_db3(source)
    looks_ros = is_ros_folder or looks_ros_sqlite

    rs_try = ExportResult(
        False, None, label, message="Skipped RealSense SDK (ROS 2 bag folder)"
    )
    prefer_sdk_first = source.is_file() and source.suffix.lower() in {
        ".bag",
        ".bd3",
        ".db3",
    } and not is_ros_folder
    if prefer_sdk_first or not looks_ros:
        rs_try = _export_realsense_bag(source, output, fourcc, label, progress)
        if rs_try.ok:
            return rs_try

    ros_try = _export_rosbag2_to_mp4(source, output, fourcc, label, progress)
    if ros_try.ok:
        return ros_try

    # If we skipped SDK because looks_ros was true but ROS failed, try SDK once.
    if looks_ros and not prefer_sdk_first and source.is_file():
        rs_try = _export_realsense_bag(source, output, fourcc, label, progress)
        if rs_try.ok:
            return rs_try

    result = _export_via_opencv(source, output, fourcc, label, progress)
    if result.ok:
        return result

    codec_hint = "h265" if "H.265" in label or "265" in fourcc else "h264"
    ff = _export_via_ffmpeg(source, output, codec_hint, label, progress)
    if ff.ok:
        return ff

    # Lone Intel .db3 next to Record MP4: lead with SDK failure, never ROS first.
    if prefer_sdk_first or (source.is_file() and not is_ros_folder):
        lead = (
            f"Could not decode RealSense recording {source.name} to MP4.\n"
            f"SDK playback: {rs_try.message}\n"
            "This is an Intel RealSense .db3/.bag, not a ROS Image bag "
            "(“no image topics” is expected and is not the real failure).\n"
            "Play the Record MP4 in Library. Export needs matching librealsense "
            "(uv sync --extra realsense).\n"
        )
        if "version" in (rs_try.message or "").lower() or "sdk" in (rs_try.message or "").lower():
            lead += (
                "This PC’s librealsense/pyrealsense2 cannot play that file "
                "(often recorded with a different SDK build). "
                "Run: uv sync --extra realsense  ·  Play the Record MP4 instead. "
                "Export cannot invent frames if the SDK cannot open the .db3.\n"
            )
    elif is_ros_folder or looks_ros:
        lead = (
            f"Could not decode ROS 2 bag {source.name} to MP4.\n"
            f"{ros_try.message}\n"
        )
    else:
        lead = (
            f"Could not decode {source.name} to MP4.\n"
            f"SDK: {rs_try.message}\n"
            f"ROS: {ros_try.message}\n"
        )
    ff_extra = ""
    if "ffmpeg not found" in (ff.message or "").lower():
        ff_extra = (
            "\n  • Install ffmpeg on PATH (helps export fallbacks and Elgato names).\n"
        )
    guidance = (
        lead
        + f"\nROS bag path: {ros_try.message}\n"
        + f"OpenCV: {result.message}\n"
        + f"ffmpeg: {ff.message}\n\n"
        "Tips:\n"
        "  • Record MP4 is unchanged — play that file in Library.\n"
        f"  • Export writes a NEW {source.stem}_h264.mp4 decoded from the bag.\n"
        "  • uv sync --extra realsense && uv run poc1 (title sdk-record-v19+)\n"
        "  • Elgato ROS2 bags are folders named *_color with metadata.yaml"
        + ff_extra
    )
    return ExportResult(False, None, label, message=guidance)
