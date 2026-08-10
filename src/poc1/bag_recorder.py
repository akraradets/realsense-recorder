"""
Intel RealSense SDK recording (legacy .bag or ROS2 .db3).

Newer librealsense builds require ``.db3`` (rosbag2/SQLite). Older builds use
``.bag``. We probe the SDK once and write the extension it accepts.

Recording is armed only at Record time — never during Start preview.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("poc1.bag")

# Cached after first successful probe: ".bag" or ".db3"
_RECORD_SUFFIX: Optional[str] = None


def can_record_bag(source: Any) -> bool:
    return getattr(source, "mode", None) == "hardware"


def recording_suffix() -> str:
    """Return the file suffix this pyrealsense2 build accepts for record_to_file."""
    global _RECORD_SUFFIX
    if _RECORD_SUFFIX:
        return _RECORD_SUFFIX
    # Default .bag; start_bag_recording flips to .db3 if the SDK rejects .bag.
    return ".bag"


def set_recording_suffix(suffix: str) -> None:
    global _RECORD_SUFFIX
    suf = suffix if suffix.startswith(".") else f".{suffix}"
    if suf not in {".bag", ".db3"}:
        suf = ".bag"
    _RECORD_SUFFIX = suf
    logger.info("RealSense SDK record suffix set to %s", suf)


def with_sdk_record_suffix(path: Path) -> Path:
    """Rewrite path to the SDK-required suffix (.bag or .db3)."""
    path = Path(path)
    return path.with_suffix(recording_suffix())


def paths_to_try(path: Path) -> list[Path]:
    """Candidate record paths: known-good suffix first, then the other."""
    path = Path(path)
    primary = with_sdk_record_suffix(path)
    other_suf = ".db3" if primary.suffix.lower() == ".bag" else ".bag"
    secondary = path.with_suffix(other_suf)
    out = [primary]
    if secondary != primary:
        out.append(secondary)
    # Also try original if different
    if path.suffix.lower() in {".bag", ".db3"} and path not in out:
        out.insert(0, path)
    # Dedupe preserving order
    seen: set[str] = set()
    unique: list[Path] = []
    for p in out:
        key = str(p.resolve()) if p.parent.exists() else str(p)
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def _restore_wanted(source: Any, wanted: tuple[int, int, int, str]) -> None:
    w, h, fps, fmt = wanted
    source.width = w
    source.height = h
    source.target_fps = fps
    source.pixel_format = fmt
    if hasattr(source, "_wanted"):
        source._wanted = wanted


def _wanted_from_source(source: Any) -> tuple[int, int, int, str]:
    wanted = (
        int(getattr(source, "width", 640)),
        int(getattr(source, "height", 480)),
        int(getattr(source, "target_fps", 30) or 30),
        str(getattr(source, "pixel_format", "bgr8") or "bgr8"),
    )
    saved_wanted = getattr(source, "_wanted", None)
    if isinstance(saved_wanted, tuple) and len(saved_wanted) == 4:
        wanted = (
            int(saved_wanted[0]),
            int(saved_wanted[1]),
            int(saved_wanted[2]),
            str(saved_wanted[3]),
        )
    return wanted


def _call_start(source: Any, *, allow_fallback: bool) -> None:
    start = getattr(source, "start")
    try:
        start(allow_fallback=allow_fallback)
    except TypeError:
        start()


def pending_bag_path(out_dir: Path, slot_id: int, prefix: str) -> Path:
    """Unused (preview must not arm recording). Kept for API compatibility."""
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in prefix) or "cam"
    return Path(out_dir).resolve() / f"{safe}_slot{slot_id}_recording{recording_suffix()}"


def prearm_bag_on_source(source: Any, bag_path: Path) -> None:
    """No-op: never arm SDK recording during preview (causes Start preview errors)."""
    source.bag_path = None
    source._bag_path = None
    source._bag_final_path = None
    if hasattr(source, "bag_start_paused"):
        source.bag_start_paused = False
    if hasattr(source, "_rs_recorder"):
        source._rs_recorder = None
    _ = bag_path


def clear_bag_on_source(source: Any) -> None:
    """Ensure preview starts without enable_record_to_file."""
    if hasattr(source, "bag_path"):
        source.bag_path = None
    if hasattr(source, "_bag_path"):
        source._bag_path = None
    if hasattr(source, "_bag_final_path"):
        source._bag_final_path = None
    if hasattr(source, "bag_start_paused"):
        source.bag_start_paused = False
    if hasattr(source, "_rs_recorder"):
        source._rs_recorder = None


def start_bag_recording(source: Any, bag_path: Path) -> Path:
    """
    Restart RealSense with SDK file recording to ``bag_path``.

    Automatically retries with ``.db3`` if this SDK rejects ``.bag`` (and vice versa).
    Raises RuntimeError if arming fails.
    """
    if not can_record_bag(source):
        raise RuntimeError(
            "RealSense recording requires a live [realsense] hardware device "
            "(not UVC twin / simulation)."
        )
    try:
        import pyrealsense2 as rs  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "pyrealsense2 is not installed — cannot write RealSense bag. "
            "Run: uv sync --extra realsense"
        ) from exc

    wanted = _wanted_from_source(source)
    last_exc: Optional[BaseException] = None
    candidates = paths_to_try(Path(bag_path))

    for candidate in candidates:
        candidate = candidate.resolve()
        candidate.parent.mkdir(parents=True, exist_ok=True)
        if candidate.exists():
            try:
                if candidate.is_file():
                    candidate.unlink()
                elif candidate.is_dir():
                    # rosbag2 sometimes uses a folder; remove empty marker files only
                    pass
            except OSError as exc:
                last_exc = exc
                continue

        for attempt in range(1, 4):
            try:
                try:
                    source.stop()
                except Exception:  # noqa: BLE001
                    pass
                time.sleep(0.3 * attempt)

                source.bag_path = candidate
                source._bag_path = candidate
                source._bag_final_path = candidate
                if hasattr(source, "bag_start_paused"):
                    source.bag_start_paused = False
                _restore_wanted(source, wanted)
                _call_start(source, allow_fallback=False)

                size = 0
                for _ in range(12):
                    try:
                        _ = source.read()
                    except Exception:  # noqa: BLE001
                        pass
                    if candidate.exists():
                        try:
                            size = candidate.stat().st_size
                        except OSError:
                            size = 0
                        if size > 0:
                            break
                    # rosbag2 may create a directory
                    if candidate.with_suffix("").exists():
                        break
                    time.sleep(0.05)

                if candidate.exists() and candidate.stat().st_size <= 0:
                    # Some SDK builds create the file on stop only — allow empty
                    # at arm time if pipeline is live.
                    if getattr(source, "_pipeline", None) is None:
                        raise RuntimeError("SDK created no record file after start")

                set_recording_suffix(candidate.suffix)
                logger.info(
                    "RealSense record armed (try %d) -> %s (exists=%s size=%s)",
                    attempt,
                    candidate,
                    candidate.exists(),
                    candidate.stat().st_size if candidate.exists() else 0,
                )
                return candidate
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                msg = str(exc).lower()
                logger.warning(
                    "Record arm failed for %s (try %d): %s",
                    candidate.name,
                    attempt,
                    exc,
                )
                try:
                    source.bag_path = None
                    source._bag_path = None
                    source._bag_final_path = None
                    if hasattr(source, "bag_start_paused"):
                        source.bag_start_paused = False
                    if candidate.exists() and candidate.is_file() and candidate.stat().st_size == 0:
                        candidate.unlink(missing_ok=True)
                except Exception:  # noqa: BLE001
                    pass
                # If SDK explicitly demands the other extension, jump to it.
                if "db3" in msg and candidate.suffix.lower() == ".bag":
                    set_recording_suffix(".db3")
                    break
                if "bag" in msg and "extension" in msg and candidate.suffix.lower() == ".db3":
                    set_recording_suffix(".bag")
                    break

    # Restore preview without recording, then fail hard.
    try:
        clear_bag_on_source(source)
        _restore_wanted(source, wanted)
        if getattr(source, "_pipeline", None) is None:
            time.sleep(0.3)
            try:
                _call_start(source, allow_fallback=False)
            except Exception:
                _restore_wanted(source, wanted)
                _call_start(source, allow_fallback=True)
    except Exception:  # noqa: BLE001
        logger.exception("Could not restore RealSense preview after record failure")

    raise RuntimeError(
        f"Could not start RealSense SDK recording ({bag_path.name}). "
        f"Last error: {last_exc}. "
        "This PC’s RealSense SDK may require .db3 instead of .bag (or the reverse). "
        "Close Intel RealSense Viewer, use USB 3, then Record again."
    )


def stop_bag_recording(source: Any) -> Optional[Path]:
    """Finalize SDK recording by stopping the pipeline, then restore preview."""
    path = getattr(source, "_bag_path", None) or getattr(source, "bag_path", None)
    final = getattr(source, "_bag_final_path", None) or path
    if not path or not can_record_bag(source):
        clear_bag_on_source(source)
        return None

    wanted = _wanted_from_source(source)
    try:
        if hasattr(source, "pause_bag"):
            try:
                source.pause_bag()
            except Exception:  # noqa: BLE001
                pass
        source.stop()
    except Exception:  # noqa: BLE001
        pass

    clear_bag_on_source(source)
    time.sleep(0.4)
    _restore_wanted(source, wanted)
    try:
        _call_start(source, allow_fallback=False)
    except Exception:
        try:
            _restore_wanted(source, wanted)
            _call_start(source, allow_fallback=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("RealSense preview restart after record failed: %s", exc)

    out = Path(final) if final is not None else Path(path)
    if out.exists() and out.stat().st_size > 0:
        logger.info("RealSense record saved: %s (%d bytes)", out, out.stat().st_size)
        return out
    # rosbag2 may write beside the path
    if out.suffix.lower() == ".db3":
        parent = out.parent
        stem = out.stem
        for child in parent.glob(f"{stem}*"):
            if child.is_file() and child.stat().st_size > 0:
                logger.info("RealSense record saved: %s (%d bytes)", child, child.stat().st_size)
                return child
    logger.error("No RealSense record file produced at %s", out)
    return None
