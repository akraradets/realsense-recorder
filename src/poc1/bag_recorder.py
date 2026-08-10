"""
Intel RealSense .bag recording (opens in RealSense Viewer).

When “Also save RealSense .bag” is checked, .bag is required together with MP4 —
arming must succeed (no silent skip).

Preferred flow:
  1. Preview starts with enable_record_to_file + recorder.pause() (pre-arm).
  2. Record calls resume() — no pipeline restart.
  3. Stop finalizes the file and restores preview.

Fallback: pause capture thread, restart pipeline with recording, retries.
"""
from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("poc1.bag")


def can_record_bag(source: Any) -> bool:
    return getattr(source, "mode", None) == "hardware"


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
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in prefix) or "cam"
    return Path(out_dir).resolve() / f".pending_{safe}_slot{slot_id}.bag"


def prearm_bag_on_source(source: Any, bag_path: Path) -> None:
    """Configure a RealSense source to open with a paused .bag writer."""
    bag_path = Path(bag_path).resolve()
    bag_path.parent.mkdir(parents=True, exist_ok=True)
    if bag_path.exists():
        try:
            bag_path.unlink()
        except OSError:
            pass
    source.bag_path = bag_path
    source._bag_path = bag_path
    source._bag_final_path = None
    source.bag_start_paused = True


def start_bag_recording(source: Any, bag_path: Path) -> Path:
    """
    Ensure RealSense .bag writing is active for this take.

    Returns the active bag path (may be a pending file that will be renamed on stop).
    Raises RuntimeError if .bag cannot be armed — callers must not continue without it
    when the user checked RealSense .bag.
    """
    if not can_record_bag(source):
        raise RuntimeError(
            "RealSense .bag requires a live [realsense] hardware device "
            "(not UVC twin / simulation)."
        )
    try:
        import pyrealsense2 as rs  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "pyrealsense2 is not installed — cannot write .bag. "
            "Run: uv sync --extra realsense"
        ) from exc

    bag_path = Path(bag_path).resolve()
    bag_path.parent.mkdir(parents=True, exist_ok=True)
    wanted = _wanted_from_source(source)

    # Fast path: preview already pre-armed a paused recorder.
    recorder = getattr(source, "_rs_recorder", None)
    current = getattr(source, "_bag_path", None) or getattr(source, "bag_path", None)
    if recorder is not None and current is not None:
        try:
            source._bag_final_path = bag_path
            if hasattr(source, "resume_bag"):
                source.resume_bag()
            else:
                recorder.resume()
            logger.info(
                "RealSense .bag resumed -> writing %s (final name %s)",
                current,
                bag_path.name,
            )
            return Path(current)
        except Exception as exc:  # noqa: BLE001
            logger.warning("resume_bag failed (%s); restarting with record_to_file", exc)

    # Restart pipeline with recording to the final stamped path.
    last_exc: Optional[BaseException] = None
    for attempt in range(1, 5):
        try:
            try:
                source.stop()
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.3 * attempt)
            if bag_path.exists() and bag_path.stat().st_size == 0:
                bag_path.unlink(missing_ok=True)
            source.bag_path = bag_path
            source._bag_path = bag_path
            source._bag_final_path = bag_path
            if hasattr(source, "bag_start_paused"):
                source.bag_start_paused = False
            _restore_wanted(source, wanted)
            _call_start(source, allow_fallback=False)

            # Confirm the SDK created the bag and frames flow.
            for _ in range(5):
                try:
                    _ = source.read()
                except Exception:  # noqa: BLE001
                    pass
                if bag_path.exists() and bag_path.stat().st_size > 0:
                    break
                time.sleep(0.05)

            if not bag_path.exists():
                raise RuntimeError("SDK did not create the .bag file")

            logger.info(
                "RealSense .bag armed (try %d) -> %s (%d bytes)",
                attempt,
                bag_path,
                bag_path.stat().st_size if bag_path.exists() else 0,
            )
            return bag_path
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            logger.warning("Bag arm try %d failed: %s", attempt, exc)
            try:
                source.bag_path = None
                source._bag_path = None
                source._bag_final_path = None
                if hasattr(source, "bag_start_paused"):
                    source.bag_start_paused = False
                if bag_path.exists() and bag_path.stat().st_size == 0:
                    bag_path.unlink(missing_ok=True)
            except Exception:  # noqa: BLE001
                pass

    # Restore preview without bag, then fail hard — user asked for .bag.
    try:
        source.bag_path = None
        source._bag_path = None
        source._bag_final_path = None
        if hasattr(source, "bag_start_paused"):
            source.bag_start_paused = False
        _restore_wanted(source, wanted)
        if getattr(source, "_pipeline", None) is None:
            time.sleep(0.3)
            try:
                _call_start(source, allow_fallback=False)
            except Exception:
                _restore_wanted(source, wanted)
                _call_start(source, allow_fallback=True)
    except Exception:  # noqa: BLE001
        logger.exception("Could not restore RealSense preview after bag failure")

    raise RuntimeError(
        f"Could not start RealSense .bag recording to {bag_path}. "
        f"Last error: {last_exc}. "
        "Close Intel RealSense Viewer and other camera apps, use a USB 3 port, "
        "then Start preview again with RealSense .bag checked."
    )


def stop_bag_recording(source: Any) -> Optional[Path]:
    """
    Finalize .bag (pipeline stop closes the file), rename to final path if needed,
    then restart preview without bag.
    """
    path = getattr(source, "_bag_path", None) or getattr(source, "bag_path", None)
    final = getattr(source, "_bag_final_path", None)
    if not path or not can_record_bag(source):
        if hasattr(source, "bag_path"):
            source.bag_path = None
        if hasattr(source, "_bag_path"):
            source._bag_path = None
        if hasattr(source, "_bag_final_path"):
            source._bag_final_path = None
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

    source.bag_path = None
    source._bag_path = None
    source._bag_final_path = None
    if hasattr(source, "bag_start_paused"):
        source.bag_start_paused = False

    time.sleep(0.35)
    _restore_wanted(source, wanted)
    try:
        _call_start(source, allow_fallback=False)
    except Exception:
        try:
            _restore_wanted(source, wanted)
            _call_start(source, allow_fallback=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("RealSense preview restart after .bag failed: %s", exc)

    p = Path(path)
    out = p
    if final is not None:
        final_p = Path(final).resolve()
        if p.exists() and p.resolve() != final_p:
            try:
                final_p.parent.mkdir(parents=True, exist_ok=True)
                if final_p.exists():
                    final_p.unlink()
                shutil.move(str(p), str(final_p))
                out = final_p
            except OSError as exc:
                logger.warning("Could not rename .bag to %s: %s", final_p, exc)
                out = p
        elif final_p.exists():
            out = final_p

    if out.exists() and out.stat().st_size > 0:
        logger.info("RealSense .bag saved: %s (%d bytes)", out, out.stat().st_size)
        return out
    logger.error(
        "No .bag file produced at %s — recording may be incomplete",
        out,
    )
    return None
