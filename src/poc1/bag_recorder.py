"""
Intel RealSense .bag recording (opens in RealSense Viewer).

When “Also save RealSense .bag” is checked, .bag is required with MP4.

Strategy (reliable on Windows):
  - Pause the capture thread
  - Restart the SDK pipeline with enable_record_to_file(final_stamped_path)
  - Verify the .bag file grows
  - On stop: pipeline.stop() finalizes the file, then restore preview without bag

No hidden “.pending_” files — the stamped camN_….bag is written directly.
"""
from __future__ import annotations

import logging
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
    """Visible temp name (kept for API compatibility; Record uses stamped path)."""
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in prefix) or "cam"
    return Path(out_dir).resolve() / f"{safe}_slot{slot_id}_recording.bag"


def prearm_bag_on_source(source: Any, bag_path: Path) -> None:
    """No-op pre-arm: bag is armed at Record to the final stamped path."""
    # Clear any previous bag state so preview stays a normal live stream.
    source.bag_path = None
    source._bag_path = None
    source._bag_final_path = None
    if hasattr(source, "bag_start_paused"):
        source.bag_start_paused = False
    if hasattr(source, "_rs_recorder"):
        source._rs_recorder = None
    _ = bag_path  # API compat


def start_bag_recording(source: Any, bag_path: Path) -> Path:
    """
    Restart RealSense with .bag writing to ``bag_path`` (required).

    Raises RuntimeError if arming fails.
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
    if bag_path.exists():
        try:
            bag_path.unlink()
        except OSError as exc:
            raise RuntimeError(
                f"Cannot overwrite existing .bag ({bag_path.name}): {exc}"
            ) from exc

    wanted = _wanted_from_source(source)
    last_exc: Optional[BaseException] = None

    for attempt in range(1, 5):
        try:
            try:
                source.stop()
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.35 * attempt)

            source.bag_path = bag_path
            source._bag_path = bag_path
            source._bag_final_path = bag_path
            if hasattr(source, "bag_start_paused"):
                source.bag_start_paused = False
            _restore_wanted(source, wanted)
            _call_start(source, allow_fallback=False)

            # Pull frames so the SDK creates / grows the bag.
            size = 0
            for _ in range(12):
                try:
                    _ = source.read()
                except Exception:  # noqa: BLE001
                    pass
                if bag_path.exists():
                    size = bag_path.stat().st_size
                    if size > 0:
                        break
                time.sleep(0.05)

            if not bag_path.exists() or bag_path.stat().st_size <= 0:
                raise RuntimeError("SDK created no .bag data after start")

            logger.info(
                "RealSense .bag armed (try %d) -> %s (%d bytes)",
                attempt,
                bag_path,
                bag_path.stat().st_size,
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

    # Restore preview without bag, then fail hard.
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
        f"Could not start RealSense .bag recording to {bag_path.name}. "
        f"Last error: {last_exc}. "
        "Close Intel RealSense Viewer, use USB 3, Start preview on RealSense, "
        "check Also save RealSense .bag, then Record again."
    )


def stop_bag_recording(source: Any) -> Optional[Path]:
    """Finalize .bag by stopping the SDK pipeline, then restore preview."""
    path = getattr(source, "_bag_path", None) or getattr(source, "bag_path", None)
    final = getattr(source, "_bag_final_path", None) or path
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
    if hasattr(source, "_rs_recorder"):
        source._rs_recorder = None

    time.sleep(0.4)
    _restore_wanted(source, wanted)
    try:
        _call_start(source, allow_fallback=False)
    except Exception:
        try:
            _restore_wanted(source, wanted)
            _call_start(source, allow_fallback=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("RealSense preview restart after .bag failed: %s", exc)

    out = Path(final) if final is not None else Path(path)
    if out.exists() and out.stat().st_size > 0:
        logger.info("RealSense .bag saved: %s (%d bytes)", out, out.stat().st_size)
        return out
    logger.error("No .bag file produced at %s", out)
    return None
