"""
Optional Intel RealSense .bag recording (opens in RealSense Viewer).

When a RealSense device is connected, recording can restart the SDK pipeline
with ``config.enable_record_to_file(...)`` so the resulting ``.bag`` opens in
Intel RealSense Viewer.

If no device / simulation / SDK missing: skip .bag, keep the MP4 proof path.
POC-1 does not fail when hardware is absent — but Deliverable 1 raises when
the user explicitly checked “Also save RealSense .bag”.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("poc1.bag")


def can_record_bag(source: Any) -> bool:
    return getattr(source, "mode", None) == "hardware"


def start_bag_recording(source: Any, bag_path: Path) -> bool:
    """
    Restart a hardware RealSense pipeline with .bag recording enabled.
    Call before enable_recording so color/depth frames still feed the MP4 path.

    Returns True only when the bag file path is armed on a live hardware stream.
    """
    if not can_record_bag(source):
        logger.warning(
            "Skipping .bag (source mode=%r is not hardware RealSense).",
            getattr(source, "mode", None),
        )
        return False
    try:
        import pyrealsense2 as rs  # noqa: F401
    except ImportError:
        logger.warning("pyrealsense2 is not installed — cannot write .bag")
        return False

    bag_path = Path(bag_path)
    bag_path.parent.mkdir(parents=True, exist_ok=True)
    # Remove a stale zero-byte file from a previous failed attempt.
    if bag_path.exists() and bag_path.stat().st_size == 0:
        bag_path.unlink(missing_ok=True)

    try:
        source.stop()
        source.bag_path = bag_path
        source._bag_path = bag_path
        source.start()
        # RealSense usually creates the bag immediately when recording starts.
        if not bag_path.exists():
            # Some SDK builds create the file slightly later — touch a marker by
            # reading one frame so the recorder is active.
            try:
                _ = source.read()
            except Exception:  # noqa: BLE001
                pass
        logger.info(
            "RealSense .bag recording armed -> %s (exists=%s size=%s)",
            bag_path,
            bag_path.exists(),
            bag_path.stat().st_size if bag_path.exists() else 0,
        )
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not arm .bag recording: %s (MP4 may continue)", exc)
        try:
            source.bag_path = None
            source._bag_path = None
            if getattr(source, "_pipeline", None) is None:
                source.start()
        except Exception:  # noqa: BLE001
            pass
        return False


def stop_bag_recording(source: Any) -> Optional[Path]:
    """
    Finalize .bag by stopping the SDK pipeline, then restart preview without bag.
    """
    path = getattr(source, "_bag_path", None) or getattr(source, "bag_path", None)
    if not path or not can_record_bag(source):
        if hasattr(source, "bag_path"):
            source.bag_path = None
        if hasattr(source, "_bag_path"):
            source._bag_path = None
        return None

    try:
        source.stop()
    except Exception:  # noqa: BLE001
        pass
    source.bag_path = None
    source._bag_path = None
    try:
        source.start()
    except Exception as exc:  # noqa: BLE001
        logger.warning("RealSense preview restart after .bag failed: %s", exc)

    p = Path(path)
    if p.exists() and p.stat().st_size > 0:
        logger.info("RealSense .bag saved: %s (%d bytes)", p, p.stat().st_size)
        return p
    logger.warning(
        "No .bag file produced at %s — check USB3 / close RealSense Viewer / device support",
        p,
    )
    return None
