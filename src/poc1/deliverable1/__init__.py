"""
Deliverable 1 (R1–R6): multi-camera list / config / stream / arm / record.

Additive package — does not modify the POC-1 single-camera pipeline modules.
Reuses Pipeline, CvCaptureSource, RealSense sources, and device helpers.
"""
from __future__ import annotations

from poc1.deliverable1.devices import (
    ConnectedCamera,
    StreamMode,
    build_frame_source,
    list_all_cameras,
    list_stream_modes,
)
from poc1.deliverable1.session import CameraSlot, MultiCamSession

__all__ = [
    "ConnectedCamera",
    "StreamMode",
    "CameraSlot",
    "MultiCamSession",
    "list_all_cameras",
    "list_stream_modes",
    "build_frame_source",
]
