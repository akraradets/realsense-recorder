"""
StreamViewer: live preview only. Reads from CameraHandler.viewer_queue,
which is drop-oldest and small on purpose -- the preview only ever needs
to show the newest frame, and must never be able to apply backpressure to
the camera_handler or the recording path. If preview rendering is slow,
frames are silently dropped here (by design), not queued up.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable, Optional

from poc1.camera_handler import DropCountingQueue, FrameEnvelope


@dataclass
class StreamViewer:
    in_queue: DropCountingQueue
    on_frame: Callable[[FrameEnvelope], None]  # e.g. push into a GUI image widget

    def __post_init__(self) -> None:
        self._thread: Optional[threading.Thread] = None
        self._running = threading.Event()

    def start(self) -> None:
        self._running.set()
        self._thread = threading.Thread(target=self._loop, name="stream-viewer", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running.clear()
        if self._thread:
            self._thread.join(timeout=5.0)

    def _loop(self) -> None:
        while self._running.is_set():
            env = self.in_queue.get(timeout=0.5)
            if env is None:
                continue
            self.on_frame(env)
