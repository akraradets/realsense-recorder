"""
CameraHandler owns the ONE synchronous read loop against the camera device.

Why synchronous/single-threaded at this layer: cv2.VideoCapture (and the
underlying OS capture APIs) are not safe to call concurrently from multiple
threads. So there is exactly one thread that calls source.read(), and it
must never block on anything downstream -- if it did, frames would back up
at the hardware/driver buffer instead of in our own queues, and we'd lose
control over *where* drops happen and whether we can measure them.

Fan-out matches the required pipeline shape:

    camera_handler --sub--> stream_viewer      (best-effort, may drop-oldest)
    camera_handler --sub--> processor(compression) --> recorder  (must not
                                                          silently drop --
                                                          logged if it does)

Each subscriber gets its own bounded queue so a slow viewer can never stall
the recorder path, and vice versa.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from poc1.frame_source import FrameSource, embed_seq_barcode

logger = logging.getLogger("poc1.camera_handler")


@dataclass
class FrameEnvelope:
    """What actually travels through the queues -- frame + metadata needed
    to detect drops/gaps later, independent of the FakeFrameSource overlay."""
    seq: int
    capture_ts: float
    frame: np.ndarray


class DropCountingQueue:
    """
    A bounded queue with an explicit, measurable drop policy instead of an
    implicit one. Two modes:

    - drop_oldest=True  (stream_viewer): if full, evict the oldest item to
      make room. Live preview only cares about the newest frame; this is
      exactly "reduce frame in order to optimize newest frame" from your
      notes.
    - drop_oldest=False (recorder path): if full, the *new* frame is dropped
      and counted/logged instead. This is the "cannot write and discard
      frames again, cannot retake data back" case -- we never overwrite
      what's already queued for disk, we surface that we're falling behind.
    """

    def __init__(self, maxsize: int, drop_oldest: bool, name: str):
        self._q: "queue.Queue[Any]" = queue.Queue(maxsize=maxsize)
        self._drop_oldest = drop_oldest
        self._name = name
        self.dropped_count = 0
        self._lock = threading.Lock()

    def put(self, item: Any) -> None:
        if self._drop_oldest:
            try:
                self._q.put_nowait(item)
            except queue.Full:
                try:
                    self._q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._q.put_nowait(item)
                except queue.Full:
                    with self._lock:
                        self.dropped_count += 1
        else:
            try:
                self._q.put_nowait(item)
            except queue.Full:
                with self._lock:
                    self.dropped_count += 1
                logger.warning(
                    "%s: queue full, dropping seq=%s (total dropped=%d)",
                    self._name, getattr(item, "seq", "?"), self.dropped_count,
                )

    def get(self, timeout: Optional[float] = None) -> Optional[Any]:
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def qsize(self) -> int:
        return self._q.qsize()

    def reset_dropped(self) -> None:
        with self._lock:
            self.dropped_count = 0

    def clear(self) -> None:
        while True:
            try:
                self._q.get_nowait()
            except queue.Empty:
                break
        self.reset_dropped()


@dataclass
class CameraHandler:
    source: FrameSource
    viewer_queue_size: int = 4          # small: only latest few matter
    processor_queue_size: int = 512     # generous buffer against jitter at 120fps

    def __post_init__(self) -> None:
        self.viewer_queue = DropCountingQueue(
            self.viewer_queue_size, drop_oldest=True, name="viewer"
        )
        self.processor_queue = DropCountingQueue(
            self.processor_queue_size, drop_oldest=False, name="processor_in"
        )
        self._thread: Optional[threading.Thread] = None
        self._running = threading.Event()
        self._recording = threading.Event()
        self._record_lock = threading.Lock()
        self._seq = 0
        self.frames_read = 0
        self.frames_to_recorder = 0

    def start(self) -> None:
        self.source.start()
        self._running.set()
        self._thread = threading.Thread(target=self._loop, name="camera-handler", daemon=True)
        self._thread.start()

    def enable_recording(self) -> None:
        """Begin fanning frames into the processor/recorder path."""
        with self._record_lock:
            self.processor_queue.reset_dropped()
            self.frames_to_recorder = 0
            self._seq = 0
            self.frames_read = 0
            reset = getattr(self.source, "reset_sequence", None)
            if callable(reset):
                reset()
            self._recording.set()

    def disable_recording(self) -> None:
        with self._record_lock:
            self._recording.clear()

    @property
    def is_recording(self) -> bool:
        return self._recording.is_set()

    def stop(self) -> None:
        with self._record_lock:
            self._recording.clear()
        self._running.clear()
        if self._thread:
            self._thread.join(timeout=5.0)
        self.source.stop()

    def _loop(self) -> None:
        while self._running.is_set():
            # Snapshot arming state before read so a mid-read enable_recording()
            # cannot attach a pre-reset synthetic barcode to seq 0.
            with self._record_lock:
                was_recording = self._recording.is_set()

            frame = self.source.read()
            if frame is None:
                if not self._running.is_set():
                    break
                time.sleep(0.001)
                continue

            # Own the buffer when the source reuses memory (OpenCV/RealSense).
            # FakeFrameSource already returns a unique copy — don't copy twice
            # (FHD@120 cannot afford it).
            if frame.flags.owndata and frame.flags.c_contiguous:
                owned = frame
            else:
                owned = np.array(frame, copy=True, order="C")

            with self._record_lock:
                recording_now = self._recording.is_set()
                if recording_now and was_recording:
                    seq = self._seq
                    self._seq += 1
                    self.frames_read += 1
                    self.frames_to_recorder += 1
                    # Stamp camera seq so verifier matches drop accounting even if
                    # the source's own overlay raced with reset_sequence().
                    embed_seq_barcode(owned, seq)
                    env = FrameEnvelope(seq=seq, capture_ts=time.time(), frame=owned)
                    self.viewer_queue.put(env)
                    self.processor_queue.put(env)
                else:
                    env = FrameEnvelope(seq=-1, capture_ts=time.time(), frame=owned)
                    self.viewer_queue.put(env)
