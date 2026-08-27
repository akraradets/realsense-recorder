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
        self.maxsize = maxsize
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

    def put_live(self, item: Any) -> None:
        """
        Keep capture realtime unless the queue is nearly full.

        Near overflow we wait instead of dropping (no-drop). Otherwise put_nowait
        so live preview does not wait on encode.
        """
        if self._drop_oldest:
            self.put(item)
            return
        high = max(1, int(self.maxsize * 0.90)) if self.maxsize > 0 else 0
        if self.maxsize > 0 and self.qsize() >= high:
            self.put_block(item, timeout=2.0)
        else:
            self.put(item)

    def put_block(self, item: Any, timeout: float = 2.0) -> bool:
        """
        Block until there is room (no-drop path for synthetic R7 proofs).

        Returns False if the timeout expired (counted as a drop).
        """
        try:
            self._q.put(item, timeout=timeout)
            return True
        except queue.Full:
            with self._lock:
                self.dropped_count += 1
            logger.warning(
                "%s: blocking put timed out, dropping seq=%s (total dropped=%d)",
                self._name, getattr(item, "seq", "?"), self.dropped_count,
            )
            return False


    def get(self, timeout: Optional[float] = None) -> Optional[Any]:
        try:
            if timeout == 0:
                return self._q.get_nowait()
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

    def discard_all(self) -> int:
        """Remove pending items without resetting drop counters. Returns count."""
        n = 0
        while True:
            try:
                self._q.get_nowait()
                n += 1
            except queue.Empty:
                break
        return n


@dataclass
class CameraHandler:
    source: FrameSource
    viewer_queue_size: int = 2          # small: only latest few matter
    processor_queue_size: int = 512     # generous buffer against jitter at 120fps

    def __post_init__(self) -> None:
        # FHD@120 needs a deeper buffer; fake proofs prefer no-drop backpressure.
        fps = int(getattr(self.source, "target_fps", 30) or 30)
        mode = str(getattr(self.source, "mode", "") or "")
        if mode == "fake" or fps >= 60:
            # ~30s at 120fps plus margin so Stop can drain without mid-take drops.
            self.processor_queue_size = max(self.processor_queue_size, 4800)
        self.viewer_queue = DropCountingQueue(
            self.viewer_queue_size, drop_oldest=True, name="viewer"
        )
        self.processor_queue = DropCountingQueue(
            self.processor_queue_size, drop_oldest=False, name="processor_in"
        )
        self._thread: Optional[threading.Thread] = None
        self._running = threading.Event()
        self._recording = threading.Event()
        self._paused = threading.Event()  # clear = reading; set = pause reads
        self._in_read = threading.Event()
        self._record_lock = threading.Lock()
        self._seq = 0
        self.frames_read = 0
        self.frames_to_recorder = 0
        # Do not block capture on encode — that made live Record look slow-mo.
        # No-drop is a deep queue + drain on Stop (processor.stop).
        self._block_processor_puts = False
        # Rolling capture timestamps so Elgato actual_fps tracks live HDMI rate
        # (open-time sample alone can under-count while DirectShow settles).
        self._delivery_ts: list[float] = []

    def start(self) -> None:
        self.source.start()
        self._paused.clear()
        self._running.set()
        self._thread = threading.Thread(target=self._loop, name="camera-handler", daemon=True)
        self._thread.start()

    def pause_reads(self, timeout: float = 2.0) -> None:
        """
        Stop calling source.read() so the device can be safely restarted
        (e.g. RealSense enable_record_to_file) without racing the capture thread.
        """
        self._paused.set()
        deadline = time.time() + timeout
        while self._in_read.is_set() and time.time() < deadline:
            time.sleep(0.005)

    def resume_reads(self) -> None:
        self._paused.clear()

    def enable_recording(self) -> None:
        """Begin fanning frames into the processor/recorder path."""
        with self._record_lock:
            self.processor_queue.reset_dropped()
            self.frames_to_recorder = 0
            self._seq = 0
            self.frames_read = 0
            self._delivery_ts = []
            # Clear stale open-time / preview FPS so HUD cannot look "live" at
            # ~59 while Record is actually starved (v26 false confidence).
            try:
                self.source.actual_fps = 0.0
            except Exception:  # noqa: BLE001
                pass
            reset = getattr(self.source, "reset_sequence", None)
            if callable(reset):
                reset()
            self._recording.set()

    def disable_recording(self) -> None:
        with self._record_lock:
            self._recording.clear()

    def wait_recorded_frames(self, timeout_s: float = 2.5) -> bool:
        """True if at least one Record-path frame arrived within timeout."""
        deadline = time.time() + max(0.05, float(timeout_s))
        while time.time() < deadline:
            if int(self.frames_read) > 0:
                return True
            time.sleep(0.02)
        return int(self.frames_read) > 0

    def recover_capture(self) -> None:
        """
        Unstick a dead DirectShow/OpenCV read loop (common on Elgato Record).

        If the capture thread is wedged inside ``source.read()``, abandon it and
        reopen the device on a fresh thread. Prefer this over a silent 0-frame take.
        """
        with self._record_lock:
            self._recording.clear()
        self._paused.clear()
        self._running.clear()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
            if self._thread.is_alive():
                logger.warning(
                    "capture thread stuck in source.read() — abandoning thread and reopening"
                )
            self._thread = None
        try:
            self.source.stop()
        except Exception:  # noqa: BLE001
            logger.exception("recover_capture: source.stop failed")
        try:
            self.source.start()
        except Exception:  # noqa: BLE001
            logger.exception("recover_capture: source.start failed")
            return
        self._delivery_ts = []
        self._running.set()
        self._thread = threading.Thread(
            target=self._loop, name="camera-handler", daemon=True
        )
        self._thread.start()
        # Prove the new thread can deliver before Record arms again.
        deadline = time.time() + 2.0
        while time.time() < deadline:
            if self._delivery_ts:
                return
            time.sleep(0.02)

    @property
    def is_recording(self) -> bool:
        return self._recording.is_set()

    def live_delivery_fps(self) -> float:
        """Recent source.read() rate (true capture, not UI paint)."""
        stamps = list(self._delivery_ts)
        if len(stamps) < 15:
            return 0.0
        elapsed = stamps[-1] - stamps[0]
        if elapsed <= 0.1:
            return 0.0
        return (len(stamps) - 1) / elapsed

    def stop(self) -> None:
        with self._record_lock:
            self._recording.clear()
        self._paused.clear()
        self._running.clear()
        if self._thread:
            self._thread.join(timeout=5.0)
        self.source.stop()

    def _note_delivery(self) -> None:
        now = time.time()
        self._delivery_ts.append(now)
        if len(self._delivery_ts) > 180:
            self._delivery_ts = self._delivery_ts[-180:]
        live = self.live_delivery_fps()
        if live < 5:
            return
        try:
            self.source.actual_fps = live
        except Exception:  # noqa: BLE001
            pass

    def _loop(self) -> None:
        while self._running.is_set():
            if self._paused.is_set():
                time.sleep(0.005)
                continue

            self._in_read.set()
            try:
                frame = self.source.read()
            finally:
                self._in_read.clear()
            if frame is None:
                if not self._running.is_set():
                    break
                time.sleep(0.001)
                continue

            self._note_delivery()

            # Own the buffer when the source reuses memory (OpenCV/RealSense).
            # FakeFrameSource already returns a unique copy — don't copy twice
            # (FHD@120 cannot afford it).
            if frame.flags.owndata and frame.flags.c_contiguous:
                owned = frame
            else:
                owned = np.array(frame, copy=True, order="C")

            # Snapshot arming under the lock, then release BEFORE any queue put.
            # Holding _record_lock across put_live (up to 2s) stalled Elgato reads
            # → FPS collapse (~11) and "0 frames on Record" even with live preview.
            with self._record_lock:
                recording_now = self._recording.is_set()
                if recording_now:
                    seq = self._seq
                    self._seq += 1
                    self.frames_read += 1
                    self.frames_to_recorder += 1
                    embed_seq_barcode(owned, seq)
                    env = FrameEnvelope(seq=seq, capture_ts=time.time(), frame=owned)
                else:
                    env = FrameEnvelope(seq=-1, capture_ts=time.time(), frame=owned)

            # Preview always; never block the capture thread on encode backlog.
            self.viewer_queue.put(env)
            if recording_now:
                if self._block_processor_puts:
                    self.processor_queue.put_live(env)
                else:
                    # Non-blocking: drop+count if full (better than stalling USB/DSHOW).
                    self.processor_queue.put(env)
