"""ROS2 color bag for Elgato/UVC (not Intel RealSense SDK .bag)."""
from __future__ import annotations

import logging
import queue
import shutil
import threading
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger("poc1.uvc_rosbag")


class UvcRos2Bag:
    """
    Write JPEG CompressedImage messages to a ROS2 sqlite bag folder.

    SQLite must stay on one thread: the worker opens the Writer. JPEG can be
    encoded on the processor thread so the bag queue stays small.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._q: queue.Queue[Optional[tuple[bytes, int]]] = queue.Queue(maxsize=256)
        self._thread: Optional[threading.Thread] = None
        self._ready = threading.Event()
        self._error: Optional[str] = None
        self.frames_written = 0
        self.dropped = 0

    def start(self) -> Path:
        if self.path.exists():
            shutil.rmtree(self.path, ignore_errors=True)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._ready.clear()
        self._thread = threading.Thread(target=self._loop, name="uvc-rosbag", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=8.0):
            raise RuntimeError(self._error or "ROS2 bag worker did not start")
        if self._error:
            raise RuntimeError(self._error)
        logger.info("Elgato/UVC ROS2 bag -> %s", self.path)
        return self.path

    def submit(self, bgr: np.ndarray, capture_ts: float) -> None:
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            return
        ns = int(capture_ts * 1e9)
        try:
            self._q.put_nowait((bytes(buf), ns))
        except queue.Full:
            self.dropped += 1

    def stop(self) -> Optional[Path]:
        self._q.put(None)
        if self._thread:
            self._thread.join(timeout=30.0)
            self._thread = None
        db3 = None
        if self.path.is_dir():
            files = list(self.path.glob("*.db3"))
            if files:
                db3 = files[0]
        logger.info(
            "UVC ROS2 bag stop: frames=%d dropped=%d file=%s",
            self.frames_written,
            self.dropped,
            db3,
        )
        return db3

    def _loop(self) -> None:
        writer = None
        try:
            from rosbags.rosbag2 import Writer
            from rosbags.typesys import Stores, get_typestore

            typestore = get_typestore(Stores.LATEST)
            Compressed = typestore.types["sensor_msgs/msg/CompressedImage"]
            Header = typestore.types["std_msgs/msg/Header"]
            Time = typestore.types["builtin_interfaces/msg/Time"]
            writer = Writer(self.path, version=Writer.VERSION_LATEST)
            writer.open()
            conn = writer.add_connection(
                "/camera/color/image_raw/compressed",
                Compressed.__msgtype__,
                typestore=typestore,
            )
            self._ready.set()
            while True:
                item = self._q.get()
                if item is None:
                    break
                jpeg, ns = item
                sec = ns // 1_000_000_000
                nsec = ns % 1_000_000_000
                msg = Compressed(
                    header=Header(
                        stamp=Time(sec=int(sec), nanosec=int(nsec)),
                        frame_id="elgato",
                    ),
                    format="jpeg",
                    data=np.frombuffer(jpeg, dtype=np.uint8),
                )
                cdr = typestore.serialize_cdr(msg, Compressed.__msgtype__)
                writer.write(conn, ns, cdr)
                self.frames_written += 1
        except Exception as exc:  # noqa: BLE001
            self._error = str(exc)
            logger.exception("ROS2 bag worker failed")
            self._ready.set()
        finally:
            if writer is not None:
                try:
                    writer.close()
                except Exception:  # noqa: BLE001
                    logger.exception("ROS2 bag close failed")
