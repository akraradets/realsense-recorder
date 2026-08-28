"""
ffmpeg DirectShow capture for Elgato @1080p120 (OBS-equivalent pin).

OpenCV CAP_DSHOW often soft-opens ~60 while OBS/ffmpeg lock 120 on the same HDMI.
This module spawns ``ffmpeg -f dshow`` and reads decoded BGR24 frames from stdout.
"""
from __future__ import annotations

import logging
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("poc1.ffmpeg_dshow")

_DSHOW_PIX: dict[str, Optional[str]] = {
    "mjpg": "mjpeg",
    "mjpeg": "mjpeg",
    "yuyv": "yuyv422",
    "yuy2": "yuyv422",
    "bgr8": None,
    "rgb8": None,
}


def find_ffmpeg() -> Optional[str]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg
    if sys.platform != "win32":
        return None
    for path in (
        Path(r"C:\ffmpeg\bin\ffmpeg.exe"),
        Path(r"C:\Program Files\ffmpeg\bin\ffmpeg.exe"),
        Path.home() / "scoop" / "apps" / "ffmpeg" / "current" / "ffmpeg.exe",
        Path.home() / "AppData" / "Local" / "Microsoft" / "WinGet" / "Links" / "ffmpeg.exe",
    ):
        if path.is_file():
            return str(path)
    return None


def dshow_input_names(open_path: Optional[str]) -> list[str]:
    """Plain DirectShow device names for ffmpeg ``-i video=…``."""
    from poc1.deliverable1.win_names import elgato_open_name_paths

    names: list[str] = []

    def _add(raw: str) -> None:
        name = raw.strip()
        if name.startswith("video="):
            name = name[6:].strip()
        if name and name not in names:
            names.append(name)

    if open_path:
        _add(open_path)
    for path in elgato_open_name_paths():
        _add(path)
    return names


def _pixel_formats_to_try(requested_fmt: str) -> list[Optional[str]]:
    fmt = (requested_fmt or "mjpg").lower()
    primary = _DSHOW_PIX.get(fmt)
    order: list[Optional[str]] = []
    if primary is not None:
        order.append(primary)
    for extra in ("mjpeg", "yuyv422", None):
        if extra not in order:
            order.append(extra)
    return order


class FfmpegDshowCaptureSource:
    """Read BGR frames from ``ffmpeg -f dshow`` stdout."""

    allow_fps_remux = True

    def __init__(
        self,
        width: int,
        height: int,
        fps: int,
        *,
        device_names: list[str],
        pixel_format: str = "mjpg",
        device_tag: str = "elgato",
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self.target_fps = int(fps) if int(fps) > 0 else 30
        self.requested_fps = self.target_fps
        self.pixel_format = pixel_format
        self.device_tag = device_tag
        self.actual_fps = 0.0
        self.actual_width = self.width
        self.actual_height = self.height
        self._device_names = [n for n in device_names if n]
        self._ffmpeg: Optional[str] = find_ffmpeg()
        self._proc: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._frame_bytes = self.width * self.height * 3
        self._queue: queue.Queue[Optional[np.ndarray]] = queue.Queue(maxsize=8)
        self._pending: Optional[np.ndarray] = None
        self._active_device = ""

    def start(self) -> None:
        if sys.platform != "win32":
            raise RuntimeError("ffmpeg DirectShow capture is Windows-only")
        if not self._ffmpeg:
            raise RuntimeError(
                "ffmpeg not found on PATH — install ffmpeg so Elgato can open at 1080p120 "
                "(same pin OBS uses)."
            )
        if not self._device_names:
            raise RuntimeError("No DirectShow device name for ffmpeg Elgato capture")

        want = self.target_fps
        last_err = "no attempt"
        for device in self._device_names:
            for pix in _pixel_formats_to_try(self.pixel_format):
                try:
                    rate = self._try_open(device, pix, measure_s=2.0)
                except Exception as exc:  # noqa: BLE001
                    last_err = str(exc)
                    logger.warning(
                        "ffmpeg dshow open failed %s %dx%d@%d pix=%s: %s",
                        device,
                        self.width,
                        self.height,
                        want,
                        pix,
                        exc,
                    )
                    continue
                if rate >= want * 0.85:
                    self.actual_fps = rate
                    self._active_device = device
                    logger.info(
                        "ffmpeg dshow LOCKED %s %dx%d@%d pix=%s measured ~%.1ffps",
                        device,
                        self.width,
                        self.height,
                        want,
                        pix,
                        rate,
                    )
                    return
                self.stop()
                last_err = f"{device} pix={pix} measured ~{rate:.1f}fps"
                logger.warning(
                    "ffmpeg dshow %s %dx%d@%d pix=%s only ~%.1ffps — retrying",
                    device,
                    self.width,
                    self.height,
                    want,
                    pix,
                    rate,
                )

        raise RuntimeError(
            f"ffmpeg could not lock Elgato at {self.width}x{self.height}@{want} "
            f"(last: {last_err}). Confirm HDMI is 1080p120 and OBS is fully exited."
        )

    def _build_cmd(self, device: str, pix: Optional[str]) -> list[str]:
        assert self._ffmpeg
        cmd = [
            self._ffmpeg,
            "-hide_banner",
            "-loglevel",
            "warning",
            "-nostdin",
            "-fflags",
            "nobuffer",
            "-flags",
            "low_delay",
            "-probesize",
            "32",
            "-analyzeduration",
            "0",
            "-f",
            "dshow",
            "-rtbufsize",
            "150M",
            "-video_size",
            f"{self.width}x{self.height}",
            "-framerate",
            str(self.target_fps),
        ]
        if pix:
            cmd.extend(["-pixel_format", pix])
        cmd.extend(
            [
                "-i",
                f"video={device}",
                "-an",
                "-pix_fmt",
                "bgr24",
                "-f",
                "rawvideo",
                "pipe:1",
            ]
        )
        return cmd

    def _spawn(self, cmd: list[str]) -> subprocess.Popen:
        creationflags = 0
        if sys.platform == "win32":
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        return subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=self._frame_bytes * 4,
            creationflags=creationflags,
        )

    def _try_open(self, device: str, pix: Optional[str], measure_s: float) -> float:
        cmd = self._build_cmd(device, pix)
        logger.info("ffmpeg dshow try: %s", " ".join(cmd[1:14]))
        proc = self._spawn(cmd)
        self._proc = proc
        self._running = True
        self._thread = threading.Thread(
            target=self._reader_loop, name="ffmpeg-dshow-read", daemon=True
        )
        self._thread.start()
        # Drain stderr in background so a full pipe cannot stall ffmpeg.
        threading.Thread(
            target=self._drain_stderr, args=(proc,), name="ffmpeg-dshow-err", daemon=True
        ).start()

        n = 0
        t0 = time.perf_counter()
        first: Optional[np.ndarray] = None
        while time.perf_counter() - t0 < measure_s:
            try:
                frame = self._queue.get(timeout=0.25)
            except queue.Empty:
                if proc.poll() is not None:
                    break
                continue
            if frame is None:
                break
            n += 1
            if first is None:
                first = frame
        elapsed = max(time.perf_counter() - t0, 0.05)
        rate = (n - 1) / elapsed if n >= 2 else 0.0
        if first is not None:
            self._pending = first
        if proc.poll() is not None and n == 0:
            err = ""
            try:
                err = (proc.stderr.read() or b"").decode("utf-8", errors="replace")[:400]
            except Exception:  # noqa: BLE001
                pass
            raise RuntimeError(err or "ffmpeg exited with no frames")
        return rate

    def _drain_stderr(self, proc: subprocess.Popen) -> None:
        if proc.stderr is None:
            return
        try:
            for line in proc.stderr:
                text = line.decode("utf-8", errors="replace").strip()
                if text:
                    logger.debug("ffmpeg: %s", text)
        except Exception:  # noqa: BLE001
            pass

    def _reader_loop(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        stdout = proc.stdout
        nbytes = self._frame_bytes
        try:
            while self._running:
                buf = stdout.read(nbytes)
                if len(buf) != nbytes:
                    break
                frame = np.frombuffer(buf, dtype=np.uint8).reshape(
                    self.height, self.width, 3
                )
                owned = np.ascontiguousarray(frame)
                try:
                    self._queue.put_nowait(owned)
                except queue.Full:
                    try:
                        self._queue.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        self._queue.put_nowait(owned)
                    except queue.Full:
                        pass
        finally:
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                pass

    def read(self) -> Optional[np.ndarray]:
        if self._pending is not None:
            frame = self._pending
            self._pending = None
            return frame
        if not self._running:
            return None
        try:
            frame = self._queue.get(timeout=2.0)
        except queue.Empty:
            return None
        return frame

    def stop(self) -> None:
        self._running = False
        proc = self._proc
        self._proc = None
        if proc is not None:
            try:
                proc.terminate()
            except Exception:  # noqa: BLE001
                pass
            try:
                proc.wait(timeout=2.0)
            except Exception:  # noqa: BLE001
                try:
                    proc.kill()
                except Exception:  # noqa: BLE001
                    pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
