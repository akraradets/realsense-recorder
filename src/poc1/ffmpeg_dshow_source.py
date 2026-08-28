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

# Elgato MJPEG is an input *codec* on dshow (-vcodec mjpeg), not -pixel_format mjpeg.
InputMode = str  # "vcodec_mjpeg" | "pixel_yuyv422" | "none"


def _input_modes_to_try(requested_fmt: str) -> list[InputMode]:
    fmt = (requested_fmt or "mjpg").lower()
    modes: list[InputMode] = []
    if fmt in {"mjpg", "mjpeg"}:
        modes.append("vcodec_mjpeg")
    elif fmt in {"yuyv", "yuy2"}:
        modes.append("pixel_yuyv422")
    else:
        modes.append("none")
    for extra in ("pixel_yuyv422", "none"):
        if extra not in modes:
            modes.append(extra)
    return modes


def _pin_args_for_mode(mode: InputMode) -> list[str]:
    if mode == "vcodec_mjpeg":
        return ["-vcodec", "mjpeg"]
    if mode == "pixel_yuyv422":
        return ["-pixel_format", "yuyv422"]
    return []


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


def _pixel_formats_to_try(requested_fmt: str) -> list[InputMode]:
    """Backward-compatible alias for tests."""
    return _input_modes_to_try(requested_fmt)


def _lock_threshold(requested_fps: int) -> float:
    """Minimum measured fps to treat high-rate (≥90 ask) as locked."""
    req = int(requested_fps) if int(requested_fps) > 0 else 30
    if req >= 90:
        return 90.0
    return req * 0.85


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
        self._active_input_mode = ""

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
        threshold = _lock_threshold(want)
        last_err = "no attempt"
        for device in self._device_names:
            for mode in _input_modes_to_try(self.pixel_format):
                for measure_s in (3.5, 5.0, 7.0):
                    try:
                        rate = self._try_open(device, mode, measure_s=measure_s)
                    except Exception as exc:  # noqa: BLE001
                        last_err = str(exc)
                        logger.warning(
                            "ffmpeg dshow open failed %s %dx%d@%d mode=%s: %s",
                            device,
                            self.width,
                            self.height,
                            want,
                            mode,
                            exc,
                        )
                        continue
                    if rate >= threshold:
                        self.actual_fps = rate
                        self._active_device = device
                        self._active_input_mode = mode
                        logger.info(
                            "ffmpeg dshow LOCKED %s %dx%d@%d mode=%s measured ~%.1ffps",
                            device,
                            self.width,
                            self.height,
                            want,
                            mode,
                            rate,
                        )
                        return
                    self.stop()
                    last_err = f"{device} mode={mode} measured ~{rate:.1f}fps"
                    logger.warning(
                        "ffmpeg dshow %s %dx%d@%d mode=%s only ~%.1ffps (need %.0f) — retrying",
                        device,
                        self.width,
                        self.height,
                        want,
                        mode,
                        rate,
                        threshold,
                    )

        raise RuntimeError(
            f"ffmpeg could not lock Elgato at {self.width}x{self.height}@{want} "
            f"(last: {last_err}). Confirm HDMI is 1080p120 and OBS is fully exited."
        )

    def _build_cmd(self, device: str, input_mode: InputMode) -> list[str]:
        """OBS-style dshow pin — framerate before size, vcodec mjpeg for Elgato."""
        return self._build_cmd_variant(device, input_mode, fps_first=True)

    def _build_cmd_variant(
        self, device: str, input_mode: InputMode, *, fps_first: bool
    ) -> list[str]:
        assert self._ffmpeg
        head = [
            self._ffmpeg,
            "-hide_banner",
            "-loglevel",
            "warning",
            "-nostdin",
            "-f",
            "dshow",
            "-rtbufsize",
            "150M",
        ]
        size_args = ["-video_size", f"{self.width}x{self.height}"]
        fps_args = ["-framerate", str(self.target_fps)]
        fmt_args = _pin_args_for_mode(input_mode)
        if fps_first:
            pin = fps_args + size_args + fmt_args
        else:
            pin = size_args + fps_args + fmt_args
        return head + pin + [
            "-i",
            f"video={device}",
            "-an",
            "-pix_fmt",
            "bgr24",
            "-f",
            "rawvideo",
            "pipe:1",
        ]

    def _cmd_variants(self, device: str, input_mode: InputMode) -> list[list[str]]:
        return [
            self._build_cmd_variant(device, input_mode, fps_first=True),
            self._build_cmd_variant(device, input_mode, fps_first=False),
        ]

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

    def _try_open(self, device: str, input_mode: InputMode, measure_s: float) -> float:
        last_rate = 0.0
        for cmd in self._cmd_variants(device, input_mode):
            rate = self._try_open_cmd(cmd, input_mode, measure_s)
            if rate > last_rate:
                last_rate = rate
            if rate >= _lock_threshold(self.target_fps):
                return rate
            self.stop()
        return last_rate

    def _try_open_cmd(
        self, cmd: list[str], input_mode: InputMode, measure_s: float
    ) -> float:
        logger.info("ffmpeg dshow try: %s", " ".join(cmd[1:18]))
        proc = self._spawn(cmd)
        self._proc = proc
        self._running = True
        self._thread = threading.Thread(
            target=self._reader_loop, name="ffmpeg-dshow-read", daemon=True
        )
        self._thread.start()
        threading.Thread(
            target=self._drain_stderr, args=(proc,), name="ffmpeg-dshow-err", daemon=True
        ).start()

        # Warmup — DirectShow can deliver a burst then settle at true rate.
        warmup_end = time.perf_counter() + 0.6
        while time.perf_counter() < warmup_end:
            try:
                frame = self._queue.get(timeout=0.15)
            except queue.Empty:
                if proc.poll() is not None:
                    break
                continue
            if frame is None:
                break
            self._pending = frame

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
                self._pending = frame
        elapsed = max(time.perf_counter() - t0, 0.05)
        rate = (n - 1) / elapsed if n >= 2 else (n / elapsed if n >= 1 and elapsed > 0.2 else 0.0)
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
