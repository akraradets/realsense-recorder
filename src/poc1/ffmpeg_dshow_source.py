"""
ffmpeg DirectShow capture for Elgato — OBS-parity high-rate path.

OpenCV CAP_DSHOW often soft-opens Elgato at ~60 while OBS locks 1080p120.
ffmpeg's dshow demuxer negotiates the same pin OBS uses:

    ffmpeg -f dshow -framerate 120 -video_size 1920x1080 -i video=<name> ...

Frames are read as raw BGR24 from stdout on the camera-handler thread only.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("poc1.ffmpeg_dshow")


def find_ffmpeg() -> Optional[str]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg
    candidates = [
        Path(r"C:\ffmpeg\bin\ffmpeg.exe"),
        Path(r"C:\Program Files\ffmpeg\bin\ffmpeg.exe"),
        Path.home() / "scoop" / "apps" / "ffmpeg" / "current" / "ffmpeg.exe",
        Path.home() / "AppData" / "Local" / "Microsoft" / "WinGet" / "Links" / "ffmpeg.exe",
    ]
    for path in candidates:
        if path.is_file():
            return str(path)
    return None


def dshow_device_name(open_path: Optional[str], fallback: str = "Elgato") -> str:
    """Strip ``video=`` prefix from OpenCV/ffmpeg open paths."""
    raw = (open_path or "").strip()
    if raw.lower().startswith("video="):
        return raw[6:].strip().strip('"')
    return raw.strip().strip('"') or fallback


def build_ffmpeg_dshow_cmd(
    ffmpeg: str,
    device_name: str,
    width: int,
    height: int,
    fps: int,
    *,
    pixel_format: Optional[str] = "mjpeg",
) -> list[str]:
    """Build an OBS-like dshow grab → raw BGR24 pipe command."""
    # Options that select the capture pin must appear before ``-i``.
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-fflags",
        "nobuffer",
        "-flags",
        "low_delay",
        "-f",
        "dshow",
        "-rtbufsize",
        "256M",
        "-framerate",
        str(int(fps)),
        "-video_size",
        f"{int(width)}x{int(height)}",
    ]
    if pixel_format:
        cmd.extend(["-pixel_format", str(pixel_format)])
    cmd.extend(
        [
            "-i",
            f"video={device_name}",
            "-an",
            "-sn",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-",
        ]
    )
    return cmd


class FfmpegDshowCapture:
    """
    Owns one ffmpeg dshow process and yields BGR frames.

    Used as a drop-in read() backend behind FormattedUvcSource when OpenCV
    cannot lock ≥90 fps on Elgato.
    """

    def __init__(
        self,
        device_name: str,
        width: int,
        height: int,
        fps: int,
        *,
        ffmpeg_path: Optional[str] = None,
    ) -> None:
        self.device_name = device_name
        self.width = int(width)
        self.height = int(height)
        self.target_fps = int(fps) if int(fps) > 0 else 120
        self._ffmpeg = ffmpeg_path or find_ffmpeg()
        self._proc: Optional[subprocess.Popen] = None
        self._frame_bytes = self.width * self.height * 3
        self.actual_fps: float = 0.0
        self._pixel_format_used: str = "mjpeg"
        self._read_lock = threading.Lock()

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self) -> float:
        """
        Start ffmpeg and measure delivery FPS. Raises RuntimeError on failure.
        Returns measured delivery rate.
        """
        if not self._ffmpeg:
            raise RuntimeError(
                "ffmpeg not found — install ffmpeg on PATH for Elgato 1080p120 "
                "(OBS-parity DirectShow capture)."
            )
        self.stop()
        last_err = ""
        best_soft = 0.0
        # Try MJPEG first (matches OBS Custom MJPG pin), then yuyv, then default.
        attempts: list[Optional[str]] = ["mjpeg", "yuyv422", None]
        for pix in attempts:
            cmd = build_ffmpeg_dshow_cmd(
                self._ffmpeg,
                self.device_name,
                self.width,
                self.height,
                self.target_fps,
                pixel_format=pix,
            )
            logger.info("Elgato ffmpeg/dshow try: %s", " ".join(cmd))
            try:
                self._proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    stdin=subprocess.DEVNULL,
                    bufsize=self._frame_bytes * 8,
                )
            except OSError as exc:
                last_err = str(exc)
                self._proc = None
                continue
            measured = self._measure(seconds=2.0)
            if measured >= 90.0:
                self.actual_fps = measured
                self._pixel_format_used = pix or "auto"
                logger.info(
                    "Elgato ffmpeg/dshow LOCKED ~%.1ffps at %dx%d (pix=%s) via %r",
                    measured,
                    self.width,
                    self.height,
                    self._pixel_format_used,
                    self.device_name,
                )
                return measured
            err = self._kill_and_stderr()
            last_err = err or f"measured ~{measured:.1f}fps"
            best_soft = max(best_soft, measured)
            logger.warning(
                "Elgato ffmpeg/dshow pin pix=%s delivered ~%.1f — trying next",
                pix,
                measured,
            )
        self.actual_fps = best_soft
        raise RuntimeError(
            f"ffmpeg/dshow could not lock {self.width}x{self.height}@"
            f"{self.target_fps} on {self.device_name!r} (best ~{best_soft:.1f}fps; "
            f"{last_err}). Confirm HDMI is 1080p120 and OBS has fully Exit."
        )

    def _kill_and_stderr(self) -> str:
        proc = self._proc
        self._proc = None
        if proc is None:
            return ""
        try:
            proc.kill()
        except Exception:  # noqa: BLE001
            pass
        try:
            out = proc.communicate(timeout=2.0)
            return (out[1] or b"").decode("utf-8", errors="replace")[:400]
        except Exception:  # noqa: BLE001
            return ""

    def _measure(self, seconds: float = 1.5) -> float:
        if not self.alive:
            return 0.0
        n = 0
        t0 = time.perf_counter()
        deadline = t0 + max(0.5, float(seconds))
        while time.perf_counter() < deadline:
            frame = self.read(timeout_s=1.2)
            if frame is None:
                if n == 0 and time.perf_counter() - t0 < 0.8:
                    continue
                break
            n += 1
        elapsed = time.perf_counter() - t0
        if n < 5 or elapsed < 0.2:
            return 0.0
        return (n - 1) / elapsed

    def read(self, timeout_s: float = 2.0) -> Optional[np.ndarray]:
        if self._proc is None or self._proc.stdout is None:
            return None
        if self._proc.poll() is not None:
            return None
        need = self._frame_bytes
        stdout = self._proc.stdout
        box: list[bytes] = []

        def _worker() -> None:
            try:
                data = stdout.read(need)
                if data:
                    box.append(data)
            except Exception:  # noqa: BLE001
                pass

        with self._read_lock:
            t = threading.Thread(target=_worker, name="ffmpeg-dshow-read", daemon=True)
            t.start()
            t.join(timeout=max(0.3, float(timeout_s)))
            if t.is_alive():
                return None
        if not box or len(box[0]) != need:
            return None
        return (
            np.frombuffer(box[0], dtype=np.uint8)
            .reshape((self.height, self.width, 3))
            .copy()
        )

    def stop(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None:
            return
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
            try:
                proc.wait(timeout=1.0)
            except Exception:  # noqa: BLE001
                pass
