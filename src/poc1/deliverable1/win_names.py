"""
Windows-friendly capture device names for Deliverable 1 (R1).

OpenCV only exposes numeric indices. This helper maps those indices to the
names Windows/DirectShow advertises (Elgato, OBS, Integrated Camera, …).
On Windows, OpenCV can also open by name: VideoCapture("video=<name>", CAP_DSHOW).
"""
from __future__ import annotations

import logging
import re
import subprocess
import sys
from functools import lru_cache

logger = logging.getLogger("poc1.d1.win_names")


def classify_capture_name(name: str) -> str:
    """Public wrapper for device class tags."""
    return _classify(name)


def _classify(name: str) -> str:
    low = name.lower()
    if any(
        k in low
        for k in (
            "elgato",
            "game capture",
            "4k capture",
            "hd60",
            "cam link",
            "capture card",
            "hd60s",
            "4k60",
        )
    ):
        return "elgato"
    if "realsense" in low or "intel(r) realsense" in low:
        return "realsense-uvc"
    if any(
        k in low
        for k in ("obs", "virtual", "manycam", "snap camera", "unity video", "ndi")
    ):
        return "virtual"
    return "uvc"


def _names_from_ffmpeg() -> list[str]:
    names: list[str] = []
    try:
        proc = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-list_devices",
                "true",
                "-f",
                "dshow",
                "-i",
                "dummy",
            ],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
        blob = (proc.stderr or "") + (proc.stdout or "")
        # Lines like: [dshow @ ...] "Elgato HD60 S+" (video)
        for match in re.finditer(
            r"\"([^\"]+)\"\s*\(video\)", blob, flags=re.IGNORECASE
        ):
            name = match.group(1).strip()
            if name and name not in names:
                names.append(name)
    except Exception as exc:  # noqa: BLE001
        logger.debug("ffmpeg device list unavailable: %s", exc)
    return names


def _names_from_pnp() -> list[str]:
    names: list[str] = []
    if sys.platform != "win32":
        return names
    # Camera + Image + MEDIA cover webcams, RealSense UVC twins, and many
    # capture cards that do not register under Class=Camera.
    ps = r"""
$ErrorActionPreference = 'SilentlyContinue'
$classes = @('Camera','Image','MEDIA','USB')
$seen = @{}
foreach ($c in $classes) {
  Get-PnpDevice -Status OK -Class $c | ForEach-Object {
    $n = $_.FriendlyName
    if (-not $n) { return }
    $low = $n.ToLower()
    $keep = $c -in @('Camera','Image') -or
            $low -match 'camera|webcam|capture|elgato|realsense|cam link|hd60|obs|virtual'
    if ($keep -and -not $seen.ContainsKey($n)) {
      $seen[$n] = $true
      $n
    }
  }
}
"""
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps],
            capture_output=True,
            text=True,
            timeout=12,
            check=False,
        )
        for line in (proc.stdout or "").splitlines():
            name = line.strip()
            if name and name not in names:
                names.append(name)
    except Exception as exc:  # noqa: BLE001
        logger.debug("PnP camera list unavailable: %s", exc)
    return names


@lru_cache(maxsize=1)
def _windows_capture_inventory() -> tuple[tuple[str, ...], bool]:
    """
    Return (names, index_aligned).

    index_aligned=True only for ffmpeg DirectShow order (matches OpenCV CAP_DSHOW).
    PnP names are useful for labels but must NOT drive open-by-index paths.
    """
    if sys.platform != "win32":
        return tuple(), False

    names = _names_from_ffmpeg()
    if names:
        logger.info("Windows capture names via ffmpeg (index-aligned): %s", names)
        return tuple(names), True

    names = _names_from_pnp()
    if names:
        logger.info("Windows capture names via PnP (label-only): %s", names)
    return tuple(names), False


def list_windows_capture_names() -> list[str]:
    names, _aligned = _windows_capture_inventory()
    return list(names)


def names_are_index_aligned() -> bool:
    _names, aligned = _windows_capture_inventory()
    return aligned


def friendly_name_for_index(index: int, fallback: str) -> tuple[str, str]:
    """
    Return (display_name, class_tag) for an OpenCV device index.

    class_tag in {elgato, realsense-uvc, virtual, uvc}
    """
    names, aligned = _windows_capture_inventory()
    if aligned and 0 <= index < len(names):
        name = names[index]
        return name, _classify(name)
    if not aligned and 0 <= index < len(names):
        # PnP order is not reliable for indices — still use as a soft label.
        name = names[index]
        return name, _classify(name)
    # Fuzzy: if only one Elgato exists and index>0, still try known names.
    for name in names:
        tag = _classify(name)
        if tag == "elgato" and index > 0:
            return f"{name} (likely index {index})", tag
    return fallback, "uvc"


def dshow_open_path(index: int) -> str | None:
    """DirectShow path OpenCV understands: video=<FriendlyName> (index-aligned only)."""
    names, aligned = _windows_capture_inventory()
    if not aligned:
        return None
    if 0 <= index < len(names):
        return f"video={names[index]}"
    return None


def dshow_open_paths_for_tag(tag: str) -> list[str]:
    """All DirectShow open paths whose friendly name classifies as tag."""
    names, _aligned = _windows_capture_inventory()
    out: list[str] = []
    for name in names:
        if _classify(name) == tag:
            out.append(f"video={name}")
    return out


def clear_name_cache() -> None:
    _windows_capture_inventory.cache_clear()
