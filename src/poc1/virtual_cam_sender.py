"""
Run this as its own process BEFORE starting the main recorder app.

It stands in for a physical camera: synthetic frames -> pyvirtualcam ->
a real OS-level virtual webcam device (on Windows this installs/uses the
OBS or Unity Capture virtual-cam backend under the hood).

Your recording pipeline then opens this exactly the way it will later
open the Elgato card or RealSense-as-webcam: via cv2.VideoCapture(index,
cv2.CAP_DSHOW) on Windows. This exercises the *real* OS capture path,
not just an in-process queue, while still giving you full programmatic
control over frame content/rate/counter for verification.

Usage:
    uv run python -m poc1.virtual_cam_sender --fps 120 --width 1920 --height 1080
"""
from __future__ import annotations

import argparse

import pyvirtualcam

from poc1.frame_source import FakeFrameSource


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=120)
    parser.add_argument("--pattern", choices=["bars", "noise"], default="bars")
    args = parser.parse_args()

    source = FakeFrameSource(
        width=args.width, height=args.height, target_fps=args.fps, pattern=args.pattern
    )
    source.start()

    # pyvirtualcam picks a backend automatically (OBS/Unity Capture on
    # Windows, v4l2loopback on Linux). If it can't negotiate the requested
    # fps, it will report the fps it actually granted -- check this printout,
    # since some Windows backends silently cap below 120.
    try:
        cam_ctx = pyvirtualcam.Camera(width=args.width, height=args.height, fps=args.fps)
    except RuntimeError as exc:
        raise SystemExit(
            f"Virtual camera failed: {exc}\n"
            "Install OBS Studio and start its Virtual Camera once, "
            "or install Unity Capture. For R7 throughput proof without a "
            "virtual device, use: uv run python -m poc1.proof"
        ) from exc

    with cam_ctx as cam:
        print(f"[virtual_cam_sender] backend={cam.backend} granted_fps={cam.fps}")
        if cam.fps < args.fps:
            print(
                f"[virtual_cam_sender] WARNING: requested {args.fps}fps but backend "
                f"only granted {cam.fps}fps -- this caps what the pipeline can prove "
                f"through this path, test throughput separately (see monitor.py note)."
            )
        try:
            while True:
                frame = source.read()
                if frame is None:
                    break
                # pyvirtualcam expects RGB; our frames are BGR (OpenCV convention)
                cam.send(frame[:, :, ::-1])
                cam.sleep_until_next_frame()
        except KeyboardInterrupt:
            pass
        finally:
            source.stop()


if __name__ == "__main__":
    main()
