# POC1 Recorder

Proves **R7** (FHD / 120fps recording with **no frame drops**) using software
sources (FakeFrameSource), optional **Elgato / capture card**, laptop webcam,
pyvirtualcam, and Intel RealSense.

## Pipeline

```
camera_handler --sub--> stream_viewer            (drop-oldest, preview only)
camera_handler --sub--> processor (compression) --> recorder
```

- **processor** owns encode (H.264 when OpenH264 is available, else **mp4v**).
- **recorder** accounts encoded frames / remuxes webcam FPS metadata; it does
  not re-encode.

Pass/fail on stop:

- `frames_read == frames_written`
- empty `sequence_gaps`
- zero drops on processor/recorder queues
- `compression_stage == "processor"`
- Fake: contiguous 32-bit barcode in every decoded MP4 frame
- CPU / RAM / disk CSV (`*_sysmon.csv`)
- JSON report (`*.report.json`)

## Setup

```bash
uv sync --extra dev

# Optional — RealSense device
uv sync --extra dev --extra realsense
```

## Prove R7 (exact command)

```bash
uv run python -m poc1.proof --seconds 5 --width 1920 --height 1080 --fps 120
```

Expect console `PASSED` and `no_frame_drops: true` in the report.

### Client proof pack (artifacts for handoff)

```bash
uv run python -m poc1.delivery
# or: uv run poc1-delivery
```

Copies video / report / sysmon / scorecard into `recordings/poc1_delivery/<stamp>/`
and writes `PASS_SUMMARY.txt` (+ `recordings/poc1_delivery/LATEST.txt`).

## Prove all sources (scorecard)

```bash
uv run python -m poc1.run_poc1_tests --seconds 3
```

Runs **5 tests**: fake / **capturecard (Elgato)** / realsense / webcam / virtualcam.

Expect `5/5 passed`, or `N/5 passed, M skipped` with clear skip reasons when
hardware (Elgato, OBS virtual cam) is absent.

```bash
# Force Elgato device index if auto-detect picks the wrong camera:
uv run python -m poc1.run_poc1_tests --seconds 3 --capture-index 1
# or: set POC1_CAPTURE_INDEX=1
```

### Elgato / capture card tips

1. Plug in the card; **close** Elgato Game Capture / 4K Capture (they lock the device).
2. Connect an HDMI source at **1080p120** if you want to claim true FHD@120 I/O.
3. GUI: `--source capturecard` (alias `--source elgato`) defaults to 1920×1080@120.
4. Use **Detect Capture Card** in the GUI or `--capture-index N` if the laptop webcam is index 0.

## GUI

```bash
# Synthetic FHD@120 (primary R7 throughput proof)
uv run python -m poc1.gui --source fake --width 1920 --height 1080 --fps 120

# Elgato / capture card (FHD@120 defaults)
uv run python -m poc1.gui --source capturecard --device-index 1

# RealSense (practical daily use)
uv run python -m poc1.gui --source realsense --width 1280 --height 720 --fps 30

# Laptop webcam
uv run python -m poc1.gui --source webcam --device-index 0 --width 1280 --height 720 --fps 30
```

**Start Preview** → **Armed** → **Record** → **Stop**.

- **Also .bag (RealSense HW)** — Viewer-compatible `.bag` when a RealSense is connected.
- **Detect Capture Card** / **Detect RealSense** — set device index automatically.

### Virtual camera (two terminals)

```bash
uv run python -m poc1.virtual_cam_sender --width 1920 --height 1080 --fps 120
uv run python -m poc1.gui --source virtualcam --device-index 1 --width 1920 --height 1080 --fps 120
```

### Codec note (H.264)

Without Cisco’s `openh264-2.5.0-win64.dll` on PATH, the **processor** uses **MPEG-4 (mp4v)**.

```bash
uv run python -m poc1.verifier recordings/proof/some.report.json
uv run python -m poc1.verifier path/to/file.mp4
uv run python -m poc1.verifier path/to/webcam.mp4 --no-barcode
```

## Tests

```bash
uv run pytest -q
```

## What each source proves

| Source | No-drop pipeline | OS / HW path | FHD@120 I/O |
|---|---|---|---|
| **fake** | yes | synthetic | **yes (proven)** |
| **capturecard** | yes | Elgato / UVC | yes when input + driver grant 120 |
| realsense | yes | RealSense SDK | at sensor profile |
| webcam | yes | UVC | usually no |
| virtualcam | yes | OBS virtual cam | if backend grants 120 |

## Client guide

See **[CLIENT_GUIDE.md](CLIENT_GUIDE.md)** for a plain-language explanation of how the system works, which files do what, which commands to run, and how that maps to the POC-1 requirements.

## POC-1 all commands check
```bash
$env:Path = "C:\Users\User\.local\bin;" + $env:Path
uv sync --extra dev
uv sync --extra dev --extra realsense   # optional

#poc1.proof — FHD@120 pipeline stress test
uv run python -m poc1.proof --seconds 5 --width 1920 --height 1080 --fps 120

#poc1.run_poc1_tests — scorecard (5 short recordings)
uv run python -m poc1.run_poc1_tests --seconds 3

#poc1.delivery — client handoff pack
uv run python -m poc1.delivery

#GUI commands
#Fake
uv run python -m poc1.gui --source fake
#Webcam
uv run python -m poc1.gui --source webcam --device-index 0
#Capture card/Elgato
uv run python -m poc1.gui --source capturecard
#Realsense
uv run python -m poc1.gui --source realsense
#Virtual cam
uv run python -m poc1.gui --source virtualcam


#poc1.verifier — check a file you already recorded
uv run python -m poc1.verifier recordings\proof\some.report.json
uv run python -m poc1.verifier path\to\file.mp4 --no-barcode

#pytest — unit tests
uv run pytest -q

#Virtual cam sender (optional)
uv run python -m poc1.virtual_cam_sender --width 1920 --height 1080 --fps 120

```
## POC-1 done checklist

- [x] Spec pipeline shape with **compression in processor**
- [x] No-drop accounting + FHD@120 fake proof (`poc1.proof`)
- [x] Sysmon CSV + JSON report + verifier
- [x] GUI (fast preview, quiet logs, arm/record)
- [x] Five-source scorecard (`run_poc1_tests`) incl. **capturecard**
- [x] RealSense `.bag` option when hardware present (MP4 always)
- [x] Client delivery pack (`poc1.delivery` → `recordings/poc1_delivery/`)
- [x] **Elgato / capture card** source (`capturecard_source`, auto-detect)
- [ ] `.db3` export — **deferred** (Deliverable 2)
- [ ] Multi-cam arming (R1–R6) — **deferred**
- [ ] R8–R10 export polish / in-app browser — **deferred**



