"""
POC1 GUI — polished recorder UI: preview, arm/record, multi-source.
"""
from __future__ import annotations

import argparse
import json
import logging
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageTk

from poc1.camera_handler import FrameEnvelope
from poc1.frame_source import FakeFrameSource
from poc1.pipeline import CvCaptureSource, Pipeline
from poc1.quiet import configure_app_logging, silence_opencv
from poc1.realsense_source import list_realsense_devices, realsense_available

configure_app_logging()
logger = logging.getLogger("poc1.gui")

# Studio palette — slate + teal (tooling, not purple/glow/cream)
_BG = "#12181f"
_SURFACE = "#1a2330"
_SURFACE2 = "#243041"
_BORDER = "#334155"
_TEXT = "#e8eef6"
_MUTED = "#94a3b8"
_ACCENT = "#2dd4bf"
_ACCENT_DIM = "#0f766e"
_DANGER = "#f87171"
_OK = "#4ade80"
_PREVIEW_W, _PREVIEW_H = 480, 270


def _apply_theme(root: tk.Tk) -> None:
    root.configure(bg=_BG)
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass

    # Prefer fonts that exist on Windows; fall back silently.
    font_ui = ("Segoe UI", 10)
    font_sm = ("Segoe UI", 9)
    font_brand = ("Segoe UI Semibold", 20)
    font_sub = ("Segoe UI", 10)
    font_btn = ("Segoe UI Semibold", 10)

    style.configure(".", background=_BG, foreground=_TEXT, font=font_ui)
    style.configure("TFrame", background=_BG)
    style.configure("Surface.TFrame", background=_SURFACE)
    style.configure("TLabel", background=_BG, foreground=_TEXT, font=font_ui)
    style.configure("Muted.TLabel", background=_BG, foreground=_MUTED, font=font_sm)
    style.configure("Surface.TLabel", background=_SURFACE, foreground=_TEXT, font=font_ui)
    style.configure("SurfaceMuted.TLabel", background=_SURFACE, foreground=_MUTED, font=font_sm)
    style.configure("Brand.TLabel", background=_BG, foreground=_TEXT, font=font_brand)
    style.configure("Sub.TLabel", background=_BG, foreground=_MUTED, font=font_sub)

    style.configure(
        "Card.TLabelframe",
        background=_SURFACE,
        foreground=_MUTED,
        bordercolor=_BORDER,
        relief="solid",
        borderwidth=1,
    )
    style.configure(
        "Card.TLabelframe.Label",
        background=_SURFACE,
        foreground=_ACCENT,
        font=("Segoe UI Semibold", 9),
    )
    style.configure("TLabelframe", background=_SURFACE, foreground=_MUTED)
    style.configure("TLabelframe.Label", background=_SURFACE, foreground=_ACCENT)

    style.configure(
        "TEntry",
        fieldbackground=_SURFACE2,
        foreground=_TEXT,
        insertcolor=_TEXT,
        bordercolor=_BORDER,
        lightcolor=_BORDER,
        darkcolor=_BORDER,
        padding=4,
    )
    style.configure(
        "TCombobox",
        fieldbackground=_SURFACE2,
        foreground=_TEXT,
        background=_SURFACE2,
        arrowcolor=_ACCENT,
        padding=4,
    )
    style.map(
        "TCombobox",
        fieldbackground=[("readonly", _SURFACE2)],
        foreground=[("readonly", _TEXT)],
    )
    style.configure(
        "TSpinbox",
        fieldbackground=_SURFACE2,
        foreground=_TEXT,
        arrowcolor=_ACCENT,
        padding=2,
    )
    style.configure(
        "TCheckbutton",
        background=_SURFACE2,
        foreground=_TEXT,
        font=font_ui,
        focuscolor=_SURFACE2,
    )
    style.map("TCheckbutton", background=[("active", _SURFACE2)])

    style.configure(
        "Ghost.TButton",
        background=_SURFACE2,
        foreground=_TEXT,
        font=font_btn,
        padding=(10, 6),
        borderwidth=1,
        relief="raised",
    )
    style.map("Ghost.TButton", background=[("active", _BORDER)])


class App:
    def __init__(
        self,
        root: tk.Tk,
        source_mode: str,
        device_index: int,
        width: int,
        height: int,
        fps: int,
        out_dir: Path,
        prefix: str = "cam1",
    ):
        self.root = root
        self.pipeline: Optional[Pipeline] = None
        self._recording = False
        self._current_photo = None
        self._preview_started = False
        self._preview_starting = False
        self._last_preview_draw = 0.0
        self._preview_interval_s = 1.0 / 30.0

        _apply_theme(root)
        root.title("POC1 Recorder")
        root.minsize(720, 640)
        root.geometry("780x700")

        # —— ALWAYS-VISIBLE action bar (pack bottom first) ——
        action = tk.Frame(root, bg=_SURFACE2, padx=12, pady=10)
        action.pack(side="bottom", fill="x")

        self.armed = tk.BooleanVar(value=True)
        ttk.Checkbutton(action, text="Armed", variable=self.armed).pack(side="left", padx=(0, 8))
        self.record_bag = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            action, text="Also .bag (RealSense HW)", variable=self.record_bag
        ).pack(side="left", padx=(0, 16))

        # tk.Button = high contrast on Windows (ttk custom styles often look blank)
        self.preview_btn = tk.Button(
            action,
            text="Start Preview",
            command=self._restart_preview,
            bg=_SURFACE,
            fg=_TEXT,
            activebackground=_BORDER,
            activeforeground=_TEXT,
            font=("Segoe UI Semibold", 10),
            relief="raised",
            bd=2,
            padx=14,
            pady=6,
            cursor="hand2",
        )
        self.preview_btn.pack(side="left", padx=(0, 8))

        self.record_btn = tk.Button(
            action,
            text="Record",
            command=self.toggle_record,
            bg="#be123c",
            fg="#ffffff",
            activebackground="#e11d48",
            activeforeground="#ffffff",
            font=("Segoe UI Semibold", 10),
            relief="raised",
            bd=2,
            padx=18,
            pady=6,
            cursor="hand2",
        )
        self.record_btn.pack(side="left")

        # —— Status under action bar ——
        status_bar = tk.Frame(root, bg=_BG, padx=12, pady=4)
        status_bar.pack(side="bottom", fill="x")
        self.status_var = tk.StringVar(value="Ready — click Start Preview")
        tk.Label(
            status_bar,
            textvariable=self.status_var,
            bg=_BG,
            fg=_MUTED,
            font=("Segoe UI", 9),
            anchor="w",
            wraplength=740,
            justify="left",
        ).pack(fill="x")
        self.report_var = tk.StringVar(value="")
        tk.Label(
            status_bar,
            textvariable=self.report_var,
            bg=_BG,
            fg=_MUTED,
            font=("Consolas", 8),
            anchor="w",
            wraplength=740,
            justify="left",
        ).pack(fill="x", pady=(2, 4))

        # —— Scrollable main content ——
        main = ttk.Frame(root, style="TFrame")
        main.pack(side="top", fill="both", expand=True, padx=16, pady=(12, 4))

        header = ttk.Frame(main, style="TFrame")
        header.pack(fill="x", pady=(0, 8))
        ttk.Label(header, text="POC1", style="Brand.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="FHD · 120fps recorder  ·  camera → processor (compress) → disk",
            style="Sub.TLabel",
        ).pack(anchor="w")

        preview_wrap = ttk.Frame(main, style="Surface.TFrame")
        preview_wrap.pack(fill="x", pady=6)
        inner = tk.Frame(preview_wrap, bg=_BORDER, padx=1, pady=1)
        inner.pack(padx=10, pady=10)
        self.preview_label = tk.Label(inner, bg="#0b0f14", bd=0)
        self.preview_label.pack()
        self._show_placeholder("Click Start Preview")

        cfg = ttk.LabelFrame(main, text="  SOURCE  ", style="Card.TLabelframe", padding=10)
        cfg.pack(fill="x", pady=6)

        self.source_var = tk.StringVar(value=source_mode)
        ttk.Label(cfg, text="Input", style="Surface.TLabel").grid(row=0, column=0, sticky="w")
        self.source_combo = ttk.Combobox(
            cfg,
            textvariable=self.source_var,
            values=["fake", "virtualcam", "webcam", "capturecard", "realsense"],
            state="readonly",
            width=14,
        )
        self.source_combo.grid(row=0, column=1, padx=(8, 4), sticky="w")
        self.source_combo.bind("<<ComboboxSelected>>", lambda _e: self._restart_preview())

        ttk.Label(cfg, text="Device #", style="Surface.TLabel").grid(
            row=0, column=2, sticky="w", padx=(12, 0)
        )
        self.device_var = tk.IntVar(value=device_index)
        ttk.Spinbox(cfg, from_=0, to=9, textvariable=self.device_var, width=5).grid(
            row=0, column=3, padx=8, sticky="w"
        )
        ttk.Label(cfg, text="0 = auto for Elgato", style="SurfaceMuted.TLabel").grid(
            row=0, column=4, sticky="w"
        )

        ttk.Label(cfg, text="Width", style="Surface.TLabel").grid(
            row=1, column=0, sticky="w", pady=(8, 0)
        )
        self.width_var = tk.IntVar(value=width)
        ttk.Entry(cfg, textvariable=self.width_var, width=8).grid(
            row=1, column=1, sticky="w", padx=8, pady=(8, 0)
        )
        ttk.Label(cfg, text="Height", style="Surface.TLabel").grid(
            row=1, column=2, sticky="w", padx=(12, 0), pady=(8, 0)
        )
        self.height_var = tk.IntVar(value=height)
        ttk.Entry(cfg, textvariable=self.height_var, width=8).grid(
            row=1, column=3, sticky="w", padx=8, pady=(8, 0)
        )
        ttk.Label(cfg, text="FPS", style="Surface.TLabel").grid(
            row=1, column=4, sticky="w", padx=(8, 0), pady=(8, 0)
        )
        self.fps_var = tk.IntVar(value=fps)
        ttk.Entry(cfg, textvariable=self.fps_var, width=6).grid(
            row=1, column=5, sticky="w", padx=8, pady=(8, 0)
        )

        self.rs_serial_var = tk.StringVar(value="")
        ttk.Label(cfg, text="RS serial", style="Surface.TLabel").grid(
            row=2, column=0, sticky="w", pady=(8, 0)
        )
        ttk.Entry(cfg, textvariable=self.rs_serial_var, width=18).grid(
            row=2, column=1, columnspan=2, sticky="w", padx=8, pady=(8, 0)
        )
        ttk.Button(
            cfg, text="Detect RealSense", style="Ghost.TButton",
            command=self._detect_realsense,
        ).grid(row=2, column=3, columnspan=2, sticky="w", padx=4, pady=(8, 0))
        ttk.Button(
            cfg, text="Detect Capture Card", style="Ghost.TButton",
            command=self._detect_capturecard,
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(8, 0))

        out = ttk.LabelFrame(main, text="  OUTPUT  ", style="Card.TLabelframe", padding=10)
        out.pack(fill="x", pady=6)
        self.out_dir = tk.StringVar(value=str(out_dir.resolve()))
        self.prefix_var = tk.StringVar(value=prefix)
        ttk.Label(out, text="Folder", style="Surface.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Entry(out, textvariable=self.out_dir, width=48).grid(
            row=0, column=1, padx=8, sticky="we"
        )
        ttk.Button(out, text="Browse…", style="Ghost.TButton", command=self._browse_folder).grid(
            row=0, column=2
        )
        ttk.Label(out, text="Prefix", style="Surface.TLabel").grid(
            row=1, column=0, sticky="w", pady=(8, 0)
        )
        ttk.Entry(out, textvariable=self.prefix_var, width=16).grid(
            row=1, column=1, sticky="w", padx=8, pady=(8, 0)
        )
        out.columnconfigure(1, weight=1)

        root.protocol("WM_DELETE_WINDOW", self._on_close)
        root.after(50, self._restart_preview)

    def _set_record_button_idle(self) -> None:
        self.record_btn.configure(
            text="Record", bg="#be123c", activebackground="#e11d48", fg="#ffffff"
        )

    def _set_record_button_recording(self) -> None:
        self.record_btn.configure(
            text="Stop", bg=_ACCENT_DIM, activebackground=_ACCENT, fg="#ffffff"
        )

    def _show_placeholder(self, text: str) -> None:
        img = np.full((_PREVIEW_H, _PREVIEW_W, 3), 18, dtype=np.uint8)
        # subtle grid
        for x in range(0, _PREVIEW_W, 40):
            img[:, x:x + 1] = 28
        for y in range(0, _PREVIEW_H, 40):
            img[y:y + 1, :] = 28
        cv2.putText(
            img, "POC1", (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (45, 212, 191), 2, cv2.LINE_AA
        )
        cv2.putText(
            img, text, (24, _PREVIEW_H // 2 + 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 210, 220), 2, cv2.LINE_AA,
        )
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self._current_photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.preview_label.configure(image=self._current_photo)

    def _browse_folder(self) -> None:
        chosen = filedialog.askdirectory(initialdir=self.out_dir.get())
        if chosen:
            self.out_dir.set(chosen)

    def _detect_realsense(self) -> None:
        if not realsense_available():
            messagebox.showwarning(
                "RealSense",
                "pyrealsense2 not installed.\nRun: uv sync --extra realsense",
            )
            return
        devices = list_realsense_devices()
        if not devices:
            messagebox.showinfo(
                "RealSense", "No RealSense devices found — simulation will be used."
            )
            return
        lines = [f"{d['name']}  serial={d['serial']}" for d in devices]
        self.rs_serial_var.set(devices[0]["serial"])
        self.source_var.set("realsense")
        messagebox.showinfo("RealSense", "Found:\n" + "\n".join(lines))

    def _detect_capturecard(self) -> None:
        from poc1.capturecard_source import list_capture_card_candidates

        try:
            candidates = list_capture_card_candidates()
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Capture card", str(exc))
            return
        if not candidates:
            messagebox.showinfo(
                "Capture card",
                "No capture card candidates found.\n"
                "Plug in Elgato, close Game Capture software, connect HDMI signal.",
            )
            return
        ranked = sorted(
            candidates,
            key=lambda c: (0 if c["index"] == 0 else 1, c["width"] * c["height"]),
            reverse=True,
        )
        best = ranked[0]
        lines = [
            f"index={c['index']}  {c['backend']}  {c['width']}x{c['height']}"
            + ("  ← likely laptop webcam" if c["index"] == 0 else "")
            for c in candidates
        ]
        if best["index"] == 0 and best["width"] < 1920:
            messagebox.showwarning(
                "Capture card",
                "Only found device index 0 (likely laptop webcam), not an Elgato.\n"
                "Plug in the capture card, then click Detect again.\n\n"
                "Seen:\n" + "\n".join(lines),
            )
            return
        self.device_var.set(best["index"])
        self.source_var.set("capturecard")
        if self.width_var.get() < 1920:
            self.width_var.set(1920)
            self.height_var.set(1080)
            self.fps_var.set(120)
        messagebox.showinfo("Capture card", "Found:\n" + "\n".join(lines))

    def _build_source(self):
        mode = self.source_var.get()
        w, h, fps = self.width_var.get(), self.height_var.get(), self.fps_var.get()
        if mode == "fake":
            return FakeFrameSource(width=w, height=h, target_fps=fps)
        if mode == "realsense":
            from poc1.realsense_source import create_realsense_source

            serial = self.rs_serial_var.get().strip() or None
            return create_realsense_source(
                width=min(w, 1280), height=min(h, 720), fps=min(fps, 30),
                serial=serial, allow_simulate=True,
            )
        if mode == "virtualcam":
            from poc1.virtualcam_source import VirtualCamCaptureSource

            return VirtualCamCaptureSource(
                width=w, height=h, target_fps=fps,
                preferred_index=self.device_var.get() if self.device_var.get() > 0 else None,
                auto_start_sender=False,
                require_os=False,
            )
        if mode in ("capturecard", "elgato"):
            from poc1.capturecard_source import create_capture_card_source

            idx = self.device_var.get()
            preferred = idx if idx > 0 else None
            return create_capture_card_source(
                width=w, height=h, fps=fps, preferred_index=preferred,
            )
        return CvCaptureSource(self.device_var.get(), w, h, fps, backend=cv2.CAP_DSHOW)

    def _stop_pipeline(self) -> None:
        if self.pipeline is not None:
            try:
                if self._recording:
                    self.pipeline.stop_recording()
                    self._recording = False
                self.pipeline.stop()
            except Exception as exc:  # noqa: BLE001
                logger.debug("stop pipeline: %s", exc)
            self.pipeline = None
        self._preview_started = False

    def _restart_preview(self) -> None:
        if self._recording:
            messagebox.showwarning("Busy", "Stop recording before changing source.")
            return
        if self._preview_starting:
            return
        self._preview_starting = True
        self.preview_btn.configure(state="disabled")
        self._show_placeholder("Starting preview…")
        self.status_var.set(f"opening {self.source_var.get()}…")
        self.report_var.set("")
        self.root.update_idletasks()

        def worker() -> None:
            err: Optional[str] = None
            source = None
            pipe = None
            try:
                silence_opencv()
                self._stop_pipeline()
                source = self._build_source()
                pipe = Pipeline(source=source, on_preview_frame=self._on_preview_frame)
                pipe.start_preview()
            except Exception as exc:  # noqa: BLE001
                err = str(exc)
                pipe = None

            def done() -> None:
                self._preview_starting = False
                self.preview_btn.configure(state="normal")
                if err or pipe is None:
                    self.pipeline = None
                    self._preview_started = False
                    self._show_placeholder("Preview failed")
                    self.status_var.set(f"preview failed: {err}")
                    messagebox.showerror("Preview failed", err or "unknown error")
                    return
                self.pipeline = pipe
                self._preview_started = True
                mode = getattr(source, "capture_mode", None) or getattr(source, "mode", "")
                extra = f" [{mode}]" if mode else ""
                note = ""
                if mode == "simulated":
                    note = " — COLOR BARS are normal (no RealSense plugged in)"
                elif mode == "loopback":
                    note = " — COLOR BARS are normal (GUI loopback, not laptop webcam)"
                self.status_var.set(
                    f"preview: {self.source_var.get()}{extra} "
                    f"{source.width}x{source.height}@{source.target_fps}{note}"
                )

            self.root.after(0, done)

        threading.Thread(target=worker, name="preview-start", daemon=True).start()

    def _on_preview_frame(self, env: FrameEnvelope) -> None:
        import time
        now = time.perf_counter()
        if now - self._last_preview_draw < self._preview_interval_s:
            return
        self._last_preview_draw = now
        small = cv2.resize(env.frame, (_PREVIEW_W, _PREVIEW_H))
        self.root.after(0, self._render_preview, small)

    def _render_preview(self, small_bgr: np.ndarray) -> None:
        rgb = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB)
        self._current_photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.preview_label.configure(image=self._current_photo)

    def toggle_record(self) -> None:
        if not self._recording:
            if not self.armed.get():
                self.status_var.set("cannot record: not armed")
                return
            self._start_recording()
        else:
            self._stop_recording()

    def _start_recording(self) -> None:
        try:
            if self.pipeline is None or not self._preview_started:
                self.status_var.set("wait for preview to start…")
                return
            from poc1.codec import choose_best_fourcc
            choose_best_fourcc()
            out = Path(self.out_dir.get())
            out.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            prefix = self.prefix_var.get().strip() or "cam1"
            output_path = out / f"{prefix}_{stamp}.mp4"
            monitor_csv = out / f"{prefix}_{stamp}_sysmon.csv"
            bag_path = None
            if self.record_bag.get() and self.source_var.get() == "realsense":
                bag_path = out / f"{prefix}_{stamp}.bag"
            self.pipeline.start_recording(output_path, monitor_csv, bag_path=bag_path)
            self._recording = True
            self._set_record_button_recording()
            bag_note = f" + {bag_path.name}" if bag_path else ""
            self.status_var.set(f"recording → {output_path.name}{bag_note}")
            self.report_var.set("")
        except Exception as exc:  # noqa: BLE001
            logger.exception("Record failed")
            messagebox.showerror("Record failed", str(exc))

    def _stop_recording(self) -> None:
        if not self.pipeline:
            return
        report = self.pipeline.stop_recording()
        self._recording = False
        self._set_record_button_idle()
        drops_ok = "OK — no frame drops" if report["no_frame_drops"] else "DROPS DETECTED"
        fps_note = ""
        if report.get("fps_corrected"):
            fps_note = (
                f" | playback fixed to ~{report.get('container_fps')}fps "
                f"(webcam hardware limit, not a bug)"
            )
        self.status_var.set(
            f"stopped [{drops_ok}] read={report['frames_read_by_camera']} "
            f"written={report['frames_written']} codec={report.get('codec', '?')}{fps_note}"
        )
        self.report_var.set(json.dumps({
            k: report[k] for k in (
                "no_frame_drops", "frames_read_by_camera", "frames_written",
                "codec", "compression_stage", "measured_fps", "container_fps",
                "fps_corrected", "bag_recorded", "bag_path",
            ) if k in report
        }, indent=2))
        out = Path(report.get("output_path") or "")
        if out:
            self._show_review_popup(out)

    def _show_review_popup(self, output_path: Path) -> None:
        popup = tk.Toplevel(self.root)
        popup.title("Recording saved")
        popup.configure(bg=_SURFACE)
        ttk.Label(
            popup,
            text=f"Saved: {output_path.name}\nReview footage now?\n(auto-closes in 5s)",
            background=_SURFACE,
        ).pack(padx=20, pady=12)
        ttk.Button(
            popup, text="Review", style="Ghost.TButton",
            command=lambda: self._open_playback(output_path),
        ).pack(pady=8)
        popup.after(5000, popup.destroy)

    def _open_playback(self, output_path: Path) -> None:
        import os
        import subprocess
        import sys

        if sys.platform == "win32":
            os.startfile(output_path)  # noqa: S606
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(output_path)])
        else:
            subprocess.Popen(["xdg-open", str(output_path)])

    def _on_close(self) -> None:
        self._stop_pipeline()
        self.root.destroy()


def main() -> None:
    silence_opencv()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        choices=["fake", "virtualcam", "webcam", "capturecard", "elgato", "realsense"],
        default="fake",
    )
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--out-dir", type=Path, default=Path("./recordings"))
    parser.add_argument("--prefix", type=str, default="cam1")
    args = parser.parse_args()

    if args.source in ("fake", "capturecard", "elgato") and args.width == 1280 and args.fps == 30:
        args.width, args.height, args.fps = 1920, 1080, 120
    if args.source == "elgato":
        args.source = "capturecard"

    root = tk.Tk()
    App(
        root,
        args.source,
        args.device_index,
        args.width,
        args.height,
        args.fps,
        args.out_dir,
        args.prefix,
    )
    root.mainloop()


if __name__ == "__main__":
    main()
