"""
Unified POC1 app — Setup / Record / Library in one window.

Launch:
  uv run poc1
  uv run python -m poc1
  uv run python -m poc1.app
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Optional

import cv2
from PIL import Image, ImageTk

from poc1.app.cards import CameraCard
from poc1.app.library import LibraryPage
from poc1.app.theme import (
    ACCENT,
    ACCENT_SOFT,
    BG,
    BLUE,
    BORDER,
    FIELD,
    GREEN,
    INK,
    MUTED,
    PANEL,
    RED,
    SURFACE,
    apply_styles,
    button,
)
from poc1.deliverable1.session import MultiCamSession
from poc1.deliverable2.review import show_review_prompt
from poc1.quiet import configure_app_logging, silence_opencv

configure_app_logging()
logger = logging.getLogger("poc1.app")


class UnifiedApp:
    """One comfortable shell over the existing multi-cam + library pipelines."""

    def __init__(self, root: tk.Tk, n_slots: int = 2, out_dir: Path | None = None):
        self.root = root
        self.session = MultiCamSession(
            n_slots=n_slots,
            out_dir=out_dir or Path("./recordings/deliverable1"),
        )
        self.cards: list[CameraCard] = []
        self.busy = False
        self._stopping = False
        self.last_reports: dict[str, dict] = {}
        self._closing = False
        self._record_tiles: list[tk.Label] = []
        self._record_photos: list = []

        root.title("POC1 Recorder")
        root.configure(bg=BG)
        root.geometry("1240x860")
        root.minsize(980, 700)
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        apply_styles(root)

        self._build_chrome()
        self._build_pages()
        self.rebuild_cards(auto_assign=False)
        self.refresh_cameras(initial=True)
        self.library.sync_folder(self.session.out_dir)
        self.refresh_record_gate()
        self._tick()

    # ── chrome ───────────────────────────────────────────────────────────

    def _build_chrome(self) -> None:
        header = tk.Frame(self.root, bg=PANEL, padx=20, pady=14)
        header.pack(fill="x")
        header.configure(highlightbackground=BORDER, highlightthickness=1)
        tk.Label(
            header,
            text="POC1 Recorder",
            bg=PANEL,
            fg=INK,
            font=("Segoe UI Semibold", 18),
        ).pack(side="left")
        tk.Label(
            header,
            text="Cameras  ·  Record  ·  Library",
            bg=PANEL,
            fg=MUTED,
            font=("Segoe UI", 10),
        ).pack(side="left", padx=16)
        self.count_var = tk.StringVar()
        tk.Label(
            header,
            textvariable=self.count_var,
            bg=ACCENT_SOFT,
            fg=ACCENT,
            font=("Segoe UI Semibold", 9),
            padx=10,
            pady=4,
        ).pack(side="right")

        self.status_bar = tk.Frame(self.root, bg=SURFACE, padx=16, pady=8)
        self.status_bar.pack(side="bottom", fill="x")
        self.status_var = tk.StringVar(
            value="Start on Setup: refresh cameras, start previews, then Record."
        )
        tk.Label(
            self.status_bar,
            textvariable=self.status_var,
            bg=SURFACE,
            fg=INK,
            anchor="w",
            font=("Segoe UI", 9),
        ).pack(fill="x")

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=12, pady=10)
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab)

    def _build_pages(self) -> None:
        self.setup_page = tk.Frame(self.notebook, bg=BG)
        self.record_page = tk.Frame(self.notebook, bg=BG)
        self.library_page = tk.Frame(self.notebook, bg=BG)
        self.notebook.add(self.setup_page, text="  Setup  ")
        self.notebook.add(self.record_page, text="  Record  ")
        self.notebook.add(self.library_page, text="  Library  ")

        self._build_setup()
        self._build_record()
        self.library = LibraryPage(self.library_page, self)
        self.library.pack(fill="both", expand=True)

    def _build_setup(self) -> None:
        tip = tk.Frame(self.setup_page, bg=ACCENT_SOFT, padx=16, pady=10)
        tip.pack(fill="x", padx=4, pady=(4, 0))
        tk.Label(
            tip,
            text="1) Refresh cameras  ·  2) Pick device & configuration  ·  "
            "3) Check Armed on cameras to record  ·  "
            "4) Start preview until live  ·  5) Go to Record",
            bg=ACCENT_SOFT,
            fg=ACCENT,
            font=("Segoe UI", 9),
            anchor="w",
        ).pack(fill="x")

        folder = tk.Frame(self.setup_page, bg=BG, padx=8, pady=10)
        folder.pack(fill="x")
        tk.Label(
            folder, text="Save folder", bg=BG, fg=MUTED, font=("Segoe UI Semibold", 9)
        ).pack(side="left")
        self.out_var = tk.StringVar(value=str(self.session.out_dir.resolve()))
        self.out_entry = tk.Entry(
            folder,
            textvariable=self.out_var,
            bg=FIELD,
            fg=INK,
            insertbackground=INK,
            relief="solid",
            bd=1,
            font=("Segoe UI", 9),
        )
        self.out_entry.pack(side="left", fill="x", expand=True, padx=10, ipady=5)
        button(folder, "Browse…", self.browse_folder, color="#475569").pack(side="left")

        toolbar = tk.Frame(self.setup_page, bg=BG, padx=8, pady=4)
        toolbar.pack(fill="x")
        self.refresh_btn = button(
            toolbar, "Refresh cameras", self.refresh_cameras, color="#475569"
        )
        self.refresh_btn.pack(side="left")
        self.add_btn = button(toolbar, "Add camera", self.add_camera, color=ACCENT)
        self.add_btn.pack(side="left", padx=6)
        self.start_all_btn = button(
            toolbar, "Start all previews", self.start_all_previews, color=BLUE
        )
        self.start_all_btn.pack(side="left")
        self.stop_all_btn = button(
            toolbar, "Stop all previews", self.stop_all_previews, color="#475569"
        )
        self.stop_all_btn.pack(side="left", padx=6)

        adv = ttk.LabelFrame(self.setup_page, text=" Advanced ", padding=8)
        adv.pack(fill="x", padx=8, pady=(4, 0))
        tk.Label(
            adv,
            text="Files are compressed MP4 (+ optional RealSense .bag). "
            "System usage CSV and JSON reports are written next to each take. "
            "Import/export .db3 or .bd3 in Library.",
            fg=MUTED,
            font=("Segoe UI", 8),
            wraplength=1000,
            justify="left",
        ).pack(anchor="w")

        shell = tk.Frame(self.setup_page, bg=BG)
        shell.pack(fill="both", expand=True, padx=4, pady=8)
        self.canvas = tk.Canvas(shell, bg=BG, highlightthickness=0, bd=0)
        scrollbar = ttk.Scrollbar(shell, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.cards_frame = tk.Frame(self.canvas, bg=BG)
        self.canvas_window = self.canvas.create_window(
            (0, 0), window=self.cards_frame, anchor="nw"
        )
        self.cards_frame.bind(
            "<Configure>",
            lambda _e: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )
        self.canvas.bind(
            "<Configure>",
            lambda e: self.canvas.itemconfigure(self.canvas_window, width=e.width),
        )
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _build_record(self) -> None:
        tip = tk.Frame(self.record_page, bg=SURFACE, padx=16, pady=10)
        tip.pack(fill="x", padx=4, pady=(4, 0))
        self.record_tip = tk.StringVar()
        tk.Label(
            tip,
            textvariable=self.record_tip,
            bg=SURFACE,
            fg=MUTED,
            font=("Segoe UI", 9),
            anchor="w",
            wraplength=1000,
            justify="left",
        ).pack(fill="x")

        actions = tk.Frame(self.record_page, bg=BG, padx=8, pady=12)
        actions.pack(fill="x")
        self.record_btn = button(
            actions,
            "●  Record",
            self.start_recording,
            color=RED,
            font=("Segoe UI Semibold", 12),
            padx=28,
            pady=12,
        )
        self.record_btn.pack(side="left")
        self.stop_record_btn = button(
            actions,
            "■  Stop",
            self.stop_recording,
            color=GREEN,
            font=("Segoe UI Semibold", 12),
            padx=28,
            pady=12,
            state="disabled",
        )
        self.stop_record_btn.pack(side="left", padx=10)
        button(actions, "Open save folder", self.open_output_folder, color="#475569").pack(
            side="right"
        )
        button(actions, "Last report", self.show_report, color="#475569").pack(
            side="right", padx=6
        )

        self.record_grid = tk.Frame(self.record_page, bg=BG)
        self.record_grid.pack(fill="both", expand=True, padx=8, pady=8)

    # ── navigation helpers ───────────────────────────────────────────────

    def set_status(self, text: str) -> None:
        self.status_var.set(text)

    def show_library(self, path: Path | None = None) -> None:
        self.notebook.select(self.library_page)
        self.library.sync_folder(Path(self.out_var.get()))
        if path is not None:
            self.library.select_path(Path(path))

    def _on_tab(self, *_args) -> None:
        try:
            current = self.notebook.select()
        except tk.TclError:
            return
        if current == str(self.library_page):
            self.library.sync_folder(Path(self.out_var.get()))
        elif current == str(self.record_page):
            self.refresh_record_gate()
            self._rebuild_record_tiles()

    def _on_mousewheel(self, event) -> None:
        try:
            if self.notebook.select() != str(self.setup_page):
                return
        except tk.TclError:
            return
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    # ── setup actions ────────────────────────────────────────────────────

    def rebuild_cards(self, auto_assign: bool = False) -> None:
        for card in self.cards:
            card.destroy()
        self.cards.clear()
        for col in range(2):
            self.cards_frame.grid_columnconfigure(col, weight=1, uniform="cards")
        for index, slot in enumerate(self.session.slots):
            card = CameraCard(self.cards_frame, self, slot)
            card.grid(row=index // 2, column=index % 2, sticky="nsew", padx=6, pady=6)
            self.cards.append(card)
        for card in self.cards:
            card.load_devices(auto_assign=auto_assign and card.slot_id == 0)
        self.count_var.set(f"{len(self.session.slots)} cameras")
        self._rebuild_record_tiles()
        self.refresh_record_gate()

    def add_camera(self) -> None:
        if self.busy:
            return
        try:
            self.session.add_slot()
            self.rebuild_cards(auto_assign=False)
            self.set_status(f"Camera {len(self.session.slots)} added.")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Add camera", str(exc))

    def remove_camera(self, slot_id: int) -> None:
        if self.busy:
            return
        if not messagebox.askyesno(
            "Remove camera", f"Remove Camera {slot_id + 1}?"
        ):
            return
        try:
            self.session.remove_slot(slot_id)
            self.rebuild_cards()
            self.set_status("Camera removed.")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Remove camera", str(exc))

    def refresh_cameras(self, initial: bool = False) -> None:
        if self.busy:
            return
        if self.session.previews_running:
            if not messagebox.askyesno(
                "Refresh cameras", "This stops all previews. Continue?"
            ):
                return
            self.session.stop_previews()
        try:
            devices = self.session.refresh_devices(include_fake=True)
            self.rebuild_cards(auto_assign=initial)
            real = [d for d in devices if d.kind != "fake"]
            names = ", ".join(d.name for d in real[:5]) or "none yet"
            self.set_status(
                f"Found {len(real)} camera(s): {names}. "
                "Start preview on each camera you plan to record."
            )
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Camera scan failed", str(exc))

    def browse_folder(self) -> None:
        if self.busy:
            return
        chosen = filedialog.askdirectory(initialdir=self.out_var.get())
        if chosen:
            self.out_var.set(chosen)
            self.session.set_out_dir(chosen)
            self.library.sync_folder(Path(chosen))

    def _sync_all(self) -> None:
        self.session.set_out_dir(self.out_var.get())
        for card in self.cards:
            card.sync_controls()

    def start_all_previews(self) -> None:
        if self.busy:
            return
        self._sync_all()
        try:
            self.session.start_previews()
            running = len([s for s in self.session.slots if s.pipeline])
            self.set_status(f"{running} preview(s) live. Open Record when ready.")
            self.refresh_record_gate()
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Preview", str(exc))

    def stop_all_previews(self) -> None:
        if self.busy:
            return
        if self.session.is_recording:
            messagebox.showwarning("Recording", "Stop recording first.")
            return
        self.session.stop_previews()
        for card in self.cards:
            card.preview.configure(image="", text="Preview stopped")
            card._photo = None
        self.set_status("All previews stopped.")
        self.refresh_record_gate()

    # ── record gate + tiles ──────────────────────────────────────────────

    def _armed_live_ready(self) -> tuple[bool, str]:
        armed = [s for s in self.session.slots if s.armed]
        if not armed:
            return False, "Arm at least one camera on Setup (Include when recording)."
        waiting = []
        for s in armed:
            if s.pipeline is None:
                waiting.append(f"Camera {s.slot_id + 1} (start preview)")
                continue
            if s.get_preview_frame() is None:
                waiting.append(f"Camera {s.slot_id + 1} (wait for live picture)")
        if waiting:
            return False, "Need live preview before Record: " + "; ".join(waiting)
        return True, f"Ready to record {len(armed)} camera(s)."

    def refresh_record_gate(self) -> None:
        if self.busy or self.session.is_recording:
            return
        ok, tip = self._armed_live_ready()
        self.record_tip.set(tip)
        self.record_btn.configure(state="normal" if ok else "disabled")

    def _rebuild_record_tiles(self) -> None:
        for child in self.record_grid.winfo_children():
            child.destroy()
        self._record_tiles = []
        self._record_photos = []
        slots = self.session.slots
        cols = 2 if len(slots) > 1 else 1
        for col in range(cols):
            self.record_grid.columnconfigure(col, weight=1, uniform="rec")
        for index, slot in enumerate(slots):
            cell = tk.Frame(
                self.record_grid,
                bg=PANEL,
                highlightbackground=BORDER,
                highlightthickness=1,
            )
            cell.grid(
                row=index // cols,
                column=index % cols,
                sticky="nsew",
                padx=6,
                pady=6,
            )
            self.record_grid.rowconfigure(index // cols, weight=1)
            title = tk.Label(
                cell,
                text=f"Camera {slot.slot_id + 1}  ·  {slot.prefix}"
                + ("  ·  armed" if slot.armed else "  ·  not armed"),
                bg=PANEL,
                fg=INK,
                font=("Segoe UI Semibold", 10),
                anchor="w",
                padx=10,
                pady=6,
            )
            title.pack(fill="x")
            tile = tk.Label(
                cell,
                bg="#0b1220",
                fg="#94a3b8",
                text="No preview",
                font=("Segoe UI", 10),
            )
            tile.pack(fill="both", expand=True, padx=8, pady=(0, 8))
            self._record_tiles.append(tile)
            self._record_photos.append(None)

    def _tick_record_tiles(self) -> None:
        if len(self._record_tiles) != len(self.session.slots):
            return
        for slot, tile, i in zip(
            self.session.slots, self._record_tiles, range(len(self._record_tiles))
        ):
            frame = slot.get_preview_frame()
            if frame is None:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            max_w = max(tile.winfo_width(), 240)
            max_h = max(tile.winfo_height(), 160)
            scale = min(max_w / max(w, 1), max_h / max(h, 1), 1.0)
            if scale < 1.0:
                rgb = cv2.resize(
                    rgb,
                    (max(1, int(w * scale)), max(1, int(h * scale))),
                    interpolation=cv2.INTER_AREA,
                )
            photo = ImageTk.PhotoImage(Image.fromarray(rgb))
            self._record_photos[i] = photo
            tile.configure(image=photo, text="")

    # ── recording ────────────────────────────────────────────────────────

    def _set_busy(self, busy: bool, recording: bool = False) -> None:
        self.busy = busy
        for card in self.cards:
            card.set_locked(busy or recording)
        state = "disabled" if busy or recording else "normal"
        for btn in (
            self.refresh_btn,
            self.add_btn,
            self.start_all_btn,
            self.stop_all_btn,
        ):
            btn.configure(state=state)
        self.out_entry.configure(state=state)
        if recording:
            self.record_btn.configure(state="disabled")
            self.stop_record_btn.configure(state="normal" if not busy else "disabled")
        elif busy:
            self.record_btn.configure(state="disabled")
            self.stop_record_btn.configure(state="disabled")
        else:
            self.stop_record_btn.configure(state="disabled")
            self.refresh_record_gate()

    def start_recording(self) -> None:
        if self.busy or self.session.is_recording:
            return
        ok, tip = self._armed_live_ready()
        if not ok:
            messagebox.showinfo("Not ready", tip)
            return
        try:
            self._sync_all()
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Recording", str(exc))
            return

        # Snapshot .bag intent before UI lock (busy used to clear the flag).
        bag_intent = {s.slot_id: bool(s.record_bag) for s in self.session.slots}

        armed = [s for s in self.session.slots if s.armed]
        heavy = [
            s
            for s in armed
            if s.mode and s.mode.width * s.mode.height * s.mode.fps >= 1920 * 1080 * 60
        ]
        if len(armed) >= 2 and heavy:
            self.set_status(
                "High-rate multi-cam — recording at configured FPS (encoder may drop if overloaded)."
            )

        self._set_busy(True)
        for s in self.session.slots:
            s.record_bag = bag_intent.get(s.slot_id, s.record_bag)
        self.set_status("Starting recording…")

        def worker() -> None:
            paths: list[Path] = []
            error: Optional[str] = None
            try:
                paths = self.session.start_recording_armed()
            except Exception as exc:  # noqa: BLE001
                logger.exception("record start failed")
                error = str(exc)
            self.root.after(0, lambda: self._record_started(paths, error))

        threading.Thread(target=worker, name="app-record-start", daemon=True).start()

    def _record_started(self, paths: list[Path], error: Optional[str]) -> None:
        if error:
            self._set_busy(False)
            self.set_status("Recording did not start.")
            messagebox.showerror("Recording", error)
            return
        self._set_busy(False, recording=True)
        self.record_tip.set(f"Recording {len(paths)} camera(s)…")
        self.set_status("Recording — press Stop when finished.")
        self.notebook.select(self.record_page)
        warnings = list(getattr(self.session, "last_start_warnings", []) or [])
        if warnings:
            messagebox.showwarning(
                "RealSense .bag skipped",
                "MP4 recording started.\n\n" + "\n\n".join(warnings),
            )

    def stop_recording(self) -> None:
        # Always allow Stop while a take is active — even if busy from a
        # hung start — otherwise the button appears to do nothing.
        if self._stopping:
            return
        if not self.session.is_recording:
            if self.busy:
                self._set_busy(False)
            self.set_status("Not recording.")
            return

        self._stopping = True
        self._set_busy(True, recording=True)
        self.record_tip.set("Stopping… saving files (please wait a moment).")
        self.set_status("Stopping recording…")

        def worker() -> None:
            reports: dict = {}
            error: Optional[str] = None
            try:
                reports = self.session.stop_recording_armed()
            except Exception as exc:  # noqa: BLE001
                logger.exception("record stop failed")
                error = str(exc)
            self.root.after(0, lambda: self._record_stopped(reports, error))

        threading.Thread(target=worker, name="app-record-stop", daemon=True).start()

    def _record_stopped(
        self, reports: dict[str, dict], error: Optional[str] = None
    ) -> None:
        self._stopping = False
        self.last_reports = reports
        self._set_busy(False)
        if error:
            self.set_status(f"Stop failed: {error}")
            messagebox.showerror("Stop recording", error)
            self.refresh_record_gate()
            return

        valid = [r for r in reports.values() if "error" not in r]
        ok = bool(valid) and all(r.get("no_frame_drops") for r in valid)
        self.set_status(
            "Saved cleanly — no frame drops."
            if ok
            else "Saved. Check Last report if something looks off."
        )
        self.record_tip.set("Recording stopped. Arm cameras and start previews to record again.")

        mismatched = [
            (slot, reports.get(slot.prefix, {}))
            for slot in self.session.slots
            if reports.get(slot.prefix, {}).get("fps_mismatch")
        ]
        # Auto-fix remaining mismatches (Elgato already auto-fixed in pipeline).
        if mismatched:
            details = "\n".join(
                f"• {slot.prefix}: requested {report.get('requested_fps', report.get('target_fps'))} fps, "
                f"delivered ~{float(report.get('measured_fps') or 0):.0f} fps "
                f"(file stamped {report.get('container_fps')} fps)"
                for slot, report in mismatched
            )
            self.set_status("Fixing playback timing for mismatched cameras…")
            self._convert_mismatched_async(mismatched)
            messagebox.showinfo(
                "Playback timing",
                f"{details}\n\n"
                "Playback speed was auto-fixed to match real time.\n"
                "For true 120fps on Elgato: set the HDMI source itself to 1080p120.",
            )

        if not ok:
            drop_notes = []
            for slot in self.session.slots:
                report = reports.get(slot.prefix) or {}
                if report.get("no_frame_drops", True):
                    continue
                drop_notes.append(
                    f"{slot.prefix}: wrote {report.get('frames_written')} of "
                    f"{report.get('frames_read_by_camera')} frames"
                )
            if drop_notes:
                messagebox.showwarning(
                    "Some frames were skipped",
                    "The PC couldn’t keep up with encode on:\n\n"
                    + "\n".join(drop_notes)
                    + "\n\nTip: for a clean FHD@120 proof, record only the synthetic "
                    "fake camera (or run poc1.proof).",
                )

        saved: list[Path] = []
        for report in reports.values():
            raw = report.get("output_path")
            if not raw:
                continue
            path = Path(str(raw))
            if path.is_file() and path.suffix.lower() in {".mp4", ".avi", ".mkv"}:
                saved.append(path)
        if saved:
            show_review_prompt(
                self.root,
                saved,
                on_review=lambda p: self.show_library(p),
                review_label="Open in Library",
            )

    def _convert_mismatched_async(self, mismatched) -> None:
        self._set_busy(True)
        self.set_status("Fixing playback speed in the background…")

        def worker() -> None:
            results: dict[str, bool] = {}
            for slot, _report in mismatched:
                try:
                    results[slot.prefix] = bool(
                        slot.pipeline and slot.pipeline.convert_container_fps()
                    )
                except Exception:  # noqa: BLE001
                    logger.exception("FPS conversion failed for %s", slot.prefix)
                    results[slot.prefix] = False
            self.root.after(0, lambda: self._conversion_done(results))

        threading.Thread(target=worker, name="app-fps-convert", daemon=True).start()

    def _conversion_done(self, results: dict[str, bool]) -> None:
        for slot in self.session.slots:
            if slot.pipeline and slot.pipeline._last_report:
                self.last_reports[slot.prefix] = slot.pipeline._last_report
        self._set_busy(False)
        failed = [prefix for prefix, ok in results.items() if not ok]
        if failed:
            messagebox.showwarning(
                "Couldn’t fix all files",
                "Still using original timing for: " + ", ".join(failed),
            )
        else:
            messagebox.showinfo(
                "Speed fixed",
                "Playback timing was updated in the background. Frame count unchanged.",
            )
        self.set_status("Ready.")

    def show_report(self) -> None:
        popup = tk.Toplevel(self.root)
        popup.title("Last recording report")
        popup.geometry("720x520")
        popup.configure(bg=BG)
        controls = tk.Frame(popup, bg=BG, padx=10, pady=8)
        controls.pack(side="bottom", fill="x")
        paths = [
            (prefix, Path(str(report.get("output_path"))))
            for prefix, report in self.last_reports.items()
            if report.get("output_path")
        ]
        for prefix, path in paths:
            button(
                controls,
                f"Open {prefix}",
                lambda p=path: self.show_library(p),
                color=BLUE,
            ).pack(side="left", padx=(0, 6))
        button(controls, "Close", popup.destroy, color="#475569").pack(side="right")
        text = tk.Text(popup, bg=PANEL, fg=INK, font=("Consolas", 9), wrap="none")
        text.pack(fill="both", expand=True, padx=10, pady=10)
        summary = {
            prefix: {
                key: report.get(key)
                for key in (
                    "no_frame_drops",
                    "frames_read_by_camera",
                    "frames_written",
                    "width",
                    "height",
                    "requested_fps",
                    "target_fps",
                    "measured_fps",
                    "fps_mismatch",
                    "container_fps",
                    "bag_recorded",
                    "bag_path",
                    "output_path",
                )
                if key in report
            }
            if "error" not in report
            else report
            for prefix, report in self.last_reports.items()
        }
        text.insert("1.0", json.dumps(summary, indent=2) if summary else "No recording yet.")
        text.configure(state="disabled")

    def open_output_folder(self) -> None:
        path = Path(self.out_var.get()).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        try:
            if sys.platform == "win32":
                os.startfile(path)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Open folder", str(exc))

    def _tick(self) -> None:
        if self._closing:
            return
        for card in self.cards:
            card.tick()
        try:
            if self.notebook.select() == str(self.record_page):
                self._tick_record_tiles()
                if not self.busy and not self.session.is_recording:
                    self.refresh_record_gate()
        except tk.TclError:
            pass
        self.root.after(50, self._tick)

    def _on_close(self) -> None:
        if self.busy:
            messagebox.showwarning("Please wait", "Finish saving before closing.")
            return
        if self.session.is_recording and not messagebox.askyesno(
            "Recording", "Stop and save, then close?"
        ):
            return
        self._closing = True
        try:
            self.library.stop_playback()
        except Exception:  # noqa: BLE001
            pass
        try:
            self.session.shutdown()
        except Exception:  # noqa: BLE001
            logger.exception("shutdown failed")
        try:
            self.canvas.unbind_all("<MouseWheel>")
        except Exception:  # noqa: BLE001
            pass
        self.root.destroy()


def main() -> None:
    silence_opencv()
    parser = argparse.ArgumentParser(description="POC1 Recorder — unified Setup / Record / Library")
    parser.add_argument("--slots", type=int, default=2, help="Initial camera slots (>=2)")
    parser.add_argument(
        "--out-dir", type=Path, default=Path("./recordings/deliverable1")
    )
    args = parser.parse_args()
    if args.slots < 2:
        parser.error("--slots must be >= 2")
    root = tk.Tk()
    UnifiedApp(root, n_slots=args.slots, out_dir=args.out_dir)
    root.mainloop()


if __name__ == "__main__":
    main()
