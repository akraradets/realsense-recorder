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

from PIL import Image, ImageTk

from poc1.app.cards import CameraCard
from poc1.app.library import LibraryPage
from poc1.preview_draw import bgr_to_rgb_fill, hud_lines_for_source, overlay_hud
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
from poc1.deliverable1.devices import is_fhd_high_rate, too_many_1080p120
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

        from poc1.bag_recorder import BUILD_ID

        root.title(f"POC1 Recorder — {BUILD_ID}")
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
        from poc1.bag_recorder import BUILD_ID

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
            text=f"  {BUILD_ID}",
            bg=PANEL,
            fg=MUTED,
            font=("Segoe UI", 9),
        ).pack(side="left", padx=(8, 0))
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
            text="MP4 + optional RealSense .db3/.bag stay in the save folder. "
            "JSON reports and sysmon CSV go in a meta/ subfolder. "
            "Close RealSense Viewer before Record so Elgato is not locked. "
            "Library can export bags to MP4.",
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
        tk.Label(
            tip,
            text="Multi-cam: RealSense/webcam @30 (or @60 if the device has it) + Elgato 1080p120 together. "
            "Do not arm two 1080p@120 cameras. HDMI must be 1080p120 for true Elgato 120.",
            bg=SURFACE,
            fg=INK,
            font=("Segoe UI Semibold", 9),
            anchor="w",
            wraplength=1000,
            justify="left",
        ).pack(fill="x")
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
            card.load_devices(auto_assign=auto_assign)
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
        self._set_busy(True)
        self.set_status("Scanning cameras… (UI stays responsive)")

        def worker() -> None:
            devices: list = []
            error: Optional[str] = None
            try:
                devices = self.session.refresh_devices(include_fake=True)
            except Exception as exc:  # noqa: BLE001
                logger.exception("camera refresh failed")
                error = str(exc)
            self.root.after(
                0,
                lambda d=devices, err=error, init=initial: self._refresh_cameras_done(
                    d, err, init
                ),
            )

        threading.Thread(target=worker, name="app-refresh-cameras", daemon=True).start()

    def _refresh_cameras_done(
        self, devices: list, error: Optional[str], initial: bool
    ) -> None:
        self._set_busy(False)
        if error:
            messagebox.showerror("Camera scan failed", error)
            return
        self.rebuild_cards(auto_assign=initial)
        real = [d for d in devices if d.kind != "fake"]
        rs_n = sum(1 for d in real if d.kind == "realsense")
        elg_n = sum(1 for d in real if d.device_tag == "elgato")
        uvc_n = sum(
            1
            for d in real
            if d.kind == "uvc" and d.device_tag not in {"elgato", "realsense-uvc"}
        )
        parts = []
        if rs_n:
            parts.append(f"RealSense×{rs_n}")
        if elg_n:
            parts.append(f"Elgato×{elg_n}")
        if uvc_n:
            parts.append(f"UVC×{uvc_n}")
        summary = ", ".join(parts) or "none yet"
        names = ", ".join(d.name for d in real[:4]) or "none yet"
        status = (
            f"Found {len(real)} camera(s) ({summary}): {names}. "
            "Start preview on each camera you plan to record."
        )
        if elg_n and sys.platform == "win32":
            try:
                from poc1.deliverable1.win_names import ffmpeg_available

                if not ffmpeg_available():
                    status += (
                        " Warning: ffmpeg missing — Elgato open may fail; "
                        "install ffmpeg on PATH, then Refresh."
                    )
            except Exception:  # noqa: BLE001
                pass
        self.set_status(status)

    def _clear_opening_placeholders(self) -> None:
        for card in self.cards:
            slot = self.session.slots[card.slot_id]
            if slot.pipeline is not None:
                continue
            try:
                text = str(card.preview.cget("text") or "")
            except Exception:  # noqa: BLE001
                continue
            if "Opening" in text:
                card.preview.configure(
                    image="", text="Start preview to see live video"
                )
                card._photo = None

    def _live_preview_count(self) -> int:
        return sum(
            1
            for s in self.session.slots
            if s.pipeline is not None and s.get_preview_frame() is not None
        )

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
        self._set_busy(True)
        self.set_status("Opening cameras… (UI stays responsive)")
        for card in self.cards:
            if self.session.slots[card.slot_id].pipeline is None:
                card.preview.configure(image="", text="Opening camera…")

        def worker() -> None:
            error: Optional[str] = None
            try:
                self.session.start_previews()
            except Exception as exc:  # noqa: BLE001
                logger.exception("start_previews failed")
                error = str(exc)
            self.root.after(0, lambda err=error: self._previews_started(err))

        threading.Thread(target=worker, name="app-preview-start", daemon=True).start()

    def start_slot_preview_async(self, slot_id: int) -> None:
        if self.busy:
            return
        try:
            self.cards[slot_id].sync_controls()
        except Exception:  # noqa: BLE001
            pass
        self._set_busy(True)
        self.set_status(f"Opening camera {slot_id + 1}…")
        try:
            self.cards[slot_id].preview.configure(image="", text="Opening camera…")
        except Exception:  # noqa: BLE001
            pass

        def worker() -> None:
            error: Optional[str] = None
            try:
                self.session.start_slot_preview(slot_id)
            except Exception as exc:  # noqa: BLE001
                logger.exception("start_slot_preview failed")
                error = str(exc)
            self.root.after(0, lambda err=error, sid=slot_id: self._slot_preview_started(sid, err))

        threading.Thread(
            target=worker, name=f"app-preview-slot{slot_id}", daemon=True
        ).start()

    def _previews_started(self, error: Optional[str]) -> None:
        self._set_busy(False)
        self._clear_opening_placeholders()
        for card in self.cards:
            try:
                card.sync_opened_mode_from_source()
            except Exception:  # noqa: BLE001
                pass
        live = self._live_preview_count()
        opened = len([s for s in self.session.slots if s.pipeline])
        waiting = opened - live
        if error:
            msg = f"{live} preview(s) showing video"
            if waiting > 0:
                msg += f" ({waiting} opened but no frames yet)"
            self.set_status(f"{msg}. Some failed.")
            messagebox.showerror(
                "Preview",
                error + "\n\nClose Zoom/Teams/Camera, Elgato 4K Capture Utility, "
                "and extra POC1 windows. For Elgato: HDMI ON, install ffmpeg, "
                "Refresh, then try Camera 2 Start preview alone.",
            )
        else:
            msg = f"{live} preview(s) showing video"
            if waiting > 0:
                msg += f" ({waiting} opened but no frames yet — wait or restart preview)"
            self.set_status(f"{msg}. Open Record when ready.")
        self.refresh_record_gate()

    def _slot_preview_started(self, slot_id: int, error: Optional[str]) -> None:
        self._set_busy(False)
        self._clear_opening_placeholders()
        if error:
            self.set_status(f"Camera {slot_id + 1} preview failed.")
            messagebox.showerror(
                f"Camera {slot_id + 1}",
                f"{error}\n\nClose other camera apps, then try again.",
            )
        else:
            try:
                self.cards[slot_id].sync_opened_mode_from_source()
            except Exception:  # noqa: BLE001
                pass
            self.set_status(f"Camera {slot_id + 1} is live.")
            self.root.after(3500, lambda: self._warn_slot_if_no_frames(slot_id))
        self.refresh_record_gate()

    def _warn_slot_if_no_frames(self, slot_id: int) -> None:
        if slot_id >= len(self.session.slots):
            return
        slot = self.session.slots[slot_id]
        if slot.pipeline is None or slot.get_preview_frame() is not None:
            return
        cam = slot.camera
        if cam is not None and cam.kind == "realsense":
            messagebox.showwarning(
                f"Camera {slot_id + 1}",
                "RealSense opened but no image arrived.\n\n"
                "Close Intel RealSense Viewer, use USB 3, pick 1280x720@30 bgr8, "
                "then Start preview again.",
            )
            return
        if cam is not None and cam.device_tag == "elgato":
            messagebox.showwarning(
                f"Camera {slot_id + 1}",
                "Elgato opened but no image yet.\n\n"
                "Use 1920x1080@120 mjpg (not bgr8), confirm HDMI is on, "
                "close Elgato 4K Capture Utility, then Start preview again.",
            )
            return
        messagebox.showwarning(
            f"Camera {slot_id + 1}",
            "Opened, but no image yet. Close other camera apps, check the webcam, "
            "then Start preview again.",
        )

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

    def _elgato_live_lines(self) -> list[str]:
        lines: list[str] = []
        for slot in self.session.slots:
            if not slot.armed or slot.pipeline is None:
                continue
            src = slot.pipeline.source
            if getattr(src, "device_tag", "") != "elgato":
                continue
            measured = float(getattr(src, "actual_fps", 0) or 0)
            requested = int(
                getattr(src, "requested_fps", 0)
                or getattr(src, "target_fps", 0)
                or 0
            )
            stamp = int(getattr(src, "target_fps", 0) or 0)
            if measured < 1:
                continue
            line = (
                f"{slot.prefix}: requested {requested} fps, delivering ~{measured:.0f}"
            )
            if requested >= 90 and measured < requested * 0.85:
                line += f" (HDMI source is not 120Hz; playback stamp ~{stamp})"
            lines.append(line)
        return lines

    def _prepare_high_rate_solo(self) -> bool:
        """Refuse two 1080p@≥90 takes. Allow one 120 + @30/@60 companions."""
        armed = [s for s in self.session.slots if s.armed]
        if too_many_1080p120([s.mode for s in armed]):
            high = [s.prefix for s in armed if is_fhd_high_rate(s.mode)]
            messagebox.showerror(
                "Too many 1080p@120 cameras",
                "Only one 1920x1080@90+ camera per Record.\n"
                f"Currently: {', '.join(high)}.\n\n"
                "Keep Elgato (or fake) 1080p120, and set the others to @30 or @60. "
                "Webcam + fake 1080p120 together drops frames.",
            )
            return False
        return True

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
                bd=0,
                highlightthickness=0,
            )
            tile.pack(fill="both", expand=True, padx=4, pady=(0, 4))
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
            tw = tile.winfo_width()
            th = tile.winfo_height()
            if tw < 40 or th < 40:
                continue
            rgb = bgr_to_rgb_fill(frame, tw, th)
            src = slot.pipeline.source if slot.pipeline else None
            if src is not None:
                rgb = overlay_hud(rgb, hud_lines_for_source(slot, src))
            photo = ImageTk.PhotoImage(Image.fromarray(rgb))
            self._record_photos[i] = photo
            tile.configure(image=photo, text="")

    # ── recording ────────────────────────────────────────────────────────

    def _set_busy(self, busy: bool, recording: bool = False) -> None:
        self.busy = busy
        for card in self.cards:
            card.set_locked(busy or recording)
            # After unlock, force checkbox to match slot intent (never silently uncheck).
            if not busy and not recording:
                card.restore_bag_checkbox()
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

    def _capture_bag_intent(self) -> dict[int, bool]:
        """Read bag checkboxes: RealSense SDK .db3/.bag or Elgato ROS2 *_color."""
        intent: dict[int, bool] = {}
        for card in self.cards:
            slot = self.session.slots[card.slot_id]
            want = bool(card.bag_var.get())
            cam = slot.camera
            if cam is None:
                want = False
            elif cam.kind == "realsense" or cam.device_tag == "elgato":
                pass  # keep checkbox
            else:
                want = False
            intent[card.slot_id] = want
            slot.record_bag = want
        self.session.bag_intent = dict(intent)
        return intent

    def start_recording(self) -> None:
        if self.busy or self.session.is_recording:
            return
        ok, tip = self._armed_live_ready()
        if not ok:
            messagebox.showinfo("Not ready", tip)
            return
        if not self._prepare_high_rate_solo():
            return
        try:
            self._sync_all()
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Recording", str(exc))
            return

        # Capture checkbox state BEFORE busy/lock — this is what Record will use.
        bag_intent = self._capture_bag_intent()
        rs_bag = []
        elgato_bag = []
        for i, want in bag_intent.items():
            if not want or not self.session.slots[i].armed:
                continue
            cam = self.session.slots[i].camera
            prefix = self.session.slots[i].prefix
            if cam and cam.kind == "realsense":
                rs_bag.append(prefix)
            elif cam and cam.device_tag == "elgato":
                elgato_bag.append(prefix)
        if rs_bag or elgato_bag:
            bits = []
            if rs_bag:
                bits.append(f"RealSense .db3/.bag: {', '.join(rs_bag)}")
            if elgato_bag:
                bits.append(f"Elgato ROS2 *_color: {', '.join(elgato_bag)}")
            self.set_status("Recording with " + " · ".join(bits))

        armed = [s for s in self.session.slots if s.armed]
        heavy = [
            s
            for s in armed
            if s.mode and s.mode.width * s.mode.height * s.mode.fps >= 1920 * 1080 * 60
        ]
        if len(armed) >= 2 and heavy:
            self.set_status(
                "High-rate multi-cam — encode will finish every queued frame on Stop."
            )

        self._set_busy(True)
        # Re-apply after lock so nothing in set_locked can drop the flag.
        for s in self.session.slots:
            s.record_bag = bag_intent.get(s.slot_id, False)
        self.set_status("Starting recording…")

        def worker() -> None:
            paths: list[Path] = []
            error: Optional[str] = None
            try:
                paths = self.session.start_recording_armed(bag_intent=bag_intent)
            except Exception as exc:  # noqa: BLE001
                logger.exception("record start failed")
                error = str(exc)
            self.root.after(0, lambda: self._record_started(paths, error))

        threading.Thread(target=worker, name="app-record-start", daemon=True).start()

    def _record_started(self, paths: list[Path], error: Optional[str]) -> None:
        if error:
            self._set_busy(False)
            for card in self.cards:
                card.restore_bag_checkbox()
            self.set_status("Recording did not start.")
            messagebox.showerror("Recording", error)
            return
        self._set_busy(False, recording=True)
        n = max(1, len(paths))
        extra = self._elgato_live_lines()
        tip = f"Recording {n} camera(s)…"
        if extra:
            tip += "  " + "  ·  ".join(extra)
        self.record_tip.set(tip)
        bags = [
            s
            for s in self.session.slots
            if s.armed and self.session.bag_intent.get(s.slot_id)
        ]
        if bags:
            bits = []
            for s in bags:
                if s.camera and s.camera.kind == "realsense":
                    bits.append(f"{s.prefix} (.db3/.bag)")
                elif s.camera and s.camera.device_tag == "elgato":
                    bits.append(f"{s.prefix} (*_color ROS2)")
                else:
                    bits.append(s.prefix)
            self.set_status(
                f"Recording — bag ON for {', '.join(bits)}. Press Stop when finished."
            )
        else:
            self.set_status("Recording — press Stop when finished.")
        self.notebook.select(self.record_page)

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
        for card in self.cards:
            card.restore_bag_checkbox()
        if error:
            self.set_status(f"Stop failed: {error}")
            messagebox.showerror("Stop recording", error)
            self.refresh_record_gate()
            return

        valid = [r for r in reports.values() if "error" not in r]
        ok = bool(valid) and all(r.get("no_frame_drops") for r in valid)
        bag_ok = [
            (prefix, r)
            for prefix, r in reports.items()
            if r.get("bag_recorded")
        ]
        bag_missing = [
            prefix
            for prefix, r in reports.items()
            if "error" not in r
            and self.session.bag_intent.get(
                next((s.slot_id for s in self.session.slots if s.prefix == prefix), -1),
                False,
            )
            and not r.get("bag_recorded")
        ]
        status = (
            "Saved cleanly — no frame drops."
            if ok
            else "Saved. Check Last report if something looks off."
        )
        hdmi = [r for r in valid if r.get("hdmi_not_120hz")]
        if hdmi:
            bits = []
            for report in hdmi:
                bits.append(
                    f"requested {report.get('requested_fps')} → measured "
                    f"~{float(report.get('measured_fps') or 0):.0f}, "
                    f"file stamped {report.get('container_fps')}"
                )
            status += (
                " HDMI source is not 120Hz ("
                + "; ".join(bits)
                + ") — playback stamp is measured rate, not a drop."
            )
        if any(r.get("r7_120_ok") for r in valid):
            status += " Elgato ~120 with no software drops."
        if bag_ok:
            status += f" .bag saved: {', '.join(p for p, _ in bag_ok)}."
        bag_drops = [
            (prefix, int(r.get("bag_dropped") or 0))
            for prefix, r in reports.items()
            if int(r.get("bag_dropped") or 0) > 0
        ]
        if bag_drops:
            status += (
                " Elgato ROS2 bag queue dropped frames: "
                + ", ".join(f"{p}×{n}" for p, n in bag_drops)
                + " (MP4 path separate)."
            )
        self.set_status(status)
        self.record_tip.set("Recording stopped. Arm cameras and start previews to record again.")
        if bag_missing:
            messagebox.showerror(
                "RealSense .bag missing",
                "MP4 was saved, but .bag was checked and not written for:\n"
                + ", ".join(bag_missing)
                + "\n\nClose RealSense Viewer, use USB 3, keep .bag checked, Record again.",
            )

        no_capture = []
        for slot in self.session.slots:
            report = reports.get(slot.prefix) or {}
            if "error" in report:
                continue
            if int(report.get("frames_read_by_camera") or 0) == 0:
                cam = slot.camera
                label = getattr(cam, "name", None) or "unassigned"
                no_capture.append(
                    f"Camera {slot.slot_id + 1} · prefix={slot.prefix!r} · {label}"
                )
        if no_capture:
            from poc1.bag_recorder import BUILD_ID

            messagebox.showerror(
                "No frames captured",
                "Preview was live but the Record path counted 0 frames on:\n"
                + "\n".join(f"• {line}" for line in no_capture)
                + f"\n\nSave folder: {self.session.out_dir}\n\n"
                "This is a Record-path miss (not encode skipping frames).\n"
                "1) Press Stop all previews, then Start preview again on that camera.\n"
                "2) Wait until the HUD shows live video, then Record.\n"
                "3) If it still fails: fully Exit OBS / Viewer / Elgato Utility "
                "(only if they are running), Refresh cameras, retry.\n"
                f"Build: {BUILD_ID}.",
            )

        mismatched = [
            (slot, reports.get(slot.prefix, {}))
            for slot in self.session.slots
            if reports.get(slot.prefix, {}).get("fps_mismatch")
            and int(reports.get(slot.prefix, {}).get("frames_read_by_camera") or 0) > 0
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
                if report.get("hdmi_not_120hz"):
                    continue
                if int(report.get("frames_read_by_camera") or 0) == 0:
                    continue
                drop_notes.append(
                    f"{slot.prefix}: wrote {report.get('frames_written')} of "
                    f"{report.get('frames_read_by_camera')} frames "
                    f"(requested {report.get('requested_fps')}, "
                    f"measured ~{float(report.get('measured_fps') or 0):.0f})"
                )
            if drop_notes:
                messagebox.showwarning(
                    "Some frames were skipped",
                    "Encode could not keep up (read ≠ written) on:\n\n"
                    + "\n".join(drop_notes)
                    + "\n\nHDMI ~60 is not this error. Keep companions at @30/@60; "
                    "only one 1080p@120 (Elgato) per take.",
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
                    "no_capture",
                    "r7_120_ok",
                    "hdmi_not_120hz",
                    "device_tag",
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
                    "bag_dropped",
                    "bag_frames_written",
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
        try:
            page = self.notebook.select()
        except tk.TclError:
            page = ""
        # Only redraw the visible page — Setup+Record both converting HD frames
        # was freezing Tk ("Not Responding").
        if page == str(self.setup_page):
            for card in self.cards:
                card.tick()
        elif page == str(self.record_page):
            self._tick_record_tiles()
            if self.session.is_recording:
                extra = self._elgato_live_lines()
                if extra:
                    self.record_tip.set("Recording…  " + "  ·  ".join(extra))
            elif not self.busy:
                self.refresh_record_gate()
        delay_ms = 80
        self.root.after(delay_ms, self._tick)

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
