"""Library page — browse, play, export (R9–R10)."""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import TYPE_CHECKING, Optional

import cv2
from PIL import Image, ImageTk

from poc1.app.theme import (
    ACCENT,
    BG,
    BLUE,
    BORDER,
    FIELD,
    GREEN,
    INK,
    MUTED,
    PANEL,
    SURFACE,
    button,
)
from poc1.deliverable2.export import export_to_mp4, list_media_files
from poc1.device_enum import quiet_opencv

if TYPE_CHECKING:
    from poc1.app.gui import UnifiedApp


class LibraryPage(tk.Frame):
    def __init__(self, master: tk.Misc, app: "UnifiedApp"):
        super().__init__(master, bg=BG)
        self.app = app
        self.files: list[Path] = []
        self.selected: Optional[Path] = None
        self._cap: Optional[cv2.VideoCapture] = None
        self._photo = None
        self._after_id = None
        self._paused = False
        self._busy = False
        self._delay = 33
        self._build()

    def _build(self) -> None:
        tip = tk.Frame(self, bg=SURFACE, padx=16, pady=10)
        tip.pack(fill="x")
        tk.Label(
            tip,
            text="Library: select MP4 → Play. Select RealSense .db3/.bag → Export to MP4 → Play. "
            "Record saves MP4 + .db3 in the folder; JSON/CSV are in meta/.",
            bg=SURFACE,
            fg=MUTED,
            font=("Segoe UI", 9),
            wraplength=900,
            justify="left",
        ).pack(anchor="w")

        bar = tk.Frame(self, bg=BG, padx=16, pady=8)
        bar.pack(fill="x")
        tk.Label(bar, text="Folder", bg=BG, fg=MUTED, font=("Segoe UI Semibold", 9)).pack(
            side="left"
        )
        self.folder_var = tk.StringVar()
        tk.Entry(
            bar,
            textvariable=self.folder_var,
            bg=FIELD,
            fg=INK,
            relief="solid",
            bd=1,
            font=("Segoe UI", 9),
        ).pack(side="left", fill="x", expand=True, padx=8, ipady=4)
        button(bar, "Browse…", self.browse_folder, color="#475569").pack(
            side="left", padx=(0, 6)
        )
        button(bar, "Refresh", self.refresh_list, color=BLUE).pack(side="left")

        body = tk.Frame(self, bg=BG, padx=16, pady=8)
        body.pack(fill="both", expand=True)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        left = tk.Frame(body, bg=PANEL, highlightbackground=BORDER, highlightthickness=1)
        left.grid(row=0, column=0, sticky="nsw", padx=(0, 10))
        tk.Label(
            left, text="Media", bg=PANEL, fg=INK, font=("Segoe UI Semibold", 10), padx=10, pady=8
        ).pack(anchor="w")
        self.listbox = tk.Listbox(
            left,
            width=36,
            height=22,
            bg=FIELD,
            fg=INK,
            selectbackground=ACCENT,
            selectforeground="#ffffff",
            relief="flat",
            font=("Segoe UI", 9),
            activestyle="none",
        )
        self.listbox.pack(fill="y", expand=True, padx=8, pady=(0, 8))
        self.listbox.bind("<<ListboxSelect>>", self._on_select)
        button(left, "Play", self.play_selected, color=GREEN).pack(
            fill="x", padx=8, pady=2
        )
        button(left, "Open in system player", self.open_system_player, color="#475569").pack(
            fill="x", padx=8, pady=(2, 10)
        )

        right = tk.Frame(body, bg=BG)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)

        self.video = tk.Label(
            right,
            bg="#0b1220",
            fg="#94a3b8",
            text="Select a video to play",
            font=("Segoe UI", 11),
        )
        self.video.grid(row=0, column=0, sticky="nsew")

        controls = tk.Frame(right, bg=PANEL, padx=12, pady=8)
        controls.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        self.info_var = tk.StringVar(value="No file selected")
        tk.Label(
            controls, textvariable=self.info_var, bg=PANEL, fg=INK, font=("Segoe UI", 9)
        ).pack(side="left")
        self.pause_btn = button(controls, "Pause", self.toggle_pause, color="#475569")
        self.pause_btn.pack(side="right", padx=(6, 0))
        button(controls, "Stop", self.stop_playback, color="#991b1b").pack(side="right")

        export = tk.LabelFrame(
            right,
            text=" Export to MP4 (H.264 or H.265 when available) ",
            bg=PANEL,
            fg=INK,
            font=("Segoe UI Semibold", 9),
            padx=12,
            pady=10,
        )
        export.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        tk.Label(export, text="Codec", bg=PANEL, fg=MUTED).grid(row=0, column=0, sticky="w")
        self.codec_var = tk.StringVar(value="h264")
        ttk.Combobox(
            export,
            textvariable=self.codec_var,
            values=["h264", "h265"],
            state="readonly",
            width=10,
            style="App.TCombobox",
        ).grid(row=0, column=1, sticky="w", padx=8)
        button(export, "Export selected", self.export_selected, color=BLUE).grid(
            row=0, column=2, padx=8
        )
        button(export, "Choose file…", self.export_browse, color="#475569").grid(
            row=0, column=3
        )
        tk.Label(
            export,
            text="Export selected bag → NEW {name}_h264.mp4 (decoded). "
            "RealSense = .db3/.bag via SDK. Elgato = *_color folder. "
            "Record MP4 is kept. Install ffmpeg for fallbacks.",
            bg=PANEL,
            fg=MUTED,
            font=("Segoe UI", 8),
            wraplength=900,
            justify="left",
        ).grid(row=1, column=0, columnspan=4, sticky="w", pady=(8, 0))
        self.export_status = tk.Label(
            export, text="", bg=PANEL, fg=MUTED, font=("Segoe UI", 8), anchor="w"
        )
        self.export_status.grid(row=2, column=0, columnspan=4, sticky="ew", pady=(4, 0))

    def sync_folder(self, folder: Path) -> None:
        self.folder_var.set(str(Path(folder).resolve()))
        self.refresh_list()
        try:
            from poc1.deliverable2.export import _find_ffmpeg

            if _find_ffmpeg() is None:
                self.export_status.configure(
                    text="Note: ffmpeg not on PATH — install it for export fallbacks / Elgato names."
                )
        except Exception:  # noqa: BLE001
            pass

    def browse_folder(self) -> None:
        chosen = filedialog.askdirectory(initialdir=self.folder_var.get() or ".")
        if chosen:
            self.folder_var.set(chosen)
            self.refresh_list()

    def refresh_list(self) -> None:
        folder = Path(self.folder_var.get() or ".")
        self.files = list_media_files(folder)
        self.listbox.delete(0, tk.END)
        for path in self.files:
            self.listbox.insert(tk.END, path.name)
        self.app.set_status(f"Library: {len(self.files)} file(s) in {folder}")

    def select_path(self, path: Path) -> None:
        path = Path(path)
        self.sync_folder(path.parent)
        try:
            idx = self.files.index(path)
        except ValueError:
            self.refresh_list()
            try:
                idx = self.files.index(path)
            except ValueError:
                return
        self.listbox.selection_clear(0, tk.END)
        self.listbox.selection_set(idx)
        self.listbox.see(idx)
        self.selected = path
        self.info_var.set(str(path))
        if path.suffix.lower() in {".mp4", ".avi", ".mkv"}:
            self.play_selected()

    def _on_select(self, *_args) -> None:
        idxs = self.listbox.curselection()
        if not idxs:
            return
        self.selected = self.files[int(idxs[0])]
        self.info_var.set(str(self.selected))

    def play_selected(self) -> None:
        if self.selected is None:
            idxs = self.listbox.curselection()
            if not idxs:
                messagebox.showinfo("Playback", "Select a media file first.")
                return
            self.selected = self.files[int(idxs[0])]
        path = self.selected

        # Elgato ROS2 bag folder (*_color) — not a playable video.
        if path.is_dir() and (
            path.name.endswith("_color") or (path / "metadata.yaml").is_file()
        ):
            sibling = self._sibling_record_mp4_for_bag(path)
            if sibling is not None:
                if messagebox.askyesno(
                    "Playback",
                    f"{path.name} is an Elgato ROS2 color bag folder "
                    "(not a video file).\n\n"
                    f"A matching Record MP4 exists:\n{sibling.name}\n\n"
                    "Play that original MP4 now?\n\n"
                    "(Cancel = stay here. Use Export selected to decode the bag "
                    "to a NEW mp4; Record MP4 is kept.)",
                ):
                    self._start_playback(sibling)
                return
            if messagebox.askyesno(
                "Playback",
                f"{path.name} is a ROS2 bag folder and cannot play directly.\n\n"
                "Export it to a NEW MP4 (decoded from the bag), then play?",
            ):
                self._run_export(path)
            return

        if path.suffix.lower() in {".bag", ".bd3", ".db3"}:
            sibling = path.with_suffix(".mp4")
            if sibling.is_file() and sibling.stat().st_size > 32:
                if messagebox.askyesno(
                    "Playback",
                    f"{path.name} is a bag/SDK file and cannot play directly.\n\n"
                    f"A matching Record MP4 exists:\n{sibling.name}\n\n"
                    "Play that original MP4 now?\n\n"
                    "(Cancel = stay here. Export selected decodes the .db3 to a NEW mp4 "
                    "and does not replace the Record file.)",
                ):
                    self._start_playback(sibling)
                return
            if messagebox.askyesno(
                "Playback",
                f"{path.name} cannot play directly.\n\n"
                "Export it to a NEW MP4 (decoded from the bag; Record MP4 is kept), then play?",
            ):
                self._run_export(path)
            return
        self._start_playback(path)

    @staticmethod
    def _sibling_record_mp4_for_bag(path: Path) -> Optional[Path]:
        """Record MP4 beside a RealSense .db3 or Elgato *_color folder."""
        path = Path(path)
        candidates: list[Path] = []
        if path.is_dir():
            name = path.name
            if name.endswith("_color"):
                candidates.append(path.parent / f"{name[:-6]}.mp4")
            candidates.append(path.parent / f"{path.name}.mp4")
        else:
            candidates.append(path.with_suffix(".mp4"))
            stem = path.stem
            if stem.endswith("_color"):
                candidates.append(path.parent / f"{stem[:-6]}.mp4")
        for candidate in candidates:
            if candidate.is_file() and candidate.stat().st_size > 32:
                return candidate
        return None

    def open_system_player(self) -> None:
        if self.selected is None:
            return
        path = self.selected
        if path.suffix.lower() not in {".mp4", ".avi", ".mkv"}:
            messagebox.showinfo("System player", "Open an MP4 (export bags first if needed).")
            return
        try:
            if sys.platform == "win32":
                os.startfile(path)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("System player", str(exc))

    def _start_playback(self, path: Path) -> None:
        self.stop_playback()
        with quiet_opencv():
            cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            cap.release()
            messagebox.showerror("Playback", f"Could not open:\n{path}")
            return
        self._cap = cap
        self._paused = False
        self.pause_btn.configure(text="Pause")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0) or 30.0
        self._delay = max(1, int(1000 / min(fps, 120.0)))
        self.info_var.set(f"{path.name}  •  {fps:.1f} fps")
        self.app.set_status(f"Playing {path.name}")
        self._draw_next()

    def _draw_next(self) -> None:
        if self._cap is None:
            return
        if not self._paused:
            ok, frame = self._cap.read()
            if not ok:
                self.app.set_status("Playback finished")
                return
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            max_w = max(self.video.winfo_width(), 320)
            max_h = max(self.video.winfo_height(), 240)
            scale = min(max_w / max(w, 1), max_h / max(h, 1), 1.0)
            if scale < 1.0:
                rgb = cv2.resize(
                    rgb,
                    (max(1, int(w * scale)), max(1, int(h * scale))),
                    interpolation=cv2.INTER_AREA,
                )
            self._photo = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.video.configure(image=self._photo, text="")
        self._after_id = self.after(self._delay, self._draw_next)

    def toggle_pause(self) -> None:
        self._paused = not self._paused
        self.pause_btn.configure(text="Resume" if self._paused else "Pause")

    def stop_playback(self) -> None:
        if self._after_id is not None:
            try:
                self.after_cancel(self._after_id)
            except tk.TclError:
                pass
            self._after_id = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self.video.configure(image="", text="Playback stopped")
        self._photo = None

    def export_selected(self) -> None:
        if self.selected is None:
            idxs = self.listbox.curselection()
            if not idxs:
                messagebox.showinfo("Export", "Select a .bag / .bd3 / .db3 (or video) file.")
                return
            self.selected = self.files[int(idxs[0])]
        self._run_export(self.selected)

    def export_browse(self) -> None:
        path = filedialog.askopenfilename(
            initialdir=self.folder_var.get() or ".",
            filetypes=[
                ("Bags / video", "*.bag *.bd3 *.db3 *.mp4 *.avi *.mkv"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self._run_export(Path(path))

    def _run_export(self, source: Path) -> None:
        if self._busy:
            return
        codec = self.codec_var.get().strip().lower() or "h264"
        out = source.with_name(f"{source.stem}_{codec}.mp4")
        self._busy = True
        self.export_status.configure(text=f"Exporting {source.name}…", fg=ACCENT)

        def worker() -> None:
            result = export_to_mp4(
                source,
                out,
                codec=codec,
                on_progress=lambda m: self.after(
                    0, lambda: self.export_status.configure(text=m)
                ),
            )
            self.after(0, lambda: self._export_done(result))

        threading.Thread(target=worker, name="app-export", daemon=True).start()

    def _export_done(self, result) -> None:
        self._busy = False
        if not result.ok:
            self.export_status.configure(text=result.message, fg="#b91c1c")
            messagebox.showerror("Export failed", result.message)
            return
        self.export_status.configure(
            text=f"{result.message}  ({result.codec_label})", fg=GREEN
        )
        self.refresh_list()
        if result.output_path and result.output_path.is_file():
            if messagebox.askyesno("Export complete", "Play the new MP4 now?"):
                self.select_path(result.output_path)
