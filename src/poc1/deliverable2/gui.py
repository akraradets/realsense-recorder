"""
Deliverable 2 GUI — R8 review prompt helper is shared; this app covers R9–R10
(browse + playback + bag/bd3 export) and can also show R8-style prompts.

Launch:
  uv run python -m poc1.deliverable2.gui
  uv run poc1-d2
"""
from __future__ import annotations

import argparse
import logging
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Optional

import cv2
from PIL import Image, ImageTk

from poc1.deliverable2.export import export_to_mp4, list_media_files
from poc1.deliverable2.review import show_review_prompt
from poc1.device_enum import quiet_opencv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("poc1.d2.gui")

BG = "#0b1220"
PANEL = "#111827"
CARD = "#1e293b"
TEXT = "#e2e8f0"
MUTED = "#94a3b8"
BLUE = "#2563eb"
GREEN = "#059669"
ACCENT = "#0ea5e9"


def _button(parent, text, command, color=BLUE, **kwargs):
    return tk.Button(
        parent,
        text=text,
        command=command,
        bg=color,
        fg="#f8fafc",
        activebackground=color,
        activeforeground="#f8fafc",
        relief="flat",
        padx=12,
        pady=6,
        cursor="hand2",
        font=("Segoe UI", 9, "bold"),
        **kwargs,
    )


class Deliverable2App:
    """R9 browse/playback + R10 export (.bag / .bd3 / .db3 → MP4 H264/H265)."""

    def __init__(self, root: tk.Tk, folder: Path | None = None):
        self.root = root
        self.root.title("Deliverable 2 — Review / Playback / Export")
        self.root.geometry("1100x720")
        self.root.minsize(900, 600)
        self.root.configure(bg=BG)
        self.folder = Path(folder or "./recordings/deliverable1")
        self.files: list[Path] = []
        self.selected: Optional[Path] = None
        self._cap: Optional[cv2.VideoCapture] = None
        self._photo = None
        self._after_id = None
        self._paused = False
        self._busy = False

        self._build()
        self.refresh_list()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build(self) -> None:
        head = tk.Frame(self.root, bg=PANEL, padx=14, pady=10)
        head.pack(fill="x")
        tk.Label(
            head,
            text="DELIVERABLE 2",
            bg=PANEL,
            fg=ACCENT,
            font=("Segoe UI", 11, "bold"),
        ).pack(side="left")
        tk.Label(
            head,
            text="R8 review prompt  •  R9 browse/playback  •  R10 bag/bd3 → MP4",
            bg=PANEL,
            fg=MUTED,
            font=("Segoe UI", 9),
        ).pack(side="left", padx=12)

        bar = tk.Frame(self.root, bg=BG, padx=14, pady=8)
        bar.pack(fill="x")
        tk.Label(bar, text="FOLDER", bg=BG, fg=MUTED, font=("Segoe UI", 8, "bold")).pack(
            side="left"
        )
        self.folder_var = tk.StringVar(value=str(self.folder.resolve()))
        tk.Entry(
            bar,
            textvariable=self.folder_var,
            bg="#1e293b",
            fg=TEXT,
            insertbackground=TEXT,
            relief="flat",
            font=("Consolas", 9),
        ).pack(side="left", fill="x", expand=True, padx=8, ipady=4)
        _button(bar, "Browse…", self.browse_folder, color="#475569").pack(
            side="left", padx=(0, 6)
        )
        _button(bar, "Refresh", self.refresh_list, color=BLUE).pack(side="left")

        body = tk.Frame(self.root, bg=BG, padx=14, pady=8)
        body.pack(fill="both", expand=True)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        # File list (R9)
        left = tk.Frame(body, bg=CARD, padx=8, pady=8)
        left.grid(row=0, column=0, sticky="nsw", padx=(0, 8))
        tk.Label(
            left, text="Media files", bg=CARD, fg=TEXT, font=("Segoe UI", 10, "bold")
        ).pack(anchor="w")
        self.listbox = tk.Listbox(
            left,
            width=42,
            height=24,
            bg="#0f172a",
            fg=TEXT,
            selectbackground=BLUE,
            relief="flat",
            font=("Consolas", 9),
        )
        self.listbox.pack(fill="y", expand=True, pady=6)
        self.listbox.bind("<<ListboxSelect>>", self._on_select)
        _button(left, "Play selected", self.play_selected, color=GREEN).pack(
            fill="x", pady=2
        )
        _button(left, "Open in system player", self.open_system_player, color="#475569").pack(
            fill="x", pady=2
        )
        _button(left, "Demo R8 prompt", self.demo_review_prompt, color="#475569").pack(
            fill="x", pady=2
        )

        # Playback + export
        right = tk.Frame(body, bg=BG)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)

        self.video = tk.Label(
            right,
            bg="#000000",
            fg=MUTED,
            text="Select an MP4 (or export a bag first) to play",
            font=("Segoe UI", 11),
        )
        self.video.grid(row=0, column=0, sticky="nsew")

        controls = tk.Frame(right, bg=PANEL, padx=10, pady=8)
        controls.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        self.info_var = tk.StringVar(value="No file selected")
        tk.Label(
            controls, textvariable=self.info_var, bg=PANEL, fg=TEXT, font=("Segoe UI", 9)
        ).pack(side="left")
        self.pause_btn = _button(controls, "Pause", self.toggle_pause, color="#475569")
        self.pause_btn.pack(side="right", padx=(6, 0))
        _button(controls, "Stop", self.stop_playback, color="#991b1b").pack(side="right")

        export = tk.LabelFrame(
            right,
            text=" R10 — Export .bag / .bd3 / .db3 → MP4 ",
            bg=CARD,
            fg=TEXT,
            font=("Segoe UI", 9, "bold"),
            padx=10,
            pady=8,
        )
        export.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        tk.Label(
            export,
            text="Codec",
            bg=CARD,
            fg=MUTED,
            font=("Segoe UI", 8, "bold"),
        ).grid(row=0, column=0, sticky="w")
        self.codec_var = tk.StringVar(value="h264")
        ttk.Combobox(
            export,
            textvariable=self.codec_var,
            values=["h264", "h265"],
            state="readonly",
            width=10,
        ).grid(row=0, column=1, sticky="w", padx=8)
        _button(export, "Export selected → MP4", self.export_selected, color=BLUE).grid(
            row=0, column=2, padx=8
        )
        _button(
            export, "Browse file to export…", self.export_browse, color="#475569"
        ).grid(row=0, column=3, padx=4)
        self.export_status = tk.Label(
            export, text="", bg=CARD, fg=MUTED, font=("Segoe UI", 8), anchor="w"
        )
        self.export_status.grid(row=1, column=0, columnspan=4, sticky="ew", pady=(8, 0))

        self.status = tk.Label(
            self.root,
            text="Ready",
            bg=PANEL,
            fg=MUTED,
            anchor="w",
            padx=14,
            font=("Segoe UI", 9),
        )
        self.status.pack(fill="x", side="bottom")

    def set_status(self, text: str) -> None:
        self.status.configure(text=text)

    def browse_folder(self) -> None:
        chosen = filedialog.askdirectory(initialdir=self.folder_var.get())
        if chosen:
            self.folder_var.set(chosen)
            self.folder = Path(chosen)
            self.refresh_list()

    def refresh_list(self) -> None:
        self.folder = Path(self.folder_var.get())
        self.files = list_media_files(self.folder)
        self.listbox.delete(0, tk.END)
        for path in self.files:
            self.listbox.insert(tk.END, path.name)
        self.set_status(f"{len(self.files)} media file(s) in {self.folder}")

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
        if path.suffix.lower() in {".bag", ".bd3", ".db3"}:
            messagebox.showinfo(
                "Playback",
                f"{path.suffix} files need export to MP4 first (R10).\n"
                "Use Export selected → MP4, then play the result.",
            )
            return
        self._start_playback(path)

    def open_system_player(self) -> None:
        if self.selected is None:
            return
        path = self.selected
        if path.suffix.lower() not in {".mp4", ".avi", ".mkv"}:
            messagebox.showinfo(
                "System player",
                "Open exported MP4 files with the system player.\n"
                "Export bags first if needed.",
            )
            return
        import os
        import subprocess
        import sys

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
        self.info_var.set(f"{path.name}  •  {fps:.2f} fps")
        self.set_status(f"Playing {path.name}")
        self._draw_next()

    def _draw_next(self) -> None:
        if self._cap is None:
            return
        if not self._paused:
            ok, frame = self._cap.read()
            if not ok:
                self.set_status("Playback finished")
                self.info_var.set(self.info_var.get() + "  •  end")
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
        self._after_id = self.root.after(self._delay, self._draw_next)

    def toggle_pause(self) -> None:
        self._paused = not self._paused
        self.pause_btn.configure(text="Resume" if self._paused else "Pause")

    def stop_playback(self) -> None:
        if self._after_id is not None:
            try:
                self.root.after_cancel(self._after_id)
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
            initialdir=self.folder_var.get(),
            filetypes=[
                ("Bags / video", "*.bag *.bd3 *.db3 *.mp4 *.avi *.mkv"),
                ("RealSense bag", "*.bag"),
                ("BD3 / DB3", "*.bd3 *.db3"),
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
        self.set_status(f"Exporting {source.name} → {out.name}")

        def worker() -> None:
            result = export_to_mp4(
                source,
                out,
                codec=codec,
                on_progress=lambda m: self.root.after(
                    0, lambda: self.export_status.configure(text=m)
                ),
            )
            self.root.after(0, lambda: self._export_done(result))

        threading.Thread(target=worker, name="d2-export", daemon=True).start()

    def _export_done(self, result) -> None:
        self._busy = False
        if not result.ok:
            self.export_status.configure(text=result.message, fg="#fca5a5")
            messagebox.showerror("Export failed", result.message)
            self.set_status("Export failed")
            return
        self.export_status.configure(
            text=f"{result.message}  ({result.codec_label})", fg="#86efac"
        )
        self.set_status(result.message)
        self.refresh_list()
        if result.output_path and result.output_path.is_file():
            if messagebox.askyesno(
                "Export complete",
                f"Saved:\n{result.output_path}\n\nPlay it now?",
            ):
                self.selected = result.output_path
                self._start_playback(result.output_path)

    def demo_review_prompt(self) -> None:
        """Manual R8 demo from this window (also wired into D1 after save)."""
        mp4s = [p for p in self.files if p.suffix.lower() == ".mp4"]
        show_review_prompt(self.root, mp4s[:3], on_review=self._start_playback)

    def _on_close(self) -> None:
        if self._busy:
            messagebox.showwarning("Busy", "Wait for export to finish.")
            return
        self.stop_playback()
        self.root.destroy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Deliverable 2 — review / playback / export")
    parser.add_argument(
        "--folder",
        type=Path,
        default=Path("./recordings/deliverable1"),
        help="Initial media folder",
    )
    args = parser.parse_args()
    root = tk.Tk()
    Deliverable2App(root, folder=args.folder)
    root.mainloop()


if __name__ == "__main__":
    main()
