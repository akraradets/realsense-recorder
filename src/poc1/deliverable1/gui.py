"""Advanced Deliverable 1 GUI: dynamic multi-camera recorder (R1-R6)."""
from __future__ import annotations

import argparse
import json
import logging
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Optional

import cv2
from PIL import Image, ImageTk

from poc1.preview_draw import bgr_to_rgb_fill
from poc1.deliverable1.devices import StreamMode
from poc1.deliverable1.session import CameraSlot, MultiCamSession
from poc1.deliverable2.review import show_review_prompt
from poc1.quiet import configure_app_logging, silence_opencv

configure_app_logging()
logger = logging.getLogger("poc1.d1.gui")

BG = "#0b1220"
PANEL = "#111c2e"
CARD = "#172338"
CARD_TOP = "#1e2d45"
FIELD = "#f8fafc"
INK = "#111827"
TEXT = "#e6edf7"
MUTED = "#91a4bd"
BORDER = "#31445f"
TEAL = "#14b8a6"
BLUE = "#2563eb"
RED = "#dc2626"
GREEN = "#16a34a"
AMBER = "#f59e0b"


def _button(master, text: str, command, color: str = CARD_TOP, **kwargs) -> tk.Button:
    options = {
        "bg": color,
        "fg": "#ffffff",
        "activebackground": color,
        "activeforeground": "#ffffff",
        "disabledforeground": "#718198",
        "font": ("Segoe UI Semibold", 9),
        "relief": "flat",
        "bd": 0,
        "padx": 12,
        "pady": 7,
        "cursor": "hand2",
    }
    options.update(kwargs)
    return tk.Button(master, text=text, command=command, **options)


class CameraCard(tk.Frame):
    """One dynamic camera card with device/config/preview/arming controls."""

    def __init__(self, master: tk.Misc, app: "Deliverable1App", slot: CameraSlot):
        super().__init__(
            master,
            bg=CARD,
            highlightbackground=BORDER,
            highlightthickness=1,
            bd=0,
        )
        self.app = app
        self.session = app.session
        self.slot_id = slot.slot_id
        self._photo: Optional[ImageTk.PhotoImage] = None
        self._cam_by_label: dict[str, str] = {}
        self._mode_by_label: dict[str, StreamMode] = {}

        header = tk.Frame(self, bg=CARD_TOP, padx=12, pady=9)
        header.pack(fill="x")
        tk.Label(
            header,
            text=f"CAMERA {slot.slot_id + 1}",
            bg=CARD_TOP,
            fg=TEXT,
            font=("Segoe UI Semibold", 11),
        ).pack(side="left")
        self.badge = tk.Label(
            header,
            text="IDLE",
            bg="#334155",
            fg="#dbeafe",
            font=("Segoe UI Semibold", 8),
            padx=8,
            pady=3,
        )
        self.badge.pack(side="left", padx=10)
        self.remove_btn = _button(
            header,
            "Remove",
            lambda: app.remove_camera(self.slot_id),
            color="#7f1d1d",
            padx=9,
            pady=4,
        )
        self.remove_btn.pack(side="right")
        if len(self.session.slots) <= 2:
            self.remove_btn.configure(state="disabled")

        form = tk.Frame(self, bg=CARD, padx=12, pady=10)
        form.pack(fill="x")
        form.grid_columnconfigure(1, weight=1)

        self.device_var = tk.StringVar()
        self.mode_var = tk.StringVar()
        self.prefix_var = tk.StringVar(value=slot.prefix)
        self.armed_var = tk.BooleanVar(value=slot.armed)
        self.bag_var = tk.BooleanVar(value=slot.record_bag)

        self._field_label(form, "Device", 0)
        self.device_combo = ttk.Combobox(
            form,
            textvariable=self.device_var,
            state="readonly",
            style="Readable.TCombobox",
        )
        self.device_combo.grid(row=0, column=1, columnspan=3, sticky="ew", pady=3)
        self.device_combo.bind("<<ComboboxSelected>>", self._on_device)

        self._field_label(form, "Configuration", 1)
        self.mode_combo = ttk.Combobox(
            form,
            textvariable=self.mode_var,
            state="readonly",
            style="Readable.TCombobox",
        )
        self.mode_combo.grid(row=1, column=1, columnspan=3, sticky="ew", pady=3)
        self.mode_combo.bind("<<ComboboxSelected>>", self._on_mode)

        self._field_label(form, "File prefix", 2)
        self.prefix_entry = tk.Entry(
            form,
            textvariable=self.prefix_var,
            bg=FIELD,
            fg=INK,
            insertbackground=INK,
            selectbackground=TEAL,
            selectforeground="#ffffff",
            relief="flat",
            width=14,
        )
        self.prefix_entry.grid(row=2, column=1, sticky="ew", pady=3, ipady=4)
        self.prefix_var.trace_add("write", lambda *_: self._sync_prefix())

        self.arm_check = tk.Checkbutton(
            form,
            text="Armed",
            variable=self.armed_var,
            command=self._sync_arm,
            bg=CARD,
            fg=TEXT,
            activebackground=CARD,
            activeforeground=TEXT,
            selectcolor="#0f766e",
        )
        self.arm_check.grid(row=2, column=2, padx=10)
        self.bag_check = tk.Checkbutton(
            form,
            text="RealSense .bag",
            variable=self.bag_var,
            command=self._sync_bag,
            bg=CARD,
            fg=TEXT,
            activebackground=CARD,
            activeforeground=TEXT,
            selectcolor="#0f766e",
        )
        self.bag_check.grid(row=2, column=3)

        # Fixed-height viewport: Label geometry must not resize the whole card
        # when the first camera frame arrives.
        self.preview_shell = tk.Frame(
            self,
            bg="#050a13",
            height=265,
            highlightbackground="#263750",
            highlightthickness=1,
        )
        self.preview_shell.pack(fill="x", padx=12, pady=(0, 8))
        self.preview_shell.pack_propagate(False)
        self.preview = tk.Label(
            self.preview_shell,
            text="Preview not started",
            bg="#050a13",
            fg=MUTED,
            font=("Segoe UI", 10),
            relief="flat",
        )
        self.preview.pack(fill="both", expand=True)

        actions = tk.Frame(self, bg=CARD, padx=12, pady=7)
        actions.pack(fill="x")
        self.start_btn = _button(
            actions, "Start preview", self.start_preview, color=BLUE
        )
        self.start_btn.pack(side="left")
        self.stop_btn = _button(
            actions, "Stop preview", self.stop_preview, color="#475569"
        )
        self.stop_btn.pack(side="left", padx=7)
        self.status_label = tk.Label(
            actions,
            text=slot.status,
            bg=CARD,
            fg=MUTED,
            anchor="e",
            font=("Segoe UI", 9),
        )
        self.status_label.pack(side="right", fill="x", expand=True)

    @staticmethod
    def _field_label(master: tk.Misc, text: str, row: int) -> None:
        tk.Label(
            master,
            text=text,
            bg=CARD,
            fg=MUTED,
            width=13,
            anchor="w",
            font=("Segoe UI", 9),
        ).grid(row=row, column=0, sticky="w")

    def load_devices(self, auto_assign: bool = False) -> None:
        labels = [d.label() for d in self.session.devices]
        self._cam_by_label = {d.label(): d.cam_id for d in self.session.devices}
        self.device_combo["values"] = ["Select a camera..."] + labels
        slot = self.session.slots[self.slot_id]

        if slot.camera:
            self.device_var.set(slot.camera.label())
            self._load_modes(slot)
            self._refresh_bag_enabled()
            return

        if auto_assign:
            used = {
                s.camera.cam_id
                for s in self.session.slots
                if s.slot_id != self.slot_id and s.camera is not None
            }
            # Prefer real hardware over virtual cams / synthetic fakes so
            # Cam1 does not auto-bind to OBS Virtual Camera.
            candidate = None
            for prefer in ("realsense", "elgato", "uvc", "fake"):
                for d in self.session.devices:
                    if d.cam_id in used:
                        continue
                    if "busy at scan" in d.name.lower() and prefer == "uvc":
                        # Prefer easy-to-open devices first; busy webcam still
                        # remains selectable manually in the dropdown.
                        continue
                    if prefer == "realsense" and d.kind == "realsense":
                        candidate = d
                        break
                    if prefer == "elgato" and d.device_tag == "elgato":
                        candidate = d
                        break
                    if (
                        prefer == "uvc"
                        and d.kind == "uvc"
                        and d.device_tag not in {"virtual", "realsense-uvc"}
                    ):
                        candidate = d
                        break
                    if prefer == "fake" and d.kind == "fake":
                        candidate = d
                        break
                if candidate:
                    break
            # Last resort: busy webcam / any remaining UVC.
            if candidate is None:
                for d in self.session.devices:
                    if d.cam_id in used:
                        continue
                    if d.kind == "uvc" and d.device_tag != "virtual":
                        candidate = d
                        break
            if candidate:
                self.device_var.set(candidate.label())
                self._on_device()
                return
        self.device_var.set("Select a camera...")
        self.mode_var.set("Select configuration...")
        self._refresh_bag_enabled()

    def _load_modes(self, slot: CameraSlot) -> None:
        self._mode_by_label = {m.label(): m for m in slot.available_modes}
        self.mode_combo["values"] = list(self._mode_by_label)
        if slot.mode:
            self.mode_var.set(slot.mode.label())

    def _on_device(self, *_args) -> None:
        if self.app.busy:
            return
        cam_id = self._cam_by_label.get(self.device_var.get())
        if not cam_id:
            self._refresh_bag_enabled()
            return
        try:
            slot = self.session.assign_camera(self.slot_id, cam_id)
            self.bag_var.set(bool(slot.record_bag))
            self._load_modes(slot)
            self._refresh_bag_enabled()
            self.preview.configure(image="", text="Preview not started")
            self._photo = None
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Camera assignment", str(exc))
            self.load_devices()

    def _on_mode(self, *_args) -> None:
        if self.app.busy:
            return
        mode = self._mode_by_label.get(self.mode_var.get())
        if mode is None:
            return
        try:
            self.session.set_mode(self.slot_id, mode)
            self.preview.configure(image="", text="Configuration changed — start preview")
            self._photo = None
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Camera configuration", str(exc))

    def _sync_prefix(self) -> None:
        self.session.set_prefix(self.slot_id, self.prefix_var.get())

    def _sync_arm(self) -> None:
        self.session.set_armed(self.slot_id, self.armed_var.get())

    def _sync_bag(self) -> None:
        self.session.slots[self.slot_id].record_bag = self.bag_var.get()

    def restore_bag_checkbox(self) -> None:
        slot = self.session.slots[self.slot_id]
        want = bool(slot.record_bag) or bool(
            getattr(self.session, "bag_intent", {}).get(self.slot_id, False)
        )
        if slot.camera is None or slot.camera.kind != "realsense":
            want = False
        slot.record_bag = want
        try:
            self.bag_var.set(want)
            self._refresh_bag_enabled(locked=False)
        except tk.TclError:
            pass

    def _refresh_bag_enabled(self, locked: bool = False) -> None:
        slot = self.session.slots[self.slot_id]
        is_rs = slot.camera is not None and slot.camera.kind == "realsense"
        # Do not clear record_bag when locking the UI for Record.
        if not is_rs:
            if self.bag_var.get():
                self.bag_var.set(False)
            slot.record_bag = False
        try:
            self.bag_check.configure(
                state="disabled" if locked or not is_rs else "normal"
            )
        except tk.TclError:
            pass

    def sync_controls(self) -> None:
        self._sync_prefix()
        self._sync_arm()
        self._sync_bag()

    def start_preview(self) -> None:
        if self.app.busy:
            return
        self.app.start_slot_preview_async(self.slot_id)

    def _warn_if_no_preview_frames(self) -> None:
        slot = self.session.slots[self.slot_id]
        if slot.pipeline is None:
            return
        if slot.get_preview_frame() is not None:
            return
        messagebox.showwarning(
            f"Camera {self.slot_id + 1} — no preview frames",
            "The device opened but no image arrived after a few seconds.\n\n"
            "Close other camera apps, confirm the physical connection "
            "(USB3 / HDMI), click Refresh, then Start preview again.",
        )
        self.status_label.configure(text="warning: no frames yet")

    def stop_preview(self) -> None:
        if self.app.busy:
            return
        self.session.stop_slot_preview(self.slot_id)
        self.preview.configure(image="", text="Preview stopped")
        self._photo = None

    def set_locked(self, locked: bool) -> None:
        combo_state = "disabled" if locked else "readonly"
        entry_state = "disabled" if locked else "normal"
        self.device_combo.configure(state=combo_state)
        self.mode_combo.configure(state=combo_state)
        self.prefix_entry.configure(state=entry_state)
        self.arm_check.configure(state=entry_state)
        self._refresh_bag_enabled(locked=locked)
        self.remove_btn.configure(
            state="disabled" if locked or len(self.session.slots) <= 2 else "normal"
        )

    def tick(self) -> None:
        slot = self.session.slots[self.slot_id]
        self.status_label.configure(text=slot.status)
        if slot.pipeline and slot.pipeline.camera_handler.is_recording:
            self.badge.configure(text="RECORDING", bg="#991b1b", fg="#fee2e2")
        elif slot.pipeline:
            self.badge.configure(text="LIVE", bg="#065f46", fg="#d1fae5")
        else:
            self.badge.configure(text="IDLE", bg="#334155", fg="#dbeafe")
        frame = slot.get_preview_frame()
        if frame is None:
            return
        tw = max(self.preview_shell.winfo_width() - 4, 320)
        th = max(self.preview_shell.winfo_height() - 4, 180)
        rgb = bgr_to_rgb_fill(frame, tw, th)
        self._photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.preview.configure(image=self._photo, text="")


class Deliverable1App:
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

        root.title("Deliverable 1 — Multi-Camera Recorder")
        root.configure(bg=BG)
        root.geometry("1180x820")
        root.minsize(900, 680)
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._configure_styles()

        # Bottom bar is packed first so Record/Stop remain visible at any size.
        self.action_bar = tk.Frame(root, bg="#142033", padx=16, pady=11)
        self.action_bar.pack(side="bottom", fill="x")
        self.record_btn = _button(
            self.action_bar,
            "●  RECORD ARMED CAMERAS",
            self.start_recording,
            color=RED,
            font=("Segoe UI Semibold", 10),
            padx=18,
            pady=9,
        )
        self.record_btn.pack(side="left")
        self.stop_record_btn = _button(
            self.action_bar,
            "■  STOP RECORDING",
            self.stop_recording,
            color=GREEN,
            font=("Segoe UI Semibold", 10),
            padx=18,
            pady=9,
            state="disabled",
        )
        self.stop_record_btn.pack(side="left", padx=8)
        self.status_var = tk.StringVar(
            value="Refresh cameras, configure slots, start previews, then record."
        )
        tk.Label(
            self.action_bar,
            textvariable=self.status_var,
            bg="#142033",
            fg=TEXT,
            anchor="w",
            font=("Segoe UI", 9),
        ).pack(side="left", fill="x", expand=True, padx=10)
        _button(
            self.action_bar, "Last report", self.show_report, color="#475569"
        ).pack(side="right")
        _button(
            self.action_bar, "Open folder", self.open_output_folder, color="#475569"
        ).pack(side="right", padx=7)

        header = tk.Frame(root, bg=PANEL, padx=18, pady=13)
        header.pack(fill="x")
        tk.Label(
            header,
            text="MULTI-CAMERA RECORDER",
            bg=PANEL,
            fg=TEXT,
            font=("Segoe UI Semibold", 15),
        ).pack(side="left")
        tk.Label(
            header,
            text="R1–R6  •  UVC / Elgato / RealSense",
            bg=PANEL,
            fg=MUTED,
            font=("Segoe UI", 9),
        ).pack(side="left", padx=14)
        self.count_var = tk.StringVar()
        tk.Label(
            header,
            textvariable=self.count_var,
            bg="#0f766e",
            fg="#ecfeff",
            font=("Segoe UI Semibold", 9),
            padx=10,
            pady=4,
        ).pack(side="right")

        folder = tk.Frame(root, bg=BG, padx=18, pady=10)
        folder.pack(fill="x")
        tk.Label(
            folder,
            text="SAVE TO",
            bg=BG,
            fg=MUTED,
            font=("Segoe UI Semibold", 9),
        ).pack(side="left")
        self.out_var = tk.StringVar(value=str(self.session.out_dir.resolve()))
        self.out_entry = tk.Entry(
            folder,
            textvariable=self.out_var,
            bg=FIELD,
            fg=INK,
            insertbackground=INK,
            selectbackground=TEAL,
            selectforeground="#ffffff",
            relief="flat",
            font=("Consolas", 9),
        )
        self.out_entry.pack(side="left", fill="x", expand=True, padx=10, ipady=6)
        _button(folder, "Browse…", self.browse_folder, color="#475569").pack(side="left")
        tk.Label(
            folder,
            text="compressed MP4 + report + sysmon CSV",
            bg=BG,
            fg=MUTED,
            font=("Segoe UI", 8),
        ).pack(side="left", padx=10)

        toolbar = tk.Frame(root, bg=BG, padx=18, pady=4)
        toolbar.pack(fill="x")
        self.refresh_btn = _button(
            toolbar, "↻  Refresh cameras", self.refresh_cameras, color="#475569"
        )
        self.refresh_btn.pack(side="left")
        self.add_btn = _button(
            toolbar, "+  Add camera slot", self.add_camera, color=TEAL
        )
        self.add_btn.pack(side="left", padx=7)
        self.start_all_btn = _button(
            toolbar, "▶  Start all previews", self.start_all_previews, color=BLUE
        )
        self.start_all_btn.pack(side="left")
        self.stop_all_btn = _button(
            toolbar, "■  Stop all previews", self.stop_all_previews, color="#475569"
        )
        self.stop_all_btn.pack(side="left", padx=7)

        # Scrollable two-column camera workspace.
        shell = tk.Frame(root, bg=BG)
        shell.pack(fill="both", expand=True, padx=12, pady=8)
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

        self.rebuild_cards(auto_assign=False)
        self.refresh_cameras(initial=True)
        self._tick()

    def _configure_styles(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure(
            "Readable.TCombobox",
            fieldbackground=FIELD,
            background=FIELD,
            foreground=INK,
            arrowcolor=INK,
            selectbackground=FIELD,
            selectforeground=INK,
        )
        style.map(
            "Readable.TCombobox",
            fieldbackground=[("readonly", FIELD), ("disabled", "#dbe3ed")],
            foreground=[("readonly", INK), ("disabled", "#64748b")],
            selectbackground=[("readonly", FIELD)],
            selectforeground=[("readonly", INK)],
        )
        self.root.option_add("*TCombobox*Listbox.background", FIELD)
        self.root.option_add("*TCombobox*Listbox.foreground", INK)
        self.root.option_add("*TCombobox*Listbox.selectBackground", TEAL)
        self.root.option_add("*TCombobox*Listbox.selectForeground", "#ffffff")

    def _on_mousewheel(self, event) -> None:
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def set_status(self, text: str) -> None:
        self.status_var.set(text)

    def rebuild_cards(self, auto_assign: bool = False) -> None:
        for card in self.cards:
            card.destroy()
        self.cards.clear()
        for col in range(2):
            self.cards_frame.grid_columnconfigure(col, weight=1, uniform="cards")
        for index, slot in enumerate(self.session.slots):
            card = CameraCard(self.cards_frame, self, slot)
            card.grid(
                row=index // 2,
                column=index % 2,
                sticky="nsew",
                padx=6,
                pady=6,
            )
            self.cards.append(card)
        for card in self.cards:
            # With one physical webcam, Camera 2 must remain unassigned. Auto
            # assignment is intentionally limited to the first card.
            card.load_devices(auto_assign=auto_assign)
        self.count_var.set(f"{len(self.session.slots)} camera slots")

    def add_camera(self) -> None:
        if self.busy:
            return
        try:
            self.session.add_slot()
            self.rebuild_cards(auto_assign=False)
            self.set_status(
                f"Camera slot {len(self.session.slots)} added. Select a device/config."
            )
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Add camera", str(exc))

    def remove_camera(self, slot_id: int) -> None:
        if self.busy:
            return
        if not messagebox.askyesno(
            "Remove camera slot",
            f"Remove Camera {slot_id + 1}? Its preview will be stopped.",
        ):
            return
        try:
            self.session.remove_slot(slot_id)
            self.rebuild_cards()
            self.set_status("Camera slot removed.")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Remove camera", str(exc))

    def refresh_cameras(self, initial: bool = False) -> None:
        if self.busy:
            return
        if self.session.previews_running:
            if not messagebox.askyesno(
                "Refresh cameras",
                "Refreshing requires stopping all previews. Continue?",
            ):
                return
            self.session.stop_previews()
        try:
            devices = self.session.refresh_devices(include_fake=True)
            self.rebuild_cards(auto_assign=initial)
            real = [d for d in devices if d.kind != "fake"]
            names = ", ".join(d.name for d in real[:6]) or "(none)"
            kinds = {d.kind for d in real} | {d.device_tag for d in real}
            tip = ""
            if "realsense" not in kinds:
                tip += " No RealSense SDK device yet — plug USB3 + close Viewer."
            if "elgato" not in kinds:
                tip += " No named Elgato yet — plug card + HDMI ON + close Elgato app."
            self.set_status(
                f"Found {len(real)} connected camera(s): {names}.{tip} "
                "Pick devices, Start preview, then Record."
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

    def _sync_all(self) -> None:
        self.session.set_out_dir(self.out_var.get())
        for card in self.cards:
            card.sync_controls()

    def start_all_previews(self) -> None:
        if self.busy:
            return
        self._sync_all()
        self._set_busy(True)
        self.set_status("Opening cameras…")
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

        threading.Thread(target=worker, name="d1-preview-start", daemon=True).start()

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
            target=worker, name=f"d1-preview-slot{slot_id}", daemon=True
        ).start()

    def _previews_started(self, error: Optional[str]) -> None:
        self._set_busy(False)
        running = len([s for s in self.session.slots if s.pipeline])
        if error:
            self.set_status(f"{running} preview(s) running. Some failed.")
            messagebox.showerror(
                "Some previews could not start",
                f"{error}\n\nWorking previews remain active. Close other camera apps, "
                "use 1280x720@30 mjpg for webcams.",
            )
        else:
            self.set_status(f"{running} preview(s) running. Arm cameras, then Record.")

    def _slot_preview_started(self, slot_id: int, error: Optional[str]) -> None:
        self._set_busy(False)
        if error:
            self.set_status(f"Camera {slot_id + 1} preview failed.")
            messagebox.showerror(
                f"Camera {slot_id + 1} preview failed",
                f"{error}\n\nClose other camera apps and try 1280x720@30 mjpg.",
            )
            return
        self.set_status(f"Camera {slot_id + 1} preview started.")
        self.root.after(3500, lambda sid=slot_id: self._warn_slot_if_no_frames(sid))

    def _warn_slot_if_no_frames(self, slot_id: int) -> None:
        if slot_id >= len(self.cards):
            return
        self.cards[slot_id]._warn_if_no_preview_frames()

    def stop_all_previews(self) -> None:
        if self.busy:
            return
        if self.session.is_recording:
            messagebox.showwarning("Recording active", "Stop recording first.")
            return
        self.session.stop_previews()
        for card in self.cards:
            card.preview.configure(image="", text="Preview stopped")
            card._photo = None
        self.set_status("All previews stopped.")

    def _set_busy(self, busy: bool, recording: bool = False) -> None:
        self.busy = busy
        for card in self.cards:
            card.set_locked(busy or recording)
            if not busy and not recording and hasattr(card, "restore_bag_checkbox"):
                card.restore_bag_checkbox()
        state = "disabled" if busy or recording else "normal"
        self.refresh_btn.configure(state=state)
        self.add_btn.configure(state=state)
        self.start_all_btn.configure(state=state)
        self.stop_all_btn.configure(state=state)
        self.out_entry.configure(state=state)
        self.record_btn.configure(state="disabled" if busy or recording else "normal")
        self.stop_record_btn.configure(
            state="normal" if recording and not busy else "disabled"
        )

    def _capture_bag_intent(self) -> dict[int, bool]:
        intent: dict[int, bool] = {}
        for card in self.cards:
            slot = self.session.slots[card.slot_id]
            want = bool(card.bag_var.get())
            cam = slot.camera
            if cam is None:
                want = False
            elif cam.kind == "realsense" or cam.device_tag == "elgato":
                pass
            else:
                want = False
            intent[card.slot_id] = want
            slot.record_bag = want
        self.session.bag_intent = dict(intent)
        return intent

    def start_recording(self) -> None:
        if self.busy or self.session.is_recording:
            return
        try:
            self._sync_all()
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Recording configuration", str(exc))
            return

        bag_intent = self._capture_bag_intent()

        armed = [s for s in self.session.slots if s.armed]
        heavy = [
            s
            for s in armed
            if s.mode
            and s.mode.width * s.mode.height * s.mode.fps >= 1920 * 1080 * 60
        ]
        if len(armed) >= 2 and heavy:
            names = ", ".join(s.prefix for s in heavy)
            self.set_status(
                f"High-rate multi-cam ({names}) — recording at configured FPS."
            )

        self._set_busy(True)
        for s in self.session.slots:
            s.record_bag = bag_intent.get(s.slot_id, False)
        self.set_status("Starting all armed recorders…")

        def worker() -> None:
            paths: list[Path] = []
            error: Optional[str] = None
            try:
                paths = self.session.start_recording_armed(bag_intent=bag_intent)
            except Exception as exc:  # noqa: BLE001
                logger.exception("multi-camera record start failed")
                error = str(exc)
            self.root.after(0, lambda: self._record_started(paths, error))

        threading.Thread(target=worker, name="d1-record-start", daemon=True).start()

    def _record_started(self, paths: list[Path], error: Optional[str]) -> None:
        if error:
            self._set_busy(False)
            self.set_status("Recording did not start; previews remain active.")
            messagebox.showerror(
                "Recording could not start",
                f"{error}\n\n"
                "Choose an advertised configuration, use a unique prefix for each "
                "armed camera, and verify the save folder.",
            )
            return
        self._set_busy(False, recording=True)
        names = ", ".join(path.name for path in paths)
        self.set_status(f"RECORDING {len(paths)} camera(s): {names}")
        warnings = list(getattr(self.session, "last_start_warnings", []) or [])
        if warnings:
            messagebox.showwarning(
                "RealSense .bag skipped",
                "MP4 recording started.\n\n" + "\n\n".join(warnings),
            )

    def stop_recording(self) -> None:
        if getattr(self, "_stopping", False):
            return
        if not self.session.is_recording:
            if self.busy:
                self._set_busy(False)
            return
        self._stopping = True
        self._set_busy(True, recording=True)
        self.set_status("Stopping and finalizing recordings…")

        def worker() -> None:
            reports: dict = {}
            error: Optional[str] = None
            try:
                reports = self.session.stop_recording_armed()
            except Exception as exc:  # noqa: BLE001
                logger.exception("multi-camera record stop failed")
                error = str(exc)
            self.root.after(0, lambda: self._record_stopped(reports, error))

        threading.Thread(target=worker, name="d1-record-stop", daemon=True).start()

    def _record_stopped(
        self, reports: dict[str, dict], error: Optional[str] = None
    ) -> None:
        self._stopping = False
        self.last_reports = reports
        self._set_busy(False)
        if error:
            self.set_status(f"Stop failed: {error}")
            messagebox.showerror("Stop recording", error)
            return
        valid = [r for r in reports.values() if "error" not in r]
        ok = bool(valid) and all(r.get("no_frame_drops") for r in valid)
        self.set_status(
            "Stopped — all armed cameras saved with no drops."
            if ok
            else "Stopped — files saved; open Last report to check drops/errors."
        )
        mismatched = [
            (slot, reports.get(slot.prefix, {}))
            for slot in self.session.slots
            if reports.get(slot.prefix, {}).get("fps_mismatch")
        ]
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
                dropped = int(report.get("dropped_processor_queue") or 0)
                drop_notes.append(
                    f"{slot.prefix}: read {report.get('frames_read_by_camera')} "
                    f"wrote {report.get('frames_written')} "
                    f"(processor dropped {dropped})"
                )
            if drop_notes:
                messagebox.showwarning(
                    "Frame drops detected",
                    "Some cameras could not encode as fast as frames arrived:\n\n"
                    + "\n".join(drop_notes)
                    + "\n\nFor exact fake FHD@120 with zero drops, arm ONLY the "
                    "synthetic fake camera (or run: uv run python -m poc1.proof).\n"
                    "Recording fake@120 together with another live camera often "
                    "overloads MPEG-4 encode on one PC.",
                )

        # R8 — post-save review prompt (auto-dismisses in 5 seconds).
        saved = []
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
                on_review=self.review_recording,
            )

    def _convert_mismatched_async(self, mismatched) -> None:
        self._set_busy(True)
        self.set_status("Converting frame rates in background thread…")

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

        threading.Thread(target=worker, name="d1-fps-convert", daemon=True).start()

    def _conversion_done(self, results: dict[str, bool]) -> None:
        for slot in self.session.slots:
            if slot.pipeline and slot.pipeline._last_report:
                self.last_reports[slot.prefix] = slot.pipeline._last_report
        self._set_busy(False)
        failed = [prefix for prefix, ok in results.items() if not ok]
        if failed:
            self.set_status(
                "FPS conversion finished with failures: " + ", ".join(failed)
            )
            messagebox.showwarning(
                "FPS conversion incomplete",
                "Could not rewrite: "
                + ", ".join(failed)
                + "\nConfigured FPS stamp was kept.",
            )
        else:
            self.set_status(
                "FPS conversion complete (background thread). Frame count unchanged."
            )
            messagebox.showinfo(
                "FPS conversion complete",
                "Selected recordings were rewritten to their measured frame rate "
                "on a background thread. Frame count is unchanged.",
            )

    def show_report(self) -> None:
        popup = tk.Toplevel(self.root)
        popup.title("Last multi-camera report")
        popup.geometry("760x560")
        popup.configure(bg=BG)
        controls = tk.Frame(popup, bg=BG, padx=10, pady=8)
        controls.pack(side="bottom", fill="x")
        paths = [
            (prefix, Path(str(report.get("output_path"))))
            for prefix, report in self.last_reports.items()
            if report.get("output_path")
        ]
        for prefix, path in paths:
            _button(
                controls,
                f"Review {prefix}",
                lambda p=path: self.review_recording(p),
                color=BLUE,
            ).pack(side="left", padx=(0, 7))
        _button(
            controls, "Open output folder", self.open_output_folder, color="#475569"
        ).pack(side="right")
        text = tk.Text(
            popup,
            bg="#07101d",
            fg=TEXT,
            insertbackground=TEXT,
            font=("Consolas", 9),
            wrap="none",
        )
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
                    "target_fps",
                    "measured_fps",
                    "fps_mismatch",
                    "container_fps",
                    "bag_recorded",
                    "output_path",
                    "report_path",
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
            import os
            import subprocess
            import sys

            if sys.platform == "win32":
                os.startfile(path)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Open output folder", str(exc))

    def review_recording(self, path: Path) -> None:
        """Built-in OpenCV playback; does not depend on Windows MP4 codecs."""
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            cap.release()
            messagebox.showerror(
                "Review recording",
                f"Could not decode:\n{path}\n\nThe original file was not changed.",
            )
            return

        popup = tk.Toplevel(self.root)
        popup.title(f"Review — {path.name}")
        popup.geometry("900x620")
        popup.configure(bg=BG)
        footer = tk.Frame(popup, bg=PANEL, padx=10, pady=8)
        footer.pack(side="bottom", fill="x")
        video = tk.Label(popup, bg="#000000", fg=TEXT, text="Loading…")
        video.pack(fill="both", expand=True, padx=10, pady=10)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        if fps <= 0:
            fps = 30.0
        delay_ms = max(1, int(1000 / min(fps, 120.0)))
        paused = {"value": False}
        after_id = {"value": None}
        photo = {"value": None}

        info = tk.Label(
            footer,
            text=f"{path.name}  •  {fps:.3f} fps",
            bg=PANEL,
            fg=TEXT,
            font=("Segoe UI", 9),
        )
        info.pack(side="left")

        def toggle_pause() -> None:
            paused["value"] = not paused["value"]
            pause_btn.configure(text="Resume" if paused["value"] else "Pause")

        pause_btn = _button(footer, "Pause", toggle_pause, color="#475569")
        pause_btn.pack(side="right")

        def draw_next() -> None:
            if not popup.winfo_exists():
                return
            if not paused["value"]:
                ok, frame = cap.read()
                if not ok:
                    info.configure(text=f"{path.name}  •  playback finished")
                    return
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w = rgb.shape[:2]
                max_w = max(popup.winfo_width() - 30, 320)
                max_h = max(popup.winfo_height() - 100, 240)
                scale = min(max_w / max(w, 1), max_h / max(h, 1), 1.0)
                if scale < 1.0:
                    rgb = cv2.resize(
                        rgb,
                        (max(1, int(w * scale)), max(1, int(h * scale))),
                        interpolation=cv2.INTER_AREA,
                    )
                photo["value"] = ImageTk.PhotoImage(Image.fromarray(rgb))
                video.configure(image=photo["value"], text="")
            after_id["value"] = popup.after(delay_ms, draw_next)

        def close_review() -> None:
            if after_id["value"] is not None:
                try:
                    popup.after_cancel(after_id["value"])
                except tk.TclError:
                    pass
            cap.release()
            popup.destroy()

        popup.protocol("WM_DELETE_WINDOW", close_review)
        draw_next()

    def _tick(self) -> None:
        if self._closing:
            return
        for card in list(self.cards):
            try:
                card.tick()
            except Exception:  # noqa: BLE001
                pass
        self.root.after(80, self._tick)

    def _on_close(self) -> None:
        if self.busy:
            messagebox.showwarning(
                "Operation in progress",
                "Wait for recording finalization/conversion to finish before closing.",
            )
            return
        if self.session.is_recording and not messagebox.askyesno(
            "Recording active", "Stop and save all recordings, then close?"
        ):
            return
        self._closing = True
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
    parser = argparse.ArgumentParser(
        description="Deliverable 1 dynamic multi-camera GUI (R1-R6)"
    )
    parser.add_argument(
        "--slots",
        type=int,
        default=2,
        help="Initial number of camera slots (>=2; more can be added in GUI)",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("./recordings/deliverable1")
    )
    args = parser.parse_args()
    if args.slots < 2:
        parser.error("--slots must be >= 2 (R3)")
    root = tk.Tk()
    Deliverable1App(root, n_slots=args.slots, out_dir=args.out_dir)
    root.mainloop()


if __name__ == "__main__":
    main()
