"""Setup-page camera card for the unified app."""
from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from typing import TYPE_CHECKING, Optional

import cv2
from PIL import Image, ImageTk

from poc1.app.theme import (
    ACCENT,
    ACCENT_SOFT,
    BORDER,
    BLUE,
    FIELD,
    INK,
    MUTED,
    PANEL,
    SURFACE,
    button,
)
from poc1.deliverable1.devices import StreamMode
from poc1.deliverable1.session import CameraSlot

if TYPE_CHECKING:
    from poc1.app.gui import UnifiedApp


class CameraCard(tk.Frame):
    """One camera lane: device, config, arm, optional RealSense .bag."""

    def __init__(self, master: tk.Misc, app: "UnifiedApp", slot: CameraSlot):
        super().__init__(
            master,
            bg=PANEL,
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

        head = tk.Frame(self, bg=SURFACE, padx=14, pady=10)
        head.pack(fill="x")
        tk.Label(
            head,
            text=f"Camera {slot.slot_id + 1}",
            bg=SURFACE,
            fg=INK,
            font=("Segoe UI Semibold", 12),
        ).pack(side="left")
        self.badge = tk.Label(
            head,
            text="Idle",
            bg="#e2e8f0",
            fg=INK,
            font=("Segoe UI Semibold", 8),
            padx=8,
            pady=2,
        )
        self.badge.pack(side="left", padx=10)
        self.remove_btn = button(
            head,
            "Remove",
            lambda: app.remove_camera(self.slot_id),
            color="#991b1b",
            padx=10,
            pady=4,
        )
        self.remove_btn.pack(side="right")
        if len(self.session.slots) <= 2:
            self.remove_btn.configure(state="disabled")

        form = tk.Frame(self, bg=PANEL, padx=14, pady=10)
        form.pack(fill="x")
        form.grid_columnconfigure(1, weight=1)

        self.device_var = tk.StringVar()
        self.mode_var = tk.StringVar()
        self.prefix_var = tk.StringVar(value=slot.prefix)
        self.armed_var = tk.BooleanVar(value=slot.armed)
        self.bag_var = tk.BooleanVar(value=slot.record_bag)

        self._label(form, "Device", 0)
        self.device_combo = ttk.Combobox(
            form, textvariable=self.device_var, state="readonly", style="App.TCombobox"
        )
        self.device_combo.grid(row=0, column=1, columnspan=3, sticky="ew", pady=3)
        self.device_combo.bind("<<ComboboxSelected>>", self._on_device)

        self._label(form, "Configuration", 1)
        self.mode_combo = ttk.Combobox(
            form, textvariable=self.mode_var, state="readonly", style="App.TCombobox"
        )
        self.mode_combo.grid(row=1, column=1, columnspan=3, sticky="ew", pady=3)
        self.mode_combo.bind("<<ComboboxSelected>>", self._on_mode)

        self._label(form, "File prefix", 2)
        self.prefix_entry = tk.Entry(
            form,
            textvariable=self.prefix_var,
            bg=FIELD,
            fg=INK,
            insertbackground=INK,
            relief="solid",
            bd=1,
            width=12,
        )
        self.prefix_entry.grid(row=2, column=1, sticky="w", pady=3, ipady=3)
        self.prefix_var.trace_add("write", lambda *_: self._sync_prefix())

        # R5 — arming (Setup only; Library has no arm checkbox by design).
        self.arm_check = tk.Checkbutton(
            form,
            text="Armed (record this camera)",
            variable=self.armed_var,
            command=self._sync_arm,
            bg=PANEL,
            fg=INK,
            activebackground=PANEL,
            activeforeground=INK,
            selectcolor=ACCENT_SOFT,
            font=("Segoe UI Semibold", 9),
        )
        self.arm_check.grid(row=2, column=2, columnspan=2, sticky="w", padx=8)

        # Honest copy: bag write only; .db3 is Library export/import.
        self.bag_check = tk.Checkbutton(
            form,
            text="Also save RealSense .bag (with MP4)",
            variable=self.bag_var,
            command=self._sync_bag,
            bg=PANEL,
            fg=INK,
            activebackground=PANEL,
            activeforeground=INK,
            selectcolor=ACCENT_SOFT,
            font=("Segoe UI Semibold", 9),
        )
        self.bag_check.grid(row=3, column=1, columnspan=3, sticky="w", pady=(4, 0))
        tk.Label(
            form,
            text="Pick a [realsense] device for .bag. .db3 is ROS2-only (Library export, not live record).",
            bg=PANEL,
            fg=MUTED,
            font=("Segoe UI", 8),
        ).grid(row=4, column=1, columnspan=3, sticky="w", pady=(0, 4))

        self.preview_shell = tk.Frame(
            self, bg="#0b1220", height=200, highlightbackground=BORDER, highlightthickness=1
        )
        self.preview_shell.pack(fill="x", padx=14, pady=(0, 8))
        self.preview_shell.pack_propagate(False)
        self.preview = tk.Label(
            self.preview_shell,
            text="Start preview to see live video",
            bg="#0b1220",
            fg="#94a3b8",
            font=("Segoe UI", 10),
        )
        self.preview.pack(fill="both", expand=True)

        actions = tk.Frame(self, bg=PANEL, padx=14, pady=8)
        actions.pack(fill="x")
        self.start_btn = button(actions, "Start preview", self.start_preview, color=BLUE)
        self.start_btn.pack(side="left")
        self.stop_btn = button(
            actions, "Stop preview", self.stop_preview, color="#475569"
        )
        self.stop_btn.pack(side="left", padx=6)
        self.status_label = tk.Label(
            actions, text=slot.status, bg=PANEL, fg=MUTED, anchor="e", font=("Segoe UI", 9)
        )
        self.status_label.pack(side="right", fill="x", expand=True)

    @staticmethod
    def _label(master: tk.Misc, text: str, row: int) -> None:
        tk.Label(
            master, text=text, bg=PANEL, fg=MUTED, width=12, anchor="w", font=("Segoe UI", 9)
        ).grid(row=row, column=0, sticky="w")

    def load_devices(self, auto_assign: bool = False) -> None:
        labels = [d.label() for d in self.session.devices]
        self._cam_by_label = {d.label(): d.cam_id for d in self.session.devices}
        self.device_combo["values"] = ["Select a camera…"] + labels
        slot = self.session.slots[self.slot_id]

        if slot.camera:
            self.device_var.set(slot.camera.label())
            self._load_modes(slot)
            return

        if auto_assign:
            used = {
                s.camera.cam_id
                for s in self.session.slots
                if s.slot_id != self.slot_id and s.camera is not None
            }
            candidate = None
            for prefer in ("realsense", "elgato", "uvc", "fake"):
                for d in self.session.devices:
                    if d.cam_id in used:
                        continue
                    if "busy at scan" in d.name.lower() and prefer == "uvc":
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
        self.device_var.set("Select a camera…")
        self.mode_var.set("Select configuration…")

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
            return
        try:
            slot = self.session.assign_camera(self.slot_id, cam_id)
            self._load_modes(slot)
            self.preview.configure(image="", text="Start preview to see live video")
            self._photo = None
            self.app.refresh_record_gate()
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Camera", str(exc))
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
            self.app.refresh_record_gate()
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Configuration", str(exc))

    def _sync_prefix(self) -> None:
        self.session.set_prefix(self.slot_id, self.prefix_var.get())

    def _sync_arm(self) -> None:
        self.session.set_armed(self.slot_id, self.armed_var.get())
        self.app.refresh_record_gate()

    def _sync_bag(self) -> None:
        self.session.slots[self.slot_id].record_bag = self.bag_var.get()

    def sync_controls(self) -> None:
        self._sync_prefix()
        self._sync_arm()
        self._sync_bag()

    def start_preview(self) -> None:
        if self.app.busy:
            return
        try:
            self.session.start_slot_preview(self.slot_id)
            self.app.set_status(f"Camera {self.slot_id + 1} is live.")
            self.app.root.after(3500, self._warn_if_no_preview_frames)
            self.app.refresh_record_gate()
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror(
                f"Camera {self.slot_id + 1}",
                f"{exc}\n\nClose other camera apps, check USB/HDMI, then try again.",
            )

    def _warn_if_no_preview_frames(self) -> None:
        slot = self.session.slots[self.slot_id]
        if slot.pipeline is None or slot.get_preview_frame() is not None:
            return
        messagebox.showwarning(
            f"Camera {self.slot_id + 1}",
            "Opened, but no image yet. Check the cable / HDMI signal, then Start preview again.",
        )

    def stop_preview(self) -> None:
        if self.app.busy:
            return
        self.session.stop_slot_preview(self.slot_id)
        self.preview.configure(image="", text="Preview stopped")
        self._photo = None
        self.app.refresh_record_gate()

    def set_locked(self, locked: bool) -> None:
        combo = "disabled" if locked else "readonly"
        entry = "disabled" if locked else "normal"
        self.device_combo.configure(state=combo)
        self.mode_combo.configure(state=combo)
        self.prefix_entry.configure(state=entry)
        self.arm_check.configure(state=entry)
        self.bag_check.configure(state=entry)
        self.remove_btn.configure(
            state="disabled" if locked or len(self.session.slots) <= 2 else "normal"
        )

    def tick(self) -> None:
        slot = self.session.slots[self.slot_id]
        self.status_label.configure(text=slot.status)
        try:
            if slot.pipeline and slot.pipeline.camera_handler.is_recording:
                self.badge.configure(text="Recording", bg="#fecaca", fg="#7f1d1d")
            elif slot.pipeline:
                self.badge.configure(text="Live", bg=ACCENT_SOFT, fg=ACCENT)
            else:
                self.badge.configure(text="Idle", bg="#e2e8f0", fg=INK)
        except tk.TclError:
            return

        frame = slot.get_preview_frame()
        if frame is None:
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        max_w = max(self.preview_shell.winfo_width() - 6, 280)
        max_h = 190
        scale = min(max_w / max(w, 1), max_h / max(h, 1), 1.0)
        if scale < 1.0:
            rgb = cv2.resize(
                rgb,
                (max(1, int(w * scale)), max(1, int(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
        self._photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.preview.configure(image=self._photo, text="")
