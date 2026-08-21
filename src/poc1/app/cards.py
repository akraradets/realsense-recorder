"""Setup-page camera card for the unified app."""
from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from typing import TYPE_CHECKING, Optional

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
from poc1.preview_draw import bgr_to_rgb_fill, hud_lines_for_source, overlay_hud
from poc1.deliverable1.devices import StreamMode, pick_auto_camera_for_slot
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

        # Written on Record only (never on Start preview). New SDKs save .db3.
        self.bag_check = tk.Checkbutton(
            form,
            text="Also save bag with MP4 (RealSense: Intel .db3/.bag · Elgato: ROS2 *_color)",
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
            text="On Record only. RealSense: Intel SDK .db3/.bag (USB3, close Viewer). "
            "Elgato: ROS2 color folder *_color (not Intel .bag). "
            "For true 120fps set mirrorless HDMI to 1080p120.",
            bg=PANEL,
            fg=MUTED,
            font=("Segoe UI", 8),
        ).grid(row=4, column=1, columnspan=3, sticky="w", pady=(0, 4))

        self.preview_shell = tk.Frame(
            self, bg="#0b1220", height=260, highlightbackground=BORDER, highlightthickness=1
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
            self._refresh_bag_enabled()
            return

        if auto_assign:
            used = {
                s.camera.cam_id
                for s in self.session.slots
                if s.slot_id != self.slot_id and s.camera is not None
            }
            candidate = pick_auto_camera_for_slot(
                self.slot_id, self.session.devices, used
            )
            if candidate:
                self.device_var.set(candidate.label())
                self._on_device()
                return
        self.device_var.set("Select a camera…")
        self.mode_var.set("Select configuration…")
        self._refresh_bag_enabled()

    def _load_modes(self, slot: CameraSlot) -> None:
        self._mode_by_label = {m.label(): m for m in slot.available_modes}
        self.mode_combo["values"] = list(self._mode_by_label)
        if slot.mode:
            self.mode_var.set(slot.mode.label())

    def sync_opened_mode_from_source(self) -> None:
        """After Elgato preview opens, align Setup dropdown with actual size/stamp FPS."""
        slot = self.session.slots[self.slot_id]
        if slot.pipeline is None:
            return
        src = slot.pipeline.source
        if getattr(src, "device_tag", "") != "elgato":
            return
        w = int(getattr(src, "width", 0) or 0)
        h = int(getattr(src, "height", 0) or 0)
        fps = int(getattr(src, "target_fps", 0) or 0)
        fmt = str(getattr(src, "pixel_format", "mjpg") or "mjpg")
        if w < 1 or h < 1 or fps < 1:
            return
        opened = StreamMode(w, h, fps, fmt)
        if slot.mode == opened:
            return
        modes = list(slot.available_modes)
        if opened not in modes:
            modes.insert(0, opened)
            slot.available_modes = modes
        slot.mode = opened
        self._load_modes(slot)

    def _on_device(self, *_args) -> None:
        if self.app.busy:
            return
        cam_id = self._cam_by_label.get(self.device_var.get())
        if not cam_id:
            self._refresh_bag_enabled()
            return
        try:
            slot = self.session.assign_camera(self.slot_id, cam_id)
            self.prefix_var.set(slot.prefix)
            self.bag_var.set(bool(slot.record_bag))
            self._load_modes(slot)
            self._refresh_bag_enabled()
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

    def set_armed_ui(self, armed: bool) -> None:
        self.armed_var.set(bool(armed))
        self.session.set_armed(self.slot_id, bool(armed))

    def _sync_bag(self) -> None:
        self.session.slots[self.slot_id].record_bag = self.bag_var.get()

    def restore_bag_checkbox(self) -> None:
        """Keep the checkbox matching the slot intent after busy/unlock."""
        slot = self.session.slots[self.slot_id]
        want = bool(slot.record_bag) or bool(
            getattr(self.session, "bag_intent", {}).get(self.slot_id, False)
        )
        if slot.camera is None or not (
            slot.camera.kind == "realsense" or slot.camera.device_tag == "elgato"
        ):
            want = False
        slot.record_bag = want
        try:
            self.bag_var.set(want)
            self._refresh_bag_enabled(locked=False)
        except tk.TclError:
            pass

    def _refresh_bag_enabled(self, locked: bool = False) -> None:
        slot = self.session.slots[self.slot_id]
        is_bag = slot.camera is not None and (
            slot.camera.kind == "realsense" or slot.camera.device_tag == "elgato"
        )
        if not is_bag:
            if self.bag_var.get():
                self.bag_var.set(False)
            slot.record_bag = False
        try:
            self.bag_check.configure(
                state="disabled" if locked or not is_bag else "normal"
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
        self._refresh_bag_enabled(locked=locked)
        self.remove_btn.configure(
            state="disabled" if locked or len(self.session.slots) <= 2 else "normal"
        )

    def tick(self) -> None:
        slot = self.session.slots[self.slot_id]
        status = slot.status
        src = slot.pipeline.source if slot.pipeline else None
        if src is not None and getattr(src, "device_tag", "") == "elgato":
            measured = float(getattr(src, "actual_fps", 0) or 0)
            wanted = int(
                getattr(src, "requested_fps", 0) or getattr(src, "target_fps", 0) or 0
            )
            if measured > 1:
                status = f"{status} | camera ~{measured:.0f}fps"
                if wanted >= 90 and measured < wanted * 0.85:
                    status += (
                        f" (Station HDMI ~{measured:.0f} — set mirrorless HDMI to "
                        f"1080p120 for true 120; file stamped ~{measured:.0f}, "
                        "not a software drop)"
                    )
        self.status_label.configure(text=status)
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
        tw = max(self.preview_shell.winfo_width() - 4, 280)
        th = max(self.preview_shell.winfo_height() - 4, 160)
        rgb = bgr_to_rgb_fill(frame, tw, th)
        if src is not None:
            rgb = overlay_hud(rgb, hud_lines_for_source(slot, src))
        self._photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.preview.configure(image=self._photo, text="")
