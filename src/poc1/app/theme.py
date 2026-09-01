"""Shared visual language for the unified POC1 app."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

# Calm light shell — readable, not a debug console.
BG = "#eef2f6"
PANEL = "#ffffff"
SURFACE = "#f8fafc"
INK = "#0f172a"
MUTED = "#64748b"
BORDER = "#d0d7e2"
ACCENT = "#0f766e"
ACCENT_SOFT = "#ccfbf1"
BLUE = "#1d4ed8"
RED = "#b91c1c"
GREEN = "#15803d"
AMBER = "#b45309"
FIELD = "#ffffff"


def apply_styles(root: tk.Misc) -> None:
    style = ttk.Style()
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass
    style.configure(
        "App.TCombobox",
        fieldbackground=FIELD,
        background=FIELD,
        foreground=INK,
        arrowcolor=INK,
    )
    style.map(
        "App.TCombobox",
        fieldbackground=[("readonly", FIELD), ("disabled", "#e2e8f0")],
        foreground=[("readonly", INK), ("disabled", "#94a3b8")],
    )
    style.configure(
        "Nav.TButton",
        font=("Segoe UI Semibold", 10),
        padding=(14, 10),
    )
    style.configure("TNotebook", background=BG, borderwidth=0)
    style.configure(
        "TNotebook.Tab",
        background=SURFACE,
        foreground=MUTED,
        padding=(18, 10),
        font=("Segoe UI Semibold", 10),
    )
    style.map(
        "TNotebook.Tab",
        background=[("selected", PANEL)],
        foreground=[("selected", INK)],
    )
    root.option_add("*TCombobox*Listbox.background", FIELD)
    root.option_add("*TCombobox*Listbox.foreground", INK)
    root.option_add("*TCombobox*Listbox.selectBackground", ACCENT)
    root.option_add("*TCombobox*Listbox.selectForeground", "#ffffff")


def button(
    master: tk.Misc,
    text: str,
    command,
    *,
    color: str = ACCENT,
    fg: str = "#ffffff",
    font=("Segoe UI Semibold", 9),
    padx: int = 14,
    pady: int = 8,
    **kwargs,
) -> tk.Button:
    return tk.Button(
        master,
        text=text,
        command=command,
        bg=color,
        fg=fg,
        activebackground=color,
        activeforeground=fg,
        disabledforeground="#94a3b8",
        font=font,
        relief="flat",
        bd=0,
        padx=padx,
        pady=pady,
        cursor="hand2",
        **kwargs,
    )
