"""
R8 — post-save review prompt that auto-dismisses after 5 seconds.
"""
from __future__ import annotations

import tkinter as tk
from pathlib import Path
from typing import Callable, Optional, Sequence


def show_review_prompt(
    parent: tk.Misc,
    paths: Sequence[Path],
    *,
    on_review: Callable[[Path], None],
    timeout_ms: int = 5000,
    title: str = "Recording saved",
    review_label: str = "Review",
) -> Optional[tk.Toplevel]:
    """
    Ask whether to review footage. Auto-destroys after timeout_ms.

    R8: message disappears in 5 seconds if the user ignores it.
    """
    existing = [Path(p) for p in paths if Path(p).is_file()]
    if not existing:
        return None

    bg = "#ffffff"
    ink = "#0f172a"
    muted = "#64748b"
    accent = "#0f766e"
    soft = "#e2e8f0"

    popup = tk.Toplevel(parent)
    popup.title(title)
    popup.configure(bg=bg)
    popup.attributes("-topmost", True)
    popup.resizable(False, False)

    names = "\n".join(f"• {p.name}" for p in existing[:6])
    if len(existing) > 6:
        names += f"\n• …and {len(existing) - 6} more"

    tk.Label(
        popup,
        text="Recording saved.\nReview footage now?\n(closes automatically in 5 seconds)",
        bg=bg,
        fg=ink,
        font=("Segoe UI", 11),
        justify="left",
    ).pack(padx=20, pady=(16, 8), anchor="w")
    tk.Label(
        popup,
        text=names,
        bg=bg,
        fg=muted,
        font=("Segoe UI", 9),
        justify="left",
    ).pack(padx=20, pady=(0, 10), anchor="w")

    row = tk.Frame(popup, bg=bg)
    row.pack(fill="x", padx=20, pady=(0, 16))

    def review_first() -> None:
        try:
            popup.destroy()
        except tk.TclError:
            pass
        on_review(existing[0])

    def dismiss() -> None:
        try:
            popup.destroy()
        except tk.TclError:
            pass

    tk.Button(
        row,
        text=review_label,
        command=review_first,
        bg=accent,
        fg="#ffffff",
        activebackground=accent,
        activeforeground="#ffffff",
        relief="flat",
        padx=14,
        pady=6,
        cursor="hand2",
        font=("Segoe UI Semibold", 9),
    ).pack(side="left", padx=(0, 8))
    tk.Button(
        row,
        text="Dismiss",
        command=dismiss,
        bg=soft,
        fg=ink,
        activebackground=soft,
        activeforeground=ink,
        relief="flat",
        padx=14,
        pady=6,
        cursor="hand2",
        font=("Segoe UI", 9),
    ).pack(side="left")

    popup.after(timeout_ms, dismiss)
    popup.update_idletasks()
    try:
        parent.update_idletasks()
        x = parent.winfo_rootx() + 40
        y = parent.winfo_rooty() + 40
        popup.geometry(f"+{x}+{y}")
    except tk.TclError:
        pass
    return popup
