"""
SystemMonitor: logs CPU / RAM / disk-write-rate at a fixed interval to a
CSV file for the duration of a recording. This is the concrete evidence
POC1's test scenario asks for ("Log the system usage (CPU, RAM, Disk)").

Also useful as a way to *separately* stress-test raw write throughput
(the ~1GB/s figure from your notes) independent of the camera/GUI --
run this alongside recorder.py writing directly, without camera_handler
in the loop at all, to isolate "can the disk sustain this" from "does
the rest of the pipeline work."
"""
from __future__ import annotations

import csv
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import psutil


@dataclass
class SystemMonitor:
    output_csv: Path
    interval_s: float = 1.0

    def __post_init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._running = threading.Event()

    def start(self) -> None:
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        self._running.set()
        self._thread = threading.Thread(target=self._loop, name="sys-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running.clear()
        if self._thread:
            self._thread.join(timeout=5.0)

    def _loop(self) -> None:
        with open(self.output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "cpu_percent", "ram_percent", "ram_used_mb", "disk_write_mb_since_start"])
            disk_start = psutil.disk_io_counters().write_bytes
            while self._running.is_set():
                cpu = psutil.cpu_percent(interval=None)
                vm = psutil.virtual_memory()
                disk_now = psutil.disk_io_counters().write_bytes
                writer.writerow([
                    time.time(),
                    cpu,
                    vm.percent,
                    round(vm.used / (1024 * 1024), 1),
                    round((disk_now - disk_start) / (1024 * 1024), 1),
                ])
                f.flush()
                time.sleep(self.interval_s)
