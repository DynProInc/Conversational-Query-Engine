"""Thread-safe append-only CSV writer (Python >= 3.9)."""
from __future__ import annotations
import csv, os, uuid
from pathlib import Path
from filelock import FileLock

DATA_DIR = Path(__file__).resolve().parent.parent / "feedback"
DATA_DIR.mkdir(parents=True, exist_ok=True)

class CSVLogger:
    """Create-on-first-use CSV with a sidecar *.lock* file."""
    def __init__(self, name: str, header: list[str]):
        self.path = DATA_DIR / name
        self.lock = FileLock(str(self.path) + ".lock")
        if not self.path.exists():
            self._write_header(header)

    def _write_header(self, header: list[str]):
        with self.lock, open(self.path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)

    def append(self, row: list):
        with self.lock, open(self.path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)
