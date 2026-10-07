"""Run the Week 1 foundation smoke demo."""

from pathlib import Path
import subprocess
import sys

subprocess.run(
    [
        sys.executable,
        str(Path(__file__).parent / "scripts" / "run_week1.py"),
    ],
    check=True,
)