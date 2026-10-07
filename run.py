"""Convenience entry point for the EduTwin Africa Streamlit app.

Use:
    python run.py

For the Week 1 foundation smoke demo:
    python smoke_demo.py

The FastAPI backend remains available through:
    python web_app.py
"""

from __future__ import annotations

import subprocess
import sys
import time
import webbrowser
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
APP = PROJECT_ROOT / "digital_twin_streamlit.py"
URL = "http://127.0.0.1:8501"


def main() -> int:
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(APP),
        "--server.address",
        "127.0.0.1",
        "--server.port",
        "8501",
        "--browser.gatherUsageStats",
        "false",
    ]

    print("Starting EduTwin Africa...")
    print(f"Opening {URL}")
    process = subprocess.Popen(command, cwd=PROJECT_ROOT)

    try:
        # Give Streamlit a moment to bind the local port before opening the browser.
        time.sleep(1.5)
        webbrowser.open(URL)
        return process.wait()
    except KeyboardInterrupt:
        print("\nEduTwin Africa stopped successfully.")
        return 0
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()


if __name__ == "__main__":
    raise SystemExit(main())
