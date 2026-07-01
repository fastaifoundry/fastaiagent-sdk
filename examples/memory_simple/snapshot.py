"""Browser snapshots for the simple `Memory` API — the Local UI.

Boots `fastaiagent ui` on the demo DB and captures:
    1. /traces/<trace_id>   -- a turn's memory.read/write spans (+ memory.persist)
    2. /memory              -- global + per-user learned facts

Prereqs: pip install playwright && python -m playwright install chromium
Run (after companion.py):  zsh -lc 'python snapshot.py'
"""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

HERE = Path(__file__).parent
SCREENSHOTS_DIR = HERE / "screenshots"
LAST_RUN = HERE / "last_run.json"
UI_HOST, UI_PORT = "127.0.0.1", 7842
BASE_URL = f"http://{UI_HOST}:{UI_PORT}"
VIEWPORT = {"width": 1440, "height": 900}


def _wait_for_ui(timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((UI_HOST, UI_PORT), timeout=0.5):
                pass
            urlopen(f"{BASE_URL}/", timeout=1.0).read(0)
            return True
        except (OSError, URLError):
            time.sleep(0.3)
        except Exception:
            return True
    return False


def _snap(page, url, out, label, settle=1600):
    print(f"  -> {label}: {url}")
    page.goto(url, wait_until="networkidle", timeout=20_000)
    page.wait_for_timeout(settle)
    page.screenshot(path=str(out), full_page=True)


def main() -> int:
    if not LAST_RUN.exists():
        print(f"Missing {LAST_RUN.name}. Run companion.py first.")
        return 1
    run = json.loads(LAST_RUN.read_text())
    trace_id, db_path = run.get("trace_id"), run.get("db_path")
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print(
            "Playwright not installed: pip install playwright && python -m playwright install chromium"
        )
        return 3

    SCREENSHOTS_DIR.mkdir(exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "fastaiagent.cli.main",
        "ui",
        "start",
        "--host",
        UI_HOST,
        "--port",
        str(UI_PORT),
        "--no-auth",
        "--no-open",
    ]
    if db_path:
        cmd += ["--db", db_path]
    log_fh = (HERE / "ui.log").open("w")
    proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT)
    try:
        if not _wait_for_ui(30.0):
            print("UI did not start; see ui.log")
            return 4
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_context(viewport=VIEWPORT).new_page()
            try:
                if trace_id:
                    _snap(
                        page,
                        f"{BASE_URL}/traces/{trace_id}",
                        SCREENSHOTS_DIR / "01-memory-trace.png",
                        "1. Trace: memory spans",
                    )
                _snap(
                    page,
                    f"{BASE_URL}/memory",
                    SCREENSHOTS_DIR / "02-memory-page.png",
                    "2. Memory page: global + per-user facts",
                )
            finally:
                browser.close()
        print(f"\nDone. PNGs in: {SCREENSHOTS_DIR}")
        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        log_fh.close()


if __name__ == "__main__":
    sys.exit(main())
