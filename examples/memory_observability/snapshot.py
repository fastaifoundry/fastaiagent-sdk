"""Browser snapshots for memory observability — the Local UI.

Boots the local UI (pointed at the DB companion.py seeded), drives headless
Chromium via Playwright, and saves PNGs into screenshots/.

Three shots:
    1. /traces/<trace_id>                 -- the trace with memory.read/write spans
    2. /traces/<trace_id> (memory span)   -- VectorBlock child: scores + snippets
    3. /memory                            -- the Memory page (learned facts)

Prereqs (once):
    pip install playwright && python -m playwright install chromium

Run (after companion.py):
    zsh -lc 'python snapshot.py'
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

UI_HOST = "127.0.0.1"
UI_PORT = 7842
BASE_URL = f"http://{UI_HOST}:{UI_PORT}"
STARTUP_TIMEOUT_S = 30.0
VIEWPORT = {"width": 1440, "height": 900}


def _wait_for_ui(timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((UI_HOST, UI_PORT), timeout=0.5):
                pass
        except OSError:
            time.sleep(0.3)
            continue
        try:
            urlopen(f"{BASE_URL}/", timeout=1.0).read(0)
            return True
        except URLError:
            time.sleep(0.3)
        except Exception:
            return True
    return False


def _snap(page, url: str, out_path: Path, label: str, settle_ms: int = 1800) -> None:
    print(f"  -> {label}: {url}")
    page.goto(url, wait_until="networkidle", timeout=20_000)
    page.wait_for_timeout(settle_ms)
    page.screenshot(path=str(out_path), full_page=True)
    print(f"     saved {out_path.name} ({out_path.stat().st_size // 1024} KB)")


def main() -> int:
    if not LAST_RUN.exists():
        print(f"Missing {LAST_RUN.name}. Run companion.py first:")
        print("  zsh -lc 'python companion.py'")
        return 1

    run = json.loads(LAST_RUN.read_text())
    trace_id = run.get("trace_id")
    db_path = run.get("db_path")

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("Playwright not installed. Run:")
        print("  pip install playwright && python -m playwright install chromium")
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

    print(f"Starting local UI on {BASE_URL} ...")
    ui_log = HERE / "ui.log"
    log_fh = ui_log.open("w", encoding="utf-8")
    proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT)

    try:
        if not _wait_for_ui(STARTUP_TIMEOUT_S):
            print(f"UI did not come up within {STARTUP_TIMEOUT_S}s. Last lines of ui.log:")
            log_fh.close()
            for line in ui_log.read_text(errors="replace").splitlines()[-25:]:
                print(f"  {line}")
            return 4
        print("UI is up. Launching headless Chromium ...")

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(viewport=VIEWPORT)
            page = context.new_page()
            try:
                if trace_id:
                    _snap(
                        page,
                        f"{BASE_URL}/traces/{trace_id}",
                        SCREENSHOTS_DIR / "01-trace-memory-spans.png",
                        "1. Trace with memory.read / memory.write spans",
                    )
                    # Hero: open the VectorBlock child span → scores + snippets.
                    try:
                        page.get_by_text("memory.read.vector", exact=True).first.click(timeout=5000)
                        page.wait_for_timeout(800)
                        page.get_by_role("tab", name="Output").click(timeout=3000)
                        page.wait_for_timeout(800)
                        page.screenshot(
                            path=str(SCREENSHOTS_DIR / "02-vectorblock-scores.png"),
                            full_page=True,
                        )
                        print("  -> 2. VectorBlock span detail: scores + snippets")
                    except Exception as err:
                        print(f"     (skipped hero shot: {err})")
                _snap(
                    page,
                    f"{BASE_URL}/memory",
                    SCREENSHOTS_DIR / "02-memory-page.png",
                    "2. Memory page (facts + scope filter + source column)",
                )
                # History: toggle "Show superseded" to reveal the audit chain.
                try:
                    page.get_by_label("Toggle superseded facts").click(timeout=3000)
                    page.wait_for_timeout(900)
                    page.screenshot(
                        path=str(SCREENSHOTS_DIR / "04-memory-superseded.png"),
                        full_page=True,
                    )
                    print("  -> 3. Memory page: Show superseded (lineage)")
                except Exception as err:
                    print(f"     (skipped superseded shot: {err})")
            finally:
                context.close()
                browser.close()

        print(f"\nDone. PNGs in: {SCREENSHOTS_DIR}")
        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        try:
            log_fh.close()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
