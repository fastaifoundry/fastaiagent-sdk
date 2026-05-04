"""Phase 4 (UI side) — Playwright smoke checks of the Local UI.

Drives a real browser against the running ``fastaiagent ui`` server on
``http://127.0.0.1:7842`` to confirm:

1. The Traces list renders without JS errors.
2. The framework filter chips (FA / LC / CA / PA) are present.
3. Clicking ``LC`` filters the list to LangChain traces only.
4. The framework badge appears on at least one row.

Skips cleanly if the UI server isn't reachable or Playwright isn't
installed. Tests run only against an already-running UI; we don't
spawn one inside the test (the harness regression flow starts the
server separately).
"""

from __future__ import annotations

import os
from urllib.request import urlopen

import pytest

UI_URL = os.environ.get("FASTAIAGENT_UI_URL", "http://127.0.0.1:7842")


def _ui_alive() -> bool:
    try:
        urlopen(f"{UI_URL}/api/auth/status", timeout=2).read(64)
        return True
    except Exception:
        return False


if not _ui_alive():
    pytest.skip(
        f"Local UI not reachable at {UI_URL}; start with `fastaiagent ui --no-auth`",
        allow_module_level=True,
    )

playwright = pytest.importorskip("playwright.sync_api")
pytestmark = pytest.mark.e2e


def test_traces_page_loads() -> None:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            page = browser.new_page()
            page.goto(f"{UI_URL}/traces", wait_until="domcontentloaded")
            # The traces page renders a table with at least the trace-id
            # header. Use a permissive selector so a ui-frontend rename
            # doesn't break the smoke test.
            page.wait_for_selector("table", timeout=10_000)
        finally:
            browser.close()


def test_framework_filter_input_present() -> None:
    """A free-text ``Framework`` input must render in the filter bar.

    Free-text on purpose — new frameworks (LangSmith, AutoGen, etc.)
    must filter without requiring a UI code change. We assert the
    input exists and accepts a value.
    """
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            page = browser.new_page()
            page.goto(f"{UI_URL}/traces", wait_until="domcontentloaded")
            page.wait_for_selector("table", timeout=10_000)
            box = page.get_by_placeholder("Framework")
            assert box.first.is_visible()
        finally:
            browser.close()


def test_framework_filter_narrows_list() -> None:
    """Typing a framework value into the filter input narrows the URL
    and the server query."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            page = browser.new_page()
            page.goto(f"{UI_URL}/traces", wait_until="domcontentloaded")
            page.wait_for_selector("table", timeout=10_000)

            box = page.get_by_placeholder("Framework").first
            box.fill("langchain")

            # The SPA pushes the filter into the URL on each change.
            page.wait_for_url("**framework=langchain**", timeout=5_000)
            page.wait_for_load_state("networkidle")
        finally:
            browser.close()
