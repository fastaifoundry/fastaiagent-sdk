"""Browser-level smoke check across the Local UI's top-level pages.

The harness changes only touched the Traces page (free-text framework
filter, no per-row badge), but the regression contract is "the rest
of the UI didn't break either". Each test here loads a top-level
page, waits for the main content to render, and asserts there were
no red console errors during load.

Skips cleanly if the UI server isn't reachable.
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

pytest.importorskip("playwright.sync_api")
pytestmark = pytest.mark.e2e


# Each tuple: (route, content-selector that proves the page rendered).
PAGES: list[tuple[str, str]] = [
    ("/", "main"),                        # Home / overview
    ("/traces", "table"),                 # Traces list
    ("/agents", "main"),                  # Agents
    ("/prompts", "main"),                 # Prompt registry
    ("/evals", "main"),                   # Eval runs
    ("/guardrail-events", "main"),        # Guardrail events
    ("/datasets", "main"),                # Datasets
    ("/threads", "main"),                 # Threads
    ("/analytics", "main"),               # Analytics
]


@pytest.mark.parametrize("route,selector", PAGES, ids=[r for r, _ in PAGES])
def test_page_loads_without_console_errors(route: str, selector: str) -> None:
    from playwright.sync_api import sync_playwright

    errors: list[str] = []

    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            page = browser.new_page()
            # Capture any ``console.error`` calls and any unhandled
            # page errors. ``pageerror`` covers JS exceptions raised
            # during render.
            page.on(
                "console",
                lambda msg: errors.append(f"[console:{msg.type}] {msg.text}")
                if msg.type == "error"
                else None,
            )
            page.on(
                "pageerror",
                lambda exc: errors.append(f"[pageerror] {exc}"),
            )

            response = page.goto(f"{UI_URL}{route}", wait_until="domcontentloaded")
            assert response is not None
            # SPA routes return 200 (the index.html); we don't assert on
            # response.status because the server falls back to index.html
            # for client-side routes.

            page.wait_for_selector(selector, timeout=10_000)
            # Give React a moment to flush any deferred fetches.
            page.wait_for_load_state("networkidle", timeout=10_000)
        finally:
            browser.close()

    # Ignore noisy known categories: source-map fetch failures from
    # Vite's dev tooling, expected 404 toasts on empty surfaces.
    real_errors = [
        e
        for e in errors
        if "source-map" not in e.lower()
        and "favicon" not in e.lower()
    ]
    assert not real_errors, f"console errors on {route}: {real_errors}"
