"""Shared harness for the 1.89.0 capture scripts (``scripts/capture-*-189 …``).

Each ``scripts/capture_<name>.py`` driver builds one :class:`Capture`, opens
pages through it and asserts on what a reader of the page sees. Every
assertion is counted; a failed one is recorded and the run carries on, so one
broken label does not hide the next. At the end :meth:`Capture.finish` also
asserts that the browser logged no console error and threw no page error
(except network errors the driver declared it expects, such as the 422 a
negative check provokes), prints a summary and returns the exit code.

Screenshots go to ``docs/ui/screenshots/`` (``OUT_DIR`` overrides it).
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from playwright.sync_api import Browser, Locator, Page, expect

REPO = Path(__file__).resolve().parents[1]
VIEWPORT = {"width": 1440, "height": 900}
DEFAULT_TIMEOUT_MS = 10_000


class Capture:
    def __init__(self, name: str, base: str) -> None:
        self.name = name
        self.base = base.rstrip("/")
        self.out = Path(os.environ.get("OUT_DIR") or REPO / "docs/ui/screenshots")
        self.out.mkdir(parents=True, exist_ok=True)
        self.count = 0
        self.failures: list[str] = []
        self.console_errors: list[str] = []
        self.allowed_console: list[re.Pattern[str]] = []
        self.shots: list[Path] = []

    # -- browser ---------------------------------------------------------------

    def new_page(self, browser: Browser) -> Page:
        page = browser.new_page(viewport=VIEWPORT, device_scale_factor=2, color_scheme="light")
        page.set_default_timeout(DEFAULT_TIMEOUT_MS)

        def on_console(msg: Any) -> None:
            if msg.type == "error":
                self.console_errors.append(f"console: {msg.text} @ {page.url}")

        def on_pageerror(exc: Any) -> None:
            self.console_errors.append(f"pageerror: {exc} @ {page.url}")

        page.on("console", on_console)
        page.on("pageerror", on_pageerror)
        return page

    def allow_console(self, pattern: str) -> None:
        """A console error the driver provokes on purpose (e.g. an expected 422)."""
        self.allowed_console.append(re.compile(pattern))

    def goto(self, page: Page, path: str, settle_ms: int = 800) -> None:
        page.goto(f"{self.base}{path}")
        self.settle(page, settle_ms)

    @staticmethod
    def settle(page: Page, ms: int = 600) -> None:
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(ms)

    def shot(self, page: Page, name: str, *, full_page: bool = False) -> None:
        path = self.out / f"{name}.png"
        page.screenshot(path=str(path), full_page=full_page)
        self.shots.append(path)
        print(f"  shot {path.relative_to(REPO) if path.is_relative_to(REPO) else path}")

    # -- assertions --------------------------------------------------------------

    def _record(self, ok: bool, label: str, detail: str = "") -> bool:
        self.count += 1
        if ok:
            print(f"  ok   {label}")
        else:
            self.failures.append(f"{label}{': ' + detail if detail else ''}")
            print(f"  FAIL {label}{': ' + detail if detail else ''}")
        return ok

    def check(self, cond: bool, label: str, detail: str = "") -> bool:
        return self._record(bool(cond), label, detail)

    def visible(self, loc: Locator, label: str, timeout: int = DEFAULT_TIMEOUT_MS) -> bool:
        try:
            expect(loc.first).to_be_visible(timeout=timeout)
            return self._record(True, label)
        except AssertionError as exc:
            return self._record(False, label, str(exc).splitlines()[0])

    def text(
        self,
        page: Page | Locator,
        text: str | re.Pattern[str],
        label: str | None = None,
        *,
        exact: bool = False,
        timeout: int = DEFAULT_TIMEOUT_MS,
    ) -> bool:
        return self.visible(
            page.get_by_text(text, exact=exact), label or f"shows {text!r}", timeout
        )

    def absent(self, page: Page | Locator, text: str, label: str | None = None) -> bool:
        """``text`` appears nowhere in the rendered page (or locator)."""
        body = page.locator("body").inner_text() if isinstance(page, Page) else page.inner_text()
        return self._record(text not in body, label or f"does not show {text!r}")

    def hidden(self, loc: Locator, label: str, timeout: int = DEFAULT_TIMEOUT_MS) -> bool:
        try:
            expect(loc).to_have_count(0, timeout=timeout)
            return self._record(True, label)
        except AssertionError as exc:
            return self._record(False, label, str(exc).splitlines()[0])

    def disabled(self, loc: Locator, label: str) -> bool:
        try:
            expect(loc.first).to_be_disabled(timeout=DEFAULT_TIMEOUT_MS)
            return self._record(True, label)
        except AssertionError as exc:
            return self._record(False, label, str(exc).splitlines()[0])

    def enabled(self, loc: Locator, label: str) -> bool:
        try:
            expect(loc.first).to_be_enabled(timeout=DEFAULT_TIMEOUT_MS)
            return self._record(True, label)
        except AssertionError as exc:
            return self._record(False, label, str(exc).splitlines()[0])

    # -- the UI's own API (to pick ids, and to check what a page claims) ---------

    def api(self, path: str, method: str = "GET", body: Any = None) -> tuple[int, Any]:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            f"{self.base}{path}",
            data=data,
            method=method,
            headers={"Content-Type": "application/json"} if data is not None else {},
        )
        try:
            with urllib.request.urlopen(req) as resp:
                raw = resp.read()
                return resp.status, json.loads(raw) if raw else None
        except urllib.error.HTTPError as exc:
            raw = exc.read()
            try:
                return exc.code, json.loads(raw)
            except ValueError:
                return exc.code, raw.decode(errors="replace")

    def get(self, path: str) -> Any:
        status, body = self.api(path)
        if status != 200:
            raise SystemExit(f"GET {path} -> {status}: {body}")
        return body

    def traces(self, query: str = "") -> list[dict[str, Any]]:
        data = self.get(f"/api/traces?page_size=200{('&' + query) if query else ''}")
        return data["rows"] if isinstance(data, dict) else data

    # -- end ---------------------------------------------------------------------

    def finish(self) -> int:
        unexpected = [
            e for e in self.console_errors if not any(p.search(e) for p in self.allowed_console)
        ]
        self._record(
            not unexpected,
            "no unexpected console or page errors",
            "; ".join(unexpected[:5]),
        )
        print(
            f"\n{self.name}: {self.count} assertions, {len(self.failures)} failed, "
            f"{len(self.shots)} screenshots"
        )
        for f in self.failures:
            print(f"  FAILED: {f}")
        return 1 if self.failures else 0


def pick_provider(page: Page, provider: str) -> None:
    """Choose a provider in the Playground's (Radix) provider select."""
    page.locator("#playground-provider").click()
    page.get_by_role("option", name=re.compile(rf"^{re.escape(provider)}\b")).first.click()
    page.wait_for_timeout(200)


def base_url() -> str:
    if len(sys.argv) < 2:
        raise SystemExit(f"usage: {sys.argv[0]} <ui base url> [args…]")
    return sys.argv[1]
