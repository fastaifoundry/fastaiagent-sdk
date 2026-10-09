"""Drive the Local UI and capture the Decisions API docs screenshots.

Called by ``scripts/capture-decisions-screenshots.sh`` with the UI's base URL and
the output directory, after examples 105/106/107 have run into the UI's database.
Each trace is picked by its content through the UI's own API, so the shots do not
depend on trace ids.

    python scripts/capture_decisions_screenshots.py http://127.0.0.1:7849 docs/ui/screenshots
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

VIEWPORT = {"width": 1440, "height": 900}


def api(base: str, path: str) -> object:
    with urllib.request.urlopen(f"{base}{path}") as resp:
        return json.loads(resp.read())


def traces(base: str) -> list[dict]:
    data = api(base, "/api/traces?limit=100")
    return data.get("rows", data) if isinstance(data, dict) else data  # type: ignore[return-value]


def spans_of(base: str, trace_id: str) -> list[dict]:
    data = api(base, f"/api/traces/{trace_id}")
    return data["spans"] if isinstance(data, dict) else data  # type: ignore[index,return-value]


def find_trace(base: str, name: str, needle: str) -> str:
    """The newest trace called ``name`` whose spans mention ``needle``."""
    for row in traces(base):
        if row["name"] != name:
            continue
        if needle in json.dumps(spans_of(base, row["trace_id"]), default=str):
            return str(row["trace_id"])
    raise SystemExit(f"no {name} trace mentioning {needle!r}")


def settle(page: Page, ms: int = 1500) -> None:
    page.wait_for_load_state("networkidle")
    page.wait_for_timeout(ms)


def open_trace(page: Page, base: str, trace_id: str) -> None:
    page.goto(f"{base}/traces/{trace_id}")
    settle(page)


def select_span(page: Page, name: str, nth: int = 0) -> None:
    page.get_by_text(name, exact=True).nth(nth).click()
    settle(page, 800)


def tab(page: Page, label: str) -> None:
    page.get_by_role("tab", name=label).or_(page.get_by_text(label, exact=True)).first.click()
    settle(page, 600)


def main() -> int:
    base, out = sys.argv[1].rstrip("/"), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)

    complaint = find_trace(base, "supervisor.call-center", "cracked lamp base")
    chain_product = find_trace(base, "chain.call-center", "phone charger")
    # Example 105 runs the agent, then its recorded rerun — the rerun is the newer one.
    reruns = [r for r in traces(base) if r["name"] == "agent.router"]
    rerun = str(reruns[0]["trace_id"])

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport=VIEWPORT, device_scale_factor=2)

        # 1. The trace list: every ticket is one supervisor / chain trace.
        page.goto(f"{base}/traces")
        settle(page)
        page.screenshot(path=out / "decisions-01-traces.png")

        # 2. A routed ticket end to end: route → worker (chat + tools) → review.
        open_trace(page, base, complaint)
        page.screenshot(path=out / "decisions-02-supervisor-trace.png")

        # 3. The route, on the supervisor's root span.
        tab(page, "Attributes")
        page.screenshot(path=out / "decisions-03-route-attributes.png")

        # 4. One Decisions API call: OTel GenAI + OpenInference attributes, the
        #    questions and the answers.
        select_span(page, "llm.openai.decisions.gpt-6-luna", 0)
        tab(page, "Attributes")
        page.screenshot(path=out / "decisions-04-decisions-span.png")
        # ...and further down the same panel: the answers, usage and cost.
        page.get_by_text("fastaiagent.decision.answers").first.scroll_into_view_if_needed()
        settle(page, 500)
        page.screenshot(path=out / "decisions-04b-decision-answers.png")

        # 5. The same desk as a Chain: the triage decision node and the queue it chose.
        open_trace(page, base, chain_product)
        page.screenshot(path=out / "decisions-05-chain-trace.png")

        # 6. Recorded replay: chat turns and the decision served from the capture.
        open_trace(page, base, rerun)
        select_span(page, "llm.openai.decisions.gpt-6-luna", 0)
        tab(page, "Attributes")
        page.screenshot(path=out / "decisions-06-replay-recorded.png")

        # 7. Cost by model — the Decisions API next to the chat workers. The
        #    breakdown sits low in a scrolling panel, so bring it into view.
        page.goto(f"{base}/analytics")
        settle(page, 2500)
        page.get_by_text("// COST BREAKDOWN").first.scroll_into_view_if_needed()
        settle(page, 800)
        page.screenshot(path=out / "decisions-07-analytics.png")

        browser.close()
    for f in sorted(out.glob("decisions-*.png")):
        print(f"  {f.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
