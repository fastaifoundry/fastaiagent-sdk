"""Drive the Local UI and capture the AutoLLM closed-loop docs screenshots.

Called by ``scripts/capture-autollm-loop-screenshots.sh`` with the UI's base URL,
the output directory and the example folder, after ``examples/autollm-loop`` has
run end to end. Runs are picked by the ids the example wrote to ``out/`` and by
name through the UI's own API, so the shots do not depend on ordering.

    python scripts/capture_autollm_loop_screenshots.py http://127.0.0.1:7863 \\
        docs/ui/screenshots examples/autollm-loop
"""

from __future__ import annotations

import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

VIEWPORT = {"width": 1440, "height": 900}


def api(base: str, path: str) -> object:
    with urllib.request.urlopen(f"{base}{path}") as resp:
        return json.loads(resp.read())


def eval_run_id(base: str, run_name: str) -> str:
    """The newest eval run called ``run_name``."""
    query = urllib.parse.urlencode({"page": 1, "page_size": 200})
    data = api(base, f"/api/evals?{query}")
    rows = data.get("rows", data) if isinstance(data, dict) else data
    for row in rows:  # type: ignore[union-attr]
        if row.get("run_name") == run_name:
            return str(row["run_id"])
    raise SystemExit(f"no eval run named {run_name!r}")


def settle(page: Page, ms: int = 1500) -> None:
    page.wait_for_load_state("networkidle")
    page.wait_for_timeout(ms)


def shot(page: Page, out: Path, name: str) -> None:
    path = out / f"autollm-loop-{name}.png"
    page.screenshot(path=str(path))
    print(f"  {path}")


def scroll_to(page: Page, heading: str) -> None:
    """Bring a card to the top of the view — the shell scrolls an inner pane, not
    the page, so this scrolls whichever pane holds the heading."""
    page.get_by_text(heading, exact=True).first.evaluate(
        "el => el.scrollIntoView({block: 'start'})"
    )
    page.wait_for_timeout(600)


def main() -> int:
    base, out, example = sys.argv[1].rstrip("/"), Path(sys.argv[2]), Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    optimize = json.loads((example / "out" / "optimize.json").read_text())
    baseline = json.loads((example / "out" / "baseline.json").read_text())
    promoted = json.loads((example / "out" / "promoted.json").read_text())
    gate = eval_run_id(base, f"ticket-triage v{promoted['version']} (gate)")

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport=VIEWPORT, device_scale_factor=2)

        page.goto(f"{base}/prompts")
        settle(page)
        shot(page, out, "01-prompts")

        page.goto(f"{base}/datasets/ticket-triage")
        settle(page)
        shot(page, out, "02-dataset")

        page.goto(f"{base}/evals/{baseline['run_id']}")
        settle(page)
        shot(page, out, "03-baseline-eval")

        page.goto(f"{base}/optimizes/{optimize['run_id']}")
        settle(page, 2500)
        shot(page, out, "04-autollm-winner")
        scroll_to(page, "Rationale")  # the trajectory table's header row
        shot(page, out, "05-autollm-trajectory")

        page.goto(f"{base}/prompts/ticket-triage")
        settle(page)
        shot(page, out, "06-prompt-versions")
        scroll_to(page, "Lineage")
        shot(page, out, "07-prompt-lineage")

        page.goto(f"{base}/evals/compare?a={baseline['run_id']}&b={gate}")
        settle(page, 2500)
        shot(page, out, "08-gate-vs-v1")

        browser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
