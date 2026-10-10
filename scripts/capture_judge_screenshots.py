"""Drive the Local UI and capture the judge-calibration docs screenshots.

Called by ``scripts/capture-judge-screenshots.sh`` with the UI's base URL and the
output directory, after ``examples/autollm/calibrate_judge.py --tune-agent`` has
run. Runs are picked by name through the UI's own API.

    python scripts/capture_judge_screenshots.py http://127.0.0.1:7864 docs/ui/screenshots
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

VIEWPORT = {"width": 1440, "height": 900}
JUDGE_RUN = "reply judge — calibrated on reviewer labels"
AGENT_RUN = "support agent — tuned by the calibrated judge"


def api(base: str, path: str) -> object:
    with urllib.request.urlopen(f"{base}{path}") as resp:
        return json.loads(resp.read())


def optimize_run(base: str, run_name: str) -> dict:
    data = api(base, "/api/optimizes")
    rows = data.get("rows", data) if isinstance(data, dict) else data
    for row in rows:  # type: ignore[union-attr]
        if row.get("run_name") == run_name:
            return api(base, f"/api/optimizes/{row['run_id']}")  # type: ignore[return-value]
    raise SystemExit(f"no AutoLLM run named {run_name!r}")


def settle(page: Page, ms: int = 1500) -> None:
    page.wait_for_load_state("networkidle")
    page.wait_for_timeout(ms)


def shot(page: Page, out: Path, name: str) -> None:
    path = out / f"judge-{name}.png"
    page.screenshot(path=str(path))
    print(f"  {path}")


def main() -> int:
    base, out = sys.argv[1].rstrip("/"), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    judge = optimize_run(base, JUDGE_RUN)
    agent = optimize_run(base, AGENT_RUN)
    baseline_eval = judge["iterations"][0]["eval_run_id"]  # the naive judge, scored

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport=VIEWPORT, device_scale_factor=2)

        page.goto(f"{base}/optimizes")
        settle(page)
        shot(page, out, "01-autollm-runs")

        page.goto(f"{base}/evals/{baseline_eval}")
        settle(page)
        shot(page, out, "02-naive-judge-eval")

        page.goto(f"{base}/optimizes/{judge['run']['run_id']}")
        settle(page, 2500)
        shot(page, out, "03-judge-winner")

        page.goto(f"{base}/optimizes/{agent['run']['run_id']}")
        settle(page, 2500)
        shot(page, out, "04-agent-winner")

        browser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
