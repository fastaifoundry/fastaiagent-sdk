"""1.89.0 Phases 6–8 (eval numbers on screen, Run eval from the UI) — seed, drive, assert.

Called by ``scripts/capture-eval-ui-screenshots.sh``:

    python scripts/capture_eval_ui.py seed                # cwd = the temp dir; live
    python scripts/capture_eval_ui.py drive <base-url>    # live (the Run eval dialog)

What it proves, on screen:

* cases first, checks second: a run's tiles count cases passed / failed /
  errored, the checks rate says how many verdicts did not apply (n/a) or could
  not run (errored), and a scorer chip says the same;
* a judge outage is an **invalid** gate, not a quality failure, and says why;
* agent cost and eval cost are real numbers (they were "—" / $0.00);
* compare shows the errored cases instead of dropping them;
* the Datasets page's Run eval dialog evaluates a registered agent with real
  scorers (here ``contains`` and an ``llm_judge`` on gpt-4o-mini) and reports
  the cases inline.

Live: OpenAI gpt-4o-mini for the agent and the judge (about 30 short calls).
The outage judge asks for a model that does not exist, so it fails for real.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

DATASET = "support-faq"
ROWS = [
    {"input": "How long does a refund take?", "expected_output": "7 business days"},
    {"input": "Do you ship to Canada?", "expected_output": "Canada"},
    {"input": "Can I return a sale item?", "expected_output": "final"},
    # The facts say "two years": a reply that converts it fails `contains`.
    {"input": "How long is the warranty, in months?", "expected_output": "24 months"},
    # Nothing to compare against: `contains` does not apply (n/a).
    {"input": "Thanks, that's all for today!"},
]

AGENT_APP = """
from fastaiagent import Agent
from fastaiagent.llm import LLMClient

FACTS = (
    "Refunds take 7 business days. We ship to the US, Canada and Mexico. "
    "Sale items are final sale and cannot be returned. The warranty is two years."
)

agent = Agent(
    name="support-agent",
    system_prompt=f"You are a support agent. Answer in one short sentence. Facts: {FACTS}",
    llm=LLMClient(provider="openai", model="gpt-4o-mini"),
)
"""

CRITERIA = "The reply answers the customer's message politely and is consistent with the facts."


def seed() -> int:
    from fastaiagent.eval import evaluate
    from fastaiagent.eval.gate import gate
    from fastaiagent.eval.llm_judge import LLMJudge
    from fastaiagent.eval.results import record_gate_result
    from fastaiagent.llm import LLMClient

    datasets = Path(".fastaiagent/datasets")
    datasets.mkdir(parents=True, exist_ok=True)
    (datasets / f"{DATASET}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ROWS))
    Path("agent_app.py").write_text(AGENT_APP)
    sys.path.insert(0, str(Path.cwd()))
    from agent_app import agent  # type: ignore[import-not-found]

    def run(name: str, judge_model: str, fail_under: list[str]) -> None:
        judge = LLMJudge(
            criteria=CRITERIA,
            llm=LLMClient(provider="openai", model=judge_model),
            name="helpfulness",
        )
        results = evaluate(
            agent.run,
            str(datasets / f"{DATASET}.jsonl"),
            scorers=["contains", judge],
            run_name=name,
            dataset_name=DATASET,
            agent_name="support-agent",
            concurrency=2,
        )
        if not results.run_id:
            results.run_id = results.persist_local(run_name=name, agent_name="support-agent")
        report = gate(results, fail_under=fail_under)
        record_gate_result(
            results.run_id,
            gate_outcome=report.outcome,
            thresholds={
                f"{c.threshold.metric}.{c.threshold.field}": c.threshold.minimum
                for c in report.checks
            },
            report=report,
        )
        cc = results.case_counts
        print(
            f"▸ {name}: gate {report.outcome}; cases {cc}; invalid_reasons={report.invalid_reasons}"
        )

    run("support-agent — gpt-4o-mini judge", "gpt-4o-mini", ["cases.pass_rate=0.9"])
    run(
        "support-agent — judge outage",
        "gpt-4o-mini-does-not-exist",
        ["helpfulness.pass_rate=0.8"],
    )
    return 0


# ---------------------------------------------------------------------------


def drive(base: str) -> int:
    from _capture189 import Capture
    from playwright.sync_api import sync_playwright

    cap = Capture("capture-eval-ui", base)
    runs = {r["run_name"]: r for r in cap.get("/api/evals?page_size=50")["rows"]}
    good = runs.get("support-agent — gpt-4o-mini judge")
    outage = runs.get("support-agent — judge outage")
    if not good or not outage:
        raise SystemExit(f"seeded runs missing: {list(runs)}")

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = cap.new_page(browser)

        # -- 1. Runs list: gate verdicts, kind, agent cost -----------------------
        print("▸ eval runs list")
        cap.goto(page, "/evals", 1000)
        good_row = page.get_by_test_id(f"eval-row-{good['run_id']}")
        outage_row = page.get_by_test_id(f"eval-row-{outage['run_id']}")
        # A live model decides whether this run passes its threshold; the page
        # must show the verdict that was recorded, whichever it is.
        good_gate = (good.get("gate") or {}).get("outcome") or (good.get("metadata") or {}).get(
            "gate_outcome"
        )
        cap.check(
            good_gate in ("passed", "failed"),
            "the working-judge run has a quality verdict",
            str(good_gate),
        )
        cap.visible(
            good_row.locator(f'[data-gate="{good_gate}"]'),
            f"gpt-4o-mini run: gate {str(good_gate).upper()} (as recorded)",
        )
        cap.visible(outage_row.locator('[data-gate="invalid"]'), "judge-outage run: gate INVALID")
        cap.visible(good_row.get_by_text(re.compile(r"^\$0\.0\d+")), "agent cost is a real number")
        cap.shot(page, "189-eval-01-runs-list")

        # -- 2. A run: cases first, checks second, real costs ----------------------
        print("▸ run detail (judge ok)")
        cap.goto(page, f"/evals/{good['run_id']}", 1200)
        head = page.get_by_test_id("eval-headline")
        for label in ("Cases passed", "Failed", "Errored", "Checks", "Agent cost", "Eval cost"):
            cap.visible(head.get_by_text(label, exact=True), f"tile {label}")
        cap.text(head, re.compile(r"\d+/\d+ · 1 n/a"), "Checks hint counts 1 n/a verdict")
        tiles = head.inner_text()
        costs = re.findall(r"\$\d+\.\d+", tiles)
        cap.check(
            len(costs) >= 2 and all(c != "$0.00" for c in costs),
            "agent cost and eval cost are priced (not — or $0.00)",
            str(costs),
        )
        chips = page.get_by_test_id("scorer-chips")
        cap.text(chips, re.compile(r"contains\s*\d+/4\s*· 1 n/a"), "contains chip: x/4 · 1 n/a")
        cap.text(chips, re.compile(r"helpfulness\s*\d+/5"), "helpfulness chip: x/5")
        ci = page.get_by_test_id("agent-ci-panel")
        cap.visible(
            ci.locator(f'[data-gate="{good_gate}"]'), f"Agent CI gate: {str(good_gate).upper()}"
        )
        cap.visible(
            ci.get_by_test_id("gate-thresholds").get_by_text("cases.pass_rate"),
            "threshold cases.pass_rate with its actual value",
        )
        cap.shot(page, "189-eval-02-run-cases-first")

        page.get_by_role("button", name="Expand case").first.click()
        cap.settle(page, 600)
        cap.visible(page.get_by_text("helpfulness").nth(2), "expanded case shows the verdict table")
        cap.shot(page, "189-eval-03-case-verdicts")

        # -- 3. The outage: invalid, errored cases, and why -------------------------
        print("▸ run detail (judge outage)")
        cap.goto(page, f"/evals/{outage['run_id']}", 1200)
        ci = page.get_by_test_id("agent-ci-panel")
        cap.visible(ci.locator('[data-gate="invalid"]'), "Agent CI gate: INVALID")
        cap.text(ci, "Why the run is invalid")
        cap.text(ci, re.compile(r"helpfulness.*could not run"), "reason names helpfulness")
        cap.text(
            page.get_by_test_id("scorer-chips"),
            re.compile(r"helpfulness\s*0/0\s*· 5 errored"),
            "helpfulness chip: 0/0 · 5 errored",
        )
        errored_tile = page.get_by_test_id("eval-headline").inner_text()
        cap.check(
            re.search(r"ERRORED\s*\n?\s*[1-9]", errored_tile.upper()) is not None,
            "Errored tile counts the cases the judge could not score",
            errored_tile,
        )
        cap.shot(page, "189-eval-04-judge-outage-invalid")

        # -- 4. Compare keeps the errored cases ---------------------------------------
        print("▸ compare")
        cap.goto(page, f"/evals/compare?a={good['run_id']}&b={outage['run_id']}", 1500)
        cap.visible(
            page.get_by_text(re.compile(r"^Errored")).first, "compare has an Errored section"
        )
        cap.shot(page, "189-eval-05-compare-errored")

        # -- 5. Run eval from the Datasets page ---------------------------------------
        print("▸ run eval dialog")
        cap.goto(page, f"/datasets/{DATASET}", 1000)
        page.get_by_role("button", name="Run eval").click()
        dialog = page.get_by_role("dialog")
        cap.visible(dialog.get_by_role("heading", name="Run eval"), "Run eval dialog opens")
        cap.check(
            "support-agent (agent)" in dialog.get_by_role("combobox", name="Agent").inner_text(),
            "target lists the registered support-agent",
        )
        cap.visible(dialog.get_by_test_id("scorers-deterministic"), "Deterministic scorers group")
        cap.visible(dialog.get_by_test_id("scorers-llm"), "LLM judges group")
        dialog.get_by_label("exact_match", exact=True).click()  # off
        dialog.get_by_label("contains", exact=True).click()
        dialog.get_by_label("llm_judge", exact=True).click()
        dialog.get_by_label("llm_judge criteria").fill(CRITERIA)
        cap.check(
            "gpt-4o-mini" in dialog.get_by_label("llm_judge judge model").inner_text(),
            "the judge model list offers gpt-4o-mini (it has a key)",
        )
        warnings = dialog.get_by_test_id("run-eval-warnings")
        cap.text(
            warnings, "contains needs expected_output; 1 of 5 rows lack it: those cases are n/a"
        )
        cap.text(warnings, "~5 judge calls (5 rows × 1 judge scorer)")
        dialog.locator("#run-eval-name").fill("ui — contains + llm_judge")
        dialog.get_by_test_id("scorers-llm").scroll_into_view_if_needed()
        cap.shot(page, "189-eval-06-run-eval-dialog")

        dialog.get_by_role("button", name="Run", exact=True).click()
        result = dialog.get_by_test_id("run-eval-result")
        cap.visible(result, "the result is shown inline", timeout=120_000)
        cap.text(result, "Eval finished")
        cap.text(result, re.compile(r"Cases .*: \d+ passed · \d+ failed · \d+ errored"))
        cap.text(
            result,
            re.compile(r"Checks .*: \d+ passed · \d+ failed · 1 n/a"),
            "checks line counts the n/a verdict",
        )
        result.scroll_into_view_if_needed()
        cap.shot(page, "189-eval-07-run-eval-result")
        ui_runs = [
            r
            for r in cap.get("/api/evals?kind=ui")["rows"]
            if r["run_name"] == "ui — contains + llm_judge"
        ]
        cap.check(
            len(ui_runs) == 1 and ui_runs[0].get("agent_name") == "support-agent",
            "the UI run is stored as kind=ui for support-agent",
            str(ui_runs[:1]),
        )

        dialog.get_by_label("Dataset check (echo)").check()
        cap.text(
            dialog.get_by_test_id("run-eval-warnings"),
            "Dataset check: inputs are echoed back; the result does not measure a model",
        )

        browser.close()
    return cap.finish()


if __name__ == "__main__":
    if sys.argv[1:2] == ["seed"]:
        raise SystemExit(seed())
    if sys.argv[1:2] == ["drive"]:
        raise SystemExit(drive(sys.argv[2]))
    raise SystemExit("usage: capture_eval_ui.py seed | drive <base-url>")
