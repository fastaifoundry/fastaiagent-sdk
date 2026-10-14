"""1.89.0 Phase 2 (data integrity) — seed, then drive the Local UI and assert.

Called by ``scripts/capture-ws1-integrity.sh``:

    python scripts/capture_ws1_integrity.py seed            # cwd = the temp dir
    python scripts/capture_ws1_integrity.py drive <base-url>

What it proves, on screen:

* the prompt editor edits the stored template (``{{@fragment}}`` kept), and a
  saved version keeps the metadata and is labelled ``ui`` (P2, P3);
* an Agent-CI session records the verdict it enforces: a regression against
  the baseline is ``failed``, an unloadable baseline ``invalid`` (E1);
* the Playground checks max_tokens and the endpoint as you type, and the
  server's 422 never echoes the submitted value (P1);
* a stream that fails part-way is an error in the Playground and an ERROR
  trace (P5).

Offline: the agent is a ``FunctionModel``, the Playground talks to a localhost
SSE server (``tests/_sse_provider.py``) that drops the connection mid-stream.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

PROMPT = "support-reply"
RAW_TEMPLATE = "{{@tone}}\nReply to {{customer}} about {{topic}}."
FRAGMENT = "Be brief and kind."
METADATA = {"owner": "support-team", "ticket": "SUP-118"}

TEST_FILE = """
import os
from fastaiagent import Agent
from fastaiagent.eval import pytest_dataset as dataset
from fastaiagent.testing import FunctionModel

ANSWERS = {
    "What are your opening hours?": "9 to 5, Monday to Friday.",
    "Do you ship abroad?": "Yes, to 40 countries.",
    "Can I return a sale item?": "No, sale items are final.",
    "How long does a refund take?": "Up to 7 business days.",
}


def responder(messages):
    question = next(str(m.content) for m in reversed(messages) if m.role == "user")
    if os.environ.get("REGRESS") and question.startswith("How long"):
        return "Refunds are instant."  # the regression this PR introduced
    return ANSWERS.get(question, "I don't know.")


agent = Agent(name="faq-bot", llm=FunctionModel(responder, model="faq-model"))


@dataset("cases.jsonl")
def test_faq(eval_case, evaluate_one):
    evaluate_one(agent.run, scorers=["exact_match"])
"""


def seed() -> int:
    from fastaiagent.prompt.registry import PromptRegistry

    db = os.environ["FASTAIAGENT_LOCAL_DB"]
    reg = PromptRegistry(path=db)
    reg.register_fragment("tone", FRAGMENT)
    reg.register(PROMPT, RAW_TEMPLATE, metadata=METADATA)
    print(f"▸ prompt {PROMPT} v1 with {{{{@tone}}}} and metadata {METADATA}")

    # Three real pytest sessions through the plugin, into the same local.db.
    ci = Path("ci")
    ci.mkdir(exist_ok=True)
    (ci / "test_faq.py").write_text(TEST_FILE)
    rows = [
        ("What are your opening hours?", "9 to 5, Monday to Friday."),
        ("Do you ship abroad?", "Yes, to 40 countries."),
        ("Can I return a sale item?", "No, sale items are final."),
        ("How long does a refund take?", "Up to 7 business days."),
    ]
    (ci / "cases.jsonl").write_text(
        "".join(json.dumps({"input": q, "expected_output": a}) + "\n" for q, a in rows)
    )
    sessions = [
        ("main", {}, ["--eval-run-name", "main"], 0),
        ("pr-42", {"REGRESS": "1"}, ["--eval-baseline", "main", "--eval-run-name", "pr-42"], 1),
        (
            "pr-43",
            {},
            ["--eval-baseline", "release-candidate", "--eval-run-name", "pr-43"],
            None,
        ),
    ]
    for name, env, args, want in sessions:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "test_faq.py", *args],
            cwd=ci,
            env={**os.environ, **env},
            capture_output=True,
            text=True,
        )
        verdict = next(
            (ln for ln in proc.stdout.splitlines() if "fastaiagent eval verdict" in ln), "(none)"
        )
        print(f"▸ pytest session {name}: exit {proc.returncode} — {verdict.strip()}")
        if want is not None and proc.returncode != want:
            print(proc.stdout[-3000:], proc.stderr[-2000:])
            raise SystemExit(f"session {name} exited {proc.returncode}, expected {want}")
    return 0


# ---------------------------------------------------------------------------


def drive(base: str) -> int:
    from _capture189 import Capture, pick_provider
    from playwright.sync_api import sync_playwright

    from tests._sse_provider import Reply, SSEProvider, openai_stream

    cap = Capture("capture-ws1-integrity", base)
    stub = SSEProvider().start()

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = cap.new_page(browser)

        # -- 1. Prompt editor keeps fragments, metadata and authorship ----------
        print("▸ prompt editor")
        cap.goto(page, f"/prompts/{PROMPT}")
        editor = page.locator("textarea").first
        cap.visible(editor, "editor textarea renders")
        value = editor.input_value()
        cap.check("{{@tone}}" in value, "editor holds the stored template ({{@tone}} kept)", value)
        cap.check(FRAGMENT not in value, "editor does not hold the resolved fragment text", value)
        cap.text(page, "Fragments (kept as live references):")
        cap.visible(
            page.get_by_test_id("prompt-fragments").get_by_text("{{@tone}}"),
            "fragment chip {{@tone}}",
        )
        cap.shot(page, "189-ws1-01-prompt-editor-raw")

        editor.fill(RAW_TEMPLATE + "\nSign off as the support team.")
        page.get_by_role("button", name="Save as new version").click()
        cap.text(page, "Saved as v2", "toast: Saved as v2")
        cap.settle(page, 800)
        cap.visible(page.get_by_text(re.compile(r"·\s*ui$")), "v2 is labelled created_by ui")
        _, v2 = cap.api(f"/api/prompts/{PROMPT}/versions/2")
        cap.check(
            "{{@tone}}" in (v2 or {}).get("template_raw", ""),
            "v2 stored with {{@tone}} as a live reference",
            str(v2),
        )
        cap.check(
            FRAGMENT in (v2 or {}).get("template", ""),
            "v2 still resolves the fragment for a run",
            str(v2),
        )
        cap.check((v2 or {}).get("metadata") == METADATA, "v2 inherited v1's metadata", str(v2))
        cap.shot(page, "189-ws1-02-prompt-saved-v2")

        # -- 2. Agent-CI sessions record the verdict they enforce ----------------
        print("▸ agent-ci gate verdicts")
        runs = {r["run_name"]: r for r in cap.get("/api/evals?page_size=50")["rows"]}
        cap.check(
            {"main", "pr-42", "pr-43"} <= set(runs),
            "three pytest sessions persisted",
            str(list(runs)),
        )
        cap.goto(page, "/evals")
        for name, outcome in (("main", "passed"), ("pr-42", "failed"), ("pr-43", "invalid")):
            row = page.get_by_test_id(f"eval-row-{runs.get(name, {}).get('run_id')}")
            cap.visible(
                row.locator(f'[data-gate="{outcome}"]'), f"run {name} shows gate {outcome.upper()}"
            )
        cap.shot(page, "189-ws1-03-eval-runs-gates")

        cap.goto(page, f"/evals/{runs['pr-42']['run_id']}")
        panel = page.get_by_test_id("agent-ci-panel")
        cap.visible(panel.locator('[data-gate="failed"]'), "pr-42 Agent CI gate: FAILED")
        cap.visible(
            panel.get_by_test_id("gate-baseline").get_by_text("main", exact=True),
            "pr-42 baseline is main",
        )
        cap.text(panel, re.compile(r"1 regressed"), "pr-42 baseline: 1 regressed")
        cap.shot(page, "189-ws1-04-gate-regression-failed")

        cap.goto(page, f"/evals/{runs['pr-43']['run_id']}")
        cap.visible(
            page.get_by_test_id("agent-ci-panel").locator('[data-gate="invalid"]'),
            "pr-43 (missing baseline) Agent CI gate: INVALID",
        )

        # -- 3. Playground: invalid input is caught as you type ------------------
        print("▸ playground invalid input")
        cap.goto(page, "/playground")
        pick_provider(page, "openai")
        page.locator("#playground-model").fill("gpt-4o-mini")
        page.locator("#playground-template").fill("Say hello to {{name}}.")
        page.locator("#var-name").fill("Ada")
        page.get_by_role("button", name=re.compile(r"^Parameters")).click()
        page.locator("#param-max-tokens").fill("0")
        cap.text(page, "max_tokens must be a whole number from 1 to 200,000.")
        run_btn = page.get_by_role("button", name="Run", exact=True)
        cap.disabled(run_btn, "Run is disabled while max_tokens is 0")
        page.get_by_role("button", name=re.compile(r"^Connection")).click()
        page.locator("#playground-base-url").fill("ftp://gateway.internal")
        cap.text(page, "The endpoint must be an http:// or https:// URL.")
        cap.visible(page.locator("#playground-template"), "the form is still on the page")
        cap.shot(page, "189-ws1-05-playground-invalid-input")

        status, body = cap.api(
            "/api/playground/run",
            "POST",
            {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "prompt_template": "hi",
                "parameters": {"max_tokens": 0},
                "api_key": "sk-proj-ShouldNeverBeEchoed000000000000000000",
            },
        )
        cap.check(status == 422, "server answers 422 for max_tokens=0", str(status))
        detail = body.get("detail") if isinstance(body, dict) else body
        cap.check(
            isinstance(detail, list) and all("input" not in d for d in detail),
            "the 422 detail carries no `input` echo",
            json.dumps(detail)[:300],
        )
        cap.check("ShouldNeverBeEchoed" not in json.dumps(body), "the 422 never echoes the token")

        # -- 4. A stream that fails part-way is an error, and an ERROR trace -----
        print("▸ playground failed stream")
        page.locator("#param-max-tokens").fill("64")
        page.locator("#playground-base-url").fill(f"{stub.url}/v1")
        page.locator("#playground-api-key").fill("stub-token")
        stub.script(Reply(events=openai_stream("Hello", " Ada", ", nice"), cut_after=2))
        cap.enabled(run_btn, "Run is enabled once the input is valid")
        run_btn.click()
        err = page.get_by_test_id("run-error")
        cap.visible(err, "the failed stream shows the error panel", timeout=20_000)
        err_text = err.inner_text()
        print(f"    error panel: {err_text!r}")
        cap.check(
            "[object Object]" not in err_text and err_text.strip() != "",
            "the error panel reads as text",
            err_text,
        )
        page.get_by_text("// RESPONSE", exact=True).scroll_into_view_if_needed()
        cap.shot(page, "189-ws1-06-playground-failed-stream")

        cap.goto(page, "/traces?status=ERROR")
        row = page.locator("tr", has_text="playground.run")
        cap.visible(row, "Traces ?status=ERROR lists the failed playground.run")
        cap.visible(row.get_by_text("ERROR", exact=True), "its status is ERROR")
        cap.visible(row.get_by_test_id("trace-source-badge"), "it carries the playground badge")
        cap.shot(page, "189-ws1-07-failed-stream-trace")
        failed = [t for t in cap.traces("status=ERROR") if t["name"] == "playground.run"]
        if failed:
            cap.goto(page, f"/traces/{failed[0]['trace_id']}")
            cap.text(page, "source: playground", "trace detail subtitle names the source")
            spans = cap.get(f"/api/traces/{failed[0]['trace_id']}")["spans"]
            llm = [s for s in spans if s["name"].startswith("llm.")]
            cap.check(
                bool(llm) and all(s["status"] == "ERROR" for s in llm),
                "the streamed llm.* span ended ERROR",
                str([s["status"] for s in llm]),
            )

        browser.close()
    stub.stop()
    return cap.finish()


if __name__ == "__main__":
    if sys.argv[1:2] == ["seed"]:
        raise SystemExit(seed())
    if sys.argv[1:2] == ["drive"]:
        sys.argv = [sys.argv[0], *sys.argv[2:]]
        raise SystemExit(drive(sys.argv[1]))
    raise SystemExit("usage: capture_ws1_integrity.py seed | drive <base-url>")
