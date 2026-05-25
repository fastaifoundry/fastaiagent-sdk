"""Example 62: Failure trace → regression test → eval suite.

The full "every production failure becomes a test" loop:

1. Run a real agent → capture trace_id
2. Load the failing trace and fork it
3. Modify the prompt to fix the behavior
4. Rerun the fork to confirm the fix
5. ``rerun.save_as_test(...)`` — append the case to a JSONL dataset
6. ``evaluate(...)`` against that dataset to catch future regressions

The same JSONL works with both string matchers (``exact_match``) and
LLM-as-judge scorers (``LLMJudge``). This example demonstrates both.

> **Looking for the full template?** ``examples/regression-from-trace/``
> ships a 5-script template (capture / analyze / fix / save_test /
> verify), seeded dataset, smoke tests, browser screenshots, and docs.
> This file (``examples/62_*``) remains as the single-file quick
> reference; reach for the template when you want CI integration and
> a real-world tool-fix walkthrough.

Usage::

    # API keys are loaded from ~/.zshrc, so run via a login shell:
    zsh -lc 'python examples/62_replay_to_regression.py'

    # Or set them inline:
    OPENAI_API_KEY=sk-... python examples/62_replay_to_regression.py
"""

from __future__ import annotations

import os
from pathlib import Path

from fastaiagent import Agent, LLMClient
from fastaiagent.eval import LLMJudge, evaluate
from fastaiagent.trace.replay import Replay


def _require_key() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        print("Skipping: OPENAI_API_KEY not set.")
        print("Run via: zsh -lc 'python examples/62_replay_to_regression.py'")
        raise SystemExit(0)


def _build_agent(system_prompt: str) -> Agent:
    return Agent(
        name="support-bot",
        system_prompt=system_prompt,
        llm=LLMClient(provider="openai", model="gpt-4.1-mini"),
    )


if __name__ == "__main__":
    _require_key()

    # .fastaiagent/ is the SDK's default local data dir and is already
    # gitignored — demo output lives here so we don't pollute the repo.
    dataset_path = Path(".fastaiagent/regression_demo/regression_tests.jsonl")
    dataset_path.parent.mkdir(parents=True, exist_ok=True)

    # ── 1. Reproduce a real failure ─────────────────────────────────────
    # The original prompt is sloppy — the agent gives a verbose answer
    # when the support team wants a single sentence.
    sloppy_prompt = "You are a customer support agent. Be helpful."
    print("Step 1: Running agent with sloppy prompt...")
    bad_agent = _build_agent(sloppy_prompt)
    bad_result = bad_agent.run("What is our refund policy?")
    print(f"  Trace ID: {bad_result.trace_id}")
    print(f"  Output:   {bad_result.output[:120]}...")
    print()

    # ── 2. Fork the failing trace ───────────────────────────────────────
    print("Step 2: Forking the failing trace at step 0...")
    assert bad_result.trace_id
    replay = Replay.load(bad_result.trace_id)
    forked = replay.fork_at(step=0)

    # ── 3. Apply the fix ────────────────────────────────────────────────
    fixed_prompt = (
        "You are a customer support agent. Answer in exactly one sentence. "
        "Our refund policy: 30 days, full refund, no questions asked."
    )
    print("Step 3: Modifying prompt with the fix...")
    forked.modify_prompt(fixed_prompt)

    # ── 4. Rerun to confirm the fix ─────────────────────────────────────
    print("Step 4: Rerunning to confirm the fix...")
    rerun = forked.rerun()
    print(f"  Fixed output: {rerun.new_output}")
    print()

    # ── 5. Save the rerun as a regression test ──────────────────────────
    # The JSONL line uses field names ``evaluate()`` reads directly.
    # ``source_trace_id`` keeps the link back to the original failure.
    print(f"Step 5: Saving as regression test → {dataset_path}")
    rerun.save_as_test(
        dataset_path,
        input="What is our refund policy?",
        expected_output=str(rerun.new_output),
        source_trace_id=bad_result.trace_id,
    )
    print()

    # ── 6a. Re-run eval with exact_match (string scorer) ────────────────
    # Using a fresh agent that has the fixed prompt baked in — the case
    # should pass deterministically.
    print("Step 6a: Running eval suite with exact_match scorer...")
    fixed_agent = _build_agent(fixed_prompt)
    results = evaluate(
        agent_fn=lambda text: fixed_agent.run(text).output,
        dataset=str(dataset_path),
        scorers=["exact_match"],
        persist=False,
    )
    exact = results.scores["exact_match"][0]
    print(f"  exact_match: passed={exact.passed} score={exact.score}")
    print()

    # ── 6b. Re-run eval with LLMJudge (LLM-as-judge scorer) ─────────────
    # LLM-as-judge is more forgiving than exact_match — it scores
    # semantic correctness, so a paraphrase still passes. Same dataset,
    # different scorer.
    print("Step 6b: Running eval suite with LLMJudge scorer...")
    judge_results = evaluate(
        agent_fn=lambda text: fixed_agent.run(text).output,
        dataset=str(dataset_path),
        scorers=[LLMJudge(criteria="correctness")],
        persist=False,
    )
    judged = judge_results.scores["llm_judge"][0]
    print(f"  llm_judge:   passed={judged.passed} score={judged.score}")
    print(f"  reason:      {judged.reason}")
    print()

    print("Done. The regression case is now in:")
    print(f"  {dataset_path.resolve()}")
    print("Future runs of evaluate() against this dataset will catch any")
    print("regression of the same failure mode.")
