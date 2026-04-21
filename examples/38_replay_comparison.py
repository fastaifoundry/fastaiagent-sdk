"""Example 38 — Agent Replay: bug-fix workflow with side-by-side comparison.

Walks an end-to-end story:

  1. Ship a customer-support agent with a **deliberately vague** system
     prompt that hallucinates a refund policy.
  2. Catch the bad run in the trace store.
  3. Load it with ``Replay.load(trace_id)``.
  4. Fork at the LLM call and rewrite the prompt with the *actual* policy.
  5. Rerun the fork against the real model.
  6. Compare original vs rerun with ``forked.compare(result)``:
      * print the divergence point
      * print both outputs side by side
      * assert the fix took effect
  7. Save the corrected case to ``.fastaiagent/regression_tests.jsonl`` so
     you never regress on this prompt bug again.
  8. Print three UI deep links so you can visually verify the fix in the
     Local UI's side-by-side replay view.

Prereqs:
    pip install 'fastaiagent[ui,openai]'
    export OPENAI_API_KEY=sk-...

Run:
    python examples/38_replay_comparison.py
    fastaiagent ui   # then open the printed /replay URL
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from fastaiagent import Agent, LLMClient
from fastaiagent.trace.replay import ForkedReplay, Replay, ReplayResult

# ─── Bug scenario ──────────────────────────────────────────────────────────
# Real refund policy (from company wiki): 7 business days.
# The v1 system prompt is too vague, so the model invents "14 days".

BUGGY_SYSTEM_PROMPT = (
    "You are a customer support agent. Help the user with any question."
)
FIXED_SYSTEM_PROMPT = (
    "You are a customer support agent. Answer refund questions using this "
    "exact policy: refunds are processed within 7 business days of us "
    "receiving the return. Be concise. Reply in one sentence."
)
QUERY = "When do refunds get processed?"


def build_agent(system_prompt: str, name: str = "support-bot") -> Agent:
    return Agent(
        name=name,
        system_prompt=system_prompt,
        llm=LLMClient(provider="openai", model="gpt-4o-mini"),
    )


def section(title: str) -> None:
    print()
    print(f"── {title} ".ljust(76, "─"))


def require_key() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is required for this example.")
        print("  export OPENAI_API_KEY=sk-...")
        sys.exit(0)


# ─── 1. Produce the buggy trace ────────────────────────────────────────────


def produce_buggy_trace() -> str:
    section("Step 1 — run the agent with the buggy prompt")
    agent = build_agent(BUGGY_SYSTEM_PROMPT)
    result = agent.run(QUERY)
    print(f"  query:    {QUERY}")
    print(f"  output:   {result.output.strip()}")
    print(f"  trace_id: {result.trace_id}")
    assert result.trace_id, "agent.run() did not emit a trace"
    return str(result.trace_id)


# ─── 2. Load the replay ────────────────────────────────────────────────────


def load_and_inspect(trace_id: str) -> Replay:
    section("Step 2 — load the replay and inspect steps")
    replay = Replay.load(trace_id)
    print(replay.summary())
    print()
    print("  Steps:")
    for step in replay.step_through():
        kind = step.attributes.get("gen_ai.request.model") or step.attributes.get(
            "tool.name"
        )
        extra = f" [{kind}]" if kind else ""
        print(f"    [{step.step}] {step.span_name}{extra}")
    return replay


# ─── 3. Fork + fix + rerun ─────────────────────────────────────────────────


def find_llm_step(replay: Replay) -> int:
    """Pick the first step whose span is an LLM call — that's where we fork."""
    for step in replay.step_through():
        if step.span_name.startswith("llm.") or "gen_ai.request.model" in step.attributes:
            return step.step
    # Fallback: fork at the second-to-last step so the agent still has work to do.
    return max(0, len(replay.steps()) - 2)


def fork_fix_rerun(replay: Replay) -> tuple[int, ForkedReplay, ReplayResult]:
    section("Step 3 — fork at the LLM call, replace the prompt, rerun")
    fork_point = find_llm_step(replay)
    print(f"  forking at step {fork_point}")
    forked = replay.fork_at(step=fork_point)
    forked.modify_prompt(FIXED_SYSTEM_PROMPT)
    print(f"  new system_prompt: {FIXED_SYSTEM_PROMPT!r}")
    rerun_result = forked.rerun()
    print(f"  rerun trace_id:    {rerun_result.trace_id}")
    return fork_point, forked, rerun_result


# ─── 4. Compare ────────────────────────────────────────────────────────────


def compare_outputs(forked: ForkedReplay, rerun_result: ReplayResult) -> None:
    section("Step 4 — side-by-side comparison")
    comparison = forked.compare(rerun_result)
    print(f"  diverged_at:     {comparison.diverged_at}")
    print(f"  original steps:  {len(comparison.original_steps)}")
    print(f"  rerun steps:     {len(comparison.new_steps)}")
    print()
    print("  ORIGINAL OUTPUT:")
    print(f"    {rerun_result.original_output!r}")
    print("  REWRITTEN OUTPUT:")
    print(f"    {rerun_result.new_output!r}")

    # Prove the fix took: the corrected policy mentions '7', the buggy one doesn't.
    original = str(rerun_result.original_output or "").lower()
    fixed = str(rerun_result.new_output or "").lower()
    if "7" in fixed and "7" not in original:
        print()
        print("  ✓ fix verified: corrected output cites the 7-day policy,")
        print("    original did not.")
    else:
        print()
        print("  ⚠ the fix may not have fully landed — inspect the diff in")
        print("    the Local UI (link below) to decide whether to iterate.")


# ─── 5. Save as regression test ────────────────────────────────────────────


def save_regression(trace_id: str, rerun_result: ReplayResult) -> Path:
    section("Step 5 — save as regression test")
    path = Path(".fastaiagent") / "regression_tests.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    case = {
        "source_trace_id": trace_id,
        "input": QUERY,
        "expected_output_contains": "7",
        "expected_output_does_not_contain": "14",
        "fixed_system_prompt": FIXED_SYSTEM_PROMPT,
        "rerun_trace_id": getattr(rerun_result, "trace_id", None),
    }
    with path.open("a") as f:
        f.write(json.dumps(case) + "\n")
    print(f"  appended case to {path}")
    return path


# ─── 6. UI deep links ──────────────────────────────────────────────────────


def print_ui_links(trace_id: str, rerun_trace_id: str | None) -> None:
    section("Step 6 — open the Local UI to see the side-by-side view")
    base = os.environ.get("FASTAIAGENT_UI_URL", "http://127.0.0.1:7842")
    print("  Start the UI in another shell:")
    print("    fastaiagent ui")
    print()
    print("  Then open one of these URLs in your browser:")
    print(f"    {base}/traces/{trace_id}")
    print("      → trace detail (span tree, tokens, cost)")
    print(f"    {base}/traces/{trace_id}/replay")
    print("      → Agent Replay page — click any span, pick 'Fork here'")
    if rerun_trace_id:
        print(f"    {base}/traces/{rerun_trace_id}")
        print("      → the rerun you just produced from code")
    print()
    print("  In the Replay page you'll see the same forked/rerun data as")
    print("  above, but rendered as a split view with the diverged step")
    print("  highlighted. The 'Save as regression test' button on that")
    print("  page writes to the same file this script just appended to.")


# ─── main ──────────────────────────────────────────────────────────────────


def main() -> None:
    require_key()
    trace_id = produce_buggy_trace()
    replay = load_and_inspect(trace_id)
    _fork_point, forked, rerun_result = fork_fix_rerun(replay)
    compare_outputs(forked, rerun_result)
    save_regression(trace_id, rerun_result)
    print_ui_links(trace_id, getattr(rerun_result, "trace_id", None))


if __name__ == "__main__":
    main()
