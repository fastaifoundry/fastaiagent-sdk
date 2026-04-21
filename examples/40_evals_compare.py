"""Example 40 — Eval runs: before-and-after with the compare view.

Walks the common "did my prompt edit help?" loop end-to-end:

  1. Define a 5-case dataset.
  2. Run ``evaluate()`` against an agent with a **vague** system prompt →
     this is the BEFORE run (usually misses one or two cases).
  3. Run ``evaluate()`` again with a tighter prompt → the AFTER run.
  4. Both runs persist to ``.fastaiagent/local.db`` automatically.
  5. Print the Local UI URLs so you can open ``/evals/compare?a=…&b=…`` and
     see exactly which cases regressed or improved.

Prereqs:
    pip install 'fastaiagent[ui,openai]'
    export OPENAI_API_KEY=sk-...

Run:
    python examples/40_evals_compare.py
    fastaiagent ui   # open the printed compare URL
"""

from __future__ import annotations

import os
import sys

from fastaiagent import Agent, LLMClient
from fastaiagent.eval import evaluate

DATASET = [
    {
        "input": "Reply with exactly the word 'ready'.",
        "expected_output": "ready",
    },
    {
        "input": "Reply with exactly the word 'yes'.",
        "expected_output": "yes",
    },
    {
        "input": "Reply with exactly the word 'done'.",
        "expected_output": "done",
    },
    {
        "input": "Reply with exactly the word 'stop'.",
        "expected_output": "stop",
    },
    {
        "input": "Reply with exactly the word 'ok'.",
        "expected_output": "ok",
    },
]


BEFORE_PROMPT = (
    # Vague + a filler directive that reliably taints the output so the "BEFORE"
    # run is actually worse than the "AFTER" one — otherwise gpt-4o-mini is good
    # enough to pass both and the compare view has nothing to show.
    "You are a friendly support agent. Always begin every reply with the "
    "word 'Sure!' followed by a space, then answer the user."
)
AFTER_PROMPT = (
    "You are a terse echo bot. When asked to reply with a single exact word, "
    "your entire response MUST be just that word — no punctuation, no "
    "capitalization changes, no extra text, no preamble."
)


def require_key() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is required for this example.")
        sys.exit(0)


def section(title: str) -> None:
    print()
    print(f"── {title} ".ljust(72, "─"))


def build_agent(system_prompt: str) -> Agent:
    return Agent(
        name="echo-bot",
        system_prompt=system_prompt,
        llm=LLMClient(provider="openai", model="gpt-4o-mini"),
    )


def main() -> None:
    require_key()

    section("Run A — vague prompt (expected to leak extra words)")
    before = evaluate(
        agent_fn=build_agent(BEFORE_PROMPT).run,
        dataset=DATASET,
        scorers=["exact_match"],
        run_name="echo-bot-v1-vague",
        dataset_name="echo-strict.jsonl",
        agent_name="echo-bot",
        agent_version="v1",
    )
    print(f"  {before.summary()}")

    section("Run B — tight prompt (should pass all 5)")
    after = evaluate(
        agent_fn=build_agent(AFTER_PROMPT).run,
        dataset=DATASET,
        scorers=["exact_match"],
        run_name="echo-bot-v2-strict",
        dataset_name="echo-strict.jsonl",
        agent_name="echo-bot",
        agent_version="v2",
    )
    print(f"  {after.summary()}")

    run_a = before.run_id
    run_b = after.run_id
    base = os.environ.get("FASTAIAGENT_UI_URL", "http://127.0.0.1:7842")

    section("Done — open the compare view")
    print(f"  {base}/evals")
    print("    ← cost + latency columns")
    print(f"  {base}/evals/{run_a}")
    print("    ← filters + scorer chips + expandable diffs")
    print(f"  {base}/evals/{run_b}")
    print(
        f"  {base}/evals/compare?a={run_a}&b={run_b}"
    )
    print(
        "\n  On the compare page you'll see regressed and improved cases "
        "rendered as side-by-side diffs. Cases that passed in both runs (or "
        "failed in both) are counted but not expanded, so the page focuses "
        "on what actually changed."
    )


if __name__ == "__main__":
    main()
