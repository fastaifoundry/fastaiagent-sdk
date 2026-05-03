"""Example 52 — Trace Comparison demo (Sprint 3).

Runs the same agent twice — once with a terse system prompt, once with a
verbose one — on the same input. Both traces land in ``./.fastaiagent/local.db``,
and the script prints the URL for the side-by-side comparison view.

The Trace Comparison view (Sprint 3) generalises Replay's "original vs
forked" diff to *any* two traces — useful for prompt A/B testing,
regression detection, and "why did Monday's run differ from Friday's"
debugging.

Prereqs:
    pip install 'fastaiagent[ui,openai]'
    export OPENAI_API_KEY=sk-...

Run:
    python examples/52_trace_compare.py
    fastaiagent ui --no-auth
    # Open the printed /traces/compare?a=…&b=… URL.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("FASTAIAGENT_UI_ENABLED", "true")

from fastaiagent import Agent, LLMClient  # noqa: E402


TERSE = (
    "You are a customer-support assistant. Reply in under 15 words. "
    "Never list bullet points; one short sentence is enough."
)

VERBOSE = (
    "You are a customer-support assistant. Walk the customer through the "
    "answer step-by-step. Use bullet points where helpful. Length is fine "
    "as long as you're being clear and complete."
)

QUESTION = "What's your refund policy and how do I start a return?"


def _run_one(name: str, system_prompt: str) -> str:
    llm = LLMClient(provider="openai", model="gpt-4o-mini")
    agent = Agent(name=name, system_prompt=system_prompt, llm=llm)
    print(f"► running '{name}'…")
    result = agent.run(QUESTION)
    print(f"  trace_id: {result.trace_id}")
    print(f"  output:   {result.output[:120]}")
    return result.trace_id


def main() -> int:
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required to run this example.")

    a = _run_one("compare-demo-terse", TERSE)
    b = _run_one("compare-demo-verbose", VERBOSE)

    print()
    print("Both traces written to ./.fastaiagent/local.db")
    print()
    print("Open the comparison view:")
    print(f"    http://127.0.0.1:7842/traces/compare?a={a}&b={b}")
    print()
    print("Or hit the API directly:")
    print(f"    curl 'http://127.0.0.1:7842/api/traces/compare?a={a}&b={b}'")
    print()
    print("Tip: from /traces, check both rows and click 'Compare' in the action bar.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
