"""Example 80: Curate an eval dataset from captured agent traces.

Demonstrates (real Agent + real LLM — needs OPENAI_API_KEY):
- Running an agent captures real traces in the local DB.
- Dataset.from_traces(...) turns those traces into eval items (one per
  agent.<name> span — root or nested in a chain/supervisor/swarm).
- Dataset.to_jsonl(...) writes them; evaluate() scores the curated set.

Good filters (all/favorites/noted) use the captured output as the gold
expected_output; failure filters (guardrail/failed) mark cases needs_review.

Run:
    zsh -lc 'python examples/80_curate_from_traces.py'
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

from fastaiagent import Agent, LLMClient
from fastaiagent.eval import Dataset, evaluate


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY to run this example.")
        sys.exit(0)

    agent = Agent(
        name="curate-demo",
        system_prompt="You are a concise FAQ bot for an online store. Answer in one sentence.",
        llm=LLMClient(provider="openai", model="gpt-4o-mini"),
    )

    print("== Run the agent (captures real traces) ==")
    for q in [
        "What is your return window?",
        "Do you ship internationally?",
        "How do I track my order?",
    ]:
        r = agent.run(q)
        print(f"  Q: {q}\n  A: {r.output}")

    # Curate only this agent's traces from the last hour. Each agent span becomes
    # one case; for filter='all' the captured output is the gold expected_output.
    print("\n== Dataset.from_traces ==")
    ds = Dataset.from_traces(filter="all", agent="curate-demo", since_hours=1)
    print(f"  curated {len(ds)} case(s)")
    for item in ds:
        print(f"   - input={item['input']!r}  expected={item['expected_output']!r}")

    out = Path(tempfile.gettempdir()) / "curate_demo_cases.jsonl"
    ds.to_jsonl(out)
    print(f"  wrote {out}")

    # Re-evaluate the curated set against the agent (answer_relevancy needs no
    # exact match, so it's robust to LLM wording drift).
    print("\n== Re-evaluate the curated set ==")
    results = evaluate(
        agent_fn=agent.run,
        dataset=str(out),
        scorers=["answer_relevancy"],
        persist=False,
    )
    print(results.summary())


if __name__ == "__main__":
    main()
