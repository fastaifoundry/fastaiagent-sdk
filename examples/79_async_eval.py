"""Example 79: Async evaluation APIs against a REAL agent.

Demonstrates (real Agent + real LLM — needs OPENAI_API_KEY):
- aevaluate()          : async batch evaluation
- agenerate_scenarios(): async scenario generation by agent introspection
- asimulate()          : async multi-turn simulation
- aharden()            : async failure-to-fix recommendations

Every `evaluate`/`simulate`/`generate_scenarios`/`harden` entry point has an
`a`-prefixed coroutine for use inside async apps (FastAPI, etc.). The sync
versions just wrap these.

Run:
    zsh -lc 'python examples/79_async_eval.py'
"""

from __future__ import annotations

import asyncio
import os
import sys

from fastaiagent import Agent, LLMClient, Scorecard
from fastaiagent.eval import agenerate_scenarios, aharden, asimulate
from fastaiagent.eval.evaluate import aevaluate


async def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY to run this example.")
        sys.exit(0)

    llm = LLMClient(provider="openai", model="gpt-4o-mini")
    agent = Agent(
        name="support",
        system_prompt="You are a brief support agent for an online store.",
        llm=llm,
    )

    print("== aevaluate ==")
    results = await aevaluate(
        agent.run,
        [
            {"input": "Do you accept returns? Answer yes or no.", "expected": "yes"},
            {"input": "Is contacting support free? Answer yes or no.", "expected": "yes"},
        ],
        scorers=["contains", "answer_relevancy"],
        persist=False,
    )
    print(results.summary())

    print("\n== agenerate_scenarios -> asimulate -> Scorecard -> aharden ==")
    scenarios = await agenerate_scenarios(agent, n=2, llm=llm, focus="refunds and returns")
    for s in scenarios:
        print(f"  • {s.name}")

    sim = await asimulate(scenarios, agent, persist=False)
    print(sim.summary())
    print(Scorecard.from_simulation(sim).summary())

    report = await aharden(agent, sim, llm=llm)
    print(report.summary())


if __name__ == "__main__":
    asyncio.run(main())
