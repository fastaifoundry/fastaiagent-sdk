"""Example 105: replay a run that made Decisions API calls — offline (1.84.0).

A real agent (``gpt-4o-mini``) triages a ticket by calling a tool that asks the
Decisions API. The run is traced to a local store. Then the trace is replayed
with ``determinism="recorded"`` and ``on_miss="error"``: every chat turn **and
every decision** is served from the capture, so the rerun makes no model calls
and fails loudly if it would need one. That is what makes a regression suite
over decision-driven agents deterministic and free.

(1.84.0 also fixed the replay of turns that only call tools — before it, a
recorded rerun skipped the tool entirely and still matched the original output.)

Usage:
    zsh -lc 'python examples/105_decision_replay.py'   # needs OPENAI_API_KEY

Output from a live run (the agent's wording varies; the replay repeats it exactly):
    live run:   Your message will be handled by the billing department.
    replay:     Your message will be handled by the billing department.
      llm.openai.gpt-4o-mini             replay.mode=recorded
      llm.openai.decisions.gpt-6-luna    replay.mode=recorded
      llm.openai.gpt-4o-mini             replay.mode=recorded
"""

from __future__ import annotations

import asyncio
import os
import tempfile


def main() -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set — it's in ~/.zshrc; run via zsh -lc.")
        return 1
    # A throwaway trace store so the example leaves your local.db alone —
    # unless you point it somewhere on purpose (the docs screenshots do).
    os.environ.setdefault("FASTAIAGENT_LOCAL_DB", os.path.join(tempfile.mkdtemp(), "local.db"))

    from fastaiagent.agent import Agent
    from fastaiagent.llm import Choice, LLMClient
    from fastaiagent.tool import FunctionTool
    from fastaiagent.trace import otel
    from fastaiagent.trace.replay import Replay

    decider = LLMClient(model="gpt-6-luna")

    async def route_ticket(text: str) -> str:
        """Route a customer support ticket; returns the department name."""
        r = await decider.adecide(
            text,
            Choice(name="department", instructions="Which department?", options=["billing", "technical", "other"]),
        )
        return str(r["department"].choice)

    agent = Agent(
        name="router",
        llm=LLMClient(provider="openai", model="gpt-4o-mini"),
        tools=[FunctionTool(name="route_ticket", fn=route_ticket)],
        system_prompt="Always call route_ticket with the user's message, then say which department will help.",
    )
    live = agent.run("I was charged twice for my renewal.")
    otel.get_tracer_provider().force_flush()
    print(f"live run:   {live.output}")

    rerun = asyncio.run(
        Replay.load(live.trace_id)
        .fork_at(step=0)
        .with_determinism("recorded", on_miss="error")
        .arerun()
    )
    print(f"replay:     {rerun.new_output}")
    spans = Replay.load(rerun.trace_id)._trace.spans
    for s in spans:
        if s.name.startswith("llm."):
            print(f"  {s.name:<34} replay.mode={s.attributes.get('replay.mode')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
