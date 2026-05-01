"""Example 31: Swarm — peer-to-peer handoff between a triage → coder/writer team.

Demonstrates:
  1. A 3-agent swarm where a triage agent routes to the right specialist
  2. Real tools on one of the peers (a fake "search" function)
  3. Shared blackboard via handoff context
  4. Constrained handoff allowlist (triage fans out; specialists terminate)
  5. Stream output with HandoffEvent to surface agent transitions

Usage:
    export OPENAI_API_KEY=sk-...   # or ANTHROPIC_API_KEY
    python examples/31_swarm_research_team.py
"""

from __future__ import annotations

import os
import sys

from fastaiagent import (
    Agent,
    FunctionTool,
    HandoffEvent,
    LLMClient,
    Swarm,
    TextDelta,
)


def _pick_llm() -> LLMClient:
    if os.environ.get("OPENAI_API_KEY"):
        print("Using OpenAI gpt-4o-mini\n")
        return LLMClient(provider="openai", model="gpt-4o-mini")
    if os.environ.get("ANTHROPIC_API_KEY"):
        print("Using Anthropic claude-haiku-4-5-20251001\n")
        return LLMClient(provider="anthropic", model="claude-haiku-4-5-20251001")
    print("Set OPENAI_API_KEY or ANTHROPIC_API_KEY.")
    sys.exit(1)


# --- A pretend search tool the coder can call ------------------------------


def pypi_lookup(package: str) -> str:
    """Pretend to look up a package on PyPI."""
    catalog = {
        "requests": "requests — HTTP for humans. Latest: 2.32.3",
        "pydantic": "pydantic — Data validation. Latest: 2.10.4",
        "rich": "rich — Terminal formatting. Latest: 13.9.4",
    }
    return catalog.get(
        package.lower(),
        f"{package!r} not in cached catalog; try the real PyPI.",
    )


search_tool = FunctionTool(
    name="pypi_lookup",
    fn=pypi_lookup,
    description="Look up a Python package on a cached PyPI snapshot.",
    parameters={
        "type": "object",
        "properties": {"package": {"type": "string"}},
        "required": ["package"],
    },
)


# --- The three peers -------------------------------------------------------


def build_swarm(llm: LLMClient) -> Swarm:
    triage = Agent(
        name="triage",
        system_prompt=(
            "You are a triage agent. Read the user's request and hand off to "
            "the right specialist immediately — do NOT answer the question "
            "yourself. Use handoff_to_coder for Python / code questions, "
            "handoff_to_writer for prose / wording questions. When handing "
            "off, summarize the request in the `reason` argument."
        ),
        llm=llm,
    )
    coder = Agent(
        name="coder",
        system_prompt=(
            "You are a senior Python developer. When asked about a specific "
            "package, call pypi_lookup(package) to get details. Answer "
            "concisely with a code example if appropriate. Do not hand off."
        ),
        llm=llm,
        tools=[search_tool],
    )
    writer = Agent(
        name="writer",
        system_prompt=(
            "You are a prose editor. Help with writing, grammar, and clarity. "
            "Answer concisely. Do not hand off."
        ),
        llm=llm,
    )

    return Swarm(
        name="triage_swarm",
        agents=[triage, coder, writer],
        entrypoint="triage",
        handoffs={
            "triage": ["coder", "writer"],
            "coder": [],
            "writer": [],
        },
        max_handoffs=3,
    )


# --- Runs ------------------------------------------------------------------


def demo_sync(swarm: Swarm, prompt: str) -> None:
    print(f"--- SYNC: {prompt!r} ---")
    result = swarm.run(prompt)
    print("Output:", result.output[:500])
    handoffs = [c for c in result.tool_calls if c.get("tool_name", "").startswith("handoff_to_")]
    print(f"Handoffs observed: {[c['tool_name'] for c in handoffs]}")
    print()


def demo_stream(swarm: Swarm, prompt: str) -> None:
    import asyncio

    async def _run() -> None:
        print(f"--- STREAM: {prompt!r} ---")
        async for event in swarm.astream(prompt):
            if isinstance(event, TextDelta):
                print(event.text, end="", flush=True)
            elif isinstance(event, HandoffEvent):
                print(
                    f"\n  [↪ handoff {event.from_agent} → {event.to_agent}: "
                    f"{event.reason}]\n",
                    flush=True,
                )
        print("\n")

    asyncio.run(_run())


def main() -> None:
    llm = _pick_llm()
    swarm = build_swarm(llm)

    demo_sync(swarm, "How do I reverse a list in Python? One-liner.")
    demo_sync(
        swarm,
        "What's the latest version of the 'requests' package, and give me "
        "a one-line install command?",
    )
    demo_stream(
        swarm,
        "Please rewrite this to be more professional: 'hey this thing kinda works lol'",
    )

    print(
        "\nTo render the swarm topology (peer-to-peer handoff edges) in "
        "the Local UI, register the Swarm with build_app:\n"
        "    from fastaiagent.ui.server import build_app\n"
        "    app = build_app(runners=[swarm])\n"
        "Then visit http://127.0.0.1:7843/workflows/swarm/triage_swarm"
    )


if __name__ == "__main__":
    main()
