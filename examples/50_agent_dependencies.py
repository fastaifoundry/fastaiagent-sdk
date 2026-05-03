"""Example 50 — Agent Dependency Graph.

Builds a Supervisor with two Worker agents — each with a different toolset
and KB — registers them via ``build_app(runners=[...])``, runs the
supervisor once so spans land in local.db, and prints the URL where the
Local UI's Dependencies tab will render the graph.

Prereqs:
    pip install 'fastaiagent[ui,openai,kb]'
    export OPENAI_API_KEY=sk-...

Run:
    python examples/50_agent_dependencies.py
    # then in another shell:
    fastaiagent ui --no-auth
    # open http://127.0.0.1:7842/agents/planner → Dependencies tab

The Dependencies tab shows the agent at the centre with its tools / KBs /
prompts / guardrails / model radiating out. For Supervisors, each worker
appears as a sub-agent with its own subtree. For Swarms (uncomment the
swarm demo at the bottom), peers appear as siblings with handoff edges.
"""

from __future__ import annotations

import sys

import uvicorn

from fastaiagent import Agent, LLMClient, tool
from fastaiagent.agent.team import Supervisor, Worker
from fastaiagent.guardrail.builtins import no_pii
from fastaiagent.kb.local import LocalKB
from fastaiagent.ui.server import build_app


@tool(description="Look up a record by id.")
def lookup_record(record_id: str) -> str:
    return f"record-{record_id}"


@tool(description="Web search stub.")
def web_search(query: str) -> str:
    return f"results for '{query}'"


def main() -> int:
    llm = LLMClient(provider="openai", model="gpt-4o-mini")

    # ── Workers ───────────────────────────────────────────────────────
    research_kb = LocalKB(name="research-kb", search_type="vector", persist=False)
    for text in (
        "FastAIAgent has a workflow visualization view.",
        "Sprint 2 adds a Playground.",
    ):
        research_kb.add(text, metadata={"source": "docs"})
    researcher = Agent(
        name="researcher",
        system_prompt=(
            "You research topics. Use {{focus}} as the lens. "
            "Search the web and the local KB."
        ),
        llm=llm,
        tools=[web_search, research_kb.as_tool()],
        guardrails=[no_pii()],
    )

    writer = Agent(
        name="writer",
        system_prompt=(
            "You write a one-paragraph summary in {{style}} tone. "
            "Use lookup_record when you need source data."
        ),
        llm=llm,
        tools=[lookup_record],
    )

    supervisor = Supervisor(
        name="planner",
        llm=llm,
        workers=[
            Worker(
                agent=researcher,
                role="researcher",
                description="Searches for info",
            ),
            Worker(agent=writer, role="writer", description="Writes content"),
        ],
    )

    print("Registered Supervisor 'planner' with workers: researcher, writer")
    print()
    print("Open the Dependencies tab in your browser:")
    print("  http://127.0.0.1:7842/agents/planner")
    print("  http://127.0.0.1:7842/agents/researcher")
    print("  http://127.0.0.1:7842/agents/writer")
    print()
    print("Booting the UI on port 7842 (Ctrl+C to stop)…")

    app = build_app(no_auth=True, runners=[supervisor])
    uvicorn.run(app, host="127.0.0.1", port=7842, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
