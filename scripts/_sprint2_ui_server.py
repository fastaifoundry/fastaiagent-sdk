"""Boot a Local UI server with Sprint 2 fixtures.

Used by ``scripts/capture-sprint2-screenshots.sh``. Sprint 2 needs:

* prompt-registry rows (for the Playground spec)
* a registered ``Supervisor`` runner (for the Agent Dependency Graph spec)

The supervisor wraps two real Worker agents (researcher, writer) — same
shape as ``examples/50_agent_dependencies.py``. The screenshot Playwright
spec drives ``/agents/planner`` and ``/agents/researcher`` against this
server.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import uvicorn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from fastaiagent import Agent, LLMClient, tool  # noqa: E402
from fastaiagent.agent.team import Supervisor, Worker  # noqa: E402
from fastaiagent.guardrail.builtins import no_pii  # noqa: E402
from fastaiagent.kb.local import LocalKB  # noqa: E402
from fastaiagent.ui.server import build_app  # noqa: E402


@tool(description="Look up a record by id.")
def lookup_record(record_id: str) -> str:
    return f"record-{record_id}"


@tool(description="Web search stub.")
def web_search(query: str) -> str:
    return f"results for '{query}'"


def _build_supervisor() -> Supervisor:
    """Mirror examples/50 so screenshot data matches the example demo."""
    llm = LLMClient(provider="openai", model="gpt-4o-mini")

    research_kb = LocalKB(
        name="research-kb", search_type="vector", persist=False
    )
    for text in (
        "FastAIAgent ships with workflow visualization.",
        "Sprint 2 adds the Prompt Playground and dependency graphs.",
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
            "You write a one-paragraph summary in {{style}} tone "
            "for the audience defined by {{audience}}. "
            "Use lookup_record or web_search when you need source data."
        ),
        llm=llm,
        tools=[lookup_record, web_search],
    )

    return Supervisor(
        name="planner",
        llm=llm,
        workers=[
            Worker(agent=researcher, role="researcher"),
            Worker(agent=writer, role="writer"),
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7845)
    parser.add_argument("--project-id", default="sprint2-demo")
    args = parser.parse_args()

    supervisor = _build_supervisor()
    app = build_app(
        db_path=args.db,
        no_auth=True,
        runners=[supervisor],
        project_id=args.project_id,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
