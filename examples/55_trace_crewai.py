"""Example 55 — Auto-trace a CrewAI run.

Spawns a one-agent / one-task crew on ``gpt-4o-mini`` (via litellm) and
shows the resulting trace in the FastAIAgent Local UI. ``ca.enable()``
is the only line of glue — every ``Crew.kickoff()`` from this point on
emits a full ``crewai.crew → crewai.agent → crewai.task → llm.<model>``
trace into ``.fastaiagent/local.db``.

Run:
    pip install "fastaiagent[crewai,ui]"
    OPENAI_API_KEY=sk-... python examples/55_trace_crewai.py
    fastaiagent ui  # then open http://127.0.0.1:7842/traces
"""

from __future__ import annotations

import os
import sys


def main() -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set — skipping example.")
        return 0

    try:
        from crewai import Agent, Crew, Process, Task
        from crewai.llm import LLM
    except ImportError:
        print(
            'crewai is not installed. Install with: pip install "fastaiagent[crewai]"'
        )
        return 0

    from fastaiagent.integrations import crewai as ca
    from fastaiagent.trace.storage import TraceStore

    ca.enable()

    llm = LLM(model="openai/gpt-4o-mini", temperature=0)
    researcher = Agent(
        role="Researcher",
        goal="Answer concisely.",
        backstory="You answer in one word.",
        llm=llm,
        allow_delegation=False,
        verbose=False,
    )
    task = Task(
        description="What is the capital of France?",
        expected_output="A single word.",
        agent=researcher,
    )
    crew = Crew(
        agents=[researcher], tasks=[task], process=Process.sequential, verbose=False
    )

    result = crew.kickoff()
    print("output:", result.raw)

    # Pretty-print the just-emitted trace tree.
    store = TraceStore.default()
    for summary in store.list_traces()[:1]:
        trace = store.get_trace(summary.trace_id)
        print(f"\nTrace {summary.trace_id} — {len(trace.spans)} spans:")
        for span in trace.spans:
            print(f"  {span.name}")
        print(
            "Open in the Local UI: "
            f"http://127.0.0.1:7842/traces/{summary.trace_id}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
