"""Example 36 — Multi-agent chain as a single trace.

Demonstrates the 0.8 workflow-root behavior: a Chain with two Agents
emits ONE trace with a `chain.<name>` root span wrapping the two child
`agent.*` spans (which in turn wrap their LLM call spans).

Prereqs:
    pip install 'fastaiagent[ui,openai]'
    export OPENAI_API_KEY=...

Run:
    python examples/36_chain_workflow.py
    fastaiagent ui          # Workflow badge shows "chain" in the traces list
"""

from __future__ import annotations

import os

os.environ.setdefault("FASTAIAGENT_UI_ENABLED", "true")

from fastaiagent import Agent, LLMClient  # noqa: E402
from fastaiagent.chain import Chain  # noqa: E402


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required to run this example.")

    llm = LLMClient(provider="openai", model="gpt-4o-mini")

    researcher = Agent(
        name="researcher",
        system_prompt=(
            "Given a topic, write 2 crisp bullet points with the key facts. "
            "Be brief."
        ),
        llm=llm,
    )
    summariser = Agent(
        name="summariser",
        system_prompt=(
            "Given research bullet points, write a one-sentence summary in plain English."
        ),
        llm=llm,
    )

    chain = Chain(name="research-then-summarise")
    chain.add_node("research", agent=researcher)
    chain.add_node("summarise", agent=summariser)
    chain.connect("research", "summarise")

    print("► running chain research-then-summarise")
    result = chain.execute({"message": "The history of SQLite"})

    preview = str(result.output)[:160]
    print("\noutput:", preview)
    print("\nOne trace, tree of spans:")
    print("    chain.research-then-summarise")
    print("      agent.researcher")
    print("        llm.chat")
    print("      agent.summariser")
    print("        llm.chat")
    print("\nOpen `fastaiagent ui` — the Workflow column shows this as a 'chain'.")


if __name__ == "__main__":
    main()
