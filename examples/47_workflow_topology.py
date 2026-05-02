"""Workflow topology — register a Chain so the Local UI can render it.

Run:

    pip install 'fastaiagent[ui,openai]'
    OPENAI_API_KEY=... python examples/47_workflow_topology.py

Open http://127.0.0.1:7843/workflows/chain/refund-flow once it boots — the
topology view renders the chain's structure (nodes, edges, the conditional
branch on ``approved``, and an HITL gate). The same registration pattern
works for ``Swarm`` and ``Supervisor``.

The expected output is the topology canvas captured at
``docs/ui/screenshots/sprint1-1-workflow-topology.png``. See
``docs/ui/workflow-visualization.md`` for a screenshot walkthrough.
"""

from __future__ import annotations

import uvicorn

from fastaiagent import Agent, Chain
from fastaiagent.chain.node import NodeType
from fastaiagent.llm import LLMClient
from fastaiagent.tool import tool
from fastaiagent.ui.server import build_app


@tool(description="Search the support docs for an answer.")
def search_docs(query: str) -> str:
    return f"results for {query!r}"


@tool(description="Process a refund for a customer.")
def process_refund(amount: float, customer_id: str) -> dict[str, str]:
    return {"status": "ok", "customer_id": customer_id, "amount": str(amount)}


def build_chain() -> Chain:
    llm = LLMClient(provider="openai", model="gpt-4o-mini")
    researcher = Agent(
        name="researcher",
        llm=llm,
        tools=[search_docs],
        system_prompt="You investigate refund eligibility.",
    )
    notifier = Agent(
        name="notifier",
        llm=llm,
        system_prompt="You write a polite rejection notice.",
    )

    chain = Chain("refund-flow")
    chain.add_node("research", agent=researcher)
    chain.add_node("approval", type=NodeType.hitl, name="Manager approval")
    chain.add_node("process", tool=process_refund)
    chain.add_node("notify_rejection", agent=notifier)

    chain.connect("research", "approval")
    chain.connect("approval", "process", condition="approved == True")
    chain.connect("approval", "notify_rejection", condition="approved == False")
    return chain


def main() -> None:
    chain = build_chain()
    app = build_app(no_auth=True, runners=[chain])
    print("Local UI on http://127.0.0.1:7843")
    print("Topology:    http://127.0.0.1:7843/workflows/chain/refund-flow")
    uvicorn.run(app, host="127.0.0.1", port=7843)


if __name__ == "__main__":
    main()
