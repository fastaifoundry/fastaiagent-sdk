"""Example 59 — Register an external agent + harness attachments.

External agents (LangGraph, CrewAI, PydanticAI) live in their own
process; the FastAIAgent Local UI knows about them via the
``external_agents`` and ``external_agent_attachments`` SQLite tables
introduced in v1.6 (schema v7). ``register_agent()`` writes the agent
row; the harness helpers (``with_guardrails(name=…)``,
``prompt_from_registry(agent=…)``, ``kb_as_retriever(agent=…)``) write
attachment rows.

This example registers a tiny LangGraph + a guardrail / prompt / KB
attachment, then queries the dependency endpoint that powers the
Agent Detail page in the UI to show the merged tree comes back.

Run:
    pip install "fastaiagent[langchain,ui]" langchain langchain-openai langgraph
    OPENAI_API_KEY=sk-... python examples/59_register_external_agent.py
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from urllib.error import URLError
from urllib.request import urlopen


def main() -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set — skipping example.")
        return 0

    try:
        from langchain_openai import ChatOpenAI
        from langgraph.prebuilt import create_react_agent
    except ImportError:
        print(
            'LangChain / LangGraph are not installed. Install with: '
            'pip install "fastaiagent[langchain]" langchain-openai langgraph'
        )
        return 0

    from fastaiagent.integrations import langchain as lc
    from fastaiagent.integrations._registry import (
        attach,
        fetch_agent,
        fetch_attachments,
    )

    name = f"example-59-agent-{uuid.uuid4().hex[:6]}"

    graph = create_react_agent(ChatOpenAI(model="gpt-4o-mini"), tools=[])
    lc.register_agent(graph, name=name)

    # Attach the harness layers. In real usage you'd pass ``agent=name``
    # to ``with_guardrails``, ``prompt_from_registry``,
    # ``kb_as_retriever`` — those write the attachment rows for you.
    # Here we call ``attach`` directly to keep the example self-contained
    # (no LLM call required).
    attach(name, "guardrail", "no_pii", position="input")
    attach(name, "prompt", "support-system", version="v1")
    attach(name, "kb", "support-kb")

    row = fetch_agent(name)
    atts = fetch_attachments(name)
    print(f"\nRegistered: {name!r}  framework={row['framework']!r}")
    print(f"Attachments ({len(atts)}):")
    for a in atts:
        print(f"  {a['kind']:9s}  {a['ref_name']:24s}  {a.get('position') or ''}")

    # Read back via the dependency endpoint (the same surface the UI's
    # Agent Detail page uses). Attempt only if the UI is running.
    print(
        f"\nThe Local UI dependency endpoint: "
        f"http://127.0.0.1:7842/api/agents/{name}/dependencies"
    )
    try:
        with urlopen(
            f"http://127.0.0.1:7842/api/agents/{name}/dependencies",
            timeout=2,
        ) as resp:
            payload = json.loads(resp.read())
        print("Endpoint payload:")
        print(
            json.dumps(
                {
                    k: payload.get(k)
                    for k in (
                        "agent",
                        "guardrails",
                        "prompts",
                        "knowledge_bases",
                        "external",
                    )
                },
                indent=2,
                default=str,
            )
        )
    except URLError:
        print(
            "  (Local UI is not running. Start it with `fastaiagent ui`,"
            " then re-run this example to see the live merge.)"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
