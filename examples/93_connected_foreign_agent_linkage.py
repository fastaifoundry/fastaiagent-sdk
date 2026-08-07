"""Example 93 — link a foreign-framework agent to the control plane.

Tracing a LangGraph / CrewAI / PydanticAI run works out of the box. *Linking*
those traces to an agent on the Enterprise plane needs two things, both added
in v1.46.0:

1. the root span must carry ``agent.name`` — pass ``name=`` to
   ``with_guardrails(...)`` (or use the ``agent_name(...)`` context manager);
2. the plane must have an agent with that name — ``register_agent()`` now
   registers with the plane as well as the local registry.

Without them the plane falls back to prefix-stripping the framework's own span
name, deriving ``"chain"`` for every LangChain user — so the trace is excluded
from per-agent analytics and reads as dark to governance coverage.

Run:
    pip install "fastaiagent[langchain]"
    OPENAI_API_KEY=sk-... \\
    FASTAIAGENT_API_KEY=fa_k_... FASTAIAGENT_TARGET=https://your-plane \\
    python examples/93_connected_foreign_agent_linkage.py
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from pathlib import Path

AGENT_NAME = "example-93-support-bot"


def _root_span() -> tuple[str, dict] | None:
    """Read back the root span this run produced, to show what was emitted."""
    db = Path.cwd() / ".fastaiagent" / "local.db"
    if not db.exists():
        return None
    conn = sqlite3.connect(str(db))
    try:
        row = conn.execute(
            "SELECT name, attributes FROM spans "
            "WHERE parent_span_id IS NULL ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
    finally:
        conn.close()
    return (row[0], json.loads(row[1] or "{}")) if row else None


def _plane_would_link_to(span_name: str, attrs: dict) -> str:
    """Mirror the plane's resolution order: attributes first, then prefix-strip."""
    for key in ("agent.name", "chain.name", "swarm.name", "workflow.name"):
        if attrs.get(key):
            return f"{attrs[key]!r}  (from attribute {key})"
    derived = span_name.split(".", 1)[1] if "." in span_name else span_name
    return f"{derived!r}  (prefix-stripped from the span name — generic!)"


def _langchain_compat() -> None:
    """Reconcile a LangChain-ecosystem version clash.

    ``langchain>=1`` removed the module-level ``verbose`` / ``debug`` /
    ``llm_cache`` globals that ``langchain-core~=0.3`` still probes in
    ``langchain_core.globals``. With both installed, constructing *or* invoking
    any chat model raises ``AttributeError``. Restoring the three attributes is
    a no-op when the installed versions already agree.
    """
    try:
        import langchain
    except ImportError:
        return
    for attr in ("verbose", "debug", "llm_cache"):
        if not hasattr(langchain, attr):
            setattr(langchain, attr, None if attr == "llm_cache" else False)


def main() -> int:
    _langchain_compat()
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set — skipping example.")
        return 0
    api_key = os.environ.get("FASTAIAGENT_API_KEY")
    target = os.environ.get("FASTAIAGENT_TARGET")
    if not (api_key and target):
        print(
            "FASTAIAGENT_API_KEY / FASTAIAGENT_TARGET are not set — this example "
            "needs a control plane to link against. Skipping."
        )
        return 0

    try:
        from langchain_openai import ChatOpenAI
        from langgraph.prebuilt import create_react_agent
    except ImportError:
        print('langgraph is not installed. Install with: pip install "fastaiagent[langchain]"')
        return 0

    import fastaiagent as fa
    from fastaiagent.integrations import langchain as lc

    fa.connect(api_key=api_key, target=target)
    try:
        lc.enable()

        graph = create_react_agent(
            ChatOpenAI(model="gpt-4o-mini", temperature=0), tools=[]
        )

        # 1. name the run so the root span carries agent.name
        guarded = lc.with_guardrails(graph, name=AGENT_NAME)

        # 2. register so the plane has an agent with that name to match
        #    (local registry + plane, when connected and scoped for agent:write)
        lc.register_agent(guarded, name=AGENT_NAME)

        result = guarded.invoke(
            {"messages": [("user", "What are your support hours?")]}
        )
        print("output:", result["messages"][-1].content[:100])
    finally:
        fa.disconnect()  # flushes pending spans to the plane

    root = _root_span()
    if root:
        span_name, attrs = root
        print(f"\nroot span name       : {span_name!r}")
        print(f"agent.name attribute : {attrs.get('agent.name')!r}")
        print(f"plane will link to   : {_plane_would_link_to(span_name, attrs)}")
        print(
            "\nThe span NAME stays generic — that is expected. The plane reads the "
            "agent.name ATTRIBUTE first, which is what makes the trace linkable."
        )

    print(
        "\nTip: for code that doesn't use with_guardrails(), wrap the call:\n"
        "    from fastaiagent.integrations import agent_name\n"
        "    with agent_name('my-agent'):\n"
        "        graph.invoke(...)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
