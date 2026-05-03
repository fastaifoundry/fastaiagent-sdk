"""Example 08: Trace a real LangChain LLM call with FastAIAgent.

Drives ``ChatOpenAI.invoke()`` through the fastaiagent LangChain callback
handler and prints the recently-emitted ``langchain.*`` spans from the
local trace store.

Requirements:
    pip install "fastaiagent[langchain]" langchain langchain-openai
    OPENAI_API_KEY in the environment

Run:
    python examples/08_trace_langchain.py
"""

from __future__ import annotations

import os
import sqlite3
import sys
import uuid
from pathlib import Path


def _missing_dep_message(name: str) -> str:
    return (
        f"Optional dependency '{name}' is not installed. "
        "Install with: pip install \"fastaiagent[langchain]\" langchain langchain-openai"
    )


def main() -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set — skipping example.")
        return 0

    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        print(_missing_dep_message(exc.name or "langchain"))
        return 0

    import fastaiagent.integrations.langchain as lc_int

    lc_int.enable()
    handler = lc_int.get_callback_handler()
    print(f"handler: {type(handler).__name__}")

    marker = uuid.uuid4().hex[:8]
    print(f"\nInvoking ChatOpenAI (marker={marker}) ...")

    llm = ChatOpenAI(model="gpt-4.1", temperature=0)
    response = llm.invoke(
        [
            SystemMessage(content="Reply in exactly one short sentence."),
            HumanMessage(content=f"Echo this exact token verbatim: {marker}"),
        ],
        config={"callbacks": [handler]},
    )
    print(f"response: {response.content!r}")

    db_path = Path.cwd() / ".fastaiagent" / "local.db"
    if not db_path.exists():
        print(f"\nNote: no local trace store found at {db_path}.")
        return 0

    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT name FROM spans WHERE name LIKE 'langchain.%' "
            "ORDER BY rowid DESC LIMIT 5"
        )
        rows = cur.fetchall()
    finally:
        conn.close()

    print(f"\nMost recent langchain.* spans in {db_path.name}:")
    if not rows:
        print("  (none — spans may still be in flight; try again in a moment)")
    else:
        for (name,) in rows:
            print(f"  - {name}")

    print("\nView all traces with: fastaiagent traces list")
    return 0


if __name__ == "__main__":
    sys.exit(main())
