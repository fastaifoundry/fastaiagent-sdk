"""Example 08: Trace a real LangChain LLM call with FastAIAgent.

Drives ``ChatOpenAI.invoke()`` through the fastaiagent LangChain callback
handler and prints the spans that call wrote to the local trace store.

Note what the span is *called*. Invoking a chat model directly fires
``on_chat_model_start``, which the handler names ``llm.{provider}.{model}`` —
so this run produces ``llm.openai.gpt-4.1``, not ``langchain.*``. A
``langchain.{name}`` span only appears for a root ``on_chain_start``, i.e.
when you invoke an actual LangChain *chain* (``prompt | llm``, an LCEL
runnable, an AgentExecutor). This example invokes the model on its own.

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


def _max_span_rowid(db_path: Path) -> int:
    """Highest ``spans.rowid`` currently stored, or 0 if there is no store yet."""
    if not db_path.exists():
        return 0
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute("SELECT COALESCE(MAX(rowid), 0) FROM spans").fetchone()
        return int(row[0]) if row else 0
    except sqlite3.OperationalError:
        # No ``spans`` table yet — the first traced call creates it.
        return 0
    finally:
        conn.close()


def _spans_since(db_path: Path, watermark: int) -> list[tuple[str, str]]:
    """``(name, trace_id)`` for every span written after ``watermark``."""
    if not db_path.exists():
        return []
    conn = sqlite3.connect(str(db_path))
    try:
        return [
            (str(name), str(trace_id))
            for name, trace_id in conn.execute(
                "SELECT name, trace_id FROM spans WHERE rowid > ? ORDER BY rowid",
                (watermark,),
            )
        ]
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()


def main() -> int:
    _langchain_compat()
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

    # Read the db path the SDK actually writes to, so FASTAIAGENT_LOCAL_DB /
    # FASTAIAGENT_TRACE_DB_PATH are honoured instead of assumed.
    from fastaiagent._internal.config import get_config

    db_path = Path(get_config().resolved_trace_db_path)

    # Watermark the span table *before* the call. Selecting "recent spans"
    # by name instead would show rows from unrelated earlier runs on a
    # lived-in local.db — and show nothing at all on a clean machine.
    watermark = _max_span_rowid(db_path)

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

    rows = _spans_since(db_path, watermark)

    print(f"\nSpans this run wrote to {db_path}:")
    if not rows:
        print("  (none — spans may still be in flight; try again in a moment)")
    else:
        for name, trace_id in rows:
            print(f"  - {name}  (trace {trace_id[:16]})")

    print("\nView all traces with: fastaiagent traces list")
    return 0


if __name__ == "__main__":
    sys.exit(main())
