"""Memory observability — seed a trace whose spans show what the agent remembered.

Runs a real Agent with a ``ComposableMemory`` (StaticBlock + VectorBlock +
SummaryBlock + FactExtractionBlock) over a few turns, so the resulting trace
carries ``memory.read`` / ``memory.write`` spans with per-block children —
including VectorBlock similarity scores and the bounded snippets it recalled.

Also seeds a couple of durable facts into the ``learned_memory`` table (what
``PersistentFactBlock`` reads back across runs) so the Local UI's Memory page
has rows to show.

Writes ``last_run.json`` (trace id + db path) for ``snapshot.py`` to consume.

Run:
    zsh -lc 'python companion.py'      # needs OPENAI_API_KEY

Then capture the UI screenshots:
    zsh -lc 'python snapshot.py'
"""

from __future__ import annotations

import json
import os
from pathlib import Path

HERE = Path(__file__).parent
DB_PATH = HERE / "memory_demo.db"
LAST_RUN = HERE / "last_run.json"


def main() -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set — export it (it lives in ~/.zshrc) and re-run via zsh -lc.")
        return 1

    # Isolate this demo's trace DB so the screenshots are clean.
    os.environ["FASTAIAGENT_LOCAL_DB"] = str(DB_PATH)
    if DB_PATH.exists():
        DB_PATH.unlink()

    from fastaiagent._internal.config import reset_config
    from fastaiagent.trace.otel import reset as reset_tracer

    reset_config()
    reset_tracer()

    from fastaiagent import Agent, LLMClient
    from fastaiagent.agent.memory import AgentMemory, ComposableMemory
    from fastaiagent.agent.memory_blocks import (
        FactExtractionBlock,
        StaticBlock,
        SummaryBlock,
        VectorBlock,
    )
    from fastaiagent.kb.backends.faiss import FaissVectorStore
    from fastaiagent.learn import Fact, MemoryStore

    llm = LLMClient(provider="openai", model="gpt-4.1")

    # FAISS-backed semantic recall over past turns. Recency weight on so the
    # score column in the trace is visibly a blend (the hero detail).
    # dedupe_against_upstream: skip recalling what the StaticBlock already pinned.
    store = FaissVectorStore(dimension=384, index_type="flat")
    memory = ComposableMemory(
        blocks=[
            StaticBlock("The user is Upendra, a QA engineer who prefers terse answers."),
            VectorBlock(
                store=store,
                top_k=3,
                min_content_chars=12,
                recency_weight=0.2,
                dedupe_against_upstream=True,
            ),
            SummaryBlock(llm=llm, keep_last=2, summarize_every=2),
            # persist=True: newly extracted facts are written to learned_memory
            # during the run, stamped with this run's trace id (source lineage).
            FactExtractionBlock(
                llm=llm,
                extract_every=1,
                persist=True,
                scope="user",
                scope_id="upendra",
            ),
        ],
        primary=AgentMemory(max_messages=20),
    )

    agent = Agent(
        name="memory-demo",
        system_prompt="You are a concise assistant. Use what you remember about the user.",
        llm=llm,
        memory=memory,
    )

    turns = [
        "My name is Upendra and I'm based in Seattle. I work on agent infrastructure.",
        "I just adopted a beagle named Biscuit and I'm allergic to cats.",
        "What do you remember about me and my pet?",
    ]
    last_trace_id = None
    for t in turns:
        result = agent.run(t)
        last_trace_id = getattr(result, "trace_id", None) or last_trace_id
        print(f"> {t}\n  {result.output}\n")

    # The FactExtractionBlock above already persisted user:upendra facts *with a
    # source trace* during the run. Here we also add an agent-scoped fact
    # manually (source = "manual") and demonstrate a supersede chain so the
    # "Show superseded" history toggle has something to show.
    fact_store = MemoryStore(db_path=str(DB_PATH))
    fact_store.add(Fact(scope="agent", scope_id="memory-demo", fact="User prefers terse answers."))
    old_id = fact_store.add(
        Fact(scope="user", scope_id="upendra", fact="Works on agent infrastructure.")
    )
    new_id = fact_store.add(
        Fact(
            scope="user",
            scope_id="upendra",
            fact="Works on agent infrastructure at FastAIAgent.",
        )
    )
    fact_store.supersede(old_id, new_id)

    # If the agent didn't surface a trace id on the result, fall back to the
    # most recent root agent span in the DB.
    if not last_trace_id:
        from fastaiagent._internal.storage import SQLiteHelper

        reset_tracer()  # flush
        with SQLiteHelper(DB_PATH) as db:
            row = db.fetchone(
                "SELECT trace_id FROM spans WHERE name LIKE 'agent.%' "
                "ORDER BY start_time DESC LIMIT 1"
            )
            last_trace_id = row["trace_id"] if row else None

    LAST_RUN.write_text(json.dumps({"trace_id": last_trace_id, "db_path": str(DB_PATH)}, indent=2))
    print(f"Wrote {LAST_RUN.name}: trace_id={last_trace_id}")
    print("Now run:  zsh -lc 'python snapshot.py'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
