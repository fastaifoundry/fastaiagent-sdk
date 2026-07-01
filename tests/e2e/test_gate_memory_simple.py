"""E2E gate — the `Memory` facade with a real agent + real LLM. No mocking.

Proves the headline promises:
- one agent definition serves many users via a per-run `user_id` resolver;
- `learn=llm` extracts + persists user facts (stamped with the run's trace);
- **two users do not cross-contaminate** — the core multi-session guarantee;
- `summarize=llm` fires on long-enough conversations.

Skips locally without a key; required on CI (E2E_REQUIRED=1).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from tests.e2e.conftest import require_env

pytestmark = pytest.mark.e2e


@dataclass
class Session:
    user_id: str


def test_memory_facade_multiuser_no_cross_contamination(tmp_path: Path, monkeypatch) -> None:
    require_env()
    from fastaiagent import Agent, LLMClient, Memory
    from fastaiagent._internal.config import reset_config
    from fastaiagent.agent.context import RunContext
    from fastaiagent.learn import MemoryStore
    from fastaiagent.trace.otel import reset as reset_tracer

    db_path = tmp_path / "local.db"
    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(db_path))
    reset_config()
    reset_tracer()

    llm = LLMClient(provider="openai", model="gpt-4.1")

    # ONE agent definition, ONE Memory — serves every user via the resolver.
    agent = Agent(
        name="assistant",
        system_prompt="You are a concise assistant. Use what you remember about the user.",
        llm=llm,
        memory=Memory(
            location=MemoryStore(db_path=str(db_path)),
            user_id=lambda ctx: ctx.state.user_id,
            learn=llm,
            summarize=llm,
        ),
    )

    alice = RunContext(state=Session(user_id="alice"))
    bob = RunContext(state=Session(user_id="bob"))

    agent.run("I have a dog named Rex and I love hiking.", context=alice)
    agent.run("I have a cat named Mia and I'm allergic to dogs.", context=bob)

    # Durable facts are isolated per user in the store.
    store = MemoryStore(db_path=str(db_path))
    alice_facts = " ".join(
        f.fact.lower() for f in store.list_active(scope="user", scope_id="alice")
    )
    bob_facts = " ".join(f.fact.lower() for f in store.list_active(scope="user", scope_id="bob"))
    assert alice_facts, "no facts learned for alice"
    assert bob_facts, "no facts learned for bob"
    assert "rex" in alice_facts and "rex" not in bob_facts
    assert "mia" in bob_facts and "mia" not in alice_facts
    # persisted during the run → carries a source trace id (lineage)
    assert all(
        f.source_trace_id and len(f.source_trace_id) == 32
        for f in store.list_active(scope="user", scope_id="alice")
    )

    # And the agent recalls the RIGHT user's pet — no cross-contamination.
    ans = agent.run("What is my pet's name? One word.", context=alice)
    assert "rex" in ans.output.lower()
    assert "mia" not in ans.output.lower()


def test_memory_facade_summarize_fires(tmp_path: Path, monkeypatch) -> None:
    require_env()
    import json as _json

    from fastaiagent import Agent, LLMClient, Memory
    from fastaiagent._internal.config import reset_config
    from fastaiagent._internal.storage import SQLiteHelper
    from fastaiagent.learn import MemoryStore
    from fastaiagent.trace.otel import reset as reset_tracer

    db_path = tmp_path / "local.db"
    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(db_path))
    reset_config()
    reset_tracer()

    llm = LLMClient(provider="openai", model="gpt-4.1")
    agent = Agent(
        name="chat",
        system_prompt="You are concise.",
        llm=llm,
        memory=Memory(location=MemoryStore(db_path=str(db_path)), window=10, summarize=llm),
    )
    # SummaryBlock defaults keep_last=10 / summarize_every=5 → needs >10
    # messages before it compresses; 8 short turns (16 messages) crosses it.
    for msg in [
        "My name is Dana.",
        "I live in Oslo.",
        "I code in Rust.",
        "I have twins.",
        "I drink black coffee.",
        "I run every morning.",
        "I play the cello.",
        "I prefer trains over planes.",
    ]:
        agent.run(msg)

    reset_tracer()
    with SQLiteHelper(db_path) as d:
        rows = d.fetchall("SELECT name, attributes FROM spans WHERE name='memory.write.summary'")
    actions = [_json.loads(r["attributes"]).get("memory.action") for r in rows]
    assert "summarized" in actions, f"SummaryBlock never summarized; actions={actions}"
