"""E2E gate — memory observability spans on a real Agent run, real LLM.

No mocking. Runs an Agent with a ComposableMemory (StaticBlock + SummaryBlock +
FactExtractionBlock) over several turns against a real model, then reads the
spans back out of local.db to assert:

- ``memory.read`` / ``memory.write`` spans are emitted per turn, nested under
  the agent span (full call-site wiring works end to end);
- FactExtractionBlock reports ``extracted_facts`` with a count;
- SummaryBlock reports ``summarized`` once its cadence triggers.

Skips locally without a key; required on CI (E2E_REQUIRED=1).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.e2e.conftest import require_env

pytestmark = pytest.mark.e2e


def _read_spans(db_path: Path) -> list[dict]:
    from fastaiagent._internal.storage import SQLiteHelper
    from fastaiagent.trace.otel import reset as reset_tracer

    reset_tracer()
    if not db_path.exists():
        return []
    with SQLiteHelper(db_path) as db:
        if not db.fetchone("SELECT name FROM sqlite_master WHERE type='table' AND name='spans'"):
            return []
        return db.fetchall("SELECT * FROM spans ORDER BY start_time")


def _attrs(row: dict) -> dict:
    return json.loads(row.get("attributes") or "{}")


def test_memory_spans_on_real_agent_run(tmp_path: Path, monkeypatch) -> None:
    require_env()
    from fastaiagent import Agent, LLMClient
    from fastaiagent._internal.config import reset_config
    from fastaiagent.agent.memory import AgentMemory, ComposableMemory
    from fastaiagent.agent.memory_blocks import (
        FactExtractionBlock,
        StaticBlock,
        SummaryBlock,
    )
    from fastaiagent.trace.otel import reset as reset_tracer

    db_path = tmp_path / "local.db"
    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(db_path))
    reset_config()
    reset_tracer()

    llm = LLMClient(provider="openai", model="gpt-4.1")
    memory = ComposableMemory(
        blocks=[
            StaticBlock("The user is a QA engineer."),
            SummaryBlock(llm=llm, keep_last=1, summarize_every=2),
            FactExtractionBlock(llm=llm, extract_every=1),
        ],
        primary=AgentMemory(max_messages=20),
    )
    agent = Agent(
        name="memory-trace-gate",
        system_prompt="You are a concise assistant. Acknowledge what the user says.",
        llm=llm,
        memory=memory,
    )

    agent.run("My name is Upendra and I live in Seattle.")
    agent.run("I prefer terse answers and I use Python daily.")
    agent.run("What do you know about me?")

    spans = _read_spans(db_path)
    names = [s["name"] for s in spans]

    # Parents emitted per turn (3 reads, 6 writes = 2 messages/turn).
    assert names.count("memory.read") == 3, f"reads: {names.count('memory.read')}"
    assert names.count("memory.write") == 6, f"writes: {names.count('memory.write')}"

    # memory.read nests under an agent span.
    agent_ids = {s["span_id"] for s in spans if s["name"].startswith("agent.")}
    reads = [s for s in spans if s["name"] == "memory.read"]
    assert any(s["parent_span_id"] in agent_ids for s in reads), (
        "memory.read not nested under agent"
    )

    # FactExtractionBlock reported extraction at least once with a count field.
    fact_writes = [_attrs(s) for s in spans if s["name"] == "memory.write.facts"]
    assert fact_writes, "no memory.write.facts child spans"
    assert any(a.get("memory.action") == "extracted_facts" for a in fact_writes)
    assert any(
        "facts_extracted" in json.loads(a.get("memory.detail", "{}"))
        for a in fact_writes
        if "memory.detail" in a
    ), "no facts_extracted detail captured"

    # SummaryBlock should have summarized once the cadence (every 2) triggered.
    summary_writes = [_attrs(s) for s in spans if s["name"] == "memory.write.summary"]
    assert any(a.get("memory.action") == "summarized" for a in summary_writes), (
        "SummaryBlock never reported a summarized action across the run"
    )


def test_persist_during_run_writes_facts_with_source_trace(tmp_path: Path, monkeypatch) -> None:
    """FactExtractionBlock(persist=True) writes durable facts to learned_memory
    during a real run, each stamped with the run's trace id (lineage)."""
    require_env()
    import json as _json

    from fastaiagent import Agent, LLMClient
    from fastaiagent._internal.config import reset_config
    from fastaiagent.agent.memory import AgentMemory, ComposableMemory
    from fastaiagent.agent.memory_blocks import FactExtractionBlock
    from fastaiagent.learn import MemoryStore
    from fastaiagent.trace.otel import reset as reset_tracer

    db_path = tmp_path / "local.db"
    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(db_path))
    reset_config()
    reset_tracer()

    llm = LLMClient(provider="openai", model="gpt-4.1")
    memory = ComposableMemory(
        blocks=[
            FactExtractionBlock(
                llm=llm, extract_every=1, persist=True, scope="user", scope_id="e2e-user"
            )
        ],
        primary=AgentMemory(max_messages=20),
    )
    agent = Agent(
        name="persist-gate",
        system_prompt="You are a concise assistant.",
        llm=llm,
        memory=memory,
    )
    agent.run("My name is Dana and I live in Portland with a cat named Mochi.")

    # Facts landed in the durable store...
    rows = MemoryStore(db_path=str(db_path)).list_active(scope="user", scope_id="e2e-user")
    assert rows, "no facts persisted to learned_memory during the run"
    # ...each with confidence 0.6 (auto) and a real 32-hex source_trace_id.
    assert all(r.confidence == 0.6 for r in rows)
    assert all(r.source_trace_id and len(r.source_trace_id) == 32 for r in rows)

    # ...and the write span recorded a persisted count.
    spans = _read_spans(db_path)
    fact_writes = [
        _json.loads(_attrs(s).get("memory.detail", "{}"))
        for s in spans
        if s["name"] == "memory.write.facts" and "memory.detail" in _attrs(s)
    ]
    assert any(d.get("persisted", 0) > 0 for d in fact_writes), "no persisted count in write span"
