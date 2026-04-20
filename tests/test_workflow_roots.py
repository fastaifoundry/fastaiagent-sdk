"""Verify Chain / Swarm / Supervisor each emit a workflow-root span.

One real SDK run per runner, no mocking of OTel — we use the SDK's own
tracer pipeline (which the LocalStorageProcessor writes to SQLite under
pytest's tmp DB).

These tests guard against regressions of the fix where multi-agent
executions would fragment into N orphan agent traces instead of one
trace with a workflow-kind root. They also verify the canonical
``fastaiagent.runner.type`` attribute is set so the UI can render a
Workflow badge without heuristics.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from fastaiagent._internal.config import reset_config
from fastaiagent._internal.storage import SQLiteHelper
from fastaiagent.trace.otel import reset as reset_tracer


@pytest.fixture(autouse=True)
def _isolated_db(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(tmp_path / "local.db"))
    reset_config()
    reset_tracer()
    yield tmp_path / "local.db"
    reset_tracer()
    reset_config()


def _read_spans(db_path: Path) -> list[dict[str, Any]]:
    with SQLiteHelper(db_path) as db:
        return db.fetchall("SELECT * FROM spans ORDER BY start_time")


def _attrs(row: dict[str, Any]) -> dict[str, Any]:
    return json.loads(row.get("attributes") or "{}")


class TestChainRoot:
    def test_chain_aexecute_emits_chain_root_span(self, _isolated_db: Path):
        from fastaiagent.chain import Chain

        chain = Chain(name="unit-chain")
        chain.add_node("echo", tool=lambda state: {"message": state.get("msg", "")})
        asyncio.run(chain.aexecute({"msg": "hi"}))

        rows = _read_spans(_isolated_db)
        roots = [r for r in rows if not r.get("parent_span_id")]
        # Every other span in the trace must be a descendant of the chain root.
        assert len(roots) == 1, (
            f"expected exactly one root span; got {len(roots)} "
            f"with names {[r['name'] for r in roots]}"
        )
        root = roots[0]
        assert root["name"] == "chain.unit-chain"
        attrs = _attrs(root)
        assert attrs.get("fastaiagent.runner.type") == "chain"
        assert attrs.get("chain.name") == "unit-chain"
        assert attrs.get("chain.node_count") == 1


class TestSwarmRoot:
    def test_swarm_arun_emits_swarm_root_span(self, _isolated_db: Path):
        from tests.conftest import MockLLMClient  # real conftest import, no stubs
        from fastaiagent import Agent
        from fastaiagent.agent.swarm import Swarm
        from fastaiagent.llm.client import LLMResponse

        a = Agent(
            name="a",
            system_prompt="you are a",
            llm=MockLLMClient([LLMResponse(content="done", finish_reason="stop")]),
        )
        b = Agent(
            name="b",
            system_prompt="you are b",
            llm=MockLLMClient([LLMResponse(content="done", finish_reason="stop")]),
        )
        swarm = Swarm(
            name="unit-swarm",
            agents=[a, b],
            entrypoint="a",
            handoffs={"a": ["b"]},
        )
        asyncio.run(swarm.arun("hi"))

        rows = _read_spans(_isolated_db)
        roots = [r for r in rows if not r.get("parent_span_id")]
        assert len(roots) == 1, (
            f"expected exactly one root span; got {len(roots)} "
            f"with names {[r['name'] for r in roots]}"
        )
        root = roots[0]
        assert root["name"] == "swarm.unit-swarm"
        attrs = _attrs(root)
        assert attrs.get("fastaiagent.runner.type") == "swarm"
        assert attrs.get("swarm.name") == "unit-swarm"
        assert attrs.get("swarm.entrypoint") == "a"


class TestSupervisorRoot:
    def test_supervisor_arun_emits_supervisor_root_span(self, _isolated_db: Path):
        from tests.conftest import MockLLMClient
        from fastaiagent import Agent
        from fastaiagent.agent.team import Supervisor, Worker
        from fastaiagent.llm.client import LLMResponse

        worker_agent = Agent(
            name="worker-a",
            system_prompt="you are worker",
            llm=MockLLMClient([LLMResponse(content="ok", finish_reason="stop")]),
        )
        supervisor = Supervisor(
            name="unit-supervisor",
            llm=MockLLMClient([LLMResponse(content="final", finish_reason="stop")]),
            workers=[Worker(agent=worker_agent, role="worker", description="")],
        )
        asyncio.run(supervisor.arun("hi"))

        rows = _read_spans(_isolated_db)
        roots = [r for r in rows if not r.get("parent_span_id")]
        assert len(roots) == 1
        root = roots[0]
        assert root["name"] == "supervisor.unit-supervisor"
        attrs = _attrs(root)
        assert attrs.get("fastaiagent.runner.type") == "supervisor"
        assert attrs.get("supervisor.name") == "unit-supervisor"
