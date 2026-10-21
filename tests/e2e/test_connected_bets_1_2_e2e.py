"""1.91.0 against the LIVE plane — what Bets 1 + 2 change that the plane stores or serves.

Each test drives a real connected SDK run against the plane at ``FASTAIAGENT_TARGET``
and reads back what the plane kept:

* ``middleware_state`` (a ``ToolBudget`` count) is replicated with the checkpoint,
  served back by ``/public/v1/checkpoints/{id}/latest``, and a run resumed from
  the plane copy keeps counting (decision B1-5 across machines).
* A streamed run ends with a ``run_end`` row on the plane too (X1).
* The ``fastaiagent.middleware`` root-span attribute reaches the plane's trace
  (decision B1-6).
* A Gemini 3.5+ thought signature survives replication and restore, so a run
  resumed from the plane copy sends it back (X11) — real ``gemini-3.8-flash``.
* A turn in which the provider ran ``web_search()`` survives the plane as
  ``Message.provider_content`` and is accepted on resume, and the
  ``tool.web_search`` span reaches the plane without its payload (2D).

NO MOCKS of the plane. The model is the SDK's ``FunctionModel`` where the test is
about the plane, and real Gemini where it is about Gemini. Run with::

    zsh -lc 'set -a; . ~/.fastaiagent-sdk-plane-e2e.env; set +a; \
      .venv/bin/python -m pytest tests/e2e/test_connected_bets_1_2_e2e.py -m e2e -v -rs'
"""

from __future__ import annotations

import os
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from tests.e2e.conftest import require_env, require_platform

pytestmark = pytest.mark.e2e


@pytest.fixture
def connected(isolated_local_db: Path) -> Iterator[Any]:
    require_env()
    require_platform()
    import fastaiagent as fa
    from fastaiagent.checkpointers import platform_replica
    from fastaiagent.client import _connection
    from fastaiagent.trace import otel

    # A fresh tracer per test, writing to THIS test's local.db — otherwise the
    # provider an earlier test built keeps writing to that test's file.
    otel.reset()
    fa.connect(api_key=os.environ["FASTAIAGENT_API_KEY"], target=os.environ["FASTAIAGENT_TARGET"])
    try:
        assert _connection.is_connected
        yield _connection
    finally:
        platform_replica.drain_all_sync()
        fa.disconnect()
        otel.reset()


def echo(text: str) -> str:
    """Echo the text."""
    return f"echo:{text}"


def approve(text: str) -> str:
    """Needs a human's approval."""
    from fastaiagent import interrupt

    interrupt(reason="approve?", context={"text": text})
    return f"approved:{text}"


def _drain(store: Any, execution_id: str, timeout: float = 25.0) -> list[dict[str, Any]]:
    """Drain to the live plane until every local checkpoint is synced=1 (a 2xx)."""
    from fastaiagent.checkpointers import platform_replica

    deadline = time.monotonic() + timeout
    rows: list[dict[str, Any]] = []
    while time.monotonic() < deadline:
        platform_replica.drain_all_sync()
        rows = store._conn().fetchall(
            "SELECT checkpoint_id, node_id, status, synced FROM checkpoints WHERE execution_id = ?",
            (execution_id,),
        )
        if rows and all(r["synced"] == 1 for r in rows):
            return rows
        time.sleep(0.5)
    raise AssertionError(f"checkpoints not synced to the plane within {timeout}s: {rows}")


def _scripted(*turns: Any):
    """A FunctionModel that plays ``turns`` in order (a tool call or a reply)."""
    from fastaiagent.testing import FunctionModel

    state = {"i": 0}

    def respond(messages):
        turn = turns[min(state["i"], len(turns) - 1)]
        state["i"] += 1
        return turn

    return FunctionModel(respond)


def _plane_spans(connected: Any, trace_id: str, timeout: float = 30.0) -> list[dict[str, Any]]:
    """The trace's spans as the plane stored them, through the console spans API
    (``/api/v1/traces/{id}/spans``); the public trace endpoint serves spans
    without attributes. The plane keeps an SDK span's raw attributes as its
    ``input`` (``_enrich_sdk_span``: "preserve raw attributes")."""
    from fastaiagent.trace.otel import get_tracer_provider
    from fastaiagent.trace.platform_export import PlatformSpanExporter
    from tests.e2e.conftest import plane_admin

    admin, jwt, domain_id = plane_admin(
        connected.target, purpose="this gate reads stored span attributes through the console API."
    )
    provider = get_tracer_provider()
    if provider is not None and hasattr(provider, "force_flush"):
        provider.force_flush()
    deadline = time.monotonic() + timeout
    status = None
    while time.monotonic() < deadline:
        PlatformSpanExporter().export([])  # deterministic drain to the live plane
        resp = admin.get(
            f"/api/v1/traces/{trace_id}/spans", headers=jwt, params={"domain_id": domain_id}
        )
        status = resp.status_code
        if status == 200 and resp.json():
            return list(resp.json())
        time.sleep(1.0)
    raise AssertionError(f"trace {trace_id} not served by the plane (last HTTP {status})")


def test_middleware_state_round_trips_through_the_plane(connected: Any, tmp_path: Path) -> None:
    from fastaiagent import Agent, FunctionTool, Resume, SQLiteCheckpointer, ToolBudget
    from fastaiagent.checkpointers import platform_replica

    def agent(store: Any, llm: Any) -> Agent:
        return Agent(
            name="budgeted",
            llm=llm,
            tools=[FunctionTool(name="echo", fn=echo), FunctionTool(name="approve", fn=approve)],
            middleware=[ToolBudget(max_calls=2)],
            checkpointer=store,
        )

    turns = (
        ("", [{"name": "echo", "arguments": {"text": "1"}}]),  # 1 of 2
        ("", [{"name": "approve", "arguments": {"text": "2"}}]),  # pauses (pre-call: 1)
        ("", [{"name": "echo", "arguments": {"text": "3"}}]),  # over budget after resume
        "should not get here",
    )
    store1 = SQLiteCheckpointer(db_path=str(tmp_path / "a.db"))
    exec_id = f"b12-budget-{uuid.uuid4().hex[:8]}"
    llm = _scripted(*turns)
    assert agent(store1, llm).run("go", execution_id=exec_id).status == "paused"
    _drain(store1, exec_id)

    served = platform_replica.fetch_latest_from_plane(exec_id)
    assert served is not None and served.status == "interrupted"
    assert served.state_snapshot.get("middleware_state") == {"tool_budget": {"used": 1}}, (
        "the plane did not serve middleware_state back verbatim: "
        f"{served.state_snapshot.get('middleware_state')!r}"
    )

    # Another machine: an empty store, restored from the plane, resumes the run.
    store2 = SQLiteCheckpointer(db_path=str(tmp_path / "b.db"))
    store2.setup()
    assert platform_replica.restore_from_plane(store2, exec_id) is not None
    resumed = agent(store2, llm).resume(exec_id, resume_value=Resume(approved=True))
    assert resumed.stop_reason == "tool_budget", (
        f"the budget did not survive the plane round trip: {resumed.output!r}"
    )


def test_a_streamed_run_ends_with_run_end_on_the_plane(connected: Any, tmp_path: Path) -> None:
    import asyncio

    from fastaiagent import Agent, FunctionTool, SQLiteCheckpointer
    from fastaiagent.checkpointers import platform_replica

    store = SQLiteCheckpointer(db_path=str(tmp_path / "s.db"))
    exec_id = f"b12-stream-{uuid.uuid4().hex[:8]}"
    agent = Agent(
        name="streamer",
        llm=_scripted(("", [{"name": "echo", "arguments": {"text": "x"}}]), "done"),
        tools=[FunctionTool(name="echo", fn=echo)],
        checkpointer=store,
    )

    async def drain_stream() -> None:
        async for _ in agent.astream("go", execution_id=exec_id):
            pass

    asyncio.run(drain_stream())
    rows = _drain(store, exec_id)
    assert any(r["node_id"] == "run_end" for r in rows)
    served = platform_replica.fetch_latest_from_plane(exec_id)
    assert served is not None
    assert (served.node_id, served.status) == ("run_end", "completed"), (
        served.node_id,
        served.status,
    )


def test_the_middleware_attribute_reaches_the_plane_trace(connected: Any) -> None:
    """Read through the console spans API (see :func:`_plane_spans`)."""
    from fastaiagent import Agent, ToolBudget

    result = Agent(
        name="attributed", llm=_scripted("hello"), middleware=[ToolBudget(max_calls=3)]
    ).run("hi")
    spans = _plane_spans(connected, result.trace_id)
    root = next(s for s in spans if s["name"] == "agent.attributed")
    # The plane keeps an SDK span's raw attributes as the span's ``input``
    # (``_enrich_sdk_span``: "preserve raw attributes"); ``metadata`` holds what
    # it derives (tokens, cost, model).
    stored = (root.get("input") or {}).get("fastaiagent.middleware")
    assert stored is not None, (
        "the plane stored the root span without fastaiagent.middleware; it kept: "
        f"{sorted((root.get('input') or {}).keys())}"
    )
    assert list(stored) == ["tool_budget"], stored


def test_a_gemini_signature_survives_the_plane(connected: Any, tmp_path: Path) -> None:
    if not os.environ.get("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY not set")
    from fastaiagent import Agent, FunctionTool, LLMClient, Resume, SQLiteCheckpointer
    from fastaiagent.checkpointers import platform_replica

    def agent(store: Any) -> Agent:
        return Agent(
            name="gemini-desk",
            system_prompt="Call approve with the text the user gives, then confirm in one line.",
            llm=LLMClient(provider="gemini", model="gemini-3.8-flash"),
            tools=[FunctionTool(name="approve", fn=approve)],
            checkpointer=store,
        )

    store1 = SQLiteCheckpointer(db_path=str(tmp_path / "g1.db"))
    exec_id = f"b12-gemini-{uuid.uuid4().hex[:8]}"
    assert agent(store1).run("Approve: refund A-17", execution_id=exec_id).status == "paused"
    local_calls = [
        tc
        for m in store1.get_last(exec_id).state_snapshot["messages"]
        for tc in (m.get("tool_calls") or [])
    ]
    assert local_calls and local_calls[0].get("signature"), "Gemini returned no signature"
    _drain(store1, exec_id)

    store2 = SQLiteCheckpointer(db_path=str(tmp_path / "g2.db"))
    store2.setup()
    restored = platform_replica.restore_from_plane(store2, exec_id)
    assert restored is not None
    served_calls = [
        tc for m in restored.state_snapshot["messages"] for tc in (m.get("tool_calls") or [])
    ]
    assert served_calls[0].get("signature") == local_calls[0]["signature"], (
        "the plane did not keep the thought signature verbatim"
    )
    done = agent(store2).resume(exec_id, resume_value=Resume(approved=True))
    assert done.status == "completed"


def test_claude_thinking_survives_the_plane(connected: Any, tmp_path: Path) -> None:
    """X17 across machines: the thinking blocks are replicated with the checkpoint,
    served back verbatim, and a run resumed from the plane copy sends them — the
    API verifies their signature (real ``claude-haiku-4-5``, extended thinking)."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")
    from fastaiagent import Agent, FunctionTool, LLMClient, Resume, SQLiteCheckpointer
    from fastaiagent.checkpointers import platform_replica

    def agent(store: Any) -> Agent:
        return Agent(
            name="claude-desk",
            system_prompt="Call approve with the text the user gives, then confirm in one line.",
            llm=LLMClient(provider="anthropic", model="claude-haiku-4-5", reasoning="low"),
            tools=[FunctionTool(name="approve", fn=approve)],
            checkpointer=store,
        )

    def blocks(snapshot: dict[str, Any]) -> list[Any]:
        return [b for m in snapshot["messages"] for b in (m.get("reasoning_blocks") or [])]

    store1 = SQLiteCheckpointer(db_path=str(tmp_path / "c1.db"))
    exec_id = f"b12-claude-{uuid.uuid4().hex[:8]}"
    assert agent(store1).run("Approve: refund A-17", execution_id=exec_id).status == "paused"
    local = blocks(store1.get_last(exec_id).state_snapshot)
    assert local and local[0]["provider"] == "anthropic", "Claude returned no thinking"
    _drain(store1, exec_id)

    store2 = SQLiteCheckpointer(db_path=str(tmp_path / "c2.db"))
    store2.setup()
    restored = platform_replica.restore_from_plane(store2, exec_id)
    assert restored is not None
    assert blocks(restored.state_snapshot) == local, "the plane did not keep the blocks verbatim"
    done = agent(store2).resume(exec_id, resume_value=Resume(approved=True))
    assert done.status == "completed"


def test_a_provider_tool_turn_survives_the_plane(connected: Any, tmp_path: Path) -> None:
    """2D across machines: a turn in which Anthropic ran ``web_search`` is
    replicated as ``Message.provider_content``, served back verbatim, and a run
    resumed from the plane copy sends it — Anthropic checks the encrypted search
    results and refuses an altered turn (real ``claude-sonnet-5-5``)."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")
    from fastaiagent import Agent, FunctionTool, LLMClient, Resume, SQLiteCheckpointer
    from fastaiagent.checkpointers import platform_replica
    from fastaiagent.tool import web_search

    def agent(store: Any) -> Agent:
        return Agent(
            name="searching-desk",
            system_prompt=(
                "Search the web for the fact asked, then call approve with it, then "
                "confirm in one line."
            ),
            llm=LLMClient(provider="anthropic", model="claude-sonnet-5-5", max_retries=2),
            tools=[web_search(max_uses=1), FunctionTool(name="approve", fn=approve)],
            checkpointer=store,
        )

    def carried(snapshot: dict[str, Any]) -> list[Any]:
        return [m["provider_content"] for m in snapshot["messages"] if m.get("provider_content")]

    store1 = SQLiteCheckpointer(db_path=str(tmp_path / "p1.db"))
    exec_id = f"b12-provider-{uuid.uuid4().hex[:8]}"
    first = agent(store1).run("What is the latest stable Python release?", execution_id=exec_id)
    if first.status != "paused":
        pytest.skip("the model answered without calling approve this time")
    local = carried(store1.get_last(exec_id).state_snapshot)
    if not local:
        pytest.skip("the model called approve before searching this time")
    assert local[0]["provider"] == "anthropic"
    _drain(store1, exec_id)

    store2 = SQLiteCheckpointer(db_path=str(tmp_path / "p2.db"))
    store2.setup()
    restored = platform_replica.restore_from_plane(store2, exec_id)
    assert restored is not None
    assert carried(restored.state_snapshot) == local, "the plane did not keep the turn verbatim"
    done = agent(store2).resume(exec_id, resume_value=Resume(approved=True))
    assert done.status == "completed"


def test_a_provider_tool_span_reaches_the_plane_without_its_payload(
    connected: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A provider-run call is a ``tool.web_search`` span with ``tool.origin =
    "provider"``; with payloads off its query and sources stay local (real
    ``gemini-3.8-flash``)."""
    if not os.environ.get("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY not set")
    from fastaiagent import Agent, LLMClient
    from fastaiagent.tool import web_search

    monkeypatch.setenv("FASTAIAGENT_TRACE_PAYLOADS", "0")
    result = Agent(
        name="grounded",
        llm=LLMClient(provider="gemini", model="gemini-3.8-flash", max_retries=2),
        tools=[web_search()],
    ).run("Search the web: what is the latest stable Python release? One line.")
    assert "web_search" in [c["tool_name"] for c in result.tool_calls]

    spans = _plane_spans(connected, result.trace_id)
    (call,) = [s for s in spans if s["name"] == "tool.web_search"]
    stored = call.get("input") or {}
    assert stored.get("tool.origin") == "provider", sorted(stored)
    assert stored.get("fastaiagent.runner.type") == "tool"
    assert "tool.result" not in stored and "tool.args" not in stored


def test_cache_and_reasoning_counts_reach_the_plane_and_the_text_does_not(
    connected: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With payloads off (decision B2-4): the counts are exported, the reasoning
    text is not. Real ``claude-haiku-4-5`` with ``cache=True`` and extended thinking."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")
    from fastaiagent import Agent, LLMClient

    monkeypatch.setenv("FASTAIAGENT_TRACE_PAYLOADS", "0")
    policy = f"Policy {uuid.uuid4()}. " + "Refunds are issued within 14 days. " * 900
    result = Agent(
        name="cached-thinker",
        system_prompt=policy,
        llm=LLMClient(provider="anthropic", model="claude-haiku-4-5", reasoning="low", cache=True),
    ).run("How many days do refunds take? One line.")
    assert result.status == "completed"

    (call,) = [s for s in _plane_spans(connected, result.trace_id) if s["name"].startswith("llm.")]
    stored = call.get("input") or {}
    assert stored.get("gen_ai.usage.cache_creation.input_tokens", 0) > 4000, sorted(stored)
    assert "gen_ai.usage.cache_read.input_tokens" in stored
    assert stored.get("gen_ai.usage.reasoning_tokens", 0) > 0
    assert stored.get("gen_ai.usage.input_tokens", 0) > 4000, "the whole prompt, cache included"
    assert "gen_ai.response.reasoning" not in stored, "reasoning text left with payloads off"


@pytest.mark.parametrize("stop", ["cancelled", "timeout"])
def test_a_stopped_run_reaches_the_plane_as_failed_with_its_reason(
    connected: Any, tmp_path: Path, stop: str
) -> None:
    """Decision B2-8: no new checkpoint status for the plane — a cancelled or timed
    out run ends ``failed`` and says why in ``run_reason``."""
    import asyncio

    from fastaiagent import Agent, AgentConfig, AgentTimeoutError, FunctionTool, SQLiteCheckpointer
    from fastaiagent.checkpointers import platform_replica

    async def slow() -> str:
        """Take a while."""
        await asyncio.sleep(10)
        return "done"

    store = SQLiteCheckpointer(db_path=str(tmp_path / "stop.db"))
    exec_id = f"b12-{stop}-{uuid.uuid4().hex[:8]}"
    agent = Agent(
        name="stoppable",
        llm=_scripted(("", [{"name": "slow", "arguments": {}}]), "done"),
        tools=[FunctionTool(name="slow", fn=slow)],
        checkpointer=store,
        config=AgentConfig(timeout_seconds=0.5) if stop == "timeout" else AgentConfig(),
    )

    async def main() -> Any:
        task = asyncio.ensure_future(agent.arun("go", execution_id=exec_id))
        if stop == "cancelled":
            await asyncio.sleep(0.4)
            agent.cancel(exec_id)
        return await task

    if stop == "timeout":
        with pytest.raises(AgentTimeoutError):
            asyncio.run(main())
    else:
        assert asyncio.run(main()).status == "cancelled"
    _drain(store, exec_id)
    served = platform_replica.fetch_latest_from_plane(exec_id)
    assert served is not None
    assert (served.node_id, served.status) == ("run_end", "failed")
    assert served.state_snapshot.get("run_reason") == stop, served.state_snapshot
