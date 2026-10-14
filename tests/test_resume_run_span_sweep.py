"""Contract sweep: a resumed run is a run, and gets its own root span (1.89.0).

``Agent.arun`` opened the ``agent.<name>`` root span; ``Agent.aresume`` called
the loop directly at both of its re-entry points and opened none. So a resume's
model calls were stored as orphan ``llm.*`` roots, the re-invoked tool ran
outside any agent, ``Agent.resume`` returned ``trace_id=None``, and nothing tied
the resumed work back to the run it continued. Agent root spans carried no
execution id at all, so even a fresh run could not be joined to its checkpoints.

Decision D7: each resume is a **new trace**, linked by attributes —
``agent.execution_id`` / ``agent.resumed_execution_id`` plus
``agent.resume.{node_id,kind,approved,resolver,reason}``.

The matrix runs every axis the defect could hide on:

* entry point — sync ``resume`` and async ``aresume``;
* what paused — a governance ``approval``, a user ``interrupt()``, a crash at a
  tool boundary, a crash at a turn boundary;
* the decision — approved and rejected;
* tracing on and off (``FASTAIAGENT_TRACE_ENABLED=0`` must not break a resume
  and must store no spans);
* topology — a standalone Agent, and an Agent nested in a Chain, a Swarm, a
  Supervisor and a routed Supervisor (each resume nests the same agent spans a
  fresh run does);
* a run that pauses again after its resume.

Real ``Agent`` / ``Chain`` / ``Swarm`` / ``Supervisor``, real
``SQLiteCheckpointer``, real OTel provider writing to a real ``local.db``. The
models are ``fastaiagent.testing.FunctionModel``; the approval pause comes from
the in-process governance plane stand-in the approval tests use.
"""

from __future__ import annotations

import asyncio
import json
import re
import sqlite3
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from fastaiagent import Agent, FunctionTool, Resume, SQLiteCheckpointer, governance, interrupt
from fastaiagent._internal.errors import ChainCheckpointError
from fastaiagent.chain.interrupt import AlreadyResumed
from fastaiagent.llm.client import LLMResponse
from fastaiagent.llm.message import ToolCall
from fastaiagent.testing import FunctionModel

TRACE_ID_RE = re.compile(r"^[0-9a-f]{32}$")
USAGE = {"prompt_tokens": 7, "completion_tokens": 5, "total_tokens": 12}
ROOT_PREFIXES = ("agent.", "chain.", "swarm.", "supervisor.")


# ---------------------------------------------------------------------------
# Environment: a private local.db and a tracer provider bound to it
# ---------------------------------------------------------------------------


@pytest.fixture(params=["traced", "untraced"])
def tracing(request, monkeypatch, tmp_path) -> Iterator[dict[str, Any]]:
    """A clean provider + a private local.db, with tracing on or off.

    The provider captures its db path when built, so both are reset together —
    the same reason ``tests/test_trace_enabled_master_switch.py`` does it.
    """
    from fastaiagent._internal import instance as _instance
    from fastaiagent._internal import project as _project
    from fastaiagent._internal.config import reset_config
    from fastaiagent.trace import otel

    db = tmp_path / "local.db"
    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(db))
    on = request.param == "traced"
    if not on:
        monkeypatch.setenv("FASTAIAGENT_TRACE_ENABLED", "0")
    reset_config()
    _project.set_project_id("test-proj")
    _instance.reset_for_testing()
    otel.reset()
    try:
        yield {"on": on, "db": db, "ckpt": str(tmp_path / "ckpt.db")}
    finally:
        otel.reset()
        _instance.reset_for_testing()
        _project.reset_for_testing()
        reset_config()


@pytest.fixture
def traced(monkeypatch, tmp_path) -> Iterator[dict[str, Any]]:
    """Tracing on only — for the cases that are about the span itself."""
    from fastaiagent._internal import instance as _instance
    from fastaiagent._internal import project as _project
    from fastaiagent._internal.config import reset_config
    from fastaiagent.trace import otel

    db = tmp_path / "local.db"
    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(db))
    reset_config()
    _project.set_project_id("test-proj")
    _instance.reset_for_testing()
    otel.reset()
    try:
        yield {"on": True, "db": db, "ckpt": str(tmp_path / "ckpt.db")}
    finally:
        otel.reset()
        _instance.reset_for_testing()
        _project.reset_for_testing()
        reset_config()


def _spans(db: Path) -> list[dict[str, Any]]:
    if not db.exists():
        return []
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    try:
        try:
            rows = conn.execute(
                "SELECT trace_id, span_id, parent_span_id, name, status, attributes "
                "FROM spans ORDER BY start_time"
            ).fetchall()
        except sqlite3.OperationalError:
            return []
    finally:
        conn.close()
    return [{**dict(r), "attrs": json.loads(r["attributes"] or "{}")} for r in rows]


def _trace(db: Path, trace_id: str) -> list[dict[str, Any]]:
    return [s for s in _spans(db) if s["trace_id"] == trace_id]


def _roots(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [s for s in spans if not s["parent_span_id"]]


def _assert_no_orphans(db: Path) -> None:
    """Every root in the store is a runner — no ``llm.*`` / ``tool.*`` roots."""
    bad = [s["name"] for s in _roots(_spans(db)) if not s["name"].startswith(ROOT_PREFIXES)]
    assert not bad, f"orphan root spans (work traced outside any run): {bad}"


def _agent_names(spans: list[dict[str, Any]]) -> list[str]:
    return sorted(s["name"] for s in spans if s["name"].startswith("agent."))


def _call(entry: str, runner: Any, execution_id: str, **kw: Any) -> Any:
    if entry == "sync":
        out = runner.resume(execution_id, **kw)
        # ``Chain.resume`` is the one async-only ``resume``.
        return asyncio.run(out) if asyncio.iscoroutine(out) else out
    return asyncio.run(runner.aresume(execution_id, **kw))


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------


def _reply(text: str) -> LLMResponse:
    return LLMResponse(content=text, finish_reason="stop", usage=dict(USAGE))


def _call_tool(name: str, args: dict[str, Any], n: int = 0) -> LLMResponse:
    return LLMResponse(
        content=None,
        tool_calls=[ToolCall(id=f"call_{name}_{n}", name=name, arguments=args)],
        finish_reason="tool_calls",
        usage=dict(USAGE),
    )


def _tool_results(messages: list[Any]) -> list[str]:
    return [str(m.content) for m in messages if m.role.value == "tool"]


def _refund_tool(ran: list[str]) -> FunctionTool:
    def refund(order: str) -> str:
        """Refund an order; a manager must approve."""
        decision = interrupt(reason="manager_approval", context={"order": order})
        if not decision.approved:
            return f"declined {order}"
        ran.append(order)
        return f"refunded {order}"

    return FunctionTool(name="refund", fn=refund)


def _refund_llm(orders: tuple[str, ...] = ("1042",)) -> FunctionModel:
    """Asks for each order's refund in turn, then answers with the last result."""

    def responder(messages: list[Any]) -> LLMResponse:
        done = _tool_results(messages)
        if len(done) < len(orders):
            return _call_tool("refund", {"order": orders[len(done)]}, len(done))
        return _reply(f"Done: {done[-1]}")

    return FunctionModel(responder)


def _refund_agent(ran: list[str], ckpt: str | None, orders=("1042",), name="refund-bot") -> Agent:
    return Agent(
        name=name,
        system_prompt="Refund orders with the refund tool.",
        llm=_refund_llm(orders),
        tools=[_refund_tool(ran)],
        checkpointer=SQLiteCheckpointer(ckpt) if ckpt else None,
    )


class _Crash(BaseException):
    """Kills the run the way a process death would: past every ``except Exception``."""


def _crash_agent(shape: str, ran: list[str], ckpt: str) -> Agent:
    """Dies once — inside the tool (``tool_boundary``) or in the model call after
    it (``turn_boundary``) — and runs clean on the resume."""
    state = {"tool": 0, "after": 0}

    def lookup(order: str) -> str:
        """Look an order up."""
        state["tool"] += 1
        if shape == "tool_boundary" and state["tool"] == 1:
            raise _Crash()
        ran.append(order)
        return f"order {order} shipped"

    def responder(messages: list[Any]) -> LLMResponse:
        done = _tool_results(messages)
        if not done:
            return _call_tool("lookup", {"order": "77"})
        state["after"] += 1
        if shape == "turn_boundary" and state["after"] == 1:
            raise _Crash()
        return _reply(f"Done: {done[-1]}")

    return Agent(
        name="crash-bot",
        llm=FunctionModel(responder),
        tools=[FunctionTool(name="lookup", fn=lookup)],
        checkpointer=SQLiteCheckpointer(ckpt),
    )


# --- the approval shape: a governance policy pause from the plane stand-in ----


@pytest.fixture
def gov_plane() -> Iterator[Any]:
    import fastaiagent
    from tests._governance_plane import reset_connection, serve

    with serve() as (state, url):
        fastaiagent.connect(api_key="fa_k_gov_test", target=url)
        try:
            yield state
        finally:
            reset_connection()


def _banker(ran: list[dict[str, Any]], ckpt: str) -> Agent:
    from tests._governance_plane import AGENT_ID

    def transfer_funds(amount: int, to: str) -> str:
        """Move money."""
        ran.append({"amount": amount, "to": to})
        return f"Transferred ${amount} to {to}."

    def responder(messages: list[Any]) -> LLMResponse:
        done = _tool_results(messages)
        if not done:
            return _call_tool("transfer_funds", {"amount": 500, "to": "Bob"})
        return _reply(f"Tool said: {done[-1]}")

    return Agent(
        name="banker",
        agent_id=AGENT_ID,
        system_prompt="Move money with transfer_funds.",
        llm=FunctionModel(responder),
        tools=[FunctionTool(name="transfer_funds", fn=transfer_funds)],
        checkpointer=SQLiteCheckpointer(ckpt),
    )


# ---------------------------------------------------------------------------
# 1. Standalone agent: entry × kind × decision × tracing
# ---------------------------------------------------------------------------

STANDALONE = [
    ("interrupt", True),
    ("interrupt", False),
    ("approval", True),
    ("approval", False),
    ("tool_boundary", None),
    ("turn_boundary", None),
]


@pytest.mark.parametrize("entry", ["sync", "async"])
@pytest.mark.parametrize(("kind", "approved"), STANDALONE, ids=[f"{k}-{a}" for k, a in STANDALONE])
def test_a_standalone_resume_is_one_linked_run(
    entry: str, kind: str, approved: bool | None, tracing: dict[str, Any], request
) -> None:
    exec_id = f"run-{uuid.uuid4().hex[:8]}"
    ran: list[Any] = []
    resume_kw: dict[str, Any] = {}
    meta = {"resolver": "alice@example.com", "reason": "manager signed off"}

    if kind == "interrupt":
        agent = _refund_agent(ran, tracing["ckpt"])
        first = agent.run("Refund order 1042.", execution_id=exec_id)
        resume_kw["resume_value"] = Resume(approved=approved, metadata=meta)
    elif kind == "approval":
        request.getfixturevalue("gov_plane")
        agent = _banker(ran, tracing["ckpt"])
        first = agent.run("Pay Bob $500.", execution_id=exec_id)
        assert first.pending_interrupt["reason"] == governance.APPROVAL_REASON
        resume_kw["resume_value"] = Resume(approved=approved, metadata=meta)
    else:
        agent = _crash_agent(kind, ran, tracing["ckpt"])
        with pytest.raises(_Crash):
            agent.run("Where is order 77?", execution_id=exec_id)
        first = None
    if first is not None:
        assert first.status == "paused", first

    result = _call(entry, agent, exec_id, **resume_kw)

    # The resume itself is right whatever the tracing switch says.
    assert result.status == "completed", result
    assert result.execution_id == exec_id
    if kind == "interrupt":
        assert ran == (["1042"] if approved else [])
        assert result.output == ("Done: refunded 1042" if approved else "Done: declined 1042")
    elif kind == "approval":
        assert len(ran) == (1 if approved else 0)
    else:
        assert ran == ["77"]

    if not tracing["on"]:
        assert result.trace_id is None
        assert _spans(tracing["db"]) == [], "tracing off still stored spans"
        return

    assert result.trace_id and TRACE_ID_RE.match(result.trace_id), result.trace_id
    if first is not None:
        assert result.trace_id != first.trace_id, "D7: a resume is a new trace"
    spans = _trace(tracing["db"], result.trace_id)
    roots = _roots(spans)
    assert [r["name"] for r in roots] == ["agent." + agent.name], roots
    root = roots[0]["attrs"]
    assert root["agent.execution_id"] == exec_id
    assert root["agent.resumed_execution_id"] == exec_id
    assert root["agent.resume.kind"] == kind
    assert root["agent.resume.node_id"].startswith("turn:")
    assert root["agent.status"] == "completed"
    assert root["agent.output"] == result.output
    assert root["agent.tokens_used"] == result.tokens_used > 0
    assert "agent.latency_ms" in root and "agent.input" in root and "agent.config" in root
    if approved is None:
        assert "agent.resume.approved" not in root
    else:
        assert root["agent.resume.approved"] is approved
        assert root["agent.resume.resolver"] == "alice@example.com"
        assert root["agent.resume.reason"] == "manager signed off"
    # The resumed model calls hang under the agent span.
    assert any(s["name"].startswith("llm.") for s in spans)
    _assert_no_orphans(tracing["db"])


def test_the_resolver_falls_back_to_approver(traced: dict[str, Any]) -> None:
    ran: list[str] = []
    agent = _refund_agent(ran, traced["ckpt"])
    agent.run("Refund order 1042.", execution_id="appr-1")
    result = agent.resume(
        "appr-1", resume_value=Resume(approved=True, metadata={"approver": "bob"})
    )
    root = _roots(_trace(traced["db"], result.trace_id))[0]["attrs"]
    assert root["agent.resume.resolver"] == "bob"
    assert "agent.resume.reason" not in root


# ---------------------------------------------------------------------------
# 2. A run that pauses again after its resume
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("entry", ["sync", "async"])
def test_a_resume_that_pauses_again_still_names_its_trace(
    entry: str, tracing: dict[str, Any]
) -> None:
    ran: list[str] = []
    agent = _refund_agent(ran, tracing["ckpt"], orders=("A", "B"))
    first = agent.run("Refund A and B.", execution_id="twice")
    assert first.status == "paused"

    again = _call(entry, agent, "twice", resume_value=Resume(approved=True))
    assert again.status == "paused", again
    assert again.pending_interrupt["context"] == {"order": "B"}
    done = _call(entry, agent, "twice", resume_value=Resume(approved=True))
    assert done.status == "completed"
    assert ran == ["A", "B"]

    if not tracing["on"]:
        assert again.trace_id is None and done.trace_id is None
        assert _spans(tracing["db"]) == []
        return
    assert again.trace_id and done.trace_id
    assert len({first.trace_id, again.trace_id, done.trace_id}) == 3
    paused_root = _roots(_trace(tracing["db"], again.trace_id))[0]["attrs"]
    assert paused_root["agent.status"] == "paused"
    assert paused_root["agent.resumed_execution_id"] == "twice"
    done_root = _roots(_trace(tracing["db"], done.trace_id))[0]["attrs"]
    assert done_root["agent.status"] == "completed"
    # Three traces, one execution: every root names it.
    roots = _roots(_spans(tracing["db"]))
    assert [r["attrs"].get("agent.execution_id") for r in roots] == ["twice"] * 3
    _assert_no_orphans(tracing["db"])


def test_a_resume_that_pauses_inside_the_reinvoked_tool_names_its_trace(
    traced: dict[str, Any],
) -> None:
    """The other pause-again path: crash recovery re-invokes a tool that has not
    reached its ``interrupt()`` yet, and it pauses there (1.88.0's fix). That
    return leaves ``aresume`` from inside the tool re-invocation, not the loop."""
    ran: list[str] = []
    store = SQLiteCheckpointer(traced["ckpt"])
    state = {"n": 0}

    def refund(order: str) -> str:
        """Refund; dies once before reaching the approval."""
        state["n"] += 1
        if state["n"] == 1:
            raise _Crash()
        decision = interrupt(reason="manager_approval", context={"order": order})
        ran.append(order)
        return "refunded" if decision.approved else "declined"

    agent = Agent(
        name="refund-bot",
        llm=_refund_llm(),
        tools=[FunctionTool(name="refund", fn=refund)],
        checkpointer=store,
    )
    with pytest.raises(_Crash):
        agent.run("Refund 1042.", execution_id="crash-pause")
    paused = agent.resume("crash-pause")
    assert paused.status == "paused"
    assert paused.trace_id and TRACE_ID_RE.match(paused.trace_id)
    root = _roots(_trace(traced["db"], paused.trace_id))[0]["attrs"]
    assert root["agent.resume.kind"] == "tool_boundary"
    assert root["agent.status"] == "paused"


# ---------------------------------------------------------------------------
# 3. A failed claim is not a run
# ---------------------------------------------------------------------------


def test_a_lost_claim_opens_no_span(traced: dict[str, Any]) -> None:
    ran: list[str] = []
    agent = _refund_agent(ran, traced["ckpt"])
    agent.run("Refund order 1042.", execution_id="claimed")
    agent.resume("claimed", resume_value=Resume(approved=True))
    before = len(_spans(traced["db"]))
    with pytest.raises(AlreadyResumed):
        agent.resume("claimed", resume_value=Resume(approved=True))
    with pytest.raises(ChainCheckpointError):
        agent.resume("never-ran")
    assert len(_spans(traced["db"])) == before, "a refused resume opened a span"


def test_trace_false_resumes_without_a_span(traced: dict[str, Any]) -> None:
    ran: list[str] = []
    agent = _refund_agent(ran, traced["ckpt"])
    agent.run("Refund order 1042.", execution_id="quiet")
    before = len(_roots(_spans(traced["db"])))
    result = agent.resume("quiet", resume_value=Resume(approved=True), trace=False)
    assert result.status == "completed" and result.trace_id is None
    assert ran == ["1042"]
    agent_roots = [s for s in _roots(_spans(traced["db"])) if s["name"].startswith("agent.")]
    assert len(agent_roots) == before


# ---------------------------------------------------------------------------
# 4. Fresh runs name their execution too
# ---------------------------------------------------------------------------


def test_arun_and_astream_stamp_the_execution_id(traced: dict[str, Any]) -> None:
    agent = Agent(name="plain", llm=FunctionModel(lambda m: _reply("hi")))
    ran = agent.run("hello", execution_id="named")
    auto = agent.run("hello")
    streamed = agent.stream("hello")
    for result in (ran, auto, streamed):
        assert result.execution_id
        root = _roots(_trace(traced["db"], result.trace_id))[0]["attrs"]
        assert root["agent.execution_id"] == result.execution_id
    assert ran.execution_id == "named"


def test_an_execution_id_alone_does_not_make_an_export_durable(traced: dict[str, Any]) -> None:
    """Every root carries ``agent.execution_id`` now; ``durable`` needs checkpoints."""
    from fastaiagent._internal.storage import SQLiteHelper
    from fastaiagent.trace.trace_export import build_export_payload

    plain = Agent(name="plain", llm=FunctionModel(lambda m: _reply("hi"))).run("hello")
    durable = Agent(
        name="kept",
        llm=FunctionModel(lambda m: _reply("hi")),
        checkpointer=SQLiteCheckpointer(str(traced["db"])),
    ).run("hello")
    with SQLiteHelper(str(traced["db"])) as db:
        plain_trace = build_export_payload(db, plain.trace_id)["trace"]
        kept = build_export_payload(db, durable.trace_id)
    assert plain_trace["execution_id"] == plain.execution_id
    assert plain_trace["durable"] is False
    assert kept["trace"]["durable"] is True and kept["checkpoints"]


def test_a_fork_is_a_linked_run(traced: dict[str, Any]) -> None:
    agent = Agent(
        name="forker",
        llm=FunctionModel(lambda m: _reply("answer")),
        checkpointer=SQLiteCheckpointer(traced["ckpt"]),
    )
    agent.run("hello", execution_id="source")
    for kw in ({}, {"input": "a different question"}):
        fork = agent.fork("source", **kw)
        assert fork.trace_id, kw
        root = _roots(_trace(traced["db"], fork.trace_id))[0]["attrs"]
        assert root["agent.execution_id"] == fork.execution_id
        assert root["agent.forked_execution_id"] == fork.execution_id
        assert root["agent.fork.of_execution"] == "source"
        assert root["agent.fork.checkpoint_id"]
    _assert_no_orphans(traced["db"])


# ---------------------------------------------------------------------------
# 5. An Agent nested in a Chain / Swarm / Supervisor: topology × entry ×
#    decision × tracing. A resume nests the same agent spans a fresh run does.
# ---------------------------------------------------------------------------


def _chain(ran: list[str], ckpt: str) -> Any:
    from fastaiagent import Chain

    chain = Chain("refund-chain", checkpoint_enabled=True, checkpointer=SQLiteCheckpointer(ckpt))
    chain.add_node("refund", agent=_refund_agent(ran, None))
    return chain


def _swarm(ran: list[str], ckpt: str) -> Any:
    from fastaiagent.agent.swarm import Swarm

    return Swarm(
        name="desk",
        agents=[_refund_agent(ran, None)],
        entrypoint="refund-bot",
        checkpointer=SQLiteCheckpointer(ckpt),
    )


def _supervisor(ran: list[str], ckpt: str) -> Any:
    from fastaiagent.agent.team import Supervisor, Worker

    def boss(messages: list[Any]) -> LLMResponse:
        done = _tool_results(messages)
        if not done:
            return _call_tool("delegate_to_refunds", {"task": "Refund order 1042."})
        return _reply(f"Boss: {done[-1]}")

    return Supervisor(
        name="boss",
        llm=FunctionModel(boss),
        workers=[Worker(agent=_refund_agent(ran, None), role="refunds", description="Refunds")],
        checkpointer=SQLiteCheckpointer(ckpt),
    )


def _routed(ran: list[str], ckpt: str, stub: Any) -> Any:
    from fastaiagent.agent.team import Supervisor, Worker

    stub.choices["worker"] = "refunds"
    return Supervisor(
        name="router",
        llm=FunctionModel(lambda m: _reply("unused")),
        workers=[
            Worker(agent=_refund_agent(ran, None), role="refunds", description="Refunds"),
            Worker(
                agent=Agent(name="other", llm=FunctionModel(lambda m: _reply("general"))),
                role="other",
                description="Anything else",
            ),
        ],
        routing="decisions",
        router_llm=stub.client(),
        fallback_worker="other",
        checkpointer=SQLiteCheckpointer(ckpt),
    )


TOPOLOGIES = {
    # name: (root span prefix, id attribute on that root)
    "chain": ("chain.refund-chain", "chain"),
    "swarm": ("swarm.desk", "swarm"),
    "supervisor": ("supervisor.boss", "supervisor"),
    "supervisor_routed": ("supervisor.router", "supervisor"),
}


@pytest.mark.parametrize("approved", [True, False], ids=["approved", "rejected"])
@pytest.mark.parametrize("entry", ["sync", "async"])
@pytest.mark.parametrize("topology", sorted(TOPOLOGIES))
def test_a_nested_resume_nests_its_agent_span(
    topology: str, entry: str, approved: bool, tracing: dict[str, Any], request
) -> None:
    ran: list[str] = []
    exec_id = f"{topology}-{uuid.uuid4().hex[:6]}"
    if topology == "chain":
        runner = _chain(ran, tracing["ckpt"])
        first = runner.execute({"input": "Refund order 1042."}, execution_id=exec_id)
    else:
        if topology == "swarm":
            runner = _swarm(ran, tracing["ckpt"])
        elif topology == "supervisor":
            runner = _supervisor(ran, tracing["ckpt"])
        else:
            runner = _routed(ran, tracing["ckpt"], request.getfixturevalue("decisions_stub"))
        first = runner.run("Refund order 1042.", execution_id=exec_id)
    assert first.status == "paused", first

    meta = {"resolver": "carol", "reason": "looks fine"}
    result = _call(entry, runner, exec_id, resume_value=Resume(approved=approved, metadata=meta))

    assert result.status == "completed", result
    assert ran == (["1042"] if approved else [])

    if not tracing["on"]:
        assert result.trace_id is None
        assert _spans(tracing["db"]) == []
        return

    root_name, kind = TOPOLOGIES[topology]
    first_spans = _trace(tracing["db"], first.trace_id)
    spans = _trace(tracing["db"], result.trace_id)
    roots = _roots(spans)
    assert [r["name"] for r in roots] == [root_name], roots
    root = roots[0]["attrs"]
    assert root[f"{kind}.execution_id"] == exec_id
    assert root.get(f"{kind}.resumed_execution_id") == exec_id
    # Same agents as the fresh run, each exactly once.
    assert _agent_names(spans) == _agent_names(first_spans), (
        f"resume nested {_agent_names(spans)}, the fresh run {_agent_names(first_spans)}"
    )
    assert "agent.refund-bot" in _agent_names(spans)
    worker = next(s for s in spans if s["name"] == "agent.refund-bot")["attrs"]
    if topology == "chain":
        # The chain owns the pause and re-runs its node from the top; the
        # decision is the chain's to record.
        assert root["chain.resume.approved"] is approved
        assert root["chain.resume.resolver"] == "carol"
        assert root["chain.resume.reason"] == "looks fine"
        assert root["chain.resume.node_id"] == "refund"
    else:
        assert worker["agent.execution_id"] == exec_id
        assert worker["agent.resumed_execution_id"] == exec_id
        assert worker["agent.resume.kind"] == "interrupt"
        assert worker["agent.resume.approved"] is approved
        assert worker["agent.resume.resolver"] == "carol"
    _assert_no_orphans(tracing["db"])


# ---------------------------------------------------------------------------
# 6. The sensitive key is gated on the way out
# ---------------------------------------------------------------------------


def test_the_resume_reason_is_a_payload_key(monkeypatch) -> None:
    from fastaiagent.trace.redaction import is_sensitive_attr

    for key in ("agent.resume.reason", "chain.resume.reason"):
        assert is_sensitive_attr(key), key
    for key in (
        "agent.resume.approved",
        "agent.resume.resolver",
        "agent.resume.kind",
        "agent.execution_id",
        "agent.resumed_execution_id",
    ):
        assert not is_sensitive_attr(key), key


def test_payloads_off_drops_the_reason_from_an_export(traced: dict[str, Any], monkeypatch) -> None:
    """What actually leaves: the exported attribute map of the resumed root."""
    from fastaiagent.trace.redaction import apply_export_policy

    ran: list[str] = []
    agent = _refund_agent(ran, traced["ckpt"])
    agent.run("Refund order 1042.", execution_id="gated")
    result = agent.resume(
        "gated", resume_value=Resume(approved=True, metadata={"reason": "SECRET-REASON"})
    )
    root = _roots(_trace(traced["db"], result.trace_id))[0]["attrs"]
    monkeypatch.setenv("FASTAIAGENT_TRACE_PAYLOADS", "0")
    from fastaiagent._internal.config import reset_config

    reset_config()
    exported = apply_export_policy(dict(root))
    assert "agent.resume.reason" not in exported
    assert exported["agent.resume.approved"] is True
    assert exported["agent.resumed_execution_id"] == "gated"
