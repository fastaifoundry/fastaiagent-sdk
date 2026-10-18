"""A resumed tool call is governed like any other (1.90.0, F5).

``Agent.aresume`` re-ran the paused tool with a bare ``tool.aexecute(...)``. So
the call that a human had just approved — the one most worth checking — ran
with none of what every other tool call gets:

* no ``tool_result`` guardrail: the result went to the model unjudged;
* no ``tool_call`` guardrail: the checkpoint saves the arguments *before* the
  guardrail, so a masked argument reached the tool unmasked on resume;
* no ``wrap_tool`` middleware, no ``tool.<name>`` span, no timeout or retries;
* and crash recovery asked **no governance at all** — the pre-tool checkpoint is
  written before the gate, so a call recovered from it may never have been asked.

Since 1.90.0 the agent loop, resume and the MCP surface share one governed
tool-call function. Real Agent / Swarm / Supervisor / Chain, real
``SQLiteCheckpointer``, real OTel provider on a private ``local.db``;
``FunctionModel`` and the in-process governance plane stand-in.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from fastaiagent import Agent, FunctionTool, Resume, SQLiteCheckpointer, governance, interrupt
from fastaiagent._internal.errors import GuardrailBlockedError
from fastaiagent.agent.middleware import AgentMiddleware
from fastaiagent.guardrail.guardrail import Guardrail, GuardrailPosition, GuardrailType
from fastaiagent.llm.client import LLMResponse
from fastaiagent.llm.message import ToolCall
from fastaiagent.testing import FunctionModel

SSN = "123-45-6789"
SSN_RE = r"\b\d{3}-\d{2}-\d{4}\b"


def _mask(position: GuardrailPosition) -> Guardrail:
    return Guardrail(
        name=f"mask-ssn-{position.value}",
        guardrail_type=GuardrailType.regex,
        position=position,
        config={"pattern": SSN_RE, "should_match": False},
        action="mask",
    )


def _block_secret() -> Guardrail:
    return Guardrail(
        name="no-secret-in-results",
        guardrail_type=GuardrailType.regex,
        position=GuardrailPosition.tool_result,
        config={"pattern": r"SECRET-TOKEN", "should_match": False},
    )


@pytest.fixture(params=["traced", "untraced"])
def tracing(request, monkeypatch, tmp_path) -> Iterator[dict[str, Any]]:
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


def _spans(db: Path, trace_id: str) -> list[dict[str, Any]]:
    from fastaiagent.trace.otel import get_tracer_provider

    provider = get_tracer_provider()
    if provider is not None and hasattr(provider, "force_flush"):
        provider.force_flush()
    if not db.exists():
        return []
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT name, attributes FROM spans WHERE trace_id = ? ORDER BY start_time",
            (trace_id,),
        ).fetchall()
    finally:
        conn.close()
    return [{"name": r["name"], "attrs": json.loads(r["attributes"] or "{}")} for r in rows]


def _tool_results(messages: list[Any]) -> list[str]:
    return [str(m.content) for m in messages if m.role.value == "tool"]


class _Model:
    """A FunctionModel that asks for one tool call, then answers, and keeps every
    message list it was sent — the evidence of what the model saw."""

    def __init__(self, tool: str, args: dict[str, Any]) -> None:
        self.seen: list[str] = []
        self.tool, self.args = tool, args

    def respond(self, messages: list[Any]) -> LLMResponse:
        self.seen.append(" ".join(str(m.content) for m in messages))
        done = _tool_results(messages)
        if not done:
            return LLMResponse(
                content=None,
                tool_calls=[ToolCall(id="call_1", name=self.tool, arguments=self.args)],
                finish_reason="tool_calls",
            )
        return LLMResponse(content=f"Done: {done[-1]}", finish_reason="stop")

    def llm(self) -> FunctionModel:
        return FunctionModel(self.respond)


def _refund_tool(ran: list[dict[str, Any]], result: str = f"refunded; holder ssn {SSN}"):
    def refund(order: str, note: str = "") -> str:
        """Refund an order; a manager must approve."""
        decision = interrupt(reason="manager_approval", context={"order": order})
        if not decision.approved:
            return f"declined {order}"
        ran.append({"order": order, "note": note})
        return result

    return FunctionTool(name="refund", fn=refund)


def _refund_agent(model: _Model, ran: list, ckpt: str | None, **kw: Any) -> Agent:
    return Agent(
        name="refund-bot",
        llm=model.llm(),
        tools=[kw.pop("tool", None) or _refund_tool(ran)],
        checkpointer=SQLiteCheckpointer(ckpt) if ckpt else None,
        **kw,
    )


# ---------------------------------------------------------------------------
# 1. tool_result guardrails judge the resumed call — every topology
# ---------------------------------------------------------------------------


def _build(topology: str, model: _Model, ran: list, ckpt: str, guardrails: list) -> Any:
    from fastaiagent import Chain
    from fastaiagent.agent.swarm import Swarm
    from fastaiagent.agent.team import Supervisor, Worker

    if topology == "agent":
        return _refund_agent(model, ran, ckpt, guardrails=guardrails)
    worker = _refund_agent(model, ran, None, guardrails=guardrails)
    if topology == "swarm":
        return Swarm(
            name="desk",
            agents=[worker],
            entrypoint="refund-bot",
            checkpointer=SQLiteCheckpointer(ckpt),
        )
    if topology == "supervisor":

        def boss(messages: list[Any]) -> LLMResponse:
            done = _tool_results(messages)
            if not done:
                return LLMResponse(
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id="d1", name="delegate_to_refunds", arguments={"task": "Refund 1042"}
                        )
                    ],
                    finish_reason="tool_calls",
                )
            model.seen.append(" ".join(str(m.content) for m in messages))
            return LLMResponse(content=f"Boss: {done[-1]}", finish_reason="stop")

        return Supervisor(
            name="boss",
            llm=FunctionModel(boss),
            workers=[Worker(agent=worker, role="refunds", description="Refunds")],
            checkpointer=SQLiteCheckpointer(ckpt),
        )
    chain = Chain("refund-chain", checkpoint_enabled=True, checkpointer=SQLiteCheckpointer(ckpt))
    chain.add_node("refund", agent=worker)
    return chain


def _start(topology: str, runner: Any, exec_id: str) -> Any:
    if topology == "chain":
        return runner.execute({"input": "Refund order 1042."}, execution_id=exec_id)
    return runner.run("Refund order 1042.", execution_id=exec_id)


@pytest.mark.parametrize("topology", ["agent", "swarm", "supervisor", "chain"])
def test_a_tool_result_mask_covers_the_resumed_call(topology: str, tracing) -> None:
    model, ran = _Model("refund", {"order": "1042"}), []
    runner = _build(topology, model, ran, tracing["ckpt"], [_mask(GuardrailPosition.tool_result)])
    exec_id = f"{topology}-{uuid.uuid4().hex[:6]}"

    first = _start(topology, runner, exec_id)
    assert first.status == "paused", first
    result = asyncio.run(runner.aresume(exec_id, resume_value=Resume(approved=True)))

    assert result.status == "completed", result
    assert ran and ran[0]["order"] == "1042"
    leaked = [s for s in model.seen if SSN in s]
    assert not leaked, f"the resumed tool's unmasked result reached the model: {leaked[0][:200]}"
    assert "[REDACTED]" in " ".join(model.seen)


def test_the_resumed_calls_firings_are_on_the_result(tracing) -> None:
    """``AgentResult.guardrails`` is the only way to see a non-halting outcome
    (a mask). The resumed call's guardrails ran before the continued loop opened
    the run's collection, so a resume reported none of them — found by the live
    run, not by the first version of this file."""
    model, ran = _Model("refund", {"order": "1042"}), []
    agent = _refund_agent(
        model, ran, tracing["ckpt"], guardrails=[_mask(GuardrailPosition.tool_result)]
    )
    exec_id = f"fr-{uuid.uuid4().hex[:6]}"

    agent.run("Refund 1042", execution_id=exec_id)
    result = asyncio.run(agent.aresume(exec_id, resume_value=Resume(approved=True)))

    fired = [(g.position, g.action_taken) for g in result.guardrails]
    assert ("tool_result", "masked") in fired, fired


def test_a_tool_result_block_stops_the_resumed_run(tracing) -> None:
    model, ran = _Model("refund", {"order": "1042"}), []
    tool = _refund_tool(ran, result="refunded; SECRET-TOKEN abc")
    agent = _refund_agent(model, ran, tracing["ckpt"], tool=tool, guardrails=[_block_secret()])
    exec_id = f"blk-{uuid.uuid4().hex[:6]}"

    assert agent.run("Refund 1042", execution_id=exec_id).status == "paused"
    with pytest.raises(GuardrailBlockedError):
        asyncio.run(agent.aresume(exec_id, resume_value=Resume(approved=True)))
    assert not any("SECRET-TOKEN" in s for s in model.seen)


# ---------------------------------------------------------------------------
# 2. tool_call guardrails: the saved arguments are the pre-guardrail ones
# ---------------------------------------------------------------------------


def test_a_tool_call_mask_is_reapplied_on_resume(tracing) -> None:
    model, ran = _Model("refund", {"order": "1042", "note": f"customer ssn {SSN}"}), []
    agent = _refund_agent(
        model, ran, tracing["ckpt"], guardrails=[_mask(GuardrailPosition.tool_call)]
    )
    exec_id = f"tc-{uuid.uuid4().hex[:6]}"

    assert agent.run("Refund 1042", execution_id=exec_id).status == "paused"
    asyncio.run(agent.aresume(exec_id, resume_value=Resume(approved=True)))

    assert ran, "the approved tool never ran"
    assert SSN not in ran[0]["note"], f"the tool got the unmasked argument: {ran[0]['note']!r}"
    assert "[REDACTED]" in ran[0]["note"]


# ---------------------------------------------------------------------------
# 3. middleware, the span, timeout
# ---------------------------------------------------------------------------


class _Recorder(AgentMiddleware):
    name = "recorder"

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def wrap_tool(self, ctx, tool, args, call_next):
        self.calls.append(dict(args))
        return await call_next(tool, {**args, "note": "from-middleware"})


def test_wrap_tool_middleware_wraps_the_resumed_call(tracing) -> None:
    model, ran = _Model("refund", {"order": "1042"}), []
    mw = _Recorder()
    agent = _refund_agent(model, ran, tracing["ckpt"], middleware=[mw])
    exec_id = f"mw-{uuid.uuid4().hex[:6]}"

    agent.run("Refund 1042", execution_id=exec_id)
    assert len(mw.calls) == 1
    asyncio.run(agent.aresume(exec_id, resume_value=Resume(approved=True)))

    assert len(mw.calls) == 2, "wrap_tool did not see the resumed call"
    assert ran[0]["note"] == "from-middleware", "the middleware's rewrite was not applied"


class _StopOnSecondCall(AgentMiddleware):
    name = "stop-on-second"

    def __init__(self) -> None:
        self.seen = 0

    async def wrap_tool(self, ctx, tool, args, call_next):
        from fastaiagent._internal.errors import StopAgent

        self.seen += 1
        if self.seen > 1:
            raise StopAgent("budget exhausted", reason="budget")
        return await call_next(tool, args)


def test_a_stop_during_the_resumed_call_ends_the_run_without_running_it(tracing) -> None:
    model, ran = _Model("refund", {"order": "1042"}), []
    agent = _refund_agent(model, ran, tracing["ckpt"], middleware=[_StopOnSecondCall()])
    exec_id = f"st-{uuid.uuid4().hex[:6]}"

    agent.run("Refund 1042", execution_id=exec_id)
    calls_before = len(model.seen)
    result = asyncio.run(agent.aresume(exec_id, resume_value=Resume(approved=True)))

    assert result.status == "completed"
    assert result.output == "budget exhausted"
    assert ran == [], "the stopper fired before call_next, so the tool must not run"
    assert len(model.seen) == calls_before, "a stopped run called the model again"


def test_the_resumed_call_has_its_tool_span(tracing) -> None:
    model, ran = _Model("refund", {"order": "1042"}), []
    agent = _refund_agent(model, ran, tracing["ckpt"])
    exec_id = f"sp-{uuid.uuid4().hex[:6]}"

    agent.run("Refund 1042", execution_id=exec_id)
    result = asyncio.run(agent.aresume(exec_id, resume_value=Resume(approved=True)))

    if not tracing["on"]:
        assert result.trace_id is None
        return
    spans = _spans(tracing["db"], result.trace_id)
    tool_spans = [s for s in spans if s["name"] == "tool.refund"]
    assert len(tool_spans) == 1, [s["name"] for s in spans]
    assert tool_spans[0]["attrs"]["tool.status"] == "ok"


def test_the_tool_timeout_applies_to_the_resumed_call(tracing) -> None:
    model, ran = _Model("refund", {"order": "1042"}), []

    async def slow_refund(order: str) -> str:
        """Refund an order; a manager must approve."""
        # Async on purpose: ``Tool.timeout`` is ``asyncio.wait_for``, which can't
        # interrupt a sync function blocking the loop — on any path, not only
        # resume. That is a separate, pre-existing limit.
        if interrupt(reason="manager_approval", context={"order": order}).approved:
            await asyncio.sleep(0.5)
            ran.append({"order": order})
        return "refunded"

    tool = FunctionTool(name="refund", fn=slow_refund, timeout=0.05)
    agent = _refund_agent(model, ran, tracing["ckpt"], tool=tool)
    exec_id = f"to-{uuid.uuid4().hex[:6]}"

    agent.run("Refund 1042", execution_id=exec_id)
    result = asyncio.run(agent.aresume(exec_id, resume_value=Resume(approved=True)))

    assert result.status == "completed"
    assert "timed out" in model.seen[-1], "the resumed call ignored Tool.timeout"


# ---------------------------------------------------------------------------
# 4. Governance: crash recovery re-asks; a resolved pause does not
# ---------------------------------------------------------------------------


class _Crash(BaseException):
    """Dies past every ``except Exception``, the way a process death would."""


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


def _crashing_banker(ran: list, ckpt: str, model: _Model) -> Agent:
    from tests._governance_plane import AGENT_ID

    state = {"calls": 0}

    def transfer_funds(amount: int, to: str) -> str:
        """Move money."""
        state["calls"] += 1
        if state["calls"] == 1:
            raise _Crash()
        ran.append({"amount": amount, "to": to})
        return f"Transferred ${amount} to {to}."

    return Agent(
        name="banker",
        agent_id=AGENT_ID,
        llm=model.llm(),
        tools=[FunctionTool(name="transfer_funds", fn=transfer_funds)],
        checkpointer=SQLiteCheckpointer(ckpt),
    )


def test_crash_recovery_asks_governance_again(gov_plane, tracing) -> None:
    """The policy allowed the call, the process died inside the tool, and the
    policy now denies it. Recovery must ask — 1.89.0 re-ran the transfer."""
    model, ran = _Model("transfer_funds", {"amount": 500, "to": "Bob"}), []
    gov_plane.gated_decision = "allow"
    agent = _crashing_banker(ran, tracing["ckpt"], model)
    exec_id = f"cr-{uuid.uuid4().hex[:6]}"
    with pytest.raises(_Crash):
        agent.run("Pay Bob $500.", execution_id=exec_id)

    gov_plane.gated_decision = "deny"
    asks_before = len(gov_plane.decide_calls)
    result = asyncio.run(agent.aresume(exec_id))

    assert len(gov_plane.decide_calls) == asks_before + 1, "recovery never asked governance"
    assert ran == [], "a call the policy now denies ran on crash recovery"
    assert result.status == "completed"
    assert "Refused by governance policy" in model.seen[-1]


def test_crash_recovery_under_an_approval_policy_pauses(gov_plane, tracing) -> None:
    model, ran = _Model("transfer_funds", {"amount": 500, "to": "Bob"}), []
    gov_plane.gated_decision = "allow"
    agent = _crashing_banker(ran, tracing["ckpt"], model)
    exec_id = f"ca-{uuid.uuid4().hex[:6]}"
    with pytest.raises(_Crash):
        agent.run("Pay Bob $500.", execution_id=exec_id)

    gov_plane.gated_decision = "require_approval"
    paused = asyncio.run(agent.aresume(exec_id))
    assert paused.status == "paused", paused
    assert paused.pending_interrupt["reason"] == governance.APPROVAL_REASON
    assert ran == []

    done = asyncio.run(agent.aresume(exec_id, resume_value=Resume(approved=True)))
    assert done.status == "completed"
    assert ran == [{"amount": 500, "to": "Bob"}]


@pytest.mark.parametrize("approved", [True, False], ids=["approved", "rejected"])
def test_a_resolved_approval_is_not_asked_again(gov_plane, approved, tracing) -> None:
    """The pause *was* the decision: an approved call runs without a second ask
    (which would pause it again forever), and a rejected one never runs."""
    from tests._governance_plane import AGENT_ID

    model, ran = _Model("transfer_funds", {"amount": 500, "to": "Bob"}), []

    def transfer_funds(amount: int, to: str) -> str:
        """Move money."""
        ran.append({"amount": amount, "to": to})
        return f"Transferred ${amount} to {to}."

    agent = Agent(
        name="banker",
        agent_id=AGENT_ID,
        llm=model.llm(),
        tools=[FunctionTool(name="transfer_funds", fn=transfer_funds)],
        checkpointer=SQLiteCheckpointer(tracing["ckpt"]),
    )
    exec_id = f"ra-{uuid.uuid4().hex[:6]}"
    first = agent.run("Pay Bob $500.", execution_id=exec_id)
    assert first.pending_interrupt["reason"] == governance.APPROVAL_REASON
    asks = len(gov_plane.decide_calls)

    result = asyncio.run(agent.aresume(exec_id, resume_value=Resume(approved=approved)))

    assert result.status == "completed", result
    assert len(gov_plane.decide_calls) == asks, "a resolved approval was asked again"
    assert ran == ([{"amount": 500, "to": "Bob"}] if approved else [])
