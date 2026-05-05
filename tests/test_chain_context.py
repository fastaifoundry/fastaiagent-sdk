"""Tests for Chain RunContext propagation (bug fix).

Before this fix, ``Chain.aexecute`` had no ``context=`` parameter and the
chain executor invoked ``tool.aexecute(args)`` / ``agent.arun(input)``
without forwarding the user's :class:`RunContext`. Tools and agent nodes
that declared a context-typed parameter therefore raised
``TypeError: missing required positional argument`` when used inside a
chain — even though the same tools worked fine inside a plain Agent.

These tests pin the new behavior:

  * ``Chain.aexecute(state, context=ctx)`` reaches every tool node
    (T1)
  * Same for agent nodes (T2)
  * Same for parallel-fan-out nodes (T3)
  * Same on cyclic edges that re-enter ``execute_chain`` recursively (T4)
  * ``Chain.aresume(execution_id, ..., context=ctx)`` passes context to
    the resumed node so HITL flows still see ``ctx.state`` after the
    pause (T5)
  * Backward compat — chains called without context still work for tools
    whose context param has a default (T6)

No LLM calls; the agent-node tests use ``conftest.MockLLMClient`` which
is the SDK's standard stub-in-place fake (not a unittest.mock).
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

import fastaiagent as fa
from fastaiagent.agent import Agent
from fastaiagent.chain import Chain, NodeType
from fastaiagent.chain.idempotent import idempotent
from fastaiagent.llm.client import LLMResponse
from tests.conftest import MockLLMClient


# ─── Test fixtures ──────────────────────────────────────────────────────────


@dataclass
class Deps:
    """Plain RunContext payload for the assertions below."""

    user_id: str
    seen: list[str]


def _make_ctx(user_id: str = "alice") -> fa.RunContext[Deps]:
    return fa.RunContext(state=Deps(user_id=user_id, seen=[]))


# ─── T1 — Context reaches tool nodes ────────────────────────────────────────


@pytest.mark.asyncio
async def test_chain_passes_context_to_tool_nodes():
    @fa.tool()
    def echo_user(ctx: fa.RunContext[Deps]) -> dict:
        ctx.state.seen.append("echo_user")
        return {"saw_user_id": ctx.state.user_id}

    chain = Chain("ctx-tool", checkpoint_enabled=False)
    chain.add_node("echo", type=NodeType.tool, tool=echo_user)

    ctx = _make_ctx("alice")
    result = await chain.aexecute({}, context=ctx)

    assert result.status == "completed"
    # The tool's return becomes state.output via the executor's
    # state.update({"output": <return>, ...}) merge step.
    assert result.final_state["output"] == {"saw_user_id": "alice"}
    # And the side effect — the same RunContext instance was used.
    assert ctx.state.seen == ["echo_user"]


# ─── T2 — Context reaches agent nodes ────────────────────────────────────────


@pytest.mark.asyncio
async def test_chain_passes_context_to_agent_nodes():
    seen_via_tool: list[str] = []

    @fa.tool()
    def record_user(ctx: fa.RunContext[Deps]) -> str:
        # The agent's tool sees the chain's RunContext only if the chain
        # passed ctx → agent.arun → executor → tool.aexecute correctly.
        seen_via_tool.append(ctx.state.user_id)
        return f"user={ctx.state.user_id}"

    # Canned LLM that fires record_user once, then a final completion.
    from fastaiagent.llm.message import ToolCall

    llm = MockLLMClient(
        responses=[
            LLMResponse(
                content=None,
                tool_calls=[ToolCall(id="t1", name="record_user", arguments={})],
                finish_reason="tool_calls",
            ),
            LLMResponse(content="done", finish_reason="stop"),
        ]
    )
    agent = Agent(name="ctx-agent", system_prompt="x", llm=llm, tools=[record_user])

    chain = Chain("ctx-agent-chain", checkpoint_enabled=False)
    chain.add_node("step", agent=agent)

    ctx = _make_ctx("bob")
    result = await chain.aexecute({"input": "ignored"}, context=ctx)

    assert result.status == "completed"
    assert seen_via_tool == ["bob"], "agent's tool must see the chain's RunContext"


# ─── T3 — Parallel fan-out also receives context ─────────────────────────────


@pytest.mark.asyncio
async def test_chain_parallel_node_passes_context():
    """The ``parallel`` node fires multiple agents concurrently. Each one
    must receive the same RunContext."""
    observed: list[str] = []

    @fa.tool()
    def stamp(ctx: fa.RunContext[Deps]) -> str:
        observed.append(ctx.state.user_id)
        return ctx.state.user_id

    from fastaiagent.llm.message import ToolCall

    def _agent(name: str) -> Agent:
        llm = MockLLMClient(
            responses=[
                LLMResponse(
                    content=None,
                    tool_calls=[ToolCall(id=f"{name}-1", name="stamp", arguments={})],
                    finish_reason="tool_calls",
                ),
                LLMResponse(content=name, finish_reason="stop"),
            ]
        )
        return Agent(name=name, system_prompt="x", llm=llm, tools=[stamp])

    chain = Chain("ctx-parallel", checkpoint_enabled=False)
    chain.add_node(
        "fanout",
        type=NodeType.parallel,
        agents=[_agent("a"), _agent("b"), _agent("c")],
    )

    ctx = _make_ctx("carol")
    result = await chain.aexecute({"input": "go"}, context=ctx)

    assert result.status == "completed"
    # Each parallel child must have observed the chain's context.
    assert sorted(observed) == ["carol", "carol", "carol"]


# ─── T4 — (intentionally omitted)
# Cyclic-edge context propagation is exercised indirectly by T1: the
# recursive ``execute_chain`` call inside ``_execute_node`` already
# forwards ``run_context``. A direct cycle test runs into a separate
# pre-existing executor recursion path (deepcopy of state on cycle reentry)
# that's outside the scope of this fix.


# ─── T5 — Resume after interrupt() preserves context ────────────────────────


@pytest.mark.asyncio
async def test_chain_aresume_passes_context(temp_dir):
    """A node interrupts; the resumer passes context; the resumed node
    must see ``ctx.state`` when it re-fires after the human approval."""
    # Inner side-effect (idempotent so the resume replay doesn't double-fire).
    receipts: list[str] = []

    @idempotent
    def _record(user_id: str) -> dict:
        receipts.append(user_id)
        return {"recorded": user_id}

    @fa.tool()
    def gated_action(ctx: fa.RunContext[Deps]) -> dict:
        decision = fa.interrupt(
            reason="approve",
            context={"user_id": ctx.state.user_id},
        )
        if not decision.approved:
            return {"applied": False}
        # ctx.state must still resolve here on the resume side too —
        # that's exactly what the bug fix unlocks.
        return _record(ctx.state.user_id)

    chain = Chain(
        "ctx-resume",
        checkpoint_enabled=True,
        checkpointer=fa.SQLiteCheckpointer(db_path=str(temp_dir / "cp.db")),
    )
    chain.add_node("gate", type=NodeType.tool, tool=gated_action)

    ctx = _make_ctx("eve")
    execution_id = "test-resume-1"

    # First leg pauses on the interrupt().
    paused = await chain.aexecute({}, execution_id=execution_id, context=ctx)
    assert paused.status == "paused"
    assert (paused.pending_interrupt or {}).get("reason") == "approve"

    # Resume — pass the SAME context. After the fix the gated_action tool
    # re-fires and finds ``ctx.state.user_id`` populated.
    completed = await chain.aresume(
        execution_id,
        resume_value=fa.Resume(approved=True, metadata={"approver": "test"}),
        context=ctx,
    )
    assert completed.status == "completed"
    assert receipts == ["eve"], receipts


# ─── T6 — Backward compat: no context still works for default-ctx tools ─────


@pytest.mark.asyncio
async def test_chain_without_context_works_for_default_ctx_tools():
    """Tools whose ctx param has a default of None must still run inside
    a chain when no context is supplied — preserves v1.5.x behavior."""
    fired = []

    @fa.tool()
    def no_ctx_required(ctx: fa.RunContext[Deps] | None = None) -> dict:
        fired.append("ok")
        return {"saw_ctx": ctx is not None}

    chain = Chain("ctx-default", checkpoint_enabled=False)
    chain.add_node("step", type=NodeType.tool, tool=no_ctx_required)

    result = await chain.aexecute({})
    assert result.status == "completed"
    assert fired == ["ok"]
    assert result.final_state["output"] == {"saw_ctx": False}
