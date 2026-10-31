"""A replay rerun can be given the run's ``RunContext`` (1.95.0, S13).

A trace does not record the context a run was given (its state is live objects),
and a rerun had no way to pass one: a tool that takes a ``RunContext`` failed on
every call with "missing 1 required positional argument: 'ctx'", the model read
the error and answered anyway, and the rerun reported success. Real agent, real
trace, a scripted OpenAI-compatible endpoint on localhost (``decisions_stub``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from fastaiagent import Agent, LLMClient, RunContext
from fastaiagent._internal.errors import ReplayError
from fastaiagent.tool import tool
from fastaiagent.trace import otel
from fastaiagent.trace.replay import Replay

TOOL_TURN = {
    "role": "assistant",
    "content": None,
    "tool_calls": [
        {
            "id": "call_plan",
            "type": "function",
            "function": {"name": "get_plan", "arguments": json.dumps({"customer": "acme"})},
        }
    ],
}
FINAL_TURN = {"role": "assistant", "content": "Acme is on a plan."}


@pytest.fixture
def recorded(
    isolated_local_db: Path, decisions_stub: Any, monkeypatch: pytest.MonkeyPatch
) -> Any:
    """``(trace_id, get_plan, seen, stub)`` for one real run of a context-taking tool."""
    monkeypatch.setenv("FASTAIAGENT_ALLOW_PRIVATE_NETWORKS", "1")
    seen: list[str] = []

    @tool(name="get_plan")
    def get_plan(ctx: RunContext[dict], customer: str) -> str:
        """Look up the customer's plan."""
        seen.append(ctx.state["plan"])
        return f"{customer} is on {ctx.state['plan']}"

    otel.reset()
    decisions_stub.chat_replies = [TOOL_TURN, FINAL_TURN]
    decisions_stub.chat_calls = 0
    agent = Agent(name="ctx-bot", llm=LLMClient(**decisions_stub.llm_kwargs()), tools=[get_plan])
    result = agent.run("Which plan is acme on?", context=RunContext(state={"plan": "pro"}))
    otel.get_tracer_provider().force_flush()
    assert seen == ["pro"]
    try:
        yield result.trace_id, get_plan, seen, decisions_stub
    finally:
        otel.reset()


def test_a_rerun_without_context_refuses_up_front(recorded: Any) -> None:
    trace_id, get_plan, seen, stub = recorded
    stub.chat_replies = [TOOL_TURN, FINAL_TURN]
    stub.chat_calls = 0
    fork = Replay.load(trace_id).fork_at(0).with_tool_override("get_plan", get_plan)
    with pytest.raises(ReplayError, match=r"get_plan.*with_context"):
        fork.rerun()
    assert stub.chat_calls == 0, "the refusal must come before the model is called"


def test_with_context_reaches_the_tool(recorded: Any) -> None:
    trace_id, get_plan, seen, stub = recorded
    stub.chat_replies = [TOOL_TURN, FINAL_TURN]
    stub.chat_calls = 0
    result = (
        Replay.load(trace_id)
        .fork_at(0)
        .with_tool_override("get_plan", get_plan)
        .with_context(RunContext(state={"plan": "team"}))
        .rerun()
    )
    assert seen == ["pro", "team"]
    assert result.tool_dispositions["get_plan"] == "override"


def test_a_stubbed_context_tool_needs_no_context(recorded: Any) -> None:
    trace_id, _, seen, stub = recorded
    stub.chat_replies = [TOOL_TURN, FINAL_TURN]
    stub.chat_calls = 0
    Replay.load(trace_id).fork_at(0).with_tool_stub("get_plan", returns="stubbed").rerun()
    assert seen == ["pro"], "a stub never runs the real tool"
