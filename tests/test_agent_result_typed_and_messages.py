"""Typed results, run info for tools, and the run's messages (1.91.0, 2H + 2J).

* ``AgentResult[T]`` — ``AgentResult`` is generic in ``parsed``; a run with
  ``output_type=Invoice`` returns ``AgentResult[Invoice]``. Plain ``AgentResult``
  is ``AgentResult[Any]`` and behaves exactly as before.
* ``RunContext.run`` — inside a run, a read-only ``RunInfo``: the execution id,
  the agent, the turn, the model and what the run has spent so far. ``None``
  outside one.
* ``AgentResult.messages`` — the run's conversation (prior ``messages=``, the
  input, the tool turns, the answer), without the system prompt, middleware
  instructions or memory context, ready to pass back as ``messages=``.

Real ``Agent`` + ``FunctionModel``.
"""

from __future__ import annotations

import asyncio
from typing import Any

from pydantic import BaseModel

from fastaiagent import Agent, AgentMemory, AgentResult, FunctionTool, RunContext, RunInfo
from fastaiagent.llm.client import LLMResponse
from fastaiagent.llm.message import MessageRole, ToolCall, UserMessage
from fastaiagent.testing import FunctionModel


class Invoice(BaseModel):
    number: str
    total: float


# ── AgentResult[T] ─────────────────────────────────────────────────────────


def test_a_typed_run_returns_its_type() -> None:
    agent = Agent(
        name="invoices",
        llm=FunctionModel(lambda m: '{"number": "A-17", "total": 12.5}'),
        output_type=Invoice,
    )
    result = agent.run("parse it")
    assert isinstance(result, AgentResult)
    assert result.parsed == Invoice(number="A-17", total=12.5)


def test_agent_result_is_generic_and_plain_is_any() -> None:
    typed = AgentResult[Invoice].model_validate({"parsed": {"number": "B-2", "total": 3}})
    assert typed.parsed == Invoice(number="B-2", total=3.0)
    assert isinstance(typed, AgentResult)
    plain = AgentResult(output="x", parsed={"anything": 1})
    assert plain.parsed == {"anything": 1}


# ── RunContext.run ─────────────────────────────────────────────────────────


def test_a_tool_sees_the_run() -> None:
    seen: list[Any] = []

    def look(ctx: RunContext[str]) -> str:
        """Look at the run."""
        seen.append(ctx.run)
        return "looked"

    turns = {"n": 0}

    def respond(messages):
        turns["n"] += 1
        if turns["n"] <= 2:
            return LLMResponse(
                content="",
                tool_calls=[ToolCall(id=f"c{turns['n']}", name="look", arguments={})],
                usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            )
        return "done"

    agent = Agent(
        name="watcher",
        llm=FunctionModel(respond, model="watch-model"),
        tools=[FunctionTool(name="look", fn=look)],
    )
    result = agent.run("go", context=RunContext("state"), execution_id="ri-1")
    first, second = seen
    assert isinstance(first, RunInfo)
    assert (first.execution_id, first.agent_name, first.model) == ("ri-1", "watcher", "watch-model")
    assert (first.turn, second.turn) == (0, 1)
    assert (first.usage.tokens, second.usage.tokens) == (15, 30)
    assert result.tokens_used >= 30


def test_run_is_none_outside_a_run() -> None:
    assert RunContext("x").run is None


def test_run_info_is_read_only() -> None:
    import dataclasses

    info = RunInfo(execution_id="e", agent_name="a", turn=0, model="m")
    try:
        info.turn = 5  # type: ignore[misc]
    except dataclasses.FrozenInstanceError:
        return
    raise AssertionError("RunInfo is writable")


# ── AgentResult.messages ───────────────────────────────────────────────────


def _tool_then_answer() -> FunctionModel:
    turns = {"n": 0}

    def respond(messages):
        turns["n"] += 1
        if turns["n"] == 1:
            return "", [{"name": "lookup", "arguments": {"order": "7"}}]
        return "It shipped."

    return FunctionModel(respond)


def lookup(order: str) -> str:
    """Look up an order."""
    return f"order {order} shipped"


def _roles(messages: list[Any]) -> list[str]:
    return [m.role.value for m in messages]


def test_messages_hold_the_run_without_the_system_prompt() -> None:
    agent = Agent(
        name="support",
        system_prompt="You are support.",
        llm=_tool_then_answer(),
        tools=[FunctionTool(name="lookup", fn=lookup)],
    )
    result = agent.run("Where is order 7?")
    assert _roles(result.messages) == ["user", "assistant", "tool", "assistant"]
    assert result.messages[0].content == "Where is order 7?"
    assert result.messages[1].tool_calls[0].name == "lookup"
    assert result.messages[-1].content == "It shipped."
    assert all(m.role != MessageRole.system for m in result.messages)


def test_messages_continue_a_conversation() -> None:
    seen: list[Any] = []

    def respond(messages):
        seen.append(list(messages))
        return f"reply {len(seen)}"

    agent = Agent(name="chat", system_prompt="Be brief.", llm=FunctionModel(respond))
    first = agent.run("Hi")
    second = agent.run("And again?", messages=first.messages)
    assert _roles(second.messages) == ["user", "assistant", "user", "assistant"]
    # The second call saw the first turn once, after one system prompt.
    assert _roles(seen[1]) == ["system", "user", "assistant", "user"]


def test_memory_context_is_not_in_messages() -> None:
    memory = AgentMemory()
    memory.add(UserMessage("remembered question"))
    agent = Agent(name="mem", llm=FunctionModel(lambda m: "ok"), memory=memory)
    result = agent.run("new question")
    assert [m.content for m in result.messages] == ["new question", "ok"]


def test_streamed_run_has_messages_too() -> None:
    agent = Agent(
        name="streamed",
        llm=_tool_then_answer(),
        tools=[FunctionTool(name="lookup", fn=lookup)],
    )
    result = agent.stream("Where is order 7?")
    assert _roles(result.messages) == ["user", "assistant", "tool", "assistant"]

    async def drain() -> None:
        async for _ in agent.astream("again"):
            pass

    asyncio.run(drain())  # astream itself is unchanged
