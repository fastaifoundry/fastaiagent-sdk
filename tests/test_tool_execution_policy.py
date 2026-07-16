"""Tool execution policy — input coercion, timeout, retry, output validation.

Runs without API keys. Exercises the real ``Tool.ainvoke`` wrapper and
``FunctionTool`` argument coercion (no mocks).
"""

from __future__ import annotations

import asyncio
import enum

import pytest
from pydantic import BaseModel

from fastaiagent._internal.errors import ToolExecutionError
from fastaiagent.tool.base import ToolResult
from fastaiagent.tool.function import FunctionTool, tool


class Priority(str, enum.Enum):
    low = "low"
    high = "high"


class Ticket(BaseModel):
    title: str
    priority: Priority


# ─── Input validation / coercion ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_primitive_argument_is_coerced() -> None:
    seen: dict[str, object] = {}

    def f(n: int) -> str:
        seen["type"] = type(n).__name__
        return str(n * 2)

    tool_obj = FunctionTool(name="double", fn=f)
    result = await tool_obj.ainvoke({"n": "21"})  # string from the wire
    assert seen["type"] == "int"
    assert result.output == "42"


@pytest.mark.asyncio
async def test_pydantic_model_argument_is_instantiated() -> None:
    seen: dict[str, object] = {}

    def file_ticket(ticket: Ticket) -> str:
        seen["type"] = type(ticket).__name__
        return ticket.title

    tool_obj = FunctionTool(name="file_ticket", fn=file_ticket)
    result = await tool_obj.ainvoke(
        {"ticket": {"title": "boom", "priority": "high"}}
    )
    assert seen["type"] == "Ticket"  # coerced from dict, not passed as dict
    assert result.output == "boom"


@pytest.mark.asyncio
async def test_invalid_arguments_return_error_not_exception() -> None:
    def file_ticket(ticket: Ticket) -> str:
        return ticket.title

    tool_obj = FunctionTool(name="file_ticket", fn=file_ticket)
    result = await tool_obj.ainvoke({"ticket": {"title": "x", "priority": "nope"}})
    assert not result.success
    assert "Invalid arguments" in (result.error or "")


@pytest.mark.asyncio
async def test_missing_required_argument_is_reported() -> None:
    def greet(name: str) -> str:
        return f"hi {name}"

    tool_obj = FunctionTool(name="greet", fn=greet)
    result = await tool_obj.ainvoke({})  # missing required 'name'
    assert not result.success
    assert "Invalid arguments" in (result.error or "")


@pytest.mark.asyncio
async def test_validate_args_false_passes_through_raw() -> None:
    seen: dict[str, object] = {}

    def f(n: int) -> str:
        seen["type"] = type(n).__name__
        return str(n)

    tool_obj = FunctionTool(name="raw", fn=f, validate_args=False)
    await tool_obj.ainvoke({"n": "21"})  # not coerced
    assert seen["type"] == "str"


@pytest.mark.asyncio
async def test_untyped_param_is_passed_through() -> None:
    seen: dict[str, object] = {}

    def f(x) -> str:  # no annotation
        seen["type"] = type(x).__name__
        return "ok"

    tool_obj = FunctionTool(name="untyped", fn=f)
    await tool_obj.ainvoke({"x": {"a": 1}})
    assert seen["type"] == "dict"


# ─── Timeout ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_timeout_raises_tool_execution_error() -> None:
    async def slow(x: int) -> int:
        await asyncio.sleep(0.5)
        return x

    tool_obj = FunctionTool(name="slow", fn=slow, timeout=0.1)
    with pytest.raises(ToolExecutionError, match="timed out"):
        await tool_obj.ainvoke({"x": 1})


@pytest.mark.asyncio
async def test_no_timeout_by_default() -> None:
    async def slow(x: int) -> int:
        await asyncio.sleep(0.05)
        return x

    tool_obj = FunctionTool(name="slow", fn=slow)
    result = await tool_obj.ainvoke({"x": 7})
    assert result.output == 7


# ─── Retry ──────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_retry_succeeds_after_transient_failures() -> None:
    calls = {"n": 0}

    def flaky(x: int) -> int:
        calls["n"] += 1
        if calls["n"] < 3:
            raise ValueError("transient")
        return x * 10

    tool_obj = FunctionTool(name="flaky", fn=flaky, max_retries=3, retry_delay=0.0)
    result = await tool_obj.ainvoke({"x": 2})
    assert result.output == 20
    assert calls["n"] == 3


@pytest.mark.asyncio
async def test_retry_exhausted_reraises() -> None:
    calls = {"n": 0}

    def always_fails(x: int) -> int:
        calls["n"] += 1
        raise ValueError("permanent")

    tool_obj = FunctionTool(name="broken", fn=always_fails, max_retries=2, retry_delay=0.0)
    with pytest.raises(ToolExecutionError):
        await tool_obj.ainvoke({"x": 1})
    assert calls["n"] == 3  # initial + 2 retries


@pytest.mark.asyncio
async def test_no_retry_by_default() -> None:
    calls = {"n": 0}

    def always_fails(x: int) -> int:
        calls["n"] += 1
        raise ValueError("boom")

    tool_obj = FunctionTool(name="broken", fn=always_fails)
    with pytest.raises(ToolExecutionError):
        await tool_obj.ainvoke({"x": 1})
    assert calls["n"] == 1


# ─── Output validation ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_output_type_coerces_return_value() -> None:
    def gives_str() -> str:
        return "5"

    tool_obj = FunctionTool(name="out", fn=gives_str, output_type=int)
    result = await tool_obj.ainvoke({})
    assert result.output == 5
    assert isinstance(result.output, int)


@pytest.mark.asyncio
async def test_output_type_mismatch_returns_error() -> None:
    def gives_bad() -> str:
        return "not a number"

    tool_obj = FunctionTool(name="outbad", fn=gives_bad, output_type=int)
    result = await tool_obj.ainvoke({})
    assert not result.success
    assert "output failed schema validation" in (result.error or "")


@pytest.mark.asyncio
async def test_output_type_coerces_to_pydantic_model() -> None:
    def gives_dict() -> dict:
        return {"title": "hello", "priority": "low"}

    tool_obj = FunctionTool(name="mk", fn=gives_dict, output_type=Ticket)
    result = await tool_obj.ainvoke({})
    assert isinstance(result.output, Ticket)
    assert result.output.priority is Priority.low


# ─── Defaults are a no-op (backward compatibility) ──────────────────────────


@pytest.mark.asyncio
async def test_ainvoke_matches_aexecute_when_unconfigured() -> None:
    def f(city: str) -> str:
        return f"weather in {city}"

    tool_obj = FunctionTool(name="w", fn=f)
    via_invoke = await tool_obj.ainvoke({"city": "Paris"})
    via_execute = await tool_obj.aexecute({"city": "Paris"})
    assert via_invoke.output == via_execute.output == "weather in Paris"


# ─── @tool decorator forwards the policy ────────────────────────────────────


@pytest.mark.asyncio
async def test_decorator_forwards_policy() -> None:
    @tool(name="dec", max_retries=1, output_type=int)
    def d(x: int) -> str:
        return str(x)

    assert d.max_retries == 1
    assert d.output_type is int
    result = await d.ainvoke({"x": "9"})
    assert result.output == 9


def test_invalid_policy_values_raise() -> None:
    def f(x: int) -> int:
        return x

    with pytest.raises(ValueError, match="timeout"):
        FunctionTool(name="a", fn=f, timeout=0)
    with pytest.raises(ValueError, match="max_retries"):
        FunctionTool(name="b", fn=f, max_retries=-1)


@pytest.mark.asyncio
async def test_error_result_is_returned_not_raised() -> None:
    # A Tool that signals failure via ToolResult.error (not an exception) must
    # have that result surfaced unchanged when retries are exhausted.
    from fastaiagent.tool.base import Tool

    class ErrTool(Tool):
        async def aexecute(self, arguments, context=None) -> ToolResult:
            return ToolResult(error="nope")

    result = await ErrTool(name="err", max_retries=1, retry_delay=0.0).ainvoke({})
    assert not result.success
    assert result.error == "nope"
