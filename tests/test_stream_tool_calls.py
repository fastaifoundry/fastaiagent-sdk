"""A streamed run lists its tool calls, as ``run`` does (1.93.0).

``Agent.stream()`` returned ``tool_calls=[]`` on every provider: the streaming
tool loop kept no per-call records. Found by the real-provider check of 1.92.0.
"""

from __future__ import annotations

import asyncio

import pytest

from fastaiagent import Agent, AgentConfig, FunctionTool
from fastaiagent.testing import TestModel


def lookup(city: str) -> str:
    """Code for a city."""
    return f"code-{city}"


def _agent(parallel: bool, middleware: list | None = None) -> Agent:
    return Agent(
        middleware=middleware,
        name="streamed",
        llm=TestModel(
            response=["", "done"],
            tool_calls=[
                {"name": "lookup", "arguments": {"city": "paris"}},
                {"name": "lookup", "arguments": {"city": "rome"}},
            ],
        ),
        tools=[FunctionTool(name="lookup", fn=lookup)],
        config=AgentConfig(parallel_tools=parallel),
    )


def _shape(calls: list[dict]) -> list[tuple]:
    return [(c["tool_name"], c["arguments"], c.get("output")) for c in calls]


@pytest.mark.parametrize("parallel", [False, True])
def test_stream_returns_the_same_tool_calls_as_run(parallel: bool) -> None:
    ran = _agent(parallel).run("go")
    streamed = _agent(parallel).stream("go")
    expected = [
        ("lookup", {"city": "paris"}, "code-paris"),
        ("lookup", {"city": "rome"}, "code-rome"),
    ]
    assert _shape(ran.tool_calls) == expected
    assert _shape(streamed.tool_calls) == expected


def test_after_run_middleware_sees_the_streamed_tool_calls() -> None:
    from fastaiagent.agent.middleware import AgentMiddleware

    seen: list[int] = []

    class Count(AgentMiddleware):
        async def after_run(self, ctx, result):  # type: ignore[no-untyped-def]
            seen.append(len(result.tool_calls))
            return result

    agent = _agent(False, middleware=[Count()])

    async def drain() -> None:
        async for _ in agent.astream("go"):
            pass

    asyncio.run(drain())
    assert seen == [2]


def test_a_list_response_keeps_its_tool_calls() -> None:
    """``TestModel(response=[...], tool_calls=[...])`` dropped the tool calls."""
    result = _agent(False).run("go")
    assert [c["tool_name"] for c in result.tool_calls] == ["lookup", "lookup"]
    assert result.output == "done"
