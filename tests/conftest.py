"""Shared test fixtures for the FastAIAgent SDK test suite."""

from __future__ import annotations

from pathlib import Path

import pytest

from fastaiagent.llm.client import LLMClient, LLMResponse
from fastaiagent.llm.message import ToolCall


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    """Provide a temporary directory for test data."""
    return tmp_path


class MockLLMClient(LLMClient):
    """A mock LLM client that returns predefined responses."""

    def __init__(self, responses: list[LLMResponse] | None = None):
        super().__init__(provider="mock", model="mock-model")
        self._responses = responses or [
            LLMResponse(content="Hello! How can I help?", finish_reason="stop")
        ]
        self._call_count = 0
        self._calls: list[dict] = []

    async def acomplete(self, messages, tools=None, **kwargs):
        self._calls.append({"messages": messages, "tools": tools, "kwargs": kwargs})
        if self._call_count < len(self._responses):
            response = self._responses[self._call_count]
        else:
            response = self._responses[-1]
        self._call_count += 1
        return response


@pytest.fixture
def mock_llm() -> MockLLMClient:
    """A mock LLM that returns a simple text response."""
    return MockLLMClient()


@pytest.fixture
def mock_llm_with_tools() -> MockLLMClient:
    """A mock LLM that makes one tool call then returns a final answer."""
    return MockLLMClient(
        responses=[
            LLMResponse(
                content=None,
                tool_calls=[ToolCall(id="call_1", name="search", arguments={"query": "test"})],
                finish_reason="tool_calls",
                usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            ),
            LLMResponse(
                content="Based on the search results, here is the answer.",
                finish_reason="stop",
                usage={"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
            ),
        ]
    )


@pytest.fixture
def recording_middleware():
    """Factory for a middleware that records every hook invocation.

    Returns a ``(middleware, records)`` tuple. ``records`` is a dict with
    keys ``before_model``, ``after_model``, ``wrap_tool`` each mapping to
    a list of capture dicts. Tests assert on these to verify ordering and
    hook semantics.
    """
    from fastaiagent.agent.middleware import AgentMiddleware

    def _factory(name: str = "rec"):
        records: dict = {"before_model": [], "after_model": [], "wrap_tool": []}

        class _Recording(AgentMiddleware):
            def __init__(self) -> None:
                self.name = name

            async def before_model(self, ctx, messages):
                records["before_model"].append(
                    {
                        "name": self.name,
                        "turn": ctx.turn,
                        "agent_name": ctx.agent_name,
                        "message_count": len(messages),
                    }
                )
                return messages

            async def after_model(self, ctx, response):
                records["after_model"].append(
                    {
                        "name": self.name,
                        "turn": ctx.turn,
                        "content": response.content,
                    }
                )
                return response

            async def wrap_tool(self, ctx, tool, args, call_next):
                records["wrap_tool"].append(
                    {
                        "name": self.name,
                        "phase": "enter",
                        "tool": tool.name,
                        "tool_call_index": ctx.tool_call_index,
                    }
                )
                result = await call_next(tool, args)
                records["wrap_tool"].append(
                    {
                        "name": self.name,
                        "phase": "exit",
                        "tool": tool.name,
                    }
                )
                return result

        return _Recording(), records

    return _factory


@pytest.fixture
def noop_middleware():
    """A canonical no-op middleware — byte-for-byte identity on every hook."""
    from fastaiagent.agent.middleware import AgentMiddleware

    class _NoOp(AgentMiddleware):
        name = "noop"

    return _NoOp()
