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
