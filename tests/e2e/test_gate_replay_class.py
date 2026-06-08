"""End-to-end quality gate — replay_class on a REAL LLM-driven tool span.

No mocks: a real OpenAI model decides to call a tool we marked
``replay_class="read_only"``. We then read the persisted tool-call span back
out of the local trace DB and assert the exact wire-contract attributes the
central Replay engine consumes:

* ``fastaiagent.tool.replay_class == "read_only"`` — the developer's explicit mark;
* ``fastaiagent.runner.type == "tool"`` — the span is classified as a tool span.

This is the live counterpart to the unit test in
``tests/test_tool_replay_class.py`` (which drives the executor directly): here
the whole agent loop runs against a real provider, proving the attribute
survives the real tool-calling path end-to-end into the stored span.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.e2e.conftest import require_env

pytestmark = pytest.mark.e2e

# The span types the Enterprise classifies as a tool span.
_TOOL_SPAN_TYPES = {"tool", "tool_call", "worker_call"}


def test_replay_class_on_real_llm_tool_span(isolated_local_db: Any) -> None:
    # This gate reads the LOCAL trace DB (not the platform), so only the core
    # LLM key is required — bypass the platform env check.
    import os

    os.environ.setdefault("E2E_SKIP_PLATFORM", "1")
    require_env()

    from fastaiagent import Agent, LLMClient
    from fastaiagent.tool import tool
    from fastaiagent.trace import otel
    from fastaiagent.trace.storage import TraceStore

    # Rebuild the tracer provider so its LocalStorageProcessor binds to the
    # temp DB that ``isolated_local_db`` configured.
    otel.reset()

    @tool(name="get_weather", replay_class="read_only")
    def get_weather(city: str) -> str:
        """Return the current weather for a city."""
        return f"The weather in {city} is sunny and 22 degrees Celsius."

    agent = Agent(
        name="replay-class-gate",
        system_prompt=(
            "You have a get_weather tool. When the user asks about the weather "
            "in a city, call get_weather with that city and report the result."
        ),
        llm=LLMClient(provider="openai", model="gpt-4.1"),
        tools=[get_weather],
    )

    result = agent.run("What's the weather in Paris right now? Use your tool.")

    # The model actually drove a tool call (no mock).
    assert result.tool_calls, "agent did not invoke the get_weather tool"
    assert result.tool_calls[0]["tool_name"] == "get_weather"

    # Read the persisted tool-call span and assert the wire-contract attrs.
    store = TraceStore(db_path=str(isolated_local_db))
    try:
        tool_attrs = None
        for summary in store.list_traces():
            for span in store.get_trace(summary.trace_id).spans:
                if span.name == "tool.get_weather":
                    tool_attrs = span.attributes
        assert tool_attrs is not None, "no tool.get_weather span was persisted"
        assert tool_attrs["fastaiagent.tool.replay_class"] == "read_only"
        assert tool_attrs["fastaiagent.runner.type"] == "tool"
        assert tool_attrs["fastaiagent.runner.type"] in _TOOL_SPAN_TYPES
    finally:
        store.close()
        otel.reset()
