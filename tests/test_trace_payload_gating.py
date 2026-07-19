"""FASTAIAGENT_TRACE_PAYLOADS gating covers every free-text span attribute.

Regression: ``agent.input`` and ``agent.output`` were written unconditionally,
so setting the flag dropped the ``gen_ai.*`` payloads and the system prompt but
still persisted the user's input and the model's final answer.

Mock-free: drives a real ``Agent`` with the shipped ``TestModel`` stand-in (no
network) and reads the spans back out of the real local SQLite store.
"""

from __future__ import annotations

from typing import Any

import pytest

from fastaiagent import Agent, TraceStore
from fastaiagent.testing import TestModel

_PAYLOAD_ATTRS = {
    "agent.input",
    "agent.output",
    "agent.system_prompt",
    "gen_ai.request.messages",
    "gen_ai.response.content",
}
_STRUCTURAL_ATTRS = {"agent.name", "gen_ai.request.model"}


@pytest.fixture(autouse=True)
def _fresh_tracer_provider() -> Any:
    """Rebuild the tracer provider per test.

    The provider is a module-level singleton that binds its storage processor
    to whichever DB path was current when it was first built, so without this
    every test after the first would write into the first test's temp DB.
    """
    from fastaiagent.trace.otel import reset

    reset()
    yield
    reset()


def _attrs_for_run(agent: Agent, prompt: str) -> set[str]:
    result = agent.run(prompt)
    assert result.trace_id
    trace = TraceStore.default().get_trace(result.trace_id)
    keys: set[str] = set()
    for span in trace.spans:
        keys.update((span.attributes or {}).keys())
    return keys


def _agent() -> Agent:
    return Agent(
        name="gating-probe",
        system_prompt="You are a helpful assistant.",
        llm=TestModel(response="hello there"),
    )


@pytest.mark.usefixtures("isolated_local_db")
def test_payloads_recorded_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FASTAIAGENT_TRACE_PAYLOADS", raising=False)

    keys = _attrs_for_run(_agent(), "what is the capital of France?")

    assert "agent.input" in keys
    assert "agent.output" in keys
    assert _STRUCTURAL_ATTRS <= keys


@pytest.mark.usefixtures("isolated_local_db")
def test_gating_drops_agent_input_and_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FASTAIAGENT_TRACE_PAYLOADS", "0")

    keys = _attrs_for_run(_agent(), "my account number is 12345")

    assert not (_PAYLOAD_ATTRS & keys), f"payload attrs leaked: {_PAYLOAD_ATTRS & keys}"


@pytest.mark.usefixtures("isolated_local_db")
def test_gating_keeps_structural_attributes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Structure must survive so cost/latency/monitoring still work."""
    monkeypatch.setenv("FASTAIAGENT_TRACE_PAYLOADS", "0")

    keys = _attrs_for_run(_agent(), "hello")

    assert _STRUCTURAL_ATTRS <= keys
    assert "agent.tokens_used" in keys
    assert "agent.latency_ms" in keys


@pytest.mark.usefixtures("isolated_local_db")
def test_gated_run_still_produces_a_usable_trace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FASTAIAGENT_TRACE_PAYLOADS", "0")
    agent = _agent()

    result: Any = agent.run("hello")

    assert result.output == "hello there"
    trace = TraceStore.default().get_trace(result.trace_id or "")
    assert len(trace.spans) >= 1
