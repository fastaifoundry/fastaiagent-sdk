"""What ``FASTAIAGENT_TRACE_PAYLOADS=0`` does and does not cover.

These are characterization tests: they pin the *current, intentional* scope of
the flag so it can't drift silently.

The flag gates the ``gen_ai.*`` payload attributes and the resolved system
prompt. It deliberately does **not** gate ``agent.input`` / ``agent.output``:
those are what ``Replay`` reconstructs a run from and what the UI search
indexes, so dropping them would be a breaking change. Callers who need those
masked should install a ``RedactionPolicy`` — both keys are already in
``SENSITIVE_ATTR_KEYS``.

Mock-free: drives a real ``Agent`` with the shipped ``TestModel`` stand-in (no
network) and reads spans back out of the real local SQLite store.
"""

from __future__ import annotations

from typing import Any

import pytest

from fastaiagent import Agent, TraceStore
from fastaiagent.testing import TestModel

# Gated by the flag.
_GATED_ATTRS = {
    "agent.system_prompt",
    "gen_ai.request.messages",
    "gen_ai.response.content",
}
# Free-text, but recorded regardless of the flag (see module docstring).
_UNGATED_PAYLOAD_ATTRS = {"agent.input", "agent.output"}
# Structural — always present.
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

    assert _GATED_ATTRS <= keys
    assert _UNGATED_PAYLOAD_ATTRS <= keys
    assert _STRUCTURAL_ATTRS <= keys


@pytest.mark.usefixtures("isolated_local_db")
def test_gating_drops_gen_ai_payloads_and_system_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FASTAIAGENT_TRACE_PAYLOADS", "0")

    keys = _attrs_for_run(_agent(), "my account number is 12345")

    assert not (_GATED_ATTRS & keys), f"gated attrs leaked: {_GATED_ATTRS & keys}"


@pytest.mark.usefixtures("isolated_local_db")
def test_gating_does_not_cover_agent_input_and_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Known, documented scope gap — pinned so it can't change silently.

    Closing it would break ``Replay`` (which reads ``agent.input``) and UI
    search. Use a ``RedactionPolicy`` to mask these instead.
    """
    monkeypatch.setenv("FASTAIAGENT_TRACE_PAYLOADS", "0")

    keys = _attrs_for_run(_agent(), "my account number is 12345")

    assert _UNGATED_PAYLOAD_ATTRS <= keys


@pytest.mark.usefixtures("isolated_local_db")
def test_gating_keeps_structural_attributes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Structure must survive so cost/latency/monitoring still work."""
    monkeypatch.setenv("FASTAIAGENT_TRACE_PAYLOADS", "0")

    keys = _attrs_for_run(_agent(), "hello")

    assert _STRUCTURAL_ATTRS <= keys
    assert "agent.tokens_used" in keys
    assert "agent.latency_ms" in keys


@pytest.mark.usefixtures("isolated_local_db")
def test_redaction_policy_masks_the_ungated_attributes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The supported way to protect agent input/output: mask, don't drop."""
    from fastaiagent import RedactionPolicy, set_redaction_policy

    monkeypatch.delenv("FASTAIAGENT_TRACE_PAYLOADS", raising=False)
    set_redaction_policy(RedactionPolicy(patterns=(r"\b\d{5}\b",), mode="capture"))
    try:
        result = _agent().run("my account number is 12345")
        trace = TraceStore.default().get_trace(result.trace_id or "")
        joined = " ".join(
            str((s.attributes or {}).get("agent.input", "")) for s in trace.spans
        )
        assert "12345" not in joined
        assert "[REDACTED]" in joined
    finally:
        set_redaction_policy(None)
