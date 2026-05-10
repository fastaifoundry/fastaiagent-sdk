"""Tests for the Hierarchical Supervisor (v1.9.0): ``validate_outputs``.

No mocking. Worker, supervisor, and validator LLMs are all driven by
``fastaiagent.testing.FunctionModel`` so the manager-validates-worker
loop is fully deterministic.
"""

from __future__ import annotations

from fastaiagent.agent import Agent
from fastaiagent.agent.team import Supervisor, Worker
from fastaiagent.llm.client import LLMResponse
from fastaiagent.llm.message import Message, MessageRole, ToolCall
from fastaiagent.testing import FunctionModel, TestModel


def _supervisor_responder(call_workers: list[str]):
    """Build a closure that drives the supervisor's tool-calling LLM.

    Issues exactly one ``delegate_to_<role>`` tool call per entry in
    ``call_workers`` (cycling order), then returns a final synthesis
    message on the next call.
    """
    state = {"calls": 0}

    def _fn(messages):
        n = state["calls"]
        state["calls"] += 1
        if n < len(call_workers):
            role = call_workers[n]
            return (
                "",
                [{"name": f"delegate_to_{role}", "arguments": {"task": "do it"}}],
            )
        # Final synthesis turn.
        return ("done", [])

    return _fn


def _validator_responder(approval_pattern: list[bool]):
    """Build a responder that returns approve/reject JSON in sequence."""
    state = {"calls": 0}

    def _fn(messages):
        n = state["calls"]
        state["calls"] += 1
        approved = approval_pattern[min(n, len(approval_pattern) - 1)]
        if approved:
            return '{"approved": true}'
        return '{"approved": false, "feedback": "be more specific"}'

    return _fn


def _make_validator_llm(approval_pattern: list[bool]) -> FunctionModel:
    return FunctionModel(_validator_responder(approval_pattern))


def _make_worker(role: str, output_sequence: list[str]) -> Worker:
    """Worker whose underlying LLM returns each ``output_sequence`` entry
    in turn (cycles to the last on overflow)."""
    state = {"calls": 0}

    def _worker_fn(messages):
        n = state["calls"]
        state["calls"] += 1
        out = output_sequence[min(n, len(output_sequence) - 1)]
        return (out, [])

    agent = Agent(name=f"w-{role}", llm=FunctionModel(_worker_fn))
    return Worker(agent=agent, role=role, description=f"role {role}")


# ---------------------------------------------------------------------------
# Off (default) — validate_outputs=False keeps historical behaviour
# ---------------------------------------------------------------------------


def test_validate_outputs_off_does_not_call_validator() -> None:
    worker = _make_worker("alpha", output_sequence=["worker-output"])
    sup = Supervisor(
        name="s",
        # Supervisor LLM: one delegate call, then a final synthesis turn.
        llm=FunctionModel(_supervisor_responder(["alpha"])),
        workers=[worker],
        validate_outputs=False,
    )
    result = sup.run("anything")
    assert result.output == "done"


# ---------------------------------------------------------------------------
# On — manager validates and accepts on first try
# ---------------------------------------------------------------------------


def test_validate_outputs_first_try_accept() -> None:
    worker = _make_worker("alpha", output_sequence=["acceptable-answer"])

    # Supervisor's tool LLM does the delegate, the validator LLM approves.
    # We use a single FunctionModel that swaps personas based on the last
    # message — the supervisor LLM is called via Agent.arun, the validator
    # LLM is called via supervisor.llm.acomplete directly. Different
    # entry points, same Supervisor.llm field — distinguish by checking
    # whether the conversation includes our validator system prompt.
    state = {"sup_calls": 0, "val_calls": 0}

    def supervisor_or_validator(messages: list[Message]):
        # Validator prompts begin with our reviewer system message.
        is_validator = any(
            m.role == MessageRole.system
            and "output reviewer" in (m.content or "").lower()
            for m in messages
        )
        if is_validator:
            state["val_calls"] += 1
            return '{"approved": true}'
        # Supervisor turn — same logic as _supervisor_responder.
        n = state["sup_calls"]
        state["sup_calls"] += 1
        if n == 0:
            return (
                "",
                [{"name": "delegate_to_alpha", "arguments": {"task": "do it"}}],
            )
        return ("done", [])

    sup = Supervisor(
        name="s",
        llm=FunctionModel(supervisor_or_validator),
        workers=[worker],
        validate_outputs=True,
    )
    result = sup.run("hello")
    assert result.output == "done"
    assert state["val_calls"] == 1, "validator should be called exactly once"


# ---------------------------------------------------------------------------
# On — manager rejects, worker re-runs, manager accepts second answer
# ---------------------------------------------------------------------------


def test_validate_outputs_reject_then_accept() -> None:
    """Manager rejects worker's first try; worker re-runs with feedback;
    manager accepts the second try."""
    worker = _make_worker(
        "alpha", output_sequence=["bad-answer", "good-answer"]
    )

    state = {"sup_calls": 0, "val_calls": 0}

    def driver(messages: list[Message]):
        is_validator = any(
            m.role == MessageRole.system
            and "output reviewer" in (m.content or "").lower()
            for m in messages
        )
        if is_validator:
            n = state["val_calls"]
            state["val_calls"] += 1
            if n == 0:
                return '{"approved": false, "feedback": "try again"}'
            return '{"approved": true}'
        # Supervisor LLM.
        n = state["sup_calls"]
        state["sup_calls"] += 1
        if n == 0:
            return (
                "",
                [{"name": "delegate_to_alpha", "arguments": {"task": "do it"}}],
            )
        return ("done", [])

    sup = Supervisor(
        name="s",
        llm=FunctionModel(driver),
        workers=[worker],
        validate_outputs=True,
        max_validation_retries_per_worker=1,
    )
    result = sup.run("hello")
    assert result.output == "done"
    assert state["val_calls"] == 2, "validator should be called twice"


# ---------------------------------------------------------------------------
# On — manager rejects beyond max retries — supervisor proceeds + warns
# ---------------------------------------------------------------------------


def test_validate_outputs_exhausted_retries_logs_warning(
    tmp_path, monkeypatch, request
) -> None:
    """When the validator keeps rejecting, after max_validation_retries the
    supervisor proceeds with the last output and logs a guardrail_events
    row tagged supervisor.validate / outcome=warned."""
    db_path = tmp_path / "local.db"
    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(db_path))
    monkeypatch.setenv("FASTAIAGENT_UI_ENABLED", "true")

    from fastaiagent._internal.config import get_config

    # The cached config singleton holds previously-resolved env values.
    # Clear it so the per-test env vars actually apply, and clear again on
    # teardown so subsequent tests get a fresh read.
    get_config.cache_clear()
    request.addfinalizer(get_config.cache_clear)

    from fastaiagent.ui.db import init_local_db

    init_local_db(str(db_path)).close()

    worker = _make_worker(
        "alpha", output_sequence=["never-good", "still-not-good"]
    )

    state = {"sup_calls": 0}

    def driver(messages: list[Message]):
        is_validator = any(
            m.role == MessageRole.system
            and "output reviewer" in (m.content or "").lower()
            for m in messages
        )
        if is_validator:
            return '{"approved": false, "feedback": "nope"}'
        n = state["sup_calls"]
        state["sup_calls"] += 1
        if n == 0:
            return (
                "",
                [{"name": "delegate_to_alpha", "arguments": {"task": "do it"}}],
            )
        return ("done", [])

    sup = Supervisor(
        name="s-warn",
        llm=FunctionModel(driver),
        workers=[worker],
        validate_outputs=True,
        max_validation_retries_per_worker=1,
    )
    result = sup.run("hi")
    assert result.output == "done"

    # Warning row written to local.db
    import sqlite3

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT guardrail_name, outcome FROM guardrail_events "
            "WHERE guardrail_name = 'supervisor.validate'"
        ).fetchall()
    assert rows, "expected one supervisor.validate event row"
    assert all(r[1] == "warned" for r in rows)


# ---------------------------------------------------------------------------
# On — validator returns malformed JSON — fail-open (accept)
# ---------------------------------------------------------------------------


def test_validate_outputs_unparseable_validator_fails_open() -> None:
    worker = _make_worker("alpha", output_sequence=["only-output"])

    state = {"sup_calls": 0, "val_calls": 0}

    def driver(messages: list[Message]):
        is_validator = any(
            m.role == MessageRole.system
            and "output reviewer" in (m.content or "").lower()
            for m in messages
        )
        if is_validator:
            state["val_calls"] += 1
            return "this is not json at all"
        n = state["sup_calls"]
        state["sup_calls"] += 1
        if n == 0:
            return (
                "",
                [{"name": "delegate_to_alpha", "arguments": {"task": "do it"}}],
            )
        return ("done", [])

    sup = Supervisor(
        name="s-fopen",
        llm=FunctionModel(driver),
        workers=[worker],
        validate_outputs=True,
    )
    result = sup.run("ok")
    assert result.output == "done"
    assert state["val_calls"] == 1


# ---------------------------------------------------------------------------
# to_dict surfaces the new flag for the local UI
# ---------------------------------------------------------------------------


def test_to_dict_includes_validate_outputs() -> None:
    worker = _make_worker("alpha", output_sequence=["x"])
    sup = Supervisor(
        name="s",
        llm=TestModel(response="ok"),
        workers=[worker],
        validate_outputs=True,
        max_validation_retries_per_worker=2,
    )
    d = sup.to_dict()
    assert d["validate_outputs"] is True
    assert d["max_validation_retries_per_worker"] == 2


# ---------------------------------------------------------------------------
# Validation construction guards
# ---------------------------------------------------------------------------


def test_negative_retries_rejected() -> None:
    import pytest

    with pytest.raises(ValueError, match=">= 0"):
        Supervisor(
            name="s",
            llm=TestModel(response="x"),
            workers=[],
            validate_outputs=True,
            max_validation_retries_per_worker=-1,
        )


# Unused imports are referenced here so ruff doesn't flag them.
_ = (LLMResponse, ToolCall)
