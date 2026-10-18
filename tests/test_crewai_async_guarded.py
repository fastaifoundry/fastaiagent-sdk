"""CrewAI's native async and batch entry points are guarded and traced (1.90.0, F4/F16).

``with_guardrails(crew)`` returned a proxy that guarded ``kickoff`` and
``kickoff_async`` and passed every other attribute to the raw crew. CrewAI 1.x
has four more ways to run a crew — ``akickoff`` (native async),
``kickoff_for_each``, ``kickoff_for_each_async`` and ``akickoff_for_each`` — and
all four ran with no guardrails at all. ``fastaiagent.integrations.crewai.enable()``
likewise never traced ``Crew.akickoff``, ``Agent.aexecute_task`` or
``Task.aexecute_sync``.

And a crew built with ``stream=True`` returns a streaming object, not text: the
output guardrails judged that object's ``repr`` and reported a clean verdict on
text they never saw (CLAUDE.md §2.4). That now refuses before the crew runs.

A real CrewAI 1.x crew; its model is a fixed ``BaseLLM`` subclass — CrewAI's own
extension point for custom models, in the role ``FunctionModel`` plays for ours.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

crewai = pytest.importorskip("crewai")

from crewai import Agent as CrewAgent  # noqa: E402
from crewai import Crew, Task  # noqa: E402
from crewai.llms.base_llm import BaseLLM  # noqa: E402

from fastaiagent.guardrail import (  # noqa: E402
    Guardrail,
    GuardrailPosition,
    GuardrailResult,
    GuardrailType,
)
from fastaiagent.integrations import crewai as fa_crewai  # noqa: E402
from fastaiagent.integrations._registry import GuardrailBlocked  # noqa: E402


@pytest.fixture(autouse=True)
def _quiet_crewai(monkeypatch):
    monkeypatch.setenv("CREWAI_DISABLE_TELEMETRY", "true")
    monkeypatch.setenv("CREWAI_TRACING_ENABLED", "false")
    # Not OTEL_SDK_DISABLED: that would switch off fastaiagent's own provider too.


class _Fixed(BaseLLM):
    """Answers every call the same way and counts the calls."""

    calls: int = 0
    answer: str = "the answer is 42"

    def call(
        self,
        messages,
        tools=None,
        callbacks=None,
        available_functions=None,
        from_task=None,
        from_agent=None,
        response_model=None,
    ):  # noqa: D102
        self.calls += 1
        return f"Thought: done\nFinal Answer: {self.answer}"

    async def acall(self, *args: Any, **kwargs: Any) -> Any:  # noqa: D102
        return self.call(*args, **kwargs)


def _crew(answer: str = "the answer is 42", **crew_kw: Any) -> tuple[Crew, _Fixed]:
    llm = _Fixed(model="fixed", answer=answer)
    agent = CrewAgent(role="analyst", goal="answer", backstory="terse", llm=llm)
    task = Task(description="Answer about {topic}", expected_output="one line", agent=agent)
    return Crew(agents=[agent], tasks=[task], name="desk", **crew_kw), llm


def _rule(name: str, position: GuardrailPosition, banned: str) -> Guardrail:
    return Guardrail(
        name=name,
        guardrail_type=GuardrailType.code,
        position=position,
        fn=lambda text: GuardrailResult(passed=banned not in text, message=f"{banned!r} found"),
    )


NO_SECRET_IN = _rule("no-secret-in", GuardrailPosition.input, "SECRET")
NO_42_OUT = _rule("no-42-out", GuardrailPosition.output, "42")


# ---------------------------------------------------------------------------
# F4: every entry point is guarded
# ---------------------------------------------------------------------------


def test_akickoff_runs_input_guardrails_before_the_crew() -> None:
    crew, llm = _crew()
    guarded = fa_crewai.with_guardrails(crew, input_guardrails=[NO_SECRET_IN])
    with pytest.raises(GuardrailBlocked, match="no-secret-in"):
        asyncio.run(guarded.akickoff(inputs={"topic": "SECRET plans"}))
    assert llm.calls == 0, "the crew ran although its input was blocked"


def test_akickoff_runs_output_guardrails() -> None:
    crew, _ = _crew()
    guarded = fa_crewai.with_guardrails(crew, output_guardrails=[NO_42_OUT])
    with pytest.raises(GuardrailBlocked, match="no-42-out"):
        asyncio.run(guarded.akickoff(inputs={"topic": "life"}))


def test_akickoff_passes_a_clean_run_through() -> None:
    crew, _ = _crew(answer="all clear")
    guarded = fa_crewai.with_guardrails(
        crew, input_guardrails=[NO_SECRET_IN], output_guardrails=[NO_42_OUT]
    )
    assert asyncio.run(guarded.akickoff(inputs={"topic": "life"})).raw == "all clear"


def _for_each(guarded: Any, how: str, inputs: list[dict[str, Any]]) -> Any:
    if how == "kickoff_for_each":
        return guarded.kickoff_for_each(inputs=inputs)
    return asyncio.run(getattr(guarded, how)(inputs=inputs))


FOR_EACH = ["kickoff_for_each", "kickoff_for_each_async", "akickoff_for_each"]


@pytest.mark.parametrize("how", FOR_EACH)
def test_for_each_checks_every_input_before_any_run(how: str) -> None:
    crew, llm = _crew()
    guarded = fa_crewai.with_guardrails(crew, input_guardrails=[NO_SECRET_IN])
    with pytest.raises(GuardrailBlocked, match="no-secret-in"):
        _for_each(guarded, how, [{"topic": "fine"}, {"topic": "SECRET"}])
    assert llm.calls == 0, "a batch with a blocked input started running"


@pytest.mark.parametrize("how", FOR_EACH)
def test_for_each_checks_every_output(how: str) -> None:
    crew, _ = _crew()
    guarded = fa_crewai.with_guardrails(crew, output_guardrails=[NO_42_OUT])
    with pytest.raises(GuardrailBlocked, match="no-42-out"):
        _for_each(guarded, how, [{"topic": "a"}, {"topic": "b"}])


@pytest.mark.parametrize("how", FOR_EACH)
def test_for_each_returns_every_result_when_clean(how: str) -> None:
    crew, _ = _crew(answer="all clear")
    guarded = fa_crewai.with_guardrails(crew, output_guardrails=[NO_42_OUT])
    results = _for_each(guarded, how, [{"topic": "a"}, {"topic": "b"}])
    assert [r.raw for r in results] == ["all clear", "all clear"]


@pytest.mark.parametrize("how", ["kickoff", "akickoff"])
def test_a_streaming_crew_cannot_be_output_guarded(how: str) -> None:
    """``stream=True`` returns a streaming object, not text; judging its repr was
    a clean verdict on nothing. Refused before the crew runs."""
    crew, llm = _crew(stream=True)
    guarded = fa_crewai.with_guardrails(crew, output_guardrails=[NO_42_OUT])
    with pytest.raises(GuardrailBlocked, match="could not run"):
        if how == "kickoff":
            guarded.kickoff(inputs={"topic": "life"})
        else:
            asyncio.run(guarded.akickoff(inputs={"topic": "life"}))
    assert llm.calls == 0


def test_a_streaming_crew_with_input_guardrails_only_still_streams() -> None:
    crew, _ = _crew(stream=True)
    guarded = fa_crewai.with_guardrails(crew, input_guardrails=[NO_SECRET_IN])
    out = guarded.kickoff(inputs={"topic": "life"})
    assert type(out).__name__ == "CrewStreamingOutput"


# ---------------------------------------------------------------------------
# F16: the native async methods are traced
# ---------------------------------------------------------------------------


@pytest.fixture
def traced_db(monkeypatch, tmp_path) -> Iterator[Path]:
    from fastaiagent._internal import instance as _instance
    from fastaiagent._internal import project as _project
    from fastaiagent._internal.config import reset_config
    from fastaiagent.trace import otel

    db = tmp_path / "local.db"
    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(db))
    reset_config()
    _project.set_project_id("test-proj")
    _instance.reset_for_testing()
    otel.reset()
    fa_crewai.enable()
    try:
        yield db
    finally:
        fa_crewai.disable()
        otel.reset()
        _instance.reset_for_testing()
        _project.reset_for_testing()
        reset_config()


def _span_names(db: Path) -> tuple[list[str], list[tuple[str, Any, dict[str, Any]]]]:
    from fastaiagent.trace.otel import get_tracer_provider

    provider = get_tracer_provider()
    if provider is not None and hasattr(provider, "force_flush"):
        provider.force_flush()
    conn = sqlite3.connect(str(db))
    try:
        rows = conn.execute("SELECT name, parent_span_id, attributes FROM spans").fetchall()
    except sqlite3.OperationalError:
        rows = []  # nothing was ever traced, so the table was never created
    finally:
        conn.close()
    return [r[0] for r in rows], [(r[0], r[1], json.loads(r[2] or "{}")) for r in rows]


def test_akickoff_is_one_traced_crew_run(traced_db: Path) -> None:
    crew, _ = _crew(answer="all clear")
    asyncio.run(crew.akickoff(inputs={"topic": "life"}))

    names, rows = _span_names(traced_db)
    crew_spans = [r for r in rows if r[0] == "crewai.crew.desk"]
    assert len(crew_spans) == 1, names
    assert crew_spans[0][1] is None, "the crew span should be the root"
    assert crew_spans[0][2].get("fastaiagent.framework") == "crewai"
    assert "crewai.agent.analyst" in names, f"Agent.aexecute_task untraced: {names}"
    assert any(n.startswith("crewai.task.") for n in names), f"Task.aexecute_sync untraced: {names}"


def test_disable_restores_the_async_methods(traced_db: Path) -> None:
    def _async_methods() -> tuple[Any, ...]:
        return (Crew.akickoff, CrewAgent.aexecute_task, Task.aexecute_sync)

    assert all(getattr(f, "_fastaiagent_patched", False) for f in _async_methods())
    fa_crewai.disable()
    assert not any(getattr(f, "_fastaiagent_patched", False) for f in _async_methods())
    fa_crewai.enable()  # the fixture disables again on teardown
