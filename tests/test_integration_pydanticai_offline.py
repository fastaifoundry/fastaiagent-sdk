"""The PydanticAI integration, offline, on pydantic-ai's own test models (1.89.0).

Two defects, neither visible to the live harness tests:

* **An agent with no system prompt failed every call.** The root-span helper
  fell back to ``agent.system_prompt``, which in current pydantic-ai is a
  decorator *method*, and iterated it: ``TypeError`` from ``run``,
  ``run_sync`` and ``run_stream``. Tracing broke the user's call. The live
  tests all set a system prompt.
* **A failed ``run_stream`` never marked or ended its root span** (P5): only
  ``record_exception``, and nothing at all when the failure happened while
  the stream was opening, because ``__aexit__`` never runs for a failed
  ``__aenter__``. The span stayed attached as current, so the next run nested
  under it, and it was never exported.

pydantic-ai's ``TestModel`` and ``FunctionModel`` are its supported offline
models, not mocks of it.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from typing import Any

import pytest

pytest.importorskip("pydantic_ai")

from pydantic_ai import Agent  # noqa: E402
from pydantic_ai.models.function import AgentInfo, FunctionModel  # noqa: E402
from pydantic_ai.models.test import TestModel  # noqa: E402


@pytest.fixture
def traced_db(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    """A fresh traced DB per test. Works across tests only because the
    integration looks its tracer up per run (1.89.0); it used to bind the
    provider that was live when ``enable()`` first ran."""
    from fastaiagent._internal import project as _project
    from fastaiagent._internal.config import reset_config
    from fastaiagent.integrations import pydanticai as pa
    from fastaiagent.trace import otel

    db_path = tmp_path / "local.db"
    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(db_path))
    for name in ("FASTAIAGENT_API_KEY", "FASTAIAGENT_TARGET"):
        monkeypatch.delenv(name, raising=False)
    reset_config()
    _project.set_project_id("test-proj")
    otel.reset()
    pa.enable()
    try:
        yield db_path
    finally:
        otel.reset()
        _project.reset_for_testing()
        reset_config()


def _roots(db: Any) -> list[dict[str, Any]]:
    with sqlite3.connect(db) as conn:
        rows = conn.execute(
            "SELECT name, status, events FROM spans WHERE name LIKE 'pydanticai.agent.%'"
        ).fetchall()
    return [{"name": n, "status": s, "events": json.loads(e or "[]")} for n, s, e in rows]


@pytest.mark.parametrize("how", ["run_sync", "run", "run_stream"])
def test_an_agent_without_a_system_prompt_runs(traced_db: Any, how: str) -> None:
    agent = Agent(TestModel(custom_output_text="ok"))

    async def stream() -> str:
        async with agent.run_stream("hi") as result:
            return "".join([chunk async for chunk in result.stream_text(delta=True)])

    if how == "run_sync":
        out = agent.run_sync("hi").output
    elif how == "run":
        out = asyncio.run(agent.run("hi")).output
    else:
        out = asyncio.run(stream())
    assert out == "ok"
    assert _roots(traced_db), "the run was not traced"


@pytest.mark.parametrize("how", ["run_sync", "run", "run_stream"])
def test_the_ui_counts_a_runs_tokens_once(traced_db: Any, how: str) -> None:
    """1.93.0: the root span carried the run's tokens and cost as per-call
    values beside PydanticAI's own ``chat`` spans, and ``run_sync`` opened a
    second root through ``run`` — the UI showed 2x to 3x the real spend."""
    from fastaiagent.trace import otel
    from fastaiagent.ui.trace_cost import rollup_spans

    # The fixture builds a new tracer provider for every test, as
    # ``fastaiagent ui --db`` does in production. PydanticAI's own spans must
    # follow it: they went to OTel's first global provider (1.93.0).
    agent = Agent(TestModel(custom_output_text="ok"))

    async def stream() -> Any:
        async with agent.run_stream("hi") as result:
            async for _ in result.stream_text(delta=True):
                pass
            return result

    if how == "run_sync":
        result = agent.run_sync("hi")
    elif how == "run":
        result = asyncio.run(agent.run("hi"))
    else:
        result = asyncio.run(stream())
    usage = result.usage() if callable(result.usage) else result.usage
    real = usage.input_tokens + usage.output_tokens
    assert real > 0

    otel.get_tracer_provider().force_flush()
    with sqlite3.connect(traced_db) as conn:
        conn.row_factory = sqlite3.Row
        spans = [dict(r) for r in conn.execute("SELECT * FROM spans")]
    assert len(_roots(traced_db)) == 1, "one root span per run"
    assert any(s["name"].startswith("chat ") for s in spans), "PydanticAI's own spans"
    assert rollup_spans(spans).tokens == real


def test_a_system_prompt_is_still_previewed(traced_db: Any) -> None:
    agent = Agent(TestModel(custom_output_text="ok"), system_prompt="Be brief.")
    agent.run_sync("hi")
    with sqlite3.connect(traced_db) as conn:
        (attrs,) = conn.execute(
            "SELECT attributes FROM spans WHERE name LIKE 'pydanticai.agent.%'"
        ).fetchone()
    assert json.loads(attrs).get("pydanticai.agent.system_prompt") == "Be brief."


def test_a_failed_run_stream_ends_error_and_detaches(traced_db: Any) -> None:
    from opentelemetry import trace as otel_trace

    async def refuse(_messages: list[Any], _info: AgentInfo) -> Any:
        raise RuntimeError("provider refused the stream")
        yield ""  # pragma: no cover — makes this an async generator

    agent = Agent(FunctionModel(stream_function=refuse))

    async def go() -> None:
        async with agent.run_stream("hi") as result:
            async for _ in result.stream_text():
                pass

    with pytest.raises(RuntimeError):
        asyncio.run(go())

    assert not otel_trace.get_current_span().get_span_context().is_valid, (
        "the failed root span is still attached as current"
    )
    roots = _roots(traced_db)
    assert roots, "the failed root span was never ended, so never stored"
    assert roots[-1]["status"] == "ERROR"
    assert any(e.get("name") == "exception" for e in roots[-1]["events"])
