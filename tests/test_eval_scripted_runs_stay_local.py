"""An eval run answered by a scripted model never reaches the plane (1.104.0).

In a skill test, GitHub Copilot had no model key, scripted the answers with a
``FunctionModel``, scored 100% by construction and recorded a passing verdict on
the plane, where it read as evidence about the agent. ``TestModel`` /
``FunctionModel`` now mark the eval run they answer (or judge), and that run is
kept local: written ``synced=1``, never queued, with a warning at the gate.

Real HTTP stand-in for the plane (``tests/_governance_plane.py``), real local.db,
real pytest sessions through ``pytester``. No mocks.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

import fastaiagent as fa
from fastaiagent.eval import LLMJudge, evaluate, gate, record_gate_result
from fastaiagent.testing import FunctionModel, TestModel
from tests._governance_plane import GovPlane, reset_connection, serve

pytest_plugins = ["pytester"]

CASES = [{"input": "say ping", "expected_output": "ping"}]
INGEST = "POST /public/v1/eval/runs/ingest"


@pytest.fixture
def plane(isolated_local_db: Path) -> Iterator[tuple[GovPlane, Path]]:
    """Connected, with eval export on (the default once connected)."""
    with serve() as (state, url):
        fa.connect(api_key="fa_k_scripted", target=url)
        try:
            from fastaiagent.eval.platform_export import eval_export_enabled

            assert eval_export_enabled()
            yield state, isolated_local_db
        finally:
            from fastaiagent.eval.platform_export import get_eval_exporter

            get_eval_exporter().shutdown()
            reset_connection()


def _row(db: Path, run_id: str) -> tuple[int, dict[str, Any]]:
    with sqlite3.connect(db) as conn:
        synced, metadata = conn.execute(
            "SELECT synced, metadata FROM eval_runs WHERE run_id = ?", (run_id,)
        ).fetchone()
    return synced, json.loads(metadata or "{}")


def _gate(results: Any, db: Path) -> None:
    report = gate(results, fail_under=["contains.pass_rate=0.5"])
    record_gate_result(
        results.run_id, gate_outcome=report.outcome, thresholds={"contains.pass_rate": 0.5}
    )


def test_a_testmodel_run_is_kept_local_and_says_so(
    plane: Any, caplog: pytest.LogCaptureFixture
) -> None:
    state, db = plane
    agent = fa.Agent(name="scripted-agent", llm=TestModel("ping"))
    results = evaluate(agent.run, CASES, scorers=["contains"])

    synced, meta = _row(db, results.run_id)
    assert meta["scripted_models"] is True
    assert synced == 1  # handled: never queued
    with caplog.at_level(logging.WARNING, logger="fastaiagent.eval.results"):
        _gate(results, db)
    assert "answered by a scripted model" in caplog.text
    time.sleep(0.5)
    assert INGEST not in state.paths


def test_a_functionmodel_run_is_kept_local(plane: Any) -> None:
    _, db = plane
    agent = fa.Agent(name="scripted-fn", llm=FunctionModel(lambda messages: "ping"))
    results = evaluate(agent.run, CASES, scorers=["contains"])
    assert _row(db, results.run_id) == (1, _row(db, results.run_id)[1])
    assert _row(db, results.run_id)[1]["scripted_models"] is True


def test_a_scripted_judge_counts_too(plane: Any) -> None:
    """The answers are real (a plain function here), the verdicts scripted."""
    _, db = plane
    judge = LLMJudge(criteria="correctness", llm=TestModel('{"score": 1, "reasoning": "ok"}'))
    results = evaluate(lambda text: "ping", CASES, scorers=[judge])
    synced, meta = _row(db, results.run_id)
    assert meta.get("scripted_models") is True
    assert synced == 1


def test_a_run_with_no_scripted_model_is_still_queued_and_offered(plane: Any) -> None:
    state, db = plane
    results = evaluate(lambda text: "ping", CASES, scorers=["contains"])

    synced, meta = _row(db, results.run_id)
    assert "scripted_models" not in meta
    assert synced == 0  # queued for the plane
    _gate(results, db)
    deadline = time.monotonic() + 10
    while INGEST not in state.paths and time.monotonic() < deadline:
        time.sleep(0.05)
    assert INGEST in state.paths


def test_disconnected_a_scripted_run_stays_quiet(
    isolated_local_db: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Export off: nothing would have gone anyway, so no warning to read."""
    results = evaluate(fa.Agent(name="offline", llm=TestModel("ping")).run, CASES, ["contains"])
    with caplog.at_level(logging.WARNING, logger="fastaiagent.eval.results"):
        _gate(results, isolated_local_db)
    assert "scripted model" not in caplog.text


# ── the pytest plugin ────────────────────────────────────────────────────────

_SUITE = """
import os
os.environ["FASTAIAGENT_LOCAL_DB"] = r"{db}"
from fastaiagent._internal.config import reset_config
reset_config()

from fastaiagent.agent import Agent
from fastaiagent.eval import case
from fastaiagent.testing import TestModel

@case(input="hello", expected="hi")
def test_a(evaluate_one):
    evaluate_one({agent}, scorers=["exact_match"])
"""


@pytest.mark.parametrize(
    ("agent", "scripted"),
    [
        ('Agent(name="g", llm=TestModel(response="hi")).run', True),
        ('(lambda text: "hi")', False),
    ],
)
def test_the_pytest_plugin_marks_a_scripted_session(
    pytester: pytest.Pytester, tmp_path: Path, agent: str, scripted: bool
) -> None:
    db = tmp_path / "plugin.db"
    pytester.makepyfile(_SUITE.format(db=db, agent=agent))
    assert pytester.runpytest("--no-header", "-p", "no:randomly").ret == 0
    os.environ.pop("FASTAIAGENT_LOCAL_DB", None)
    with sqlite3.connect(db) as conn:
        (metadata,) = conn.execute(
            "SELECT metadata FROM eval_runs ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
    assert bool(json.loads(metadata or "{}").get("scripted_models")) is scripted


# ── a blocked run says which trace it was ───────────────────────────────────


@pytest.fixture
def spans(isolated_local_db: Path) -> Iterator[Any]:
    """A fresh tracer provider with an in-memory exporter (OTel's own)."""
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from fastaiagent.trace import otel

    otel.reset()
    exporter = InMemorySpanExporter()
    otel.get_tracer_provider().add_span_processor(SimpleSpanProcessor(exporter))
    try:
        yield exporter
    finally:
        otel.reset()


def _trace_ids(exporter: Any) -> set[str]:
    return {format(s.context.trace_id, "032x") for s in exporter.get_finished_spans()}


def _pii_agent(name: str) -> Any:
    from fastaiagent.guardrail import no_pii

    return fa.Agent(name=name, llm=TestModel("SSN 123-45-6789"), guardrails=[no_pii()])


def test_a_blocked_run_carries_its_trace_id(spans: Any) -> None:
    with pytest.raises(fa.GuardrailBlockedError) as err:
        _pii_agent("blocked").run("hi")
    assert err.value.trace_id in _trace_ids(spans)


def test_a_blocked_streamed_run_carries_its_trace_id(spans: Any) -> None:
    import asyncio

    agent = _pii_agent("blocked-stream")

    async def drain() -> None:
        async for _ in agent.astream("hi"):
            pass

    with pytest.raises(fa.GuardrailBlockedError) as err:
        asyncio.run(drain())
    assert err.value.trace_id in _trace_ids(spans)


def test_with_tracing_off_a_blocked_run_has_no_trace_id(tmp_path: Path) -> None:
    """Tracing is decided once per process, so this runs in its own."""
    import subprocess
    import sys

    code = (
        "import fastaiagent as fa\n"
        "from fastaiagent.guardrail import no_pii\n"
        "from fastaiagent.testing import TestModel\n"
        "agent = fa.Agent(name='u', llm=TestModel('SSN 123-45-6789'), guardrails=[no_pii()])\n"
        "try:\n"
        "    agent.run('hi')\n"
        "except fa.GuardrailBlockedError as err:\n"
        "    print('trace_id=' + repr(err.trace_id))\n"
    )
    env = {
        **os.environ,
        "FASTAIAGENT_TRACE_ENABLED": "0",
        "FASTAIAGENT_LOCAL_DB": str(tmp_path / "off.db"),
    }
    out = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
    ).stdout
    assert "trace_id=None" in out
