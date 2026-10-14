"""The learning loop mines only the traces you point it at (1.81.0).

``run_extraction`` (the ``fastaiagent learn`` CLI) read the newest 100 traces
of every agent and every date, filed all of their facts under the one
``scope_id`` it was given, re-mined its own extraction calls on the next run,
and counted re-found facts as "written".

No mocking: real agent runs write real traces to a temp ``local.db``; the
extractor is the SDK's offline ``FunctionModel`` so its calls can be counted.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fastaiagent import Agent
from fastaiagent._internal.config import reset_config
from fastaiagent.learn import MemoryStore, run_extraction
from fastaiagent.llm.message import UserMessage
from fastaiagent.testing import FunctionModel, TestModel
from fastaiagent.trace.otel import reset as reset_tracer
from fastaiagent.trace.storage import TraceStore

_HEADER = "You are extracting durable facts from a completed agent trace."


@pytest.fixture
def db(tmp_path: Path, monkeypatch):
    p = tmp_path / "local.db"
    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(p))
    reset_config()
    reset_tracer()
    Agent(name="alpha", llm=TestModel(response="alpha answer")).run("alpha question one")
    Agent(name="alpha", llm=TestModel(response="alpha answer")).run("alpha question two")
    Agent(name="beta", llm=TestModel(response="beta answer")).run("beta question")
    TestModel(response="stray").complete([UserMessage("a stray model call")])
    reset_tracer()  # flush spans to local.db
    yield p
    reset_tracer()
    reset_config()


def _extractor() -> FunctionModel:
    def responder(messages):
        prompt = messages[-1].content or ""
        trace_text = prompt.split("Trace:", 1)[1]
        if "alpha question" in trace_text:
            return json.dumps(["Alpha users ask numbered questions"])
        if "beta question" in trace_text:
            return json.dumps(["Beta users ask about beta"])
        return json.dumps(["Something from a trace that is not an agent run"])

    return FunctionModel(responder)


def _run(db: Path, extractor: FunctionModel, **kwargs):
    kwargs.setdefault("scope", "agent")
    kwargs.setdefault("scope_id", "alpha")
    return run_extraction(
        llm=extractor,
        store=MemoryStore(db_path=str(db)),
        trace_store=TraceStore(db_path=str(db)),
        **kwargs,
    )


def _facts(db: Path, scope_id: str = "alpha") -> set[str]:
    rows = MemoryStore(db_path=str(db)).list_active(scope="agent", scope_id=scope_id)
    return {f.fact for f in rows}


def test_agent_name_mines_only_that_agents_traces(db):
    extractor = _extractor()
    results = _run(db, extractor, agent_name="alpha")

    assert len(extractor.calls) == 2  # the two alpha runs
    assert len(results) == 2
    assert _facts(db) == {"Alpha users ask numbered questions"}


def test_a_second_run_skips_traces_already_processed(db):
    _run(db, _extractor(), agent_name="alpha")
    again = _extractor()
    assert _run(db, again, agent_name="alpha") == []
    assert again.calls == []


def test_reprocess_reports_nothing_new(db):
    _run(db, _extractor(), agent_name="alpha")
    results = _run(db, _extractor(), agent_name="alpha", reprocess=True)
    assert len(results) == 2
    assert all(r.written_ids and r.new_ids == [] for r in results)


def test_new_ids_count_only_rows_inserted_now(db):
    results = _run(db, _extractor(), agent_name="alpha")
    # Both alpha traces yield the same fact: inserted once, matched once.
    assert sorted(len(r.new_ids) for r in results) == [0, 1]


def test_the_extractors_own_calls_are_never_mined(db):
    _run(db, _extractor(), agent_name="alpha")  # leaves learn.extract traces behind
    reset_tracer()
    assert TraceStore(db_path=str(db)).list_traces(name_filter="learn.extract")

    extractor = _extractor()
    _run(db, extractor, scope_id="all-agents")  # no agent filter
    assert len(extractor.calls) == 4  # alpha x2, beta, the stray call
    for call in extractor.calls:
        assert (call["messages"][-1].content or "").count(_HEADER) == 1


def test_max_traces_caps_a_run(db):
    extractor = _extractor()
    _run(db, extractor, scope_id="all-agents", max_traces=1)
    assert len(extractor.calls) == 1


def test_dry_run_leaves_the_traces_unprocessed(db):
    _run(db, _extractor(), agent_name="alpha", dry_run=True)
    extractor = _extractor()
    _run(db, extractor, agent_name="alpha")
    assert len(extractor.calls) == 2


# --- the CLI's guard on user/project scope -----------------------------------


def _cli(args: list[str]):
    from typer.testing import CliRunner

    from fastaiagent.cli.main import app

    return CliRunner().invoke(app, ["learn", *args])


def test_cli_user_scope_needs_an_agent_filter(db):
    out = _cli(["--scope", "user", "--scope-id", "u42", "--allow-personal"])
    assert out.exit_code == 2
    assert "needs --agent" in out.output


def test_cli_user_scope_needs_an_explicit_attribution(db):
    out = _cli(["--scope", "user", "--scope-id", "u42", "--allow-personal", "--agent", "alpha"])
    assert out.exit_code == 2
    assert "needs --attribute-all" in out.output


def test_cli_user_scope_needs_a_scope_id(db):
    out = _cli(["--scope", "user", "--allow-personal", "--agent", "alpha", "--attribute-all"])
    assert out.exit_code == 2
    assert "needs --scope-id" in out.output


def _replay_an_alpha_run(db: Path) -> str:
    """A real replay rerun of one alpha trace; returns the rerun's trace id."""
    from fastaiagent.trace.replay import Replay

    store = TraceStore(db_path=str(db))
    original = store.list_traces(last_hours=None, agent_name="alpha", limit=1)[0]
    rerun = Replay.load(original.trace_id, store=store).fork_at(0)
    rerun = rerun.modify_input("alpha question REPLAYED").with_determinism("recorded").rerun()
    reset_tracer()
    return str(rerun.trace_id)


def test_replays_are_not_mined_by_default(db):
    """1.89.0: a replay's output is counterfactual. Mining it stored facts that
    never happened (the audit found 10 of them from one session)."""
    _replay_an_alpha_run(db)
    extractor = _extractor()
    _run(db, extractor, agent_name="alpha")
    mined = [c["messages"][-1].content or "" for c in extractor.calls]
    assert len(mined) == 2, "only the two real alpha runs should be mined"
    assert not any("REPLAYED" in m for m in mined)


def test_include_experiments_mines_replays_too(db):
    _replay_an_alpha_run(db)
    extractor = _extractor()
    _run(db, extractor, agent_name="alpha", include_experiments=True)
    assert any("REPLAYED" in (c["messages"][-1].content or "") for c in extractor.calls)
