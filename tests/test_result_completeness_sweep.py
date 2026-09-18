"""Contract sweep: every result-producing path stamps a ``trace_id``.

``Agent._arun_traced`` has stamped it since the field existed; nothing else
did. A swarm run, a streamed run, a resumed swarm and a chain all opened (or
should have opened) a real root span and then handed back a result that could
not name it.

The silent consequence is in ``eval/evaluate.py``, which does
``getattr(output, "trace_id", None)`` onto every ``EvalCaseRecord``: agent
cases carried a trace id, swarm cases carried ``None``, and the eval reported
the same scores either way — so eval-to-trace linking, ``eval/curate.py``, the
UI's per-case open-trace affordance and ``ReplayResult.to_eval_case`` all had
nothing to point at, and ``Replay.compare()`` treats a falsy id as
``rerun_failed`` forever.

The sweep is a registry keyed by entry point, so adding a new result-producing
path without stamping it fails ``test_registry_covers_the_public_surface``.
Everything runs offline against ``conftest.MockLLMClient`` — a mock-driven run
still produces a real OTel root span and a real trace id, so nothing here
skips.
"""

from __future__ import annotations

import re
import uuid

import pytest

from fastaiagent.agent import Agent, Swarm
from fastaiagent.agent.team import Supervisor, Worker
from fastaiagent.chain import Chain
from fastaiagent.checkpointers import SQLiteCheckpointer
from fastaiagent.llm.client import LLMResponse

from .conftest import MockLLMClient

TRACE_ID_RE = re.compile(r"^[0-9a-f]{32}$")


def _agent(name: str = "solo", responses=None) -> Agent:
    return Agent(name=name, system_prompt="be brief", llm=MockLLMClient(responses))


def _swarm(checkpointer=None, name: str = "pair") -> Swarm:
    a = Agent(name="alpha", llm=MockLLMClient([LLMResponse(content="alpha done")]))
    b = Agent(name="beta", llm=MockLLMClient([LLMResponse(content="beta done")]))
    return Swarm(name=name, agents=[a, b], entrypoint="alpha", checkpointer=checkpointer)


def _supervisor() -> Supervisor:
    worker = Agent(name="worker", llm=MockLLMClient([LLMResponse(content="worker output")]))
    return Supervisor(
        name="boss",
        llm=MockLLMClient([LLMResponse(content="final answer")]),
        workers=[Worker(agent=worker, role="worker", description="does the work")],
    )


def _chain() -> Chain:
    chain = Chain(f"pipeline-{uuid.uuid4().hex[:8]}", checkpoint_enabled=False)
    chain.add_node("only", agent=_agent("node-agent"))
    return chain


def _swarm_resume(tmp_path):
    """Crash-recovery resume: run once, then resume the same execution.

    ``aresume`` opened no root span at all — a bigger hole than a missing id.
    """
    from fastaiagent.chain.checkpoint import Checkpoint

    cp = SQLiteCheckpointer(db_path=str(tmp_path / "cp.db"))
    cp.setup()
    exec_id = str(uuid.uuid4())
    swarm = _swarm(checkpointer=cp, name="resumable")
    first = swarm.run("hi", execution_id=exec_id)
    assert first.status == "completed"
    # Rewrite the run-end tombstone to ``failed`` so the run is resumable —
    # the crash-recovery shape, without needing a real crash.
    for row in cp.list(exec_id, limit=500):
        if row.step_type == "run_end":
            cp.put(
                Checkpoint(
                    checkpoint_id=str(uuid.uuid4()),
                    chain_name=row.chain_name,
                    execution_id=exec_id,
                    node_id=row.node_id,
                    node_index=row.node_index,
                    step_type="run_end",
                    status="failed",
                    state_snapshot=dict(row.state_snapshot),
                    agent_path=row.agent_path,
                )
            )
            break
    return swarm.resume(exec_id)


# Every result-producing entry point this sweep covers. A path is a name and
# a zero-or-tmp_path callable that produces the result.
PATH_RUNNERS = {
    "agent.run": lambda: _agent().run("hi"),
    "agent.arun": lambda: _run(_agent().arun("hi")),
    "agent.stream": lambda: _agent().stream("hi"),
    "swarm.run": lambda: _swarm().run("hi"),
    "swarm.arun": lambda: _run(_swarm().arun("hi")),
    "swarm.stream": lambda: _swarm().stream("hi"),
    "swarm.resume": _swarm_resume,
    "supervisor.run": lambda: _supervisor().run("hi"),
    "supervisor.stream": lambda: _supervisor().stream("hi"),
    "chain.execute": lambda: _chain().execute({"message": "hi"}),
}

_NEEDS_TMP_PATH = {"swarm.resume"}


def _run(coro):
    from fastaiagent._internal.async_utils import run_sync

    return run_sync(coro)


@pytest.mark.parametrize("path", sorted(PATH_RUNNERS))
def test_every_result_path_stamps_a_trace_id(path: str, tmp_path) -> None:
    runner = PATH_RUNNERS[path]
    result = runner(tmp_path) if path in _NEEDS_TMP_PATH else runner()
    tid = getattr(result, "trace_id", None)
    assert tid is not None, f"{path} returned a result with no trace_id"
    assert TRACE_ID_RE.match(tid), f"{path} trace_id {tid!r} is not a 32-hex OTel id"
    assert tid != "0" * 32, f"{path} trace_id is the all-zero invalid span context"


def test_registry_covers_the_public_surface() -> None:
    """Fails when a new result-producing entry point has no sweep case."""
    surface: set[str] = {"chain.execute"}
    for topology, cls in (("agent", Agent), ("swarm", Swarm), ("supervisor", Supervisor)):
        for entry in ("run", "stream", "resume", "fork"):
            if hasattr(cls, entry):
                surface.add(f"{topology}.{entry}")
    surface.update({"agent.arun", "swarm.arun"})
    # These delegate their result wholesale to a path that IS covered:
    # ``Agent.fork``/``Agent.resume`` return from ``Agent.arun``, and
    # ``Supervisor.resume`` returns from its inner agent's ``arun``. Listed
    # explicitly so a change that stops delegating becomes a visible decision.
    delegated = {"agent.fork", "agent.resume", "supervisor.resume"}
    uncovered = surface - set(PATH_RUNNERS) - delegated
    assert not uncovered, (
        f"new result-producing entry point(s) with no sweep coverage: {sorted(uncovered)}"
    )


@pytest.fixture
def fresh_tracing(monkeypatch, tmp_path):
    """A private local.db AND a tracer provider pointed at it.

    ``LocalStorageProcessor`` captures its db path when the provider is built,
    and the provider is a module singleton — so any earlier test that ran an
    agent under ``isolated_local_db`` pins span WRITES to a temp file that
    ``TraceStore.default()`` no longer reads. Resetting both together is what
    ``tests/test_trace_enabled_master_switch.py`` does, and it is the only way
    to read back a span you just wrote inside a full-suite run.
    """
    from fastaiagent._internal.config import reset_config
    from fastaiagent.trace import otel

    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(tmp_path / "local.db"))
    reset_config()
    otel.reset()
    yield
    otel.reset()
    reset_config()


def _spans_for(trace_id: str | None):
    """Spans of one trace. Requires the ``fresh_tracing`` fixture."""
    from fastaiagent.trace.storage import TraceStore

    return TraceStore.default().get_trace(trace_id or "").spans


# ---------------------------------------------------------------------------
# Agent.stream: trace_id is not the only field the missing root span zeroed
# ---------------------------------------------------------------------------


def test_agent_stream_reports_tokens_and_execution_id() -> None:
    """``Agent.stream`` accepted a ``trace`` parameter and opened no root span.

    Pulling only ``trace_id`` forward would leave ``tokens_used`` at 0 and
    ``execution_id`` at "" — all three come from the root span the streamed
    run should have been opening all along.
    """
    llm = MockLLMClient(
        [
            LLMResponse(
                content="streamed reply",
                finish_reason="stop",
                usage={"prompt_tokens": 7, "completion_tokens": 5, "total_tokens": 12},
            )
        ]
    )
    result = Agent(name="streamer", llm=llm).stream("hi")
    assert result.output == "streamed reply"
    assert result.tokens_used == 12, f"streamed run reported {result.tokens_used} tokens"
    assert result.execution_id, "streamed run reported an empty execution_id"


def test_agent_stream_opens_a_root_span(fresh_tracing) -> None:
    """The root span is what the UI renders a streamed run as."""
    result = Agent(name="span-streamer", llm=MockLLMClient()).stream("hi")
    names = [sp.name for sp in _spans_for(result.trace_id)]
    assert any(n == "agent.span-streamer" for n in names), (
        f"no agent root span in the streamed trace: {names}"
    )


def test_agent_stream_trace_false_still_streams() -> None:
    """``trace=False`` must still work — it is the escape hatch the parameter
    always promised, now that it means something."""
    result = Agent(name="untraced", llm=MockLLMClient()).stream("hi", trace=False)
    assert result.output
    assert result.trace_id is None


# ---------------------------------------------------------------------------
# Swarm: the firings hole agent.py closed in 1.64.0
# ---------------------------------------------------------------------------


def test_swarm_result_carries_guardrail_firings() -> None:
    """A warn/mask firing inside a swarm must reach the caller.

    ``agent.py`` closed this in 1.64.0; the swarm's own result construction
    sites dropped ``collected_firings()`` on the floor, so a non-halting
    outcome inside a swarm was invisible to the caller.
    """
    from fastaiagent.guardrail import Guardrail, GuardrailPosition, GuardrailType

    rule = Guardrail(
        name="watch",
        guardrail_type=GuardrailType.regex,
        position=GuardrailPosition.output,
        config={"pattern": r"\d{3}-\d{2}-\d{4}"},
        action="warn",
    )
    a = Agent(
        name="leaky",
        llm=MockLLMClient([LLMResponse(content="the ssn is 123-45-6789")]),
        guardrails=[rule],
    )
    swarm = Swarm(name="warned", agents=[a], entrypoint="leaky")
    result = swarm.run("hi")
    assert result.guardrails, "swarm result dropped every guardrail firing"
    assert any(g.fired() for g in result.guardrails)


# ---------------------------------------------------------------------------
# The silent failure the whole defect produced
# ---------------------------------------------------------------------------


def test_eval_over_a_swarm_records_a_trace_id() -> None:
    """``evaluate()`` stamps ``EvalCaseRecord.trace_id`` from the result.

    Agent cases carried one, swarm cases carried ``None``, and the eval
    "succeeded" with identical scores — so nothing downstream had a trace to
    open. Measured with an identical dataset before the fix.
    """
    from fastaiagent.eval import evaluate

    swarm = _swarm(name="evaluated")
    results = evaluate(
        agent_fn=lambda q: swarm.run(q),
        dataset=[{"input": "hi", "expected_output": "alpha done"}],
        scorers=["contains"],
    )
    assert results.cases, "evaluate recorded no cases"
    for case in results.cases:
        assert case.trace_id is not None, "swarm eval case has no trace to open"
        assert TRACE_ID_RE.match(case.trace_id)
