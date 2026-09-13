"""What topology the plane is told a run has (durability audit D8).

``resource_type`` is supposed to say whether a run is an agent, a chain, a swarm
or a supervisor. It was a guess based on string prefixes that only ever returned
two of the four — and it asked the question *per checkpoint* rather than per run,
so a single swarm produced a mixture. Measured on a real 2-agent swarm before the
fix::

    handoff:0                 swarm:deskflow                -> chain
    turn:0                    swarm:deskflow/agent:triage   -> agent
    run_end                   swarm:deskflow                -> chain

The plane derives a run's type from whichever checkpoint is latest, so the type
it displayed depended on which row happened to be newest.

Why it is worth fixing before anyone is on a plane rather than after: the replica
is **append-only**. It stores what it was sent and never rewrites it, so every
swarm replicated before this fix stays labelled `chain` forever. Fixing while
there is no traffic costs zero bad history.

NO MOCKS of the thing under test: real ``Swarm`` and ``Supervisor`` runs through
the real executors, writing real rows to a real store, classified by the real
``_to_wire``. The model is the repo's ``MockLLMClient`` — a real ``LLMClient``
subclass — because none of this depends on what a model says, and the rows that
matter (a swarm's ``handoff:0``, a supervisor's turn rows) are written before any
model call resolves.
"""

from __future__ import annotations

import pytest

from fastaiagent import Agent, SQLiteCheckpointer
from fastaiagent.agent.swarm import Swarm
from fastaiagent.agent.team import Supervisor, Worker
from fastaiagent.checkpointers.platform_replica import _resource_type, _to_wire


def _store(tmp_path, name="cp.db") -> SQLiteCheckpointer:
    cp = SQLiteCheckpointer(db_path=str(tmp_path / name))
    cp.setup()
    return cp


def _wire_rows(store: SQLiteCheckpointer) -> list[dict]:
    return [_to_wire(r) for r in store.fetch_unsynced(100, None)]


# ── real runs ──────────────────────────────────────────────────────────────


def test_a_swarm_run_reports_swarm_on_every_row(tmp_path, mock_llm) -> None:
    """One run, one answer. The mixture was the defect, not just the label."""
    store = _store(tmp_path)
    a = Agent(name="triage", system_prompt="Triage.", llm=mock_llm)
    b = Agent(name="specialist", system_prompt="Answer.", llm=mock_llm)
    swarm = Swarm(
        name="deskflow",
        agents=[a, b],
        entrypoint="triage",
        handoffs={"triage": ["specialist"], "specialist": []},
        checkpointer=store,
    )
    swarm.run("hello", execution_id="ex-swarm")

    rows = _wire_rows(store)
    assert rows, "the swarm wrote no checkpoints"
    kinds = {r["resource_type"] for r in rows}
    assert kinds == {"swarm"}, f"one run reported as {sorted(kinds)}"
    # A swarm is agent-shaped, so its identity rides in agent_id. Sending both
    # ids null would replace a wrong label with no label at all.
    assert all(r["agent_id"] and r["chain_id"] is None for r in rows)


def test_a_supervisor_run_reports_supervisor_on_every_row(tmp_path, mock_llm) -> None:
    store = _store(tmp_path)
    worker = Agent(name="researcher", system_prompt="Research.", llm=mock_llm)
    sup = Supervisor(
        name="deskteam",
        llm=mock_llm,
        workers=[Worker(agent=worker, role="researcher", description="Researches")],
        checkpointer=store,
    )
    sup.run("hello", execution_id="ex-sup")

    rows = _wire_rows(store)
    assert rows, "the supervisor wrote no checkpoints"
    kinds = {r["resource_type"] for r in rows}
    assert kinds == {"supervisor"}, f"one run reported as {sorted(kinds)}"
    assert all(r["agent_id"] and r["chain_id"] is None for r in rows)


def test_a_plain_agent_and_a_plain_chain_are_unchanged(tmp_path, mock_llm) -> None:
    """The two values that already worked must keep working.

    A regression here would be worse than the bug: those are the shapes that
    actually replicate today.
    """
    from fastaiagent.chain import Chain, NodeType
    from fastaiagent.tool.function import FunctionTool

    store = _store(tmp_path, "agent.db")
    Agent(name="solo", system_prompt="Answer.", llm=mock_llm, checkpointer=store).run(
        "hi", execution_id="ex-agent"
    )
    agent_rows = _wire_rows(store)
    assert {r["resource_type"] for r in agent_rows} == {"agent"}
    assert all(r["agent_id"] and r["chain_id"] is None for r in agent_rows)

    chain_store = _store(tmp_path, "chain.db")
    chain = Chain("pipeline", checkpoint_enabled=True, checkpointer=chain_store)
    chain.add_node(
        "only",
        tool=FunctionTool(name="only", fn=lambda value: {"v": value}),
        type=NodeType.tool,
        input_mapping={"value": "{{state.v}}"},
    )
    chain.execute({"v": 1}, execution_id="ex-chain")
    chain_rows = _wire_rows(chain_store)
    assert {r["resource_type"] for r in chain_rows} == {"chain"}
    assert all(r["chain_id"] and r["agent_id"] is None for r in chain_rows)


# ── the classifier itself ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("agent_path", "node_id", "expected"),
    [
        # A swarm's own rows and its children's rows must agree.
        ("swarm:desk", "handoff:0", "swarm"),
        ("swarm:desk/agent:triage", "turn:0", "swarm"),
        ("swarm:desk/agent:triage/tool:x", "turn:0/tool:x", "swarm"),
        # Likewise a supervisor and its workers.
        ("supervisor:team", "turn:0", "supervisor"),
        ("supervisor:team/worker:research", "turn:1", "supervisor"),
        ("supervisor:team/worker:research/tool:y", "turn:1/tool:y", "supervisor"),
        # A standalone agent, by path and by the node_id fallback.
        ("agent:solo", "turn:0", "agent"),
        (None, "turn:3", "agent"),
        # Anything else is a chain.
        (None, "review", "chain"),
        ("", "", "chain"),
    ],
)
def test_classification_is_rooted_in_the_path(agent_path, node_id, expected) -> None:
    """Every row of a run must classify by its ROOT, not by its own depth.

    The three-deep swarm and supervisor cases are the ones that used to come back
    ``agent`` — they are why a single run reported two different types.
    """
    assert _resource_type({"agent_path": agent_path, "node_id": node_id}) == expected


def test_the_plane_would_accept_every_value_we_send() -> None:
    """Pin the wire vocabulary to what the door takes.

    The plane widened its ``resource_type`` Literal to these four in PR #128
    *before* the SDK sent them, on purpose — the reverse order would 422 every
    swarm checkpoint. If a fifth topology is ever added here, the plane has to
    accept it first.
    """
    from fastaiagent.checkpointers.platform_replica import _AGENT_SHAPED

    accepted = {"agent", "chain", "swarm", "supervisor"}
    assert _AGENT_SHAPED <= accepted
    produced = {
        _resource_type({"agent_path": p, "node_id": n})
        for p, n in (
            ("swarm:s", "handoff:0"),
            ("supervisor:t", "turn:0"),
            ("agent:a", "turn:0"),
            (None, "node"),
        )
    }
    assert produced == accepted
