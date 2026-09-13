"""Which agent produced a run, on the wire (durability audit D8, identity half).

``agent_id`` carried the agent's **name**, so nothing joined the checkpoint
replica to the plane's Agents inventory — the console's run-detail page
deliberately refuses to render an agent link for exactly that reason. It now
carries the plane's UUID when this process has one.

Three properties, and the second and third are the ones that make it safe:

1. a registered agent's rows carry the UUID;
2. anything without a plane object — a Chain, a Swarm *container*, an agent that
   has not run — falls back to the **name**, because that is a legitimate value
   the plane must handle anyway, not a failure;
3. restore still rebuilds the agent's **name**. ``_wire_to_checkpoint`` derives
   ``Checkpoint.chain_name`` from what comes back, and resume matches that
   against the runner — so a UUID arriving there would produce a run named
   ``ed1be3bc-…`` that nothing could resume.

NO MOCKS of the thing under test: the real ``_to_wire`` / ``_wire_to_checkpoint``
against rows written by real executors into a real store, and the real
``_pushed`` registry populated the way ``push_agent`` populates it.
"""

from __future__ import annotations

import pytest

from fastaiagent import Agent, SQLiteCheckpointer
from fastaiagent._platform import push as _push
from fastaiagent.checkpointers.platform_replica import _to_wire, _wire_to_checkpoint

_UUID = "ed1be3bc-cd16-4664-8852-9bd98105d69d"


@pytest.fixture
def clean_registry():
    """Registration state is process-global; leaking it across tests is a trap."""
    _push.reset_registration_state()
    yield _push
    _push.reset_registration_state()


def _register(name: str, agent_id: str | None) -> None:
    """Populate ``_pushed`` exactly as a successful ``push_agent`` would."""
    with _push._lock:
        _push._pushed[name] = _push.PushResult(agent_id=agent_id, name=name)


def _store(tmp_path, name="cp.db") -> SQLiteCheckpointer:
    cp = SQLiteCheckpointer(db_path=str(tmp_path / name))
    cp.setup()
    return cp


def _wire(store: SQLiteCheckpointer) -> list[dict]:
    return [_to_wire(r) for r in store.fetch_unsynced(100, None)]


# ── the accessor ───────────────────────────────────────────────────────────


def test_accessor_returns_none_for_an_unregistered_name(clean_registry) -> None:
    assert _push.pushed_agent_id("never-pushed") is None


def test_accessor_returns_the_uuid_once_registered(clean_registry) -> None:
    _register("solo", _UUID)
    assert _push.pushed_agent_id("solo") == _UUID


def test_a_push_that_returned_no_id_is_not_mistaken_for_one(clean_registry) -> None:
    """``PushResult.agent_id`` is Optional — an older plane may answer without it."""
    _register("solo", None)
    assert _push.pushed_agent_id("solo") is None


# ── the wire ───────────────────────────────────────────────────────────────


def test_a_registered_agents_rows_carry_the_uuid(clean_registry, tmp_path, mock_llm) -> None:
    store = _store(tmp_path)
    agent = Agent(name="pricing", system_prompt="Answer.", llm=mock_llm, checkpointer=store)
    _register("pricing", _UUID)
    agent.run("hi", execution_id="ex-registered")

    rows = _wire(store)
    assert rows, "the agent wrote no checkpoints"
    assert {r["agent_id"] for r in rows} == {_UUID}
    assert all(r["chain_id"] is None for r in rows)


def test_an_unregistered_agent_falls_back_to_its_name(clean_registry, tmp_path, mock_llm) -> None:
    """An agent that has not run yet, or a plane that never answered, is normal.

    Registration fires on first *run*, not on construction, so None here is an
    ordinary state — never an error, and never a reason to drop the field.
    """
    store = _store(tmp_path)
    Agent(name="unregistered", system_prompt="Answer.", llm=mock_llm, checkpointer=store).run(
        "hi", execution_id="ex-unregistered"
    )
    assert {r["agent_id"] for r in _wire(store)} == {"unregistered"}


def test_a_chain_keeps_its_name_because_it_has_no_plane_object(tmp_path, clean_registry) -> None:
    """A Chain is an orchestration *of* agents, not an agent.

    Nothing pushes a plane object for the container, so ``chain_id`` is a name
    and always will be — which is why the plane has to accept one.
    """
    from fastaiagent.chain import Chain, NodeType
    from fastaiagent.tool.function import FunctionTool

    store = _store(tmp_path)
    chain = Chain("pipeline", checkpoint_enabled=True, checkpointer=store)
    chain.add_node(
        "only",
        tool=FunctionTool(name="only", fn=lambda value: {"v": value}),
        type=NodeType.tool,
        input_mapping={"value": "{{state.v}}"},
    )
    chain.execute({"v": 1}, execution_id="ex-chain")

    rows = _wire(store)
    assert {r["chain_id"] for r in rows} == {"pipeline"}
    assert all(r["agent_id"] is None for r in rows)


def test_the_uuid_is_resolved_at_drain_time_not_write_time(
    clean_registry, tmp_path, mock_llm
) -> None:
    """THE placement argument, as a test.

    Registration runs on a daemon thread and routinely lands *after* a run's
    first checkpoints are written. Resolving at write time would freeze the name
    into those rows and the UUID into later ones — two identities for one run,
    which breaks the join this feature exists to create. Because ``_to_wire``
    runs in the drain, a registration that lands mid-run still applies to
    every row of it.
    """
    store = _store(tmp_path)
    agent = Agent(name="late", system_prompt="Answer.", llm=mock_llm, checkpointer=store)
    agent.run("hi", execution_id="ex-late")  # nothing registered yet

    assert {r["agent_id"] for r in _wire(store)} == {"late"}, "precondition: no id yet"

    _register("late", _UUID)  # registration lands afterwards, as it really does

    rows = _wire(store)
    assert {r["agent_id"] for r in rows} == {_UUID}, (
        "the id must be resolved when the batch is drained, not when the row was written"
    )


# ── restore ────────────────────────────────────────────────────────────────


def test_restore_rebuilds_the_name_not_the_uuid(clean_registry, tmp_path, mock_llm) -> None:
    """The hazard this change introduces, and the mitigation, in one test.

    Resume matches ``Checkpoint.chain_name`` against the runner. If restore
    handed back the UUID the run would come back named ``ed1be3bc-…`` and
    nothing would resume it.
    """
    store = _store(tmp_path)
    Agent(name="pricing", system_prompt="Answer.", llm=mock_llm, checkpointer=store).run(
        "hi", execution_id="ex-round-trip"
    )
    _register("pricing", _UUID)

    for wire in _wire(store):
        assert wire["agent_id"] == _UUID
        assert wire["metadata"]["chain_name"] == "pricing"
        assert _wire_to_checkpoint(wire).chain_name == "pricing"


def test_a_row_written_before_this_change_still_restores(clean_registry) -> None:
    """Rows already on the plane have no ``metadata.chain_name`` — their
    ``agent_id`` IS the name. The fallback order has to keep working for them,
    or upgrading the SDK would strand every checkpoint already replicated."""
    legacy = {
        "checkpoint_id": "c1",
        "execution_id": "ex-legacy",
        "agent_id": "pricing",
        "chain_id": None,
        "node_id": "turn:0",
        "status": "completed",
        "state_snapshot": {},
        "metadata": {"agent_path": "agent:pricing"},
    }
    assert _wire_to_checkpoint(legacy).chain_name == "pricing"

    legacy_chain = {**legacy, "agent_id": None, "chain_id": "pipeline", "node_id": "review"}
    assert _wire_to_checkpoint(legacy_chain).chain_name == "pipeline"


def test_identity_never_breaks_replication(clean_registry, tmp_path, mock_llm, monkeypatch) -> None:
    """Identity is metadata for a console. A lookup problem must not cost a row.

    Pinned because the resolver reaches into another module's lock from inside
    the drain — the one place in the SDK where a checkpoint is at stake. If the
    registry ever raises, the row must still replicate, carrying the name.
    """
    store = _store(tmp_path)
    Agent(name="pricing", system_prompt="Answer.", llm=mock_llm, checkpointer=store).run(
        "hi", execution_id="ex-safe"
    )

    def _explode(_name: str) -> str | None:
        raise RuntimeError("registry exploded")

    monkeypatch.setattr("fastaiagent._platform.push.pushed_agent_id", _explode)

    rows = _wire(store)
    assert rows, "a registry failure lost the batch"
    assert {r["agent_id"] for r in rows} == {"pricing"}, "it should fall back to the name"
    assert all(r["checkpoint_id"] and r["state_snapshot"] is not None for r in rows)
