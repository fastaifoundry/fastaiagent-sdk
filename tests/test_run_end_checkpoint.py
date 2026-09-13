"""The run-end marker, and the resume bug it exists to fix (durability audit D5).

Before this, nothing ever wrote a terminal checkpoint. A run that finished and a
run that died the instant after its last step were **byte-identical** — both left
a ``completed`` checkpoint as the newest row. Two things followed:

  * the plane could not say a run had finished, so its console rendered
    "last step done" and its ``failed`` filter could never match;
  * ``aresume`` / ``Chain.resume`` on a run that had *already completed*
    silently re-executed it — re-calling the model and re-firing every
    side-effecting tool not wrapped in ``@idempotent``.

The second is the one these tests are really about. The marker is the mechanism;
refusing to re-run a finished run is the fix.

NO MOCKS of the thing under test: a real ``Chain``, a real ``Swarm``, real
``SQLiteCheckpointer`` storage. The LLM is the repo's ``MockLLMClient`` fixture —
a real ``LLMClient`` subclass, not ``unittest.mock`` — because none of this
depends on what a model says, and a live model would make the assertions flaky
for no gain (see ``test_multi_agent_integration``, which is exactly that).
"""

from __future__ import annotations

import pytest

from fastaiagent import SQLiteCheckpointer
from fastaiagent.chain import Chain, NodeType
from fastaiagent.chain.checkpoint import RUN_END, is_run_end, latest_resumable
from fastaiagent.chain.interrupt import AlreadyResumed, Resume, interrupt
from fastaiagent.tool.function import FunctionTool


def _store(tmp_path, name="cp.db") -> SQLiteCheckpointer:
    cp = SQLiteCheckpointer(db_path=str(tmp_path / name))
    cp.setup()
    return cp


def _double(value: int) -> dict:
    return {"doubled": int(value) * 2}


def _boom(value: int) -> dict:
    raise RuntimeError(f"node blew up on {value}")


def _approval(amount: int) -> dict:
    decision = interrupt(reason="approve?", context={"amount": amount})
    return {"approved": bool(getattr(decision, "approved", decision))}


def _simple_chain(store, name="run-end-chain", *, explode=False) -> Chain:
    chain = Chain(name, checkpoint_enabled=True, checkpointer=store)
    chain.add_node(
        "first",
        tool=FunctionTool(name="first", fn=_double),
        type=NodeType.tool,
        input_mapping={"value": "{{state.amount}}"},
    )
    chain.add_node(
        "second",
        tool=FunctionTool(name="second", fn=_boom if explode else _double),
        type=NodeType.tool,
        input_mapping={"value": "{{state.output.doubled}}"},
    )
    chain.connect("first", "second")
    return chain


# ── the marker itself ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_completed_chain_writes_exactly_one_run_end(tmp_path) -> None:
    store = _store(tmp_path)
    result = await _simple_chain(store).aexecute({"amount": 2}, execution_id="ex-ok")

    rows = store.list("ex-ok")
    markers = [c for c in rows if is_run_end(c)]
    assert len(markers) == 1, f"expected exactly one terminal row, got {len(markers)}"
    assert markers[0].status == "completed"
    assert markers[0].step_type == RUN_END
    # It must sort LAST under both orderings the SDK uses — ``list`` reads
    # node_index ASC, ``get_last`` reads created_at DESC.
    assert rows[-1] is not None and is_run_end(rows[-1])
    assert is_run_end(store.get_last("ex-ok"))
    assert result.status == "completed"


@pytest.mark.asyncio
async def test_a_node_that_raises_writes_a_failed_marker_and_still_raises(tmp_path) -> None:
    """The marker must never swallow the exception that caused it."""
    store = _store(tmp_path)
    chain = _simple_chain(store, explode=True)

    with pytest.raises(Exception) as excinfo:
        await chain.aexecute({"amount": 2}, execution_id="ex-boom")
    assert "blew up" in str(excinfo.value)

    marker = store.get_last("ex-boom")
    assert is_run_end(marker)
    assert marker is not None and marker.status == "failed"
    # The cause is recorded, so an operator reading the replica knows why.
    assert "blew up" in marker.state_snapshot.get("run_error", "")


@pytest.mark.asyncio
async def test_a_paused_run_gets_no_marker(tmp_path) -> None:
    """A pause is not an ending.

    A row written after the ``interrupted`` one would sit in front of the four
    ``latest.status == "interrupted"`` guards that make a HITL resume demand a
    ``Resume`` value — turning "you must approve this" into a silent re-run.
    """
    store = _store(tmp_path)
    chain = Chain("paused-chain", checkpoint_enabled=True, checkpointer=store)
    chain.add_node(
        "approval",
        tool=FunctionTool(name="approval", fn=_approval),
        type=NodeType.tool,
        input_mapping={"amount": "{{state.amount}}"},
    )
    result = await chain.aexecute({"amount": 5}, execution_id="ex-paused")

    assert result.status == "paused"
    assert not any(is_run_end(c) for c in store.list("ex-paused"))
    latest = store.get_last("ex-paused")
    assert latest is not None and latest.status == "interrupted"


@pytest.mark.asyncio
async def test_resuming_a_paused_run_marks_it_when_it_finishes(tmp_path) -> None:
    store = _store(tmp_path)
    chain = Chain("hitl-chain", checkpoint_enabled=True, checkpointer=store)
    chain.add_node(
        "approval",
        tool=FunctionTool(name="approval", fn=_approval),
        type=NodeType.tool,
        input_mapping={"amount": "{{state.amount}}"},
    )
    await chain.aexecute({"amount": 5}, execution_id="ex-hitl")
    assert not any(is_run_end(c) for c in store.list("ex-hitl"))

    resumed = await chain.resume("ex-hitl", resume_value=Resume(approved=True))
    assert resumed.status == "completed"
    marker = store.get_last("ex-hitl")
    assert is_run_end(marker) and marker is not None and marker.status == "completed"


# ── the resume bug the marker fixes ────────────────────────────────────────


@pytest.mark.asyncio
async def test_resuming_a_finished_chain_is_refused_not_replayed(tmp_path) -> None:
    """THE headline. Pre-change this silently re-ran the whole chain.

    ``Chain.resume`` finds its restart node by matching the latest checkpoint's
    ``node_id`` against the topological order. On a finished run that matched the
    LAST node, so ``start_node`` became None — which ``execute_chain`` reads as
    "start at index 0" — and the entire chain re-executed, re-firing every
    side effect in it. Nothing raised, nothing warned.
    """
    store = _store(tmp_path)
    chain = _simple_chain(store)
    await chain.aexecute({"amount": 2}, execution_id="ex-done")
    before = len(store.list("ex-done"))

    with pytest.raises(AlreadyResumed) as excinfo:
        await chain.resume("ex-done")
    assert "already finished" in str(excinfo.value)

    # And it must refuse without writing anything — a refusal is not a run.
    assert len(store.list("ex-done")) == before


@pytest.mark.asyncio
async def test_a_failed_run_stays_resumable_and_does_not_replay_from_the_top(
    tmp_path,
) -> None:
    """The other half of the asymmetry: crash recovery is the whole feature.

    A ``failed`` marker must be stepped over, not treated as a re-entry point —
    otherwise its unmatched ``node_id`` sends ``Chain.resume`` back to node 0.
    """
    store = _store(tmp_path)
    with pytest.raises(Exception):
        await _simple_chain(store, explode=True).aexecute({"amount": 2}, execution_id="ex-retry")

    assert store.get_last("ex-retry").status == "failed"
    # The resume target steps back past the tombstone to the last real node.
    target = latest_resumable(store, "ex-retry")
    assert target is not None and target.node_id == "first"

    # A repaired chain resumes from "second", not from the top: "first" runs once.
    healthy = _simple_chain(store)
    resumed = await healthy.resume("ex-retry")
    assert resumed.status == "completed"
    firsts = [c for c in store.list("ex-retry") if c.node_id == "first"]
    assert len(firsts) == 1, "the chain replayed from node 0 instead of resuming"


def test_is_run_end_reads_a_pre_v20_row_as_not_an_ending(tmp_path) -> None:
    """A checkpoint written before this column existed has ``step_type=None``.

    Reading NULL as "not a run end" is what keeps every pre-upgrade run
    resumable — the alternative, guessing, would invent history.
    """
    from fastaiagent.chain.checkpoint import Checkpoint

    assert not is_run_end(Checkpoint(node_id="turn:0", status="completed"))
    assert not is_run_end(None)
    assert is_run_end(Checkpoint(node_id=RUN_END, step_type=RUN_END, status="completed"))


# ── step_type on the ordinary write points ─────────────────────────────────


@pytest.mark.asyncio
async def test_every_checkpoint_is_classified(tmp_path) -> None:
    """``step_type`` is what the plane needs to render a run; NULL tells it nothing."""
    store = _store(tmp_path)
    await _simple_chain(store).aexecute({"amount": 2}, execution_id="ex-typed")
    assert [c.step_type for c in store.list("ex-typed")] == ["node", "node", RUN_END]


@pytest.mark.asyncio
async def test_step_type_survives_the_replication_round_trip(tmp_path) -> None:
    """``_to_wire`` → plane → ``_wire_to_checkpoint`` must not drop it.

    The plane has always had this key. Losing it on the way out would leave the
    console exactly as blind as it was before D5.
    """
    from fastaiagent.checkpointers.platform_replica import _to_wire, _wire_to_checkpoint

    store = _store(tmp_path)
    await _simple_chain(store).aexecute({"amount": 2}, execution_id="ex-wire")
    rows = store.fetch_unsynced(50, None)
    assert rows, "nothing to replicate"

    wire = [_to_wire(r) for r in rows]
    assert [w["step_type"] for w in wire] == ["node", "node", RUN_END]
    back = [_wire_to_checkpoint(w) for w in wire]
    assert [c.step_type for c in back] == ["node", "node", RUN_END]
    assert is_run_end(back[-1]) and back[-1].status == "completed"
