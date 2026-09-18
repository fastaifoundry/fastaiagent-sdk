"""A forked branch that dies leaves a tombstone, like every other run shape.

``Chain.aexecute``, ``Chain.resume`` and ``Agent._arun_core`` have each written a
``run_end`` marker on the failure path since the durability audit (D5, 1.65.0)
and 1.67.0. The two **fork** entry points never did: ``Chain.afork`` called
``execute_chain`` with no ``try/except`` at all, and ``Agent.afork`` delegated to
a core that only marks the run when the *agent* carries a checkpointer — while a
fork itself works off ``self._checkpointer or SQLiteCheckpointer()``, so a fork
driven from the default store wrote its ``__fork_origin__`` lineage row and no
tombstone anywhere.

Why it matters is why it mattered for D5: without the row a branch that crashed
is byte-identical to one that simply stopped, and ``latest_resumable`` hands back
the last real checkpoint — so resuming a dead fork silently re-runs it.

No mocks: a real ``Chain`` / ``Agent``, a real ``SQLiteCheckpointer``, and the
repo's ``MockLLMClient`` (a real ``LLMClient`` subclass) where a model is needed.
"""

from __future__ import annotations

import sqlite3

import pytest

from fastaiagent import SQLiteCheckpointer
from fastaiagent.agent import Agent
from fastaiagent.chain import Chain, NodeType
from fastaiagent.chain.checkpoint import is_run_end
from fastaiagent.llm.client import LLMResponse
from fastaiagent.tool.function import FunctionTool

from .conftest import MockLLMClient

#: Flipped on for the forked branch only, so the source run succeeds and the
#: branch is the thing that dies.
BLOW_UP = {"yes": False}


@pytest.fixture(autouse=True)
def _reset_blowup():
    BLOW_UP["yes"] = False
    yield
    BLOW_UP["yes"] = False


def _store(tmp_path, name="fork.db") -> SQLiteCheckpointer:
    cp = SQLiteCheckpointer(db_path=str(tmp_path / name))
    cp.setup()
    return cp


def _markers(store: SQLiteCheckpointer, execution_id: str):
    return [c for c in store.list(execution_id, limit=500) if is_run_end(c)]


def _fork_id(store: SQLiteCheckpointer, source_id: str) -> str:
    """The forked execution id, read back out of the store.

    ``afork`` returns nothing when it raises, so the branch is located by its
    lineage row — the ``fork_origin`` checkpoint written under the new id.
    """
    conn = sqlite3.connect(store.db_path)
    try:
        rows = conn.execute(
            "SELECT execution_id FROM checkpoints WHERE step_type = 'fork_origin' "
            "ORDER BY rowid"
        ).fetchall()
    finally:
        conn.close()
    ids = [r[0] for r in rows if r[0] != source_id]
    assert ids, "no fork_origin row — the fork never started"
    return str(ids[-1])


# ---------------------------------------------------------------------------
# Chain.afork
# ---------------------------------------------------------------------------


def _double(value: int) -> dict:
    return {"doubled": int(value) * 2}


def _second(value: int) -> dict:
    if BLOW_UP["yes"]:
        raise RuntimeError("the branch blew up")
    return {"doubled": int(value) * 2}


def _forkable_chain(store: SQLiteCheckpointer) -> Chain:
    chain = Chain("forked-chain", checkpoint_enabled=True, checkpointer=store)
    chain.add_node(
        "first",
        tool=FunctionTool(name="first", fn=_double),
        type=NodeType.tool,
        input_mapping={"value": "{{state.amount}}"},
    )
    chain.add_node(
        "second",
        tool=FunctionTool(name="second", fn=_second),
        type=NodeType.tool,
        input_mapping={"value": "{{state.output.doubled}}"},
    )
    chain.connect("first", "second")
    return chain


@pytest.mark.asyncio
async def test_chain_fork_that_dies_writes_exactly_one_failed_marker(tmp_path) -> None:
    store = _store(tmp_path)
    chain = _forkable_chain(store)
    await chain.aexecute({"amount": 2}, execution_id="chain-src")

    first = next(c for c in store.list("chain-src", limit=500) if c.node_id == "first")
    BLOW_UP["yes"] = True
    with pytest.raises(Exception) as excinfo:
        await chain.afork("chain-src", checkpoint_id=first.checkpoint_id)
    assert "blew up" in str(excinfo.value)

    fork_id = _fork_id(store, "chain-src")
    markers = _markers(store, fork_id)
    assert len(markers) == 1, (
        f"a forked chain that raised wrote {len(markers)} run_end rows; the branch "
        "is indistinguishable from one that simply stopped"
    )
    assert markers[0].status == "failed"
    assert "blew up" in markers[0].state_snapshot.get("run_error", "")


@pytest.mark.asyncio
async def test_chain_fork_that_finishes_writes_a_completed_marker(tmp_path) -> None:
    """The mirror of the failure case: ``aexecute`` and ``resume`` both close a
    run that ends, and a fork is a run."""
    store = _store(tmp_path, "fork-ok.db")
    chain = _forkable_chain(store)
    await chain.aexecute({"amount": 2}, execution_id="chain-ok-src")

    first = next(c for c in store.list("chain-ok-src", limit=500) if c.node_id == "first")
    result = await chain.afork("chain-ok-src", checkpoint_id=first.checkpoint_id)

    markers = _markers(store, result.execution_id)
    assert len(markers) == 1, f"forked chain wrote {len(markers)} run_end rows"
    assert markers[0].status == "completed"


# ---------------------------------------------------------------------------
# Agent.afork
# ---------------------------------------------------------------------------
#
# ``Agent.afork`` delegates the branch to ``_arun_core``, whose own failure
# handler already writes the marker — but only ``if self._checkpointer is not
# None``. ``afork`` itself runs off ``self._checkpointer or
# SQLiteCheckpointer()``, so the branch of a checkpointer-less agent is written
# into a real store (its ``__fork_origin__`` row proves it) and then closed by
# nobody. Both shapes are covered here: the delegating one to pin that it stays
# at exactly one row, the other because it has none.


class _BoomLLM(MockLLMClient):
    """An LLM that dies mid-branch.

    A *tool* that raises would not do: the executor catches a tool error and
    feeds it back to the model as a tool result, so the run completes. The model
    call is the failure that actually propagates out of ``_arun_core``.
    """

    async def acomplete(self, messages, tools=None, **kwargs):
        raise RuntimeError("the agent branch blew up")


def _ok_llm() -> MockLLMClient:
    return MockLLMClient(
        [LLMResponse(content="done", finish_reason="stop", usage={"total_tokens": 3})]
    )


@pytest.mark.asyncio
async def test_agent_fork_that_dies_writes_exactly_one_failed_marker(tmp_path) -> None:
    """The delegating shape — a regression guard, not a red proof.

    ``_arun_core`` marks this one today; adding a marker in ``afork`` as well
    would give the branch two tombstones, which is its own kind of wrong.
    """
    store = _store(tmp_path, "agent-fork.db")
    await Agent(name="forker", llm=_ok_llm(), checkpointer=store).arun(
        "go", execution_id="agent-src"
    )

    with pytest.raises(Exception) as excinfo:
        await Agent(name="forker", llm=_BoomLLM(), checkpointer=store).afork("agent-src")
    assert "blew up" in str(excinfo.value)

    fork_id = _fork_id(store, "agent-src")
    markers = _markers(store, fork_id)
    assert len(markers) == 1, f"a forked agent run that raised wrote {len(markers)} run_end rows"
    assert markers[0].status == "failed"


@pytest.mark.asyncio
async def test_agent_fork_marks_the_branch_without_an_agent_checkpointer(
    tmp_path, monkeypatch
) -> None:
    """The gap: a fork can be durable while the agent is not.

    ``afork`` resolves its own store, writes the lineage row into it, and — before
    this fix — left the branch with no tombstone at all, because ``_arun_core``
    only marks a run when ``self._checkpointer`` is set.
    """
    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(tmp_path / "default.db"))
    from fastaiagent._internal.config import reset_config

    reset_config()
    try:
        store = SQLiteCheckpointer()
        store.setup()
        await Agent(name="forker2", llm=_ok_llm(), checkpointer=store).arun(
            "go", execution_id="agent-src-2"
        )

        with pytest.raises(Exception):
            await Agent(name="forker2", llm=_BoomLLM()).afork("agent-src-2")

        fork_id = _fork_id(store, "agent-src-2")
        markers = _markers(store, fork_id)
        assert len(markers) == 1, (
            f"a fork driven off the default store wrote {len(markers)} run_end rows"
        )
        assert markers[0].status == "failed"
    finally:
        reset_config()


@pytest.mark.asyncio
async def test_agent_fork_that_finishes_without_an_agent_checkpointer_is_closed(
    tmp_path, monkeypatch
) -> None:
    """A branch that ends is closed whichever way it ended."""
    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(tmp_path / "default-ok.db"))
    from fastaiagent._internal.config import reset_config

    reset_config()
    try:
        store = SQLiteCheckpointer()
        store.setup()
        await Agent(name="forker3", llm=_ok_llm(), checkpointer=store).arun(
            "go", execution_id="agent-src-3"
        )

        result = await Agent(name="forker3", llm=_ok_llm()).afork("agent-src-3")

        markers = _markers(store, result.execution_id)
        assert len(markers) == 1, f"forked agent run wrote {len(markers)} run_end rows"
        assert markers[0].status == "completed"
    finally:
        reset_config()
