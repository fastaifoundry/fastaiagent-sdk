"""End-to-end gate — Agent.afork branches a run with a modified input.

No mocks: a real OpenAI model answers an original question, then ``afork``
re-asks a DIFFERENT question under a fresh execution_id. The fork's answer
diverges, and the original execution's checkpoints are left intact.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.e2e.conftest import require_env

pytestmark = pytest.mark.e2e


def test_agent_afork_diverges_and_leaves_original_intact(tmp_path: Path) -> None:
    import os

    os.environ.setdefault("E2E_SKIP_PLATFORM", "1")
    require_env()

    from fastaiagent import Agent, LLMClient
    from fastaiagent.checkpointers import SQLiteCheckpointer

    cp = SQLiteCheckpointer(db_path=str(tmp_path / "ckpt.db"))
    agent = Agent(
        name="fork-agent",
        system_prompt="Answer in as few words as possible.",
        llm=LLMClient(provider="openai", model="gpt-4.1"),
        checkpointer=cp,
    )

    orig = agent.run("What is 2 + 2? Reply with just the number.", execution_id="orig")
    assert orig.execution_id == "orig"
    assert "4" in (orig.output or "")
    orig_ckpts = cp.list("orig")
    assert orig_ckpts, "expected checkpoints for the original run"

    # Fork with a DIFFERENT question -> divergent answer under a fresh id.
    fork = agent.fork("orig", input="What is the capital of France? One word.")
    assert fork.execution_id != "orig"
    assert fork.execution_id != ""
    assert "paris" in (fork.output or "").lower()

    # The original execution is untouched by the fork.
    orig_after = cp.list("orig")
    assert len(orig_after) == len(orig_ckpts)
    assert "4" in (orig.output or "")


def test_agent_afork_without_input_branches_from_a_real_turn(tmp_path: Path) -> None:
    """The no-input fork, against a live model, over what the checkpointer holds.

    Since 1.65.0 every finished run ends with a ``run_end`` tombstone, and
    ``afork`` was still calling ``get_last`` raw. ``Agent.afork`` was soft-broken
    rather than hard-broken by that — it ran, and produced a branch whose lineage
    was wrong in three ways at once:

    * ``parent_checkpoint_id`` pointed at the tombstone rather than a turn;
    * the origin's ``state_snapshot`` carried ``run_status: "completed"`` from
      the marker into the new run's state, so the branch began life labelled
      finished;
    * ``node_index`` was the marker's inflated last+1, which put the branch's
      seed row *after* the first step the branch then wrote.

    None of that raised. It is asserted here rather than in a unit test because
    the snapshot it reads is the one a real model's turn actually wrote.
    """
    import os

    os.environ.setdefault("E2E_SKIP_PLATFORM", "1")
    require_env()

    from fastaiagent import Agent, LLMClient
    from fastaiagent.chain.checkpoint import is_run_end
    from fastaiagent.checkpointers import SQLiteCheckpointer

    cp = SQLiteCheckpointer(db_path=str(tmp_path / "noinput.db"))
    agent = Agent(
        name="fork-agent-noinput",
        system_prompt="Answer in as few words as possible.",
        llm=LLMClient(provider="openai", model="gpt-4.1"),
        checkpointer=cp,
    )

    agent.run("What is the capital of Japan? One word.", execution_id="src")
    rows = cp.list("src")
    assert is_run_end(rows[-1]), "the run should end with a terminal marker"
    last_turn = next(c for c in reversed(rows) if not is_run_end(c))

    fork = agent.fork("src")
    assert fork.execution_id != "src"

    origin = next(c for c in cp.list(fork.execution_id) if c.node_id == "__fork_origin__")
    assert origin.parent_checkpoint_id == last_turn.checkpoint_id, (
        "the branch's lineage points at the run-end tombstone, not at a turn"
    )
    assert "run_status" not in origin.state_snapshot, (
        "the branch inherited the finished run's status into its own state"
    )
    assert origin.node_index == last_turn.node_index

    # The seed row sorts FIRST under ``list``'s node_index ordering and is never
    # what ``get_last`` hands back. With the marker's inflated index it sorted
    # past the branch's own first step instead.
    branch_rows = cp.list(fork.execution_id)
    assert branch_rows[0].node_id == "__fork_origin__"
    assert cp.get_last(fork.execution_id).node_id != "__fork_origin__"
