"""A resumed or forked chain still sees earlier nodes' results (1.95.0, S15).

``resume`` and ``afork`` ran forward with an empty ``node_results``, so a node
whose input templates ``{{node_results.<earlier>.output}}`` got an empty value
and finished "completed" with the wrong result. The meeting-notes example's
``merge`` lost its analysis that way. Real chains, real SQLite checkpoints.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from fastaiagent.chain import Chain, NodeType
from fastaiagent.checkpointers import SQLiteCheckpointer
from fastaiagent.tool import FunctionTool


def _chain(tmp_path: Path, *, fail_merge_once: bool = False) -> tuple[Chain, SQLiteCheckpointer]:
    store = SQLiteCheckpointer(db_path=str(tmp_path / "ckpt.db"))
    store.setup()
    failures = [True] if fail_merge_once else []

    def extract(text: str) -> str:
        """Extract the facts."""
        return f"facts of {text}"

    def summarize(note: str) -> str:
        """Summarize."""
        return "summary"

    def merge(facts: str) -> str:
        """Merge the facts into notes."""
        if failures:
            failures.pop()
            raise RuntimeError("merge crashed")
        return f"notes from [{facts}]"

    chain = Chain("notes", checkpointer=store, checkpoint_enabled=True)
    chain.add_node(
        "extract",
        type=NodeType.tool,
        tool=FunctionTool(name="extract", fn=extract),
        input_mapping={"text": "{{state.text}}"},
    )
    # Overwrites state.output, so merge can only reach extract via node_results.
    chain.add_node(
        "summarize",
        type=NodeType.tool,
        tool=FunctionTool(name="summarize", fn=summarize),
        input_mapping={"note": "x"},
    )
    chain.add_node(
        "merge",
        type=NodeType.tool,
        tool=FunctionTool(name="merge", fn=merge),
        input_mapping={"facts": "{{node_results.extract.output}}"},
    )
    chain.connect("extract", "summarize")
    chain.connect("summarize", "merge")
    return chain, store


def _merged(result) -> str:
    return result.node_results["merge"]["output"]


def test_a_fork_keeps_the_results_of_the_nodes_before_it(tmp_path: Path) -> None:
    chain, store = _chain(tmp_path)
    original = chain.execute({"text": "the call"}, execution_id="run-1")
    assert _merged(original) == "notes from [facts of the call]"

    after_summarize = next(c for c in store.list("run-1") if c.node_id == "summarize")
    forked = chain.fork("run-1", checkpoint_id=after_summarize.checkpoint_id)
    assert forked.status == "completed"
    assert _merged(forked) == "notes from [facts of the call]"
    # The result still lists only what the fork ran; the restored results are
    # for templates and conditions inside the run.
    assert set(forked.node_results) == {"merge"}


def test_a_resume_after_a_crash_keeps_the_earlier_results(tmp_path: Path) -> None:
    chain, _ = _chain(tmp_path, fail_merge_once=True)
    with pytest.raises(Exception, match="merge crashed"):
        chain.execute({"text": "the call"}, execution_id="run-2")

    resumed = asyncio.run(chain.resume("run-2"))
    assert resumed.status == "completed"
    assert _merged(resumed) == "notes from [facts of the call]"
    assert set(resumed.node_results) == {"merge"}, "only what the resume ran"
