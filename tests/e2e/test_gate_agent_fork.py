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
