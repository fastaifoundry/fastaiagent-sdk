"""A resumed Swarm keeps a multimodal request intact (1.99.0).

``Swarm.aresume`` read the request back with ``str(snapshot["current_input"])``,
but the snapshot stores an Image/PDF as a JSON-safe dict. Every agent after the
resume was briefed with that dict's text — the photo became a ``repr`` of base64
— instead of the original parts.

No mocks: the offline ``FunctionModel`` from ``fastaiagent.testing``, a real
``SQLiteCheckpointer`` and a real image fixture.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import fastaiagent as fa
from fastaiagent import Image
from fastaiagent.chain.interrupt import Resume, interrupt
from fastaiagent.checkpointers import SQLiteCheckpointer
from fastaiagent.testing import FunctionModel
from fastaiagent.tool import FunctionTool

PHOTO = Path(__file__).parent / "fixtures" / "multimodal" / "cat.jpg"


def _approve(note: str) -> str:
    decision = interrupt("adjuster_approval", {"note": note})
    return "approved" if decision.approved else "rejected"


def test_the_agent_after_a_resume_still_gets_the_photo(tmp_path: Path) -> None:
    seen: list[Any] = []

    def triage(messages: list[Any]) -> Any:
        return "", [{"name": "handoff_to_assessor", "arguments": {"reason": "photo claim"}}]

    def assessor(messages: list[Any]) -> Any:
        if messages[-1].role.value == "tool":
            return "", [{"name": "handoff_to_payer", "arguments": {"reason": "approved"}}]
        return "", [{"name": "approve", "arguments": {"note": "dent on bumper"}}]

    def payer(messages: list[Any]) -> str:
        seen.append(messages[-1].content)
        return "paid"

    swarm = fa.Swarm(
        name="claims",
        agents=[
            fa.Agent(name="triage", llm=FunctionModel(triage)),
            fa.Agent(
                name="assessor",
                llm=FunctionModel(assessor),
                tools=[FunctionTool(name="approve", fn=_approve)],
            ),
            fa.Agent(name="payer", llm=FunctionModel(payer)),
        ],
        entrypoint="triage",
        checkpointer=SQLiteCheckpointer(str(tmp_path / "ckpt.db")),
    )

    paused = swarm.run(["Assess this car damage claim.", Image.from_file(PHOTO)])
    assert paused.status == "paused"

    done = swarm.resume(paused.execution_id, resume_value=Resume(approved=True))

    assert done.status == "completed" and done.output == "paid"
    parts = seen[-1]
    assert isinstance(parts, list), f"the payer was briefed with text only: {str(parts)[:120]}"
    assert any(isinstance(p, Image) for p in parts)
    assert any(isinstance(p, str) and "Assess this car damage claim." in p for p in parts)


def test_crash_recovery_re_runs_the_agent_with_the_photo(tmp_path: Path) -> None:
    """The other resume path: no pause, the run died mid-hop, and ``resume``
    re-runs the active agent with the request it had been given."""
    seen: list[Any] = []
    calls = {"n": 0}

    def triage(messages: list[Any]) -> Any:
        return "", [{"name": "handoff_to_assessor", "arguments": {"reason": "photo claim"}}]

    def assessor(messages: list[Any]) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("provider went away")
        seen.append(messages[-1].content)
        return "dent on bumper, approve"

    swarm = fa.Swarm(
        name="claims-crash",
        agents=[
            fa.Agent(name="triage", llm=FunctionModel(triage)),
            fa.Agent(name="assessor", llm=FunctionModel(assessor)),
        ],
        entrypoint="triage",
        checkpointer=SQLiteCheckpointer(str(tmp_path / "ckpt.db")),
    )
    execution_id = "claims-crash-1"
    try:
        swarm.run(
            ["Assess this car damage claim.", Image.from_file(PHOTO)], execution_id=execution_id
        )
    except Exception:
        pass  # the crash under test

    done = swarm.resume(execution_id)

    assert done.output == "dent on bumper, approve"
    assert isinstance(seen[-1], list) and any(isinstance(p, Image) for p in seen[-1])
