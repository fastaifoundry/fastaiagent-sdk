"""Platform-initiated runs are durable and joinable (durability audit D4b).

``grep -c checkpoint fastaiagent/runner/execute.py`` returned **0**. Every run the
plane dispatched — every live-playground click, every eval case — ran with no
checkpointer and no ``execution_id``: non-durable, unresumable, and invisible in
the plane's Durability view. The plane could dispatch work and then hold no
record of its state.

Two things had to be true, and each is a test below: the run must be
**checkpointed**, and it must be checkpointed under an id the plane already knows
— its own ``command_id`` — because that is what joins the replica back to the
command without a wire change. ``CommandResult`` is the frozen shape reported to
the plane; adding a field to it would not be.

NO MOCKS of the thing under test: the real ``execute_command`` entry point, real
``Agent.from_dict`` reconstruction from a real ``to_dict`` payload, real
``SQLiteCheckpointer`` storage. The model is the repo's ``MockLLMClient`` — a
real ``LLMClient`` subclass, not ``unittest.mock`` — because none of this depends
on what a model says.
"""

from __future__ import annotations

import uuid

import pytest

from fastaiagent import Agent, SQLiteCheckpointer
from fastaiagent.runner.execute import execute_command


@pytest.fixture
def runner_db(tmp_path, monkeypatch):
    """Point the runner's default checkpointer at a throwaway database.

    The runner builds ``SQLiteCheckpointer()`` with no path — it uses whatever
    the config resolves — so this is the seam that keeps the test isolated
    without reaching into the function.
    """
    from fastaiagent._internal.config import reset_config

    db = tmp_path / "runner.db"
    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(db))
    reset_config()
    yield db
    reset_config()


def _agent_config(mock_llm) -> dict:
    agent = Agent(
        name="runner-durability-probe",
        system_prompt="Answer in one word.",
        llm=mock_llm,
    )
    cfg = agent.to_dict()
    # ``from_dict`` rebuilds the client from this; keep the real endpoint shape.
    cfg["llm_endpoint"] = mock_llm.to_dict()
    return cfg


def _rows(db, execution_id: str) -> list[dict]:
    store = SQLiteCheckpointer(db_path=str(db))
    store.setup()
    try:
        return [
            {"node_id": c.node_id, "step_type": c.step_type, "status": c.status}
            for c in store.list(execution_id)
        ]
    finally:
        store.close()


@pytest.mark.asyncio
async def test_live_playground_checkpoints_under_the_command_id(
    runner_db, mock_llm, monkeypatch
) -> None:
    """The join key. Without it the replica cannot be tied to the command."""
    monkeypatch.setattr(
        "fastaiagent.llm.client.LLMClient.from_dict", staticmethod(lambda _d: mock_llm)
    )
    command_id = f"cmd-{uuid.uuid4().hex[:8]}"
    result = await execute_command(
        {
            "type": "live_playground",
            "command_id": command_id,
            "payload": {"agent": _agent_config(mock_llm), "input": "hello"},
        }
    )

    assert result.status == "completed"
    rows = _rows(runner_db, command_id)
    assert rows, "a platform-initiated run wrote no checkpoints"
    assert any(r["step_type"] == "llm_call" for r in rows), rows
    # The run-end marker from D5 rides along, so the plane can see it finished.
    assert rows[-1]["step_type"] == "run_end" and rows[-1]["status"] == "completed"


@pytest.mark.asyncio
async def test_each_eval_case_is_its_own_execution(runner_db, mock_llm, monkeypatch) -> None:
    """A suite is N runs, not one.

    Collapsing them onto one execution_id would interleave their checkpoints
    into a history no resume could read, and the console would show one run
    where the operator dispatched several.
    """
    monkeypatch.setattr(
        "fastaiagent.llm.client.LLMClient.from_dict", staticmethod(lambda _d: mock_llm)
    )
    command_id = f"cmd-{uuid.uuid4().hex[:8]}"
    result = await execute_command(
        {
            "type": "eval_run",
            "command_id": command_id,
            "payload": {
                "agent": _agent_config(mock_llm),
                "cases": [
                    {"case_id": "case-a", "input": "first"},
                    {"case_id": "case-b", "input": "second"},
                ],
            },
        }
    )

    assert result.status == "completed"
    assert [o["case_id"] for o in result.result["outputs"]] == ["case-a", "case-b"]
    for case_id in ("case-a", "case-b"):
        rows = _rows(runner_db, f"{command_id}-{case_id}")
        assert rows, f"{case_id} wrote no checkpoints"
        assert rows[-1]["step_type"] == "run_end"
    # And the two histories are genuinely separate.
    assert _rows(runner_db, command_id) == []


@pytest.mark.asyncio
async def test_the_off_switch_restores_the_old_footprint(runner_db, mock_llm, monkeypatch) -> None:
    """This turns on disk writes and plane ingest on hosts that had neither.

    An operator has to be able to say no, and saying no must still run the job.
    """
    monkeypatch.setattr(
        "fastaiagent.llm.client.LLMClient.from_dict", staticmethod(lambda _d: mock_llm)
    )
    monkeypatch.setenv("FASTAIAGENT_RUNNER_CHECKPOINTS", "0")
    command_id = f"cmd-{uuid.uuid4().hex[:8]}"
    result = await execute_command(
        {
            "type": "live_playground",
            "command_id": command_id,
            "payload": {"agent": _agent_config(mock_llm), "input": "hello"},
        }
    )

    assert result.status == "completed", "the off switch must not break the job"
    assert _rows(runner_db, command_id) == []


@pytest.mark.asyncio
async def test_a_command_with_no_id_still_runs(runner_db, mock_llm, monkeypatch) -> None:
    """Durable locally, just not joinable — never a failure."""
    monkeypatch.setattr(
        "fastaiagent.llm.client.LLMClient.from_dict", staticmethod(lambda _d: mock_llm)
    )
    result = await execute_command(
        {
            "type": "live_playground",
            "payload": {"agent": _agent_config(mock_llm), "input": "hello"},
        }
    )
    assert result.status == "completed"


@pytest.mark.asyncio
async def test_tool_exec_is_untouched(runner_db, monkeypatch) -> None:
    """One tool call, no agent loop, nothing to checkpoint.

    Pinned because ``tool_exec`` is the hosted-MCP hot path: giving it a
    checkpointer would add a database write to every remote tool call for a
    run state that does not exist.
    """
    from fastaiagent.tool.function import FunctionTool
    from fastaiagent.tool.registry import ToolRegistry

    def ping(value: str) -> str:
        return f"pong:{value}"

    ToolRegistry.register(FunctionTool(name="zz_runner_ping", fn=ping))
    command_id = f"cmd-{uuid.uuid4().hex[:8]}"
    result = await execute_command(
        {
            "type": "tool_exec",
            "command_id": command_id,
            "payload": {
                "tool_exec": {
                    "tool_type": "connector",
                    "exposed_name": "zz_runner_ping",
                    "arguments": {"value": "x"},
                    "connector": {},
                }
            },
        }
    )

    assert result.status == "completed"
    assert result.result["success"] is True
    assert _rows(runner_db, command_id) == []
