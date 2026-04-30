"""Tests for the v1.0 durability CLI commands (Phase 9).

Covers ``fastaiagent list-pending``, ``fastaiagent inspect``,
``fastaiagent resume``, and ``fastaiagent setup-checkpointer`` —
all four commands wired into ``cli/main.py``.
"""

from __future__ import annotations

import json
import re
import textwrap
import uuid
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from fastaiagent import (
    Chain,
    FunctionTool,
    SQLiteCheckpointer,
    interrupt,
)
from fastaiagent.chain.node import NodeType
from fastaiagent.cli.main import app

runner = CliRunner()


# ---------- Shared fixture: chain that pauses on first run --------------


def _approval_fn(amount: str) -> dict[str, Any]:
    n = int(amount)
    if n > 1000:
        decision = interrupt(
            reason="manager_approval",
            context={"amount": n, "policy": "high-value"},
        )
        return {"approved": decision.approved, "approver": decision.metadata.get("approver")}
    return {"approved": True, "auto": True}


def _final_fn(approved: str) -> dict[str, Any]:
    return {"final_approved": str(approved).lower() in ("true", "1", "yes")}


def _build_chain(ckpt_db: str) -> Chain:
    chain = Chain(
        "cli-test-chain",
        checkpoint_enabled=True,
        checkpointer=SQLiteCheckpointer(db_path=ckpt_db),
    )
    chain.add_node(
        "approval",
        tool=FunctionTool(name="approval_tool", fn=_approval_fn),
        type=NodeType.tool,
        input_mapping={"amount": "{{state.amount}}"},
    )
    chain.add_node(
        "final",
        tool=FunctionTool(name="final_tool", fn=_final_fn),
        type=NodeType.tool,
        input_mapping={"approved": "{{state.output.approved}}"},
    )
    chain.connect("approval", "final")
    return chain


@pytest.fixture
def paused_db(tmp_path: Path) -> tuple[str, str]:
    """Build a chain, run it to pause, return (ckpt_db, execution_id)."""
    ckpt_db = str(tmp_path / "cli-cp.db")
    chain = _build_chain(ckpt_db)
    execution_id = f"cli-{uuid.uuid4().hex[:8]}"
    paused = chain.execute({"amount": 50_000}, execution_id=execution_id)
    assert paused.status == "paused"
    return ckpt_db, execution_id


# ---------- list-pending ------------------------------------------------


def test_list_pending_shows_paused_row(paused_db: tuple[str, str]) -> None:
    ckpt_db, execution_id = paused_db
    result = runner.invoke(app, ["list-pending", "--db-path", ckpt_db])
    assert result.exit_code == 0, result.output
    # Rich may truncate longer columns under a narrow test terminal, so
    # check landmarks that are short enough to render in full.
    assert "Pending Interrupts (1)" in result.output
    assert execution_id in result.output  # execution_id column is wide enough
    assert "approval" in result.output  # node_id column shows it in full


def test_list_pending_empty(tmp_path: Path) -> None:
    db = str(tmp_path / "empty.db")
    # Force schema creation so the read path works on a fresh DB.
    SQLiteCheckpointer(db_path=db).setup()
    result = runner.invoke(app, ["list-pending", "--db-path", db])
    assert result.exit_code == 0, result.output
    assert "No pending interrupts" in result.output


# ---------- inspect -----------------------------------------------------


def test_inspect_shows_checkpoint_history(paused_db: tuple[str, str]) -> None:
    ckpt_db, execution_id = paused_db
    result = runner.invoke(app, ["inspect", execution_id, "--db-path", ckpt_db])
    assert result.exit_code == 0, result.output
    assert "cli-test-chain" in result.output
    assert "approval" in result.output
    assert "interrupted" in result.output


def test_inspect_unknown_execution_exits_1(tmp_path: Path) -> None:
    db = str(tmp_path / "empty.db")
    SQLiteCheckpointer(db_path=db).setup()
    result = runner.invoke(app, ["inspect", "no-such-id", "--db-path", db])
    assert result.exit_code == 1


# ---------- resume ------------------------------------------------------


def _write_runner_module(tmp_path: Path, ckpt_db: str) -> str:
    """Write a tiny module that re-builds the chain so `--runner` can import it.

    The package name is unique per call so Python's import cache doesn't
    feed a stale module from a previous test into this one.
    """
    pkg_name = f"_resume_runner_pkg_{uuid.uuid4().hex[:8]}"
    pkg = tmp_path / pkg_name
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    module_file = pkg / "chain.py"
    module_file.write_text(
        textwrap.dedent(
            f"""
            from fastaiagent import Chain, FunctionTool, SQLiteCheckpointer
            from fastaiagent.chain.node import NodeType
            from tests.test_cli_resume import _approval_fn, _final_fn

            chain = Chain(
                "cli-test-chain",
                checkpoint_enabled=True,
                checkpointer=SQLiteCheckpointer(db_path={ckpt_db!r}),
            )
            chain.add_node(
                "approval",
                tool=FunctionTool(name="approval_tool", fn=_approval_fn),
                type=NodeType.tool,
                input_mapping={{"amount": "{{{{state.amount}}}}"}},
            )
            chain.add_node(
                "final",
                tool=FunctionTool(name="final_tool", fn=_final_fn),
                type=NodeType.tool,
                input_mapping={{"approved": "{{{{state.output.approved}}}}"}},
            )
            chain.connect("approval", "final")
            """
        )
    )
    return f"{pkg_name}.chain:chain"


def test_resume_completes_chain_via_cli(
    paused_db: tuple[str, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ckpt_db, execution_id = paused_db
    spec = _write_runner_module(tmp_path, ckpt_db)
    monkeypatch.syspath_prepend(str(tmp_path))

    result = runner.invoke(
        app,
        [
            "resume",
            execution_id,
            "--runner",
            spec,
            "--value",
            json.dumps({"approved": True, "metadata": {"approver": "alice"}}),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "status: completed" in result.output

    # The pending row was atomically claimed — list-pending no longer shows it.
    pending = runner.invoke(app, ["list-pending", "--db-path", ckpt_db])
    assert pending.exit_code == 0
    assert execution_id[:6] not in pending.output


def test_resume_double_call_returns_already_resumed(
    paused_db: tuple[str, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ckpt_db, execution_id = paused_db
    spec = _write_runner_module(tmp_path, ckpt_db)
    monkeypatch.syspath_prepend(str(tmp_path))

    first = runner.invoke(
        app,
        [
            "resume",
            execution_id,
            "--runner",
            spec,
            "--value",
            json.dumps({"approved": True}),
        ],
    )
    assert first.exit_code == 0, first.output

    second = runner.invoke(
        app,
        [
            "resume",
            execution_id,
            "--runner",
            spec,
            "--value",
            json.dumps({"approved": False}),
        ],
    )
    # Exit code 2 is what the resume command uses for AlreadyResumed.
    assert second.exit_code == 2, second.output
    assert "AlreadyResumed" in second.output


def test_resume_rejects_non_runner_module(
    paused_db: tuple[str, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, execution_id = paused_db
    # Point --runner at something that doesn't have .aresume.
    monkeypatch.syspath_prepend(str(tmp_path))
    pkg = tmp_path / "_bad_runner_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "thing.py").write_text("not_a_runner = 42\n")

    result = runner.invoke(
        app,
        [
            "resume",
            execution_id,
            "--runner",
            "_bad_runner_pkg.thing:not_a_runner",
            "--value",
            json.dumps({"approved": True}),
        ],
    )
    assert result.exit_code != 0
    assert "aresume" in result.output


# ---------- setup-checkpointer ------------------------------------------


def test_setup_checkpointer_sqlite(tmp_path: Path) -> None:
    db = str(tmp_path / "provisioned.db")
    result = runner.invoke(
        app,
        ["setup-checkpointer", "--backend", "sqlite", "--connection-string", db],
    )
    assert result.exit_code == 0, result.output
    assert "SQLite checkpointer ready" in result.output

    # Re-running is idempotent.
    result2 = runner.invoke(
        app,
        ["setup-checkpointer", "--backend", "sqlite", "--connection-string", db],
    )
    assert result2.exit_code == 0


def test_setup_checkpointer_postgres_requires_connection_string() -> None:
    result = runner.invoke(app, ["setup-checkpointer", "--backend", "postgres"])
    assert result.exit_code != 0
    plain = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
    assert "is required for the postgres backend" in plain


def test_setup_checkpointer_unknown_backend(tmp_path: Path) -> None:
    db = str(tmp_path / "x.db")
    result = runner.invoke(
        app,
        ["setup-checkpointer", "--backend", "redis", "--connection-string", db],
    )
    assert result.exit_code != 0
    assert "Unknown" in result.output
