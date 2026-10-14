"""The replay tool plan: what a rerun may run, and how to allow more (1.89.0).

Default ``block``: a rerun runs only tools you supply (``with_tool_override``,
``with_tool_stub``) or allow (``with_tool_policy``). REST/MCP tools rebuilt from
the trace run only when named in ``allow``, and never with redacted credentials.
Stubs and callable overrides never touch the global tool registry.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest

from fastaiagent._internal.errors import ReplayError
from tests._replay_isolation_harness import (
    hits,
    prepare_world,
    script_model,
    side_lines,
    spans_of,
    teardown_world,
)


@pytest.fixture
def world(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capture_server: Any, decisions_stub: Any
) -> Any:
    try:
        yield prepare_world(tmp_path, monkeypatch, capture_server, decisions_stub)
    finally:
        teardown_world()


def _fork(w: Any) -> Any:
    from fastaiagent.trace.replay import Replay

    script_model(w.stub)
    return Replay.load(w.original_trace_id).fork_at(0)


def test_by_default_every_tool_is_blocked_and_counted(world: Any) -> None:
    result = _fork(world).with_determinism("recorded").rerun()
    assert result.tool_dispositions == {
        "charge_card": "blocked",
        "issue_refund": "blocked",
        "notify": "blocked",
    }
    assert result.blocked_tool_calls == {"charge_card": 1, "issue_refund": 1, "notify": 1}
    assert len(side_lines(world)) == 1 and hits(world, "/refunds") == 1
    assert result.source_trace_id == world.original_trace_id
    assert result.determinism == "recorded" and result.exported is False


def test_live_policy_runs_function_tools_but_not_rebuilt_endpoints(world: Any) -> None:
    result = _fork(world).with_determinism("recorded").with_tool_policy("live").rerun()
    assert result.tool_dispositions["charge_card"] == "live"
    assert result.tool_dispositions["issue_refund"] == "blocked"
    assert len(side_lines(world)) == 2, "live policy must run the function tool exactly once"
    assert hits(world, "/refunds") == 1, "a REST tool rebuilt from the trace must not re-fire"


def test_allowing_a_rest_tool_runs_it(world: Any) -> None:
    result = (
        _fork(world)
        .with_determinism("recorded")
        .with_tool_policy("block", allow=["issue_refund"])
        .rerun()
    )
    assert result.tool_dispositions["issue_refund"] == "live"
    assert result.tool_dispositions["charge_card"] == "blocked"
    assert hits(world, "/refunds") == 2


def test_a_rest_tool_with_redacted_credentials_is_never_live(world: Any) -> None:
    from fastaiagent.tool.rest import RESTTool

    forked = _fork(world).with_tool_policy("live", allow=["secure_api"])
    spec = RESTTool(
        name="secure_api", url=f"{world.server.url}/secure", headers={"Authorization": "Bearer x"}
    ).to_dict()
    tools, dispositions, blocked = forked._build_tool_plan([spec])
    assert dispositions == {"secure_api": "blocked"}
    assert "redacted" in blocked["secure_api"].reason


def test_policy_error_refuses_before_anything_runs_in_recorded_mode(world: Any) -> None:
    import sqlite3

    with sqlite3.connect(world.db) as conn:
        before = conn.execute("SELECT COUNT(*) FROM spans").fetchone()[0]
    with pytest.raises(ReplayError, match="charge_card"):
        _fork(world).with_determinism("recorded").with_tool_policy("error").rerun()
    with sqlite3.connect(world.db) as conn:
        after = conn.execute("SELECT COUNT(*) FROM spans").fetchone()[0]
    assert after == before, "the refused rerun still wrote spans"


def test_policy_error_raises_after_a_live_rerun_calls_a_blocked_tool(world: Any) -> None:
    with pytest.raises(ReplayError, match="tool policy 'error'"):
        _fork(world).with_tool_policy("error").rerun()
    assert len(side_lines(world)) == 1


def test_a_callable_override_is_wrapped_and_never_registered(world: Any) -> None:
    from fastaiagent.tool.registry import ToolRegistry

    real = ToolRegistry.get("charge_card")
    result = (
        _fork(world)
        .with_determinism("recorded")
        .with_tool_override("charge_card", lambda **_: "stubbed")
        .rerun()
    )
    assert result.tool_dispositions["charge_card"] == "override"
    assert ToolRegistry.get("charge_card") is real, "the override took over the global registry"
    assert len(side_lines(world)) == 1


def test_a_stub_keeps_its_function_and_never_registers(world: Any) -> None:
    from fastaiagent.tool.registry import ToolRegistry

    real = ToolRegistry.get("charge_card")
    forked = _fork(world).with_tool_stub("charge_card", {"ok": True})
    stub = forked._tool_overrides["charge_card"]
    assert stub.fn() == {"ok": True}
    assert ToolRegistry.get("charge_card") is real
    result = forked.with_determinism("recorded").rerun()
    assert result.tool_dispositions["charge_card"] == "stub"


def test_a_globally_registered_override_warns(world: Any, caplog: pytest.LogCaptureFixture) -> None:
    import fastaiagent as fa

    def stub_charge(amount: float) -> str:
        return "fake"

    registered = fa.FunctionTool(name="charge_card", fn=stub_charge)  # registers globally
    try:
        with caplog.at_level(logging.WARNING):
            _fork(world).with_tool_override("charge_card", registered)
        assert "registered globally" in caplog.text
    finally:
        fa.FunctionTool(name="charge_card", fn=world.agent.tools[0].fn)  # restore


def test_export_is_an_opt_in(world: Any) -> None:
    result = _fork(world).with_determinism("recorded").with_export(True).rerun()
    spans = spans_of(world.db, str(result.trace_id))
    assert result.exported is True
    assert all(s["synced"] == 0 for s in spans)
    assert all(s["attributes"].get("fastaiagent.source") == "replay" for s in spans)


def test_the_env_escape_hatch_restores_live_tools(
    world: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FASTAIAGENT_REPLAY_TOOL_POLICY", "live")
    result = _fork(world).with_determinism("recorded").rerun()
    assert result.tool_dispositions["charge_card"] == "live"
    assert len(side_lines(world)) == 2


def test_an_unknown_env_policy_warns_and_blocks(
    world: Any, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("FASTAIAGENT_REPLAY_TOOL_POLICY", "lvie")
    with caplog.at_level(logging.WARNING):
        result = _fork(world).with_determinism("recorded").rerun()
    assert "FASTAIAGENT_REPLAY_TOOL_POLICY='lvie'" in caplog.text
    assert result.tool_dispositions["charge_card"] == "blocked"
    assert len(side_lines(world)) == 1


@pytest.mark.parametrize(
    ("env", "flag", "expected"),
    [
        (None, [], "blocked"),
        ("live", [], "live"),
        ("live", ["--tools", "block"], "blocked"),
    ],
)
def test_the_cli_honours_the_env_policy_unless_told_otherwise(
    world: Any, monkeypatch: pytest.MonkeyPatch, env: str | None, flag: list[str], expected: str
) -> None:
    from typer.testing import CliRunner

    from fastaiagent.cli.main import app

    if env is None:
        monkeypatch.delenv("FASTAIAGENT_REPLAY_TOOL_POLICY", raising=False)
    else:
        monkeypatch.setenv("FASTAIAGENT_REPLAY_TOOL_POLICY", env)
    script_model(world.stub)
    out = world.db.parent / "rerun.json"
    args = ["replay", "fork", world.original_trace_id, "--step", "0", "--mode", "recorded"]
    res = CliRunner().invoke(app, [*args, "--output", str(out), *flag])
    assert res.exit_code == 0, res.output
    assert json.loads(out.read_text())["tool_dispositions"]["charge_card"] == expected


def test_no_modified_text_reaches_the_replay_attributes(world: Any) -> None:
    secret = "SECRET-PROMPT-TEXT-xyz"
    result = (
        _fork(world)
        .modify_prompt(secret)
        .modify_input("SECRET-INPUT-abc")
        .with_determinism("recorded")
        .rerun()
    )
    root = next(s for s in spans_of(world.db, str(result.trace_id)) if not s["parent_span_id"])
    replay_attrs = {
        k: v for k, v in root["attributes"].items() if k.startswith("fastaiagent.replay.")
    }
    blob = json.dumps(replay_attrs)
    assert "SECRET" not in blob
    assert set(json.loads(replay_attrs["fastaiagent.replay.modifications"])) == {"prompt", "input"}
