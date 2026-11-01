"""The CLI tells a script what happened (1.97.0).

A coding agent or a CI job reads a command's exit code and parses its stdout.
Each case here once exited ``0`` on a failure, or printed "JSON" that did not
parse: a mistyped KB name created an empty KB, a mistyped path became a
document, ``connect`` to a dead or wrong address printed "✓ Connected",
``learn`` exited 0 with every model call failed, ``--format csv`` printed JSON.

Real commands through Typer's runner, real SQLite, a real local HTTP server for
``connect``; no mocks. ``traces prune`` has its own case in
``test_connected_visibility.py``.
"""

from __future__ import annotations

import json
import socket
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import fastaiagent as fa
from fastaiagent.cli import auth as auth_cli
from fastaiagent.cli.main import app
from fastaiagent.testing import TestModel

runner = CliRunner()


def _recorded_trace() -> str:
    from fastaiagent.trace import otel

    otel.reset()
    try:
        result = fa.Agent(name="greeter", llm=TestModel(response="hi")).run("q")
        otel.get_tracer_provider().force_flush()
    finally:
        otel.reset()
    assert result.trace_id
    return result.trace_id


# ── connect ──────────────────────────────────────────────────────────────────


@pytest.fixture
def credentials(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    path = tmp_path / "home" / ".fastaiagent" / "credentials.toml"
    monkeypatch.setattr(auth_cli, "_CREDENTIALS_PATH", path)
    for name in ("FASTAIAGENT_API_KEY", "FASTAIAGENT_TARGET"):
        monkeypatch.delenv(name, raising=False)
    yield path
    fa.disconnect()


def _dead_url() -> str:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return f"http://127.0.0.1:{s.getsockname()[1]}"


def test_connect_to_an_unreachable_plane_saves_but_says_unverified(
    isolated_local_db: Path, credentials: Path
) -> None:
    result = runner.invoke(app, ["connect", "--api-key", "fa_k_test", "--target", _dead_url()])
    assert result.exit_code == 1, result.output
    assert "Connected" not in result.stdout
    assert "not verified" in result.stderr
    assert credentials.exists()


@pytest.fixture
def not_a_plane() -> Any:
    """A web server that answers 404 to everything, as one that is not a plane does.

    Not ``capture_server`` with one scripted 404: on CI another request can take
    that 404 first, and the key check then gets a 200.
    """
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    paths: list[str] = []

    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:
            pass

        def _answer(self) -> None:
            paths.append(self.path)
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()

        do_GET = _answer  # noqa: N815
        do_POST = _answer  # noqa: N815

    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", paths
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_connect_to_a_server_that_is_not_a_plane_saves_nothing(
    isolated_local_db: Path, credentials: Path, not_a_plane: Any
) -> None:
    url, paths = not_a_plane
    result = runner.invoke(app, ["connect", "--api-key", "fa_k_test", "--target", url])
    assert result.exit_code == 1, result.output
    assert "/public/v1/auth/check" in paths
    assert "HTTP 404" in result.stderr and "Nothing was saved" in result.stderr
    assert not credentials.exists()


def test_connect_to_a_plane_that_verifies_the_key(
    isolated_local_db: Path, credentials: Path, capture_server: Any
) -> None:
    capture_server.set_response_sequence([(200, {"domain_id": "d", "project_id": "p"})])
    result = runner.invoke(
        app, ["connect", "--api-key", "fa_k_test", "--target", capture_server.url]
    )
    assert result.exit_code == 0, result.output
    assert "✓ Connected" in result.stdout
    assert credentials.exists()


# ── kb ───────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "argv",
    [
        ["kb", "status", "--name", "nope"],
        ["kb", "clear", "--name", "nope"],
        ["kb", "delete", "a.md", "--name", "nope"],
    ],
)
def test_kb_commands_refuse_a_kb_that_does_not_exist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, argv: list[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, argv)
    assert result.exit_code == 1, result.output
    assert "No KB named 'nope'" in result.stderr
    assert not (tmp_path / ".fastaiagent" / "kb" / "nope").exists()  # nothing created


def test_kb_add_refuses_a_path_that_does_not_exist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["kb", "add", "nonexistent_dir"])
    assert result.exit_code == 1, result.output
    assert "No such file or directory: nonexistent_dir" in result.stderr
    assert not (tmp_path / ".fastaiagent" / "kb").exists()


# ── learn ────────────────────────────────────────────────────────────────────


def test_learn_exits_1_when_the_extraction_fails(
    isolated_local_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _recorded_trace()
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)  # every model call fails
    result = runner.invoke(app, ["learn", "--max-traces", "1"])
    assert result.exit_code == 1, result.output
    assert "1 of 1 traces failed" in result.stderr


def test_the_hidden_learn_extract_alias_is_a_usage_error() -> None:
    result = runner.invoke(app, ["learn", "extract"])
    assert result.exit_code == 2, result.output
    assert "fastaiagent learn" in result.stderr


# ── output that says it is JSON ─────────────────────────────────────────────


def test_traces_export_refuses_a_format_it_does_not_have(isolated_local_db: Path) -> None:
    result = runner.invoke(app, ["traces", "export", "any", "--format", "csv"])
    assert result.exit_code == 2, result.output
    assert "only 'json'" in result.stderr


def test_replay_fork_prints_json(isolated_local_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    trace_id = _recorded_trace()
    for name in ("FASTAIAGENT_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    result = runner.invoke(app, ["replay", "fork", trace_id, "--mode", "recorded"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["new_output"] == "hi" and payload["trace_id"]


# ── eval run: a run that could not happen is invalid (3), not failed (1) ─────


def test_eval_run_with_a_missing_dataset_is_invalid(
    isolated_local_db: Path, tmp_path: Path
) -> None:
    agent_file = tmp_path / "agent_mod.py"
    agent_file.write_text("def agent(q):\n    return q\n")
    missing = str(tmp_path / "missing.jsonl")
    result = runner.invoke(
        app, ["eval", "run", "--dataset", missing, "--agent", f"{agent_file}:agent"]
    )
    assert result.exit_code == 3, result.output
    assert "dataset not found" in result.stderr
    assert "Traceback" not in result.output


def test_eval_run_with_an_agent_that_fails_to_import_is_invalid(
    isolated_local_db: Path, tmp_path: Path
) -> None:
    data = tmp_path / "d.jsonl"
    data.write_text('{"input": "q", "expected_output": "q"}\n')
    agent_file = tmp_path / "bad_agent_mod.py"
    agent_file.write_text("import a_module_that_does_not_exist_197\nagent = None\n")
    result = runner.invoke(
        app, ["eval", "run", "--dataset", str(data), "--agent", f"{agent_file}:agent"]
    )
    assert result.exit_code == 3, result.output
    assert "could not load the agent" in result.stderr
    assert "ModuleNotFoundError" in result.stderr
