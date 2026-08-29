"""Tests for `fastaiagent ui` CLI subcommands.

We avoid starting a real uvicorn — the start command itself is too invasive to
unit-test end-to-end (it blocks the process). Instead we test:

- Missing-extras path (friendly error).
- First-run credential prompt writes a valid auth.json.
- `reset-password` removes the file.
- The `ui` subcommand is registered in the main CLI's help output.
- Legacy-migration autoprompt runs on first start.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# The `ui start` command lazy-imports fastapi + uvicorn + bcrypt + itsdangerous.
# Running these tests without the `[ui]` extra would exit non-zero with a
# ModuleNotFoundError inside the Typer runner; skip the whole module instead.
pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")
pytest.importorskip("bcrypt")
pytest.importorskip("itsdangerous")

from typer.testing import CliRunner  # noqa: E402

from fastaiagent.cli.main import app as main_app  # noqa: E402
from fastaiagent.cli.ui import ui_app

runner = CliRunner()


class TestMainAppWiring:
    def test_ui_subcommand_listed(self):
        result = runner.invoke(main_app, ["--help"])
        assert result.exit_code == 0
        assert "ui" in result.output.lower()


class TestResetPassword:
    def test_removes_existing_auth_file(self, tmp_path: Path):
        auth_path = tmp_path / "auth.json"
        auth_path.write_text(json.dumps({"username": "x"}))
        result = runner.invoke(ui_app, ["reset-password", "--auth-file", str(auth_path)])
        assert result.exit_code == 0
        assert not auth_path.exists()

    def test_noop_when_absent(self, tmp_path: Path):
        auth_path = tmp_path / "auth.json"
        result = runner.invoke(ui_app, ["reset-password", "--auth-file", str(auth_path)])
        assert result.exit_code == 0
        assert "did not exist" in result.output.lower()


class TestStartFirstRun:
    """The first-run prompt writes auth.json and then tries to run uvicorn.

    We intercept the uvicorn.run call to avoid actually starting a server —
    but the prompt itself (bcrypt, auth.json) is exercised for real.
    """

    def test_prompts_and_writes_auth_file(
        self, monkeypatch, tmp_path: Path, capsys
    ):
        import fastaiagent.cli.ui as ui_module

        calls: dict[str, object] = {}

        def fake_run(app, host, port, log_level):  # noqa: ARG001
            calls["host"] = host
            calls["port"] = port

        monkeypatch.setattr("uvicorn.run", fake_run)
        monkeypatch.setattr(
            "getpass.getpass",
            lambda prompt="": "correct-horse-battery-staple",
        )
        # Suppress browser open
        monkeypatch.setattr(
            "webbrowser.open_new_tab", lambda url: None  # noqa: ARG005
        )

        auth_path = tmp_path / "auth.json"
        db_path = tmp_path / "local.db"

        result = runner.invoke(
            ui_app,
            [
                "start",
                "--auth-file",
                str(auth_path),
                "--db",
                str(db_path),
                "--no-open",
                "--port",
                "7999",
            ],
            input="testuser\n",
        )

        assert result.exit_code == 0, result.output
        assert auth_path.exists()
        payload = json.loads(auth_path.read_text())
        assert payload["username"] == "testuser"
        # bcrypt-hashed
        assert payload["password_hash"].startswith("$2")
        assert calls["host"] == "127.0.0.1"
        assert calls["port"] == 7999

        # Touched = unused binding silenced
        _ = ui_module

    def test_no_auth_skips_prompt(self, monkeypatch, tmp_path: Path):
        def fake_run(app, host, port, log_level):  # noqa: ARG001
            return None

        monkeypatch.setattr("uvicorn.run", fake_run)

        auth_path = tmp_path / "auth.json"
        db_path = tmp_path / "local.db"

        result = runner.invoke(
            ui_app,
            [
                "start",
                "--auth-file",
                str(auth_path),
                "--db",
                str(db_path),
                "--no-auth",
                "--no-open",
            ],
        )
        assert result.exit_code == 0, result.output
        # No auth.json should have been created.
        assert not auth_path.exists()

    def test_existing_auth_file_skips_prompt(self, monkeypatch, tmp_path: Path):
        from fastaiagent.ui.auth import create_auth_file

        def fake_run(app, host, port, log_level):  # noqa: ARG001
            return None

        monkeypatch.setattr("uvicorn.run", fake_run)

        auth_path = tmp_path / "auth.json"
        create_auth_file("upendra", "secret", path=auth_path)
        db_path = tmp_path / "local.db"

        result = runner.invoke(
            ui_app,
            [
                "start",
                "--auth-file",
                str(auth_path),
                "--db",
                str(db_path),
                "--no-open",
            ],
        )
        assert result.exit_code == 0, result.output
        payload = json.loads(auth_path.read_text())
        assert payload["username"] == "upendra"


class TestAutoMigrate:
    def test_legacy_files_are_migrated_on_start(
        self, monkeypatch, tmp_path: Path
    ):
        """Seed a legacy traces.db, run `ui start`, verify rows land in local.db."""
        from fastaiagent._internal.storage import SQLiteHelper

        legacy_root = tmp_path / "legacy"
        legacy_root.mkdir()
        monkeypatch.chdir(legacy_root)

        legacy_dir = legacy_root / ".fastaiagent"
        legacy_dir.mkdir()
        legacy_traces = legacy_dir / "traces.db"
        with SQLiteHelper(legacy_traces) as db:
            db.execute(
                """CREATE TABLE spans (
                    span_id TEXT PRIMARY KEY,
                    trace_id TEXT NOT NULL,
                    parent_span_id TEXT,
                    name TEXT,
                    start_time TEXT,
                    end_time TEXT,
                    status TEXT,
                    attributes TEXT,
                    events TEXT
                )"""
            )
            db.execute(
                "INSERT INTO spans (span_id, trace_id, name) VALUES (?, ?, ?)",
                ("s1", "t1", "legacy"),
            )

        auth_path = legacy_root / "auth.json"
        db_path = legacy_root / "local.db"

        def fake_run(app, host, port, log_level):  # noqa: ARG001
            return None

        monkeypatch.setattr("uvicorn.run", fake_run)

        result = runner.invoke(
            ui_app,
            [
                "start",
                "--auth-file",
                str(auth_path),
                "--db",
                str(db_path),
                "--no-auth",
                "--no-open",
            ],
        )
        assert result.exit_code == 0, result.output

        with SQLiteHelper(db_path) as db:
            rows = db.fetchall("SELECT span_id FROM spans")
        assert any(r["span_id"] == "s1" for r in rows)


# ─── --agent: register real agents with the server ────────────────────────
#
# `fastaiagent ui` historically called build_app() without `runners=`, so
# ctx.runners was always empty: approval resume 503'd, dataset eval could
# only echo, and the agents directory couldn't list an agent until it had
# run. These cover the flag that fills it.


_AGENT_MODULE = '''
from fastaiagent import Agent
from fastaiagent.testing import TestModel

agent = Agent(name="support-bot", llm=TestModel(response="ok"))
other = Agent(name="billing-bot", llm=TestModel(response="ok"))
not_a_runner = {"nope": True}
'''


@pytest.fixture
def agent_module(tmp_path: Path) -> Path:
    path = tmp_path / "my_agents.py"
    path.write_text(_AGENT_MODULE)
    return path


class TestLoadRunners:
    def test_resolves_a_file_target(self, agent_module: Path):
        from fastaiagent.cli.ui import _load_runners

        runners = _load_runners([f"{agent_module}:agent"])
        assert [r.name for r in runners] == ["support-bot"]

    def test_result_is_accepted_by_build_app(self, agent_module: Path, tmp_path: Path):
        """The contract we validate locally must match build_app's."""
        from fastaiagent.cli.ui import _load_runners
        from fastaiagent.ui.server import build_app

        runners = _load_runners([f"{agent_module}:agent", f"{agent_module}:other"])
        app = build_app(
            db_path=str(tmp_path / "cli.db"), no_auth=True, runners=runners
        )
        assert set(app.state.context.runners) == {"support-bot", "billing-bot"}

    def test_resolves_a_dotted_module_target(self, tmp_path: Path, monkeypatch):
        """`pkg.module:attr` is documented in the help text, so cover it."""
        from fastaiagent.cli.ui import _load_runners

        pkg = tmp_path / "cli_agent_pkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (pkg / "defs.py").write_text(_AGENT_MODULE)
        monkeypatch.syspath_prepend(str(tmp_path))

        runners = _load_runners(["cli_agent_pkg.defs:agent"])
        assert [r.name for r in runners] == ["support-bot"]

    def test_empty_and_none_are_noops(self):
        from fastaiagent.cli.ui import _load_runners

        assert _load_runners(None) == []
        assert _load_runners([]) == []

    def test_spec_without_colon_is_reported_not_raised(self, capsys):
        from fastaiagent.cli.ui import _load_runners

        assert _load_runners(["nocolon"]) == []
        assert "nocolon" in capsys.readouterr().out

    def test_missing_attribute_is_reported_not_raised(self, agent_module: Path, capsys):
        from fastaiagent.cli.ui import _load_runners

        assert _load_runners([f"{agent_module}:nope"]) == []
        assert "nope" in capsys.readouterr().out

    def test_module_that_raises_at_import_does_not_propagate(self, tmp_path: Path, capsys):
        from fastaiagent.cli.ui import _load_runners

        bad = tmp_path / "explodes.py"
        bad.write_text("raise RuntimeError('boom at import')")
        assert _load_runners([f"{bad}:agent"]) == []
        assert "boom at import" in capsys.readouterr().out

    def test_non_runner_target_is_rejected_before_build_app(
        self, agent_module: Path, capsys
    ):
        """build_app raises on a bad entry; we must catch it first so a typo
        can't abort startup after the password prompt."""
        from fastaiagent.cli.ui import _load_runners

        assert _load_runners([f"{agent_module}:not_a_runner"]) == []
        out = capsys.readouterr().out
        assert "dict" in out
        assert "Supervisor" in out  # names the accepted types

    def test_duplicate_names_collapse_to_one(self, agent_module: Path, capsys):
        from fastaiagent.cli.ui import _load_runners

        runners = _load_runners([f"{agent_module}:agent", f"{agent_module}:agent"])
        assert [r.name for r in runners] == ["support-bot"]
        assert "duplicate" in capsys.readouterr().out.lower()

    def test_good_targets_survive_alongside_bad_ones(self, agent_module: Path):
        from fastaiagent.cli.ui import _load_runners

        runners = _load_runners(
            ["nocolon", f"{agent_module}:agent", f"{agent_module}:missing"]
        )
        assert [r.name for r in runners] == ["support-bot"]


class TestAgentFlagWiring:
    """The flag must reach build_app from both entry points."""

    @pytest.mark.parametrize("argv_prefix", [[], ["start"]])
    def test_flag_reaches_build_app(
        self, monkeypatch, tmp_path: Path, agent_module: Path, argv_prefix
    ):
        import fastaiagent.cli.ui as ui_module

        seen: dict[str, object] = {}

        real_build_app = ui_module.__dict__.get("build_app")
        assert real_build_app is None  # imported lazily inside _start_server

        from fastaiagent.ui import server as server_module

        original = server_module.build_app

        def spy(**kwargs):
            seen["runners"] = kwargs.get("runners")
            return original(**kwargs)

        monkeypatch.setattr(server_module, "build_app", spy)
        monkeypatch.setattr("uvicorn.run", lambda app, host, port, log_level: None)

        result = runner.invoke(
            ui_app,
            [
                *argv_prefix,
                "--no-auth",
                "--no-open",
                "--db",
                str(tmp_path / "flag.db"),
                "--auth-file",
                str(tmp_path / "auth.json"),
                "--agent",
                f"{agent_module}:agent",
                "--agent",
                f"{agent_module}:other",
            ],
        )
        assert result.exit_code == 0, result.output
        names = [r.name for r in seen["runners"]]
        assert names == ["support-bot", "billing-bot"]
        assert "support-bot" in result.output

    def test_ui_starts_even_when_every_target_is_broken(
        self, monkeypatch, tmp_path: Path
    ):
        started: dict[str, object] = {}
        monkeypatch.setattr(
            "uvicorn.run",
            lambda app, host, port, log_level: started.setdefault("ok", True),
        )
        result = runner.invoke(
            ui_app,
            [
                "start",
                "--no-auth",
                "--no-open",
                "--db",
                str(tmp_path / "broken.db"),
                "--auth-file",
                str(tmp_path / "auth.json"),
                "--agent",
                "does/not/exist.py:agent",
            ],
        )
        assert result.exit_code == 0, result.output
        assert started.get("ok") is True
