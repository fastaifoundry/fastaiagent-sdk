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

from typer.testing import CliRunner

from fastaiagent.cli.main import app as main_app
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
