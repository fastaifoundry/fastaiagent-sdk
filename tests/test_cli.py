"""Tests for fastaiagent.cli module."""

from __future__ import annotations

from typer.testing import CliRunner

from fastaiagent.cli.main import app

runner = CliRunner()


class TestCLI:
    def test_version(self):
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        from fastaiagent._version import __version__

        assert __version__ in result.output

    def test_help(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "FastAIAgent" in result.output

    def test_traces_help(self):
        result = runner.invoke(app, ["traces", "--help"])
        assert result.exit_code == 0
        assert "list" in result.output

    def test_replay_help(self):
        result = runner.invoke(app, ["replay", "--help"])
        assert result.exit_code == 0

    def test_eval_help(self):
        result = runner.invoke(app, ["eval", "--help"])
        assert result.exit_code == 0
        assert "run" in result.output

    def test_prompts_help(self):
        result = runner.invoke(app, ["prompts", "--help"])
        assert result.exit_code == 0
        assert "list" in result.output

    def test_kb_help(self):
        result = runner.invoke(app, ["kb", "--help"])
        assert result.exit_code == 0
        assert "status" in result.output
