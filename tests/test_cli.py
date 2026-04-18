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
        assert "list" in result.output  # added in 0.6.1

    # ------------------------------------------------------------------
    # 0.6.1 additions
    # ------------------------------------------------------------------

    def test_version_shows_installed_extras(self):
        """`fastaiagent version` annotates which optional extras are present."""
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        # At minimum, core is there. If any extras resolved, they're bracketed.
        assert "fastaiagent" in result.output
        # `[extras]` token only appears when at least one extra is installed;
        # test-env always has pytest deps but not every extra, so just assert
        # the token shape when present.
        if "[" in result.output:
            assert "]" in result.output

    def test_top_level_lists_new_commands(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        for cmd in ("connect", "disconnect", "agent", "auth", "mcp"):
            assert cmd in result.output

    def test_kb_list_on_missing_root_is_graceful(self, tmp_path):
        """`kb list` against a nonexistent dir should exit cleanly, not crash."""
        missing = tmp_path / "does-not-exist"
        result = runner.invoke(app, ["kb", "list", "--path", str(missing)])
        assert result.exit_code == 0
        assert "No KBs" in result.output or "does not exist" in result.output

    def test_kb_list_enumerates_persistent_kbs(self, tmp_path):
        """Create two persistent KBs; `kb list` should surface both."""
        from fastaiagent.kb import LocalKB
        from fastaiagent.kb.embedding import SimpleEmbedder

        for name in ("alpha", "bravo"):
            kb = LocalKB(
                name=name,
                path=str(tmp_path),
                embedder=SimpleEmbedder(dimensions=16),
                search_type="keyword",  # skip vectors; we just want the SQLite file
            )
            kb.add(f"seed content for {name}")
            kb.close()

        result = runner.invoke(app, ["kb", "list", "--path", str(tmp_path)])
        assert result.exit_code == 0, result.output
        assert "alpha" in result.output
        assert "bravo" in result.output

    def test_agent_help_documents_serve(self):
        result = runner.invoke(app, ["agent", "--help"])
        assert result.exit_code == 0
        assert "serve" in result.output

    def test_agent_serve_rejects_invalid_target_spec(self):
        """`agent serve foo` (no colon) should fail with a clear error."""
        result = runner.invoke(app, ["agent", "serve", "not-a-valid-spec"])
        assert result.exit_code != 0

    def test_replay_help_lists_fork(self):
        result = runner.invoke(app, ["replay", "--help"])
        assert result.exit_code == 0
        assert "fork" in result.output

    def test_auth_status_when_not_connected(self, tmp_path, monkeypatch):
        """Without creds, `auth status` exits non-zero with a hint."""
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.delenv("FASTAIAGENT_API_KEY", raising=False)
        result = runner.invoke(app, ["auth", "status"])
        assert result.exit_code != 0
        assert "Not connected" in result.output

    def test_auth_env_when_not_connected(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.delenv("FASTAIAGENT_API_KEY", raising=False)
        result = runner.invoke(app, ["auth", "env"])
        assert result.exit_code != 0
