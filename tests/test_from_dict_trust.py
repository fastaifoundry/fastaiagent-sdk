"""security_audit_2 N5 — deserialization trust-boundary guards.

``Agent.from_dict`` / ``LLMClient.from_dict`` can be fed by an untrusted source
(a replayed trace read from local.db, or a runner job payload from the plane),
and the runner ``--connect`` channel must not be plaintext to a remote plane.
"""

from __future__ import annotations

import logging

import pytest
import typer

from fastaiagent.cli.runner import _require_secure_connect
from fastaiagent.llm.client import LLMClient


class TestLLMClientFromDictTrust:
    def test_serialized_api_key_is_ignored(self, caplog, monkeypatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with caplog.at_level(logging.WARNING):
            c = LLMClient.from_dict(
                {"provider": "openai", "model": "gpt-4o", "api_key": "sk-INJECTED"}
            )
        assert c.api_key != "sk-INJECTED"
        assert any("Ignoring" in r.message for r in caplog.records)

    def test_local_http_base_url_allowed(self) -> None:
        # Local LLMs (Ollama etc.) are legitimate — not blocked.
        c = LLMClient.from_dict(
            {"provider": "openai", "model": "m", "base_url": "http://localhost:11434/v1"}
        )
        assert c.base_url == "http://localhost:11434/v1"

    @pytest.mark.parametrize("bad", ["file:///etc/passwd", "ftp://x/y", "gopher://z"])
    def test_bad_base_url_scheme_rejected(self, bad: str) -> None:
        with pytest.raises(ValueError, match="scheme"):
            LLMClient.from_dict({"provider": "openai", "model": "m", "base_url": bad})

    def test_no_base_url_is_fine(self) -> None:
        assert LLMClient.from_dict({"provider": "openai", "model": "m"}) is not None


class TestRunnerConnectScheme:
    def test_https_allowed(self) -> None:
        _require_secure_connect("https://app.fastaiagent.net")  # no raise

    @pytest.mark.parametrize("url", ["http://localhost:20001", "http://127.0.0.1:9", "http://[::1]:9"])
    def test_loopback_http_allowed(self, url: str) -> None:
        _require_secure_connect(url)  # no raise

    def test_remote_http_refused(self, monkeypatch) -> None:
        monkeypatch.delenv("FASTAIAGENT_RUNNER_ALLOW_INSECURE", raising=False)
        with pytest.raises(typer.Exit) as exc:
            _require_secure_connect("http://plane.example.com")
        assert exc.value.exit_code == 2

    def test_insecure_optout_allows_remote_http(self, monkeypatch) -> None:
        monkeypatch.setenv("FASTAIAGENT_RUNNER_ALLOW_INSECURE", "1")
        _require_secure_connect("http://plane.internal.corp")  # no raise
