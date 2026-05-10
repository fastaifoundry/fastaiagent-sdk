"""Regression tests for security_review_1.md M8 + M10 — platform client.

* M8: ``__repr__`` / ``str()`` of the platform connection objects must
  never spill the API key into logs, tracebacks, REPL output, or
  pytest assertion diffs.
* M10: ``_normalize_target`` must refuse to silently rewrite a bare
  hostname into ``http://`` (the previous behaviour smuggled
  unencrypted requests for any host that *looked* local).
"""

from __future__ import annotations

import pytest


def test_m8_connection_repr_redacts_api_key():
    """``repr(_Connection(api_key="sk-real-secret"))`` must NOT contain
    the raw key. Show only the last 4 chars + length so a developer can
    confirm wiring without leaking the secret.
    """
    from fastaiagent.client import _Connection

    c = _Connection()
    c.api_key = "sk-real-platform-key-with-many-chars-AB12"
    text = repr(c)
    assert "sk-real-platform-key-with-many-chars" not in text
    assert "AB12" in text  # last 4 still present so it's debuggable
    assert "***" in text


def test_m8_platformapi_repr_redacts_api_key():
    """Same property for ``PlatformAPI``."""
    from fastaiagent._platform.api import PlatformAPI

    api = PlatformAPI(api_key="sk-real-platform-key-with-many-chars-CD34")
    text = repr(api)
    assert "sk-real-platform-key-with-many-chars" not in text
    assert "CD34" in text
    assert "***" in text


def test_m8_unset_key_repr_does_not_crash():
    """When no key has been wired in yet, repr renders ``<unset>``."""
    from fastaiagent.client import _Connection

    text = repr(_Connection())
    assert "<unset>" in text or "len=0" in text


def test_m10_normalize_target_rejects_missing_scheme():
    """M10 — bare hostnames that used to be auto-promoted to ``http://``
    now raise so the user picks a scheme deliberately.
    """
    from fastaiagent.client import _normalize_target

    with pytest.raises(ValueError, match="scheme"):
        _normalize_target("localhost:7842")
    with pytest.raises(ValueError, match="scheme"):
        _normalize_target("api.example.com")


def test_m10_normalize_target_accepts_explicit_scheme():
    """Both http and https are accepted; trailing slashes are stripped."""
    from fastaiagent.client import _normalize_target

    assert _normalize_target("http://localhost:7842/") == "http://localhost:7842"
    assert _normalize_target("https://app.fastaiagent.net/") == "https://app.fastaiagent.net"


def test_m10_normalize_target_rejects_unknown_scheme():
    from fastaiagent.client import _normalize_target

    with pytest.raises(ValueError, match="scheme"):
        _normalize_target("ftp://example.com")


def test_m10_normalize_target_passes_through_empty():
    """Empty string is not a connection target — left untouched (caller
    handles the "not configured" branch)."""
    from fastaiagent.client import _normalize_target

    assert _normalize_target("") == ""
