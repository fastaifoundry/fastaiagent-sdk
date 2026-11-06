"""``connect()`` has no default target (1.100.0).

Until 1.100.0 ``target`` defaulted to ``https://app.fastaiagent.net``, a host
that no longer resolves, so a ``connect(api_key=...)`` copied from the README
reached nothing. Every account now has its own program address, so the target
comes from the argument, else ``FASTAIAGENT_TARGET``, else a clear error.

Real ``connect()`` against a closed local port (its real unreachable-plane
path); no mocks.
"""

from __future__ import annotations

import socket
from pathlib import Path
from typing import Any

import pytest

import fastaiagent as fa
from fastaiagent.client import _connection


def _dead_url() -> str:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return f"http://127.0.0.1:{s.getsockname()[1]}"


@pytest.fixture
def clean(isolated_local_db: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.delenv("FASTAIAGENT_TARGET", raising=False)
    fa.disconnect()
    yield monkeypatch
    fa.disconnect()


def test_no_target_and_no_env_raises_and_does_not_connect(clean: Any) -> None:
    with pytest.raises(ValueError, match="FASTAIAGENT_TARGET"):
        fa.connect(api_key="fa_k_test")
    assert _connection.is_connected is False


def test_env_supplies_the_target(clean: Any) -> None:
    url = _dead_url()
    clean.setenv("FASTAIAGENT_TARGET", url + "/")
    fa.connect(api_key="fa_k_test")
    assert _connection.target == url


def test_argument_wins_over_env(clean: Any) -> None:
    clean.setenv("FASTAIAGENT_TARGET", _dead_url())
    url = _dead_url()
    fa.connect(api_key="fa_k_test", target=url)
    assert _connection.target == url


def test_a_bad_target_leaves_the_previous_connection_as_it_was(clean: Any) -> None:
    url = _dead_url()
    fa.connect(api_key="fa_k_first", target=url)
    with pytest.raises(ValueError, match="scheme"):
        fa.connect(api_key="fa_k_second", target="plane.example.com")
    assert _connection.api_key == "fa_k_first"
    assert _connection.target == url
