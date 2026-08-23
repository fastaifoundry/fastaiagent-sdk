"""security_audit_2 N7 — ``export_checkpoints=False`` stops checkpoint STATE
from replicating to the plane, while local durability is untouched.

The single choke point is ``platform_replica._drain_guarded`` (used by the
write-kick, connect-drain, and disconnect-drain paths). These tests drive it
directly with a stub connection — no live plane.
"""

from __future__ import annotations

import pytest

import fastaiagent.checkpointers.platform_replica as replica
import fastaiagent.client as client_mod


class _StubConn:
    is_connected = True

    def __init__(self, export_checkpoints: bool) -> None:
        self.export_checkpoints = export_checkpoints


class _StubCheckpointer:
    def __init__(self) -> None:
        self.drained = False

    def fetch_unsynced(self, **_kw):
        return []

    def mark_synced(self, _ids):  # pragma: no cover - not reached when gated
        pass


@pytest.fixture
def patched(monkeypatch):
    drained = {"count": 0}
    monkeypatch.setattr(
        replica, "_drain_checkpointer", lambda cp, conn: drained.__setitem__("count", drained["count"] + 1)
    )
    return drained


def test_drain_skipped_when_export_checkpoints_false(monkeypatch, patched):
    monkeypatch.setattr(client_mod, "_connection", _StubConn(export_checkpoints=False))
    replica._drain_guarded(_StubCheckpointer())
    assert patched["count"] == 0


def test_drain_runs_when_export_checkpoints_true(monkeypatch, patched):
    monkeypatch.setattr(client_mod, "_connection", _StubConn(export_checkpoints=True))
    replica._drain_guarded(_StubCheckpointer())
    assert patched["count"] == 1


def test_connection_default_is_true():
    # A fresh connection replicates by default (non-breaking).
    from fastaiagent.client import _Connection

    assert _Connection().export_checkpoints is True
