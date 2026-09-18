"""security_audit_2 N7 — ``export_checkpoints=False`` stops checkpoint STATE
from replicating to the plane, while local durability is untouched.

The single choke point is ``platform_replica._drain_guarded`` (used by the
write-kick, connect-drain, and disconnect-drain paths). These tests drive it
directly with a stub connection — no live plane.
"""

from __future__ import annotations

import json
import time

import pytest

import fastaiagent.checkpointers.platform_replica as replica
import fastaiagent.client as client_mod
from fastaiagent.chain.checkpoint import Checkpoint
from fastaiagent.checkpointers import SQLiteCheckpointer


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


# ── the environment half (1.67.0) ──────────────────────────────────────────
#
# Everything above drives ``_connection.export_checkpoints`` directly, which is
# exactly why the defect shipped: nothing here ever asked how the *environment*
# resolves to that attribute. ``FASTAIAGENT_EXPORT_CHECKPOINTS`` was parsed as
# ``!= "0"``, so ``false`` / ``no`` / ``off`` — the spellings
# docs/configuration/environment-variables.md invites — kept replicating
# checkpoint state to the plane. These cases go through the real ``connect()``
# and then a real HTTP server, and assert a canary string never leaves.

CANARY = "CANARY-checkpoint-egress-7b1f2e"

FALSE_SPELLINGS = ["0", "false", "no", "off", "OFF", "  False  "]
TRUE_SPELLINGS = ["1", "true", "yes", "on", ""]


@pytest.fixture
def seeded_checkpointer(isolated_local_db):
    """A checkpointer holding one checkpoint whose state carries the canary.

    Seeded **before** ``connect()``: connect kicks a background drain that opens
    ``local.db`` on a daemon thread, and racing it from the main thread just
    produces ``database is locked`` noise that has nothing to do with the switch.
    """
    cp = SQLiteCheckpointer(db_path=str(isolated_local_db))
    cp.setup()
    cp.put(
        Checkpoint(
            execution_id="ex-egress",
            chain_name="egress-probe",
            node_id="turn:0",
            status="interrupted",
            state_snapshot={"secret": CANARY},
        )
    )
    try:
        yield cp
    finally:
        cp.close()


def _ingest_bodies(capture_server):
    return [r for r in capture_server.requests if "/checkpoints/ingest" in r["path"]]


def _wait_for_ingest(capture_server, timeout: float = 5.0):
    """Poll until an ingest lands. ``connect()`` kicks the drain on a daemon
    thread, and ``_drain_guarded`` no-ops when that drain already holds the
    per-checkpointer lock, so the send is legitimately asynchronous."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        bodies = _ingest_bodies(capture_server)
        if bodies:
            return bodies
        time.sleep(0.05)
    return _ingest_bodies(capture_server)


def _settle(seconds: float = 1.0) -> None:
    """Give any background drain a chance to send, so "nothing was sent" means it."""
    time.sleep(seconds)


@pytest.mark.parametrize("value", FALSE_SPELLINGS)
def test_env_false_spellings_send_nothing_and_never_leak_the_canary(
    value, monkeypatch, seeded_checkpointer, capture_server
):
    import fastaiagent

    monkeypatch.setenv("FASTAIAGENT_EXPORT_CHECKPOINTS", value)
    fastaiagent.connect(api_key="fa_k_test", target=capture_server.url, auto_register=False)
    try:
        assert client_mod._connection.export_checkpoints is False, (
            f"FASTAIAGENT_EXPORT_CHECKPOINTS={value!r} did not disable replication"
        )
        replica._drain_guarded(seeded_checkpointer)
        _settle()
        assert _ingest_bodies(capture_server) == []
        blob = json.dumps([r["body"] for r in capture_server.requests], default=str)
        assert CANARY not in blob, f"the canary left the process with value {value!r}"
    finally:
        fastaiagent.disconnect()


@pytest.mark.parametrize("value", TRUE_SPELLINGS)
def test_env_true_and_empty_spellings_still_replicate(
    value, monkeypatch, seeded_checkpointer, capture_server
):
    """The control. Empty means *unset*, not off — see env.py's module docstring."""
    import fastaiagent

    monkeypatch.setenv("FASTAIAGENT_EXPORT_CHECKPOINTS", value)
    fastaiagent.connect(api_key="fa_k_test", target=capture_server.url, auto_register=False)
    try:
        assert client_mod._connection.export_checkpoints is True
        replica._drain_guarded(seeded_checkpointer)
        bodies = _wait_for_ingest(capture_server)
        assert bodies, f"nothing was replicated with value {value!r}"
        assert CANARY in json.dumps([r["body"] for r in bodies], default=str)
    finally:
        fastaiagent.disconnect()


def test_an_unparseable_value_fails_closed(monkeypatch, seeded_checkpointer, capture_server):
    """Signed off for 1.67.0: a typo on an egress opt-out does not egress."""
    import fastaiagent

    monkeypatch.setenv("FASTAIAGENT_EXPORT_CHECKPOINTS", "flase")
    fastaiagent.connect(api_key="fa_k_test", target=capture_server.url, auto_register=False)
    try:
        assert client_mod._connection.export_checkpoints is False
        replica._drain_guarded(seeded_checkpointer)
        _settle()
        assert _ingest_bodies(capture_server) == []
    finally:
        fastaiagent.disconnect()


def test_the_kwarg_still_beats_the_environment(monkeypatch, isolated_local_db, capture_server):
    import fastaiagent

    monkeypatch.setenv("FASTAIAGENT_EXPORT_CHECKPOINTS", "off")
    fastaiagent.connect(
        api_key="fa_k_test",
        target=capture_server.url,
        auto_register=False,
        export_checkpoints=True,
    )
    try:
        assert client_mod._connection.export_checkpoints is True
    finally:
        fastaiagent.disconnect()
