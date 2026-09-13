"""Resume pulls a run from the plane when this machine has never seen it (D4).

``restore_from_plane`` shipped as a helper **nothing called**. ``aresume``,
``Chain.resume``, the CLI and the local UI all consulted local storage only, so
on a fresh machine a resume failed even though the plane was holding the state.
The ``export_checkpoints`` docstring promised cross-machine restore; the code
never did it. That gap is what "restore anywhere" was supposed to be.

NO MOCKS: a real ``Chain`` with a real ``interrupt()``, real
``SQLiteCheckpointer`` stores, and a real ``ThreadingHTTPServer`` speaking the
real wire — ingest on the way out, ``/latest`` on the way back. The whole point
is the round trip, so a stub that did not actually serve the row back would
prove nothing.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from fastaiagent import SQLiteCheckpointer
from fastaiagent.chain import Chain, NodeType
from fastaiagent.chain.interrupt import Resume, interrupt
from fastaiagent.checkpointers import platform_replica
from fastaiagent.tool.function import FunctionTool

_API_KEY = "fa_k_restore_test"


class _Plane:
    """Stores what it is sent and serves the newest row back, like the real door."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.rows: dict[str, list[dict]] = {}
        self.latest_gets: list[str] = []

    def add(self, items: list[dict]) -> int:
        with self.lock:
            n = 0
            for c in items:
                if not c.get("checkpoint_id"):
                    continue
                bucket = self.rows.setdefault(c["execution_id"], [])
                if any(r["checkpoint_id"] == c["checkpoint_id"] for r in bucket):
                    continue  # insert-only, like the plane
                bucket.append(c)
                n += 1
            return n

    def latest(self, execution_id: str) -> dict | None:
        with self.lock:
            self.latest_gets.append(execution_id)
            bucket = self.rows.get(execution_id) or []
            if not bucket:
                return None
            # The plane orders by the CLIENT clock first (its ``_recency_key``),
            # which is what makes a restored-and-resumed run report correctly.
            return sorted(bucket, key=lambda r: (r.get("created_at") or "", r["checkpoint_id"]))[-1]


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a: Any, **k: Any) -> None:
        pass

    def _json(self, code: int, obj: dict | None) -> None:
        body = json.dumps(obj or {}).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        st: _Plane = self.server.state  # type: ignore[attr-defined]
        if self.path == "/public/v1/auth/check":
            self._json(200, {"ok": True, "domain_id": "d", "project_id": "p", "scopes": []})
        elif self.path.startswith("/public/v1/checkpoints/") and self.path.endswith("/latest"):
            ex = self.path.split("/checkpoints/", 1)[1].rsplit("/latest", 1)[0]
            row = st.latest(ex)
            self._json(200, row) if row else self._json(404, {"detail": "not found"})
        else:
            self._json(404, {"detail": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        st: _Plane = self.server.state  # type: ignore[attr-defined]
        n = int(self.headers.get("Content-Length", 0) or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        if self.path.endswith("/checkpoints/ingest"):
            self._json(201, {"ingested": st.add(body.get("checkpoints") or [])})
        else:
            self._json(404, {"detail": "not found"})


@pytest.fixture
def plane():
    state = _Plane()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.state = state  # type: ignore[attr-defined]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    host, port = server.server_address[:2]
    try:
        yield state, f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
def _clean_platform():
    import fastaiagent
    from fastaiagent.client import _connection

    yield
    try:
        fastaiagent.disconnect()
    except Exception:
        pass
    for attr, val in (
        ("api_key", None),
        ("target", "https://app.fastaiagent.net"),
        ("project", None),
        ("project_id", None),
        ("domain_id", None),
        ("export_checkpoints", True),
    ):
        setattr(_connection, attr, val)


def _approval(amount: int) -> dict:
    decision = interrupt(reason="approve?", context={"amount": amount})
    return {"approved": bool(getattr(decision, "approved", decision))}


def _finalize(approved: bool) -> dict:
    return {"final": bool(approved)}


def _chain(store, name="restore-chain") -> Chain:
    c = Chain(name, checkpoint_enabled=True, checkpointer=store)
    c.add_node(
        "approval",
        tool=FunctionTool(name="approval", fn=_approval),
        type=NodeType.tool,
        input_mapping={"amount": "{{state.amount}}"},
    )
    c.add_node(
        "finalize",
        tool=FunctionTool(name="finalize", fn=_finalize),
        type=NodeType.tool,
        input_mapping={"approved": "{{state.output.approved}}"},
    )
    c.connect("approval", "finalize")
    return c


def _store(tmp_path, name) -> SQLiteCheckpointer:
    cp = SQLiteCheckpointer(db_path=str(tmp_path / name))
    cp.setup()
    platform_replica._REGISTRY.discard(cp)  # drive replication explicitly
    return cp


def _drain(store) -> None:
    from fastaiagent.client import _connection

    lock = platform_replica._lock_for(store)
    with lock:
        platform_replica._drain_checkpointer(store, _connection)


async def _pause_and_replicate(tmp_path, url, execution_id):
    """Run a chain to its interrupt on store A and replicate it to the plane."""
    import fastaiagent

    fastaiagent.connect(api_key=_API_KEY, target=url)
    store_a = _store(tmp_path, "a.db")
    result = await _chain(store_a).aexecute({"amount": 500}, execution_id=execution_id)
    assert result.status == "paused"
    _drain(store_a)
    rows = store_a._conn().fetchall("SELECT synced FROM checkpoints", ())
    assert rows and all(r["synced"] == 1 for r in rows), "the pause did not replicate"
    return store_a


# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_fresh_machine_resumes_a_run_it_has_never_seen(
    plane, _clean_platform, isolated_local_db, tmp_path
) -> None:
    """THE gap. No manual ``restore_from_plane`` call anywhere in this test."""
    state, url = plane
    await _pause_and_replicate(tmp_path, url, "ex-anywhere")

    # A different machine: an empty store that has never heard of this run.
    store_b = _store(tmp_path, "b.db")
    assert store_b.get_last("ex-anywhere") is None
    assert not store_b.list_pending_interrupts()

    resumed = await _chain(store_b).resume("ex-anywhere", resume_value=Resume(approved=True))

    assert resumed.status == "completed"
    assert resumed.final_state.get("output", {}).get("final") is True
    assert "ex-anywhere" in state.latest_gets, "the plane was never consulted"


@pytest.mark.asyncio
async def test_a_local_row_is_never_overwritten_by_the_replica(
    plane, _clean_platform, isolated_local_db, tmp_path
) -> None:
    """The local copy wins while it exists, and the plane is not even asked.

    Two reasons, and the second is the one with teeth: ``put`` is a plain INSERT
    on the primary key, so restoring over a row the store already holds would
    raise — and the local store is the source of truth for a run on its own
    machine. A resume must never prefer the replica.
    """
    state, url = plane
    store_a = await _pause_and_replicate(tmp_path, url, "ex-local")
    before = len(state.latest_gets)

    resumed = await _chain(store_a).resume("ex-local", resume_value=Resume(approved=True))

    assert resumed.status == "completed"
    assert state.latest_gets[before:] == [], "the plane was consulted despite a local copy"


@pytest.mark.asyncio
async def test_disconnected_resume_fails_exactly_as_before(
    plane, _clean_platform, isolated_local_db, tmp_path
) -> None:
    """No connection, no behaviour change — the restore is purely additive."""
    import fastaiagent
    from fastaiagent._internal.errors import ChainCheckpointError

    state, url = plane
    await _pause_and_replicate(tmp_path, url, "ex-offline")
    fastaiagent.disconnect()

    store_b = _store(tmp_path, "b.db")
    before = len(state.latest_gets)
    with pytest.raises((ChainCheckpointError, Exception)) as excinfo:
        await _chain(store_b).resume("ex-offline", resume_value=Resume(approved=True))
    assert "ex-offline" in str(excinfo.value)
    assert state.latest_gets[before:] == [], "a disconnected SDK still called the plane"


@pytest.mark.asyncio
async def test_the_env_switch_keeps_a_deleted_run_deleted(
    plane, _clean_platform, isolated_local_db, tmp_path, monkeypatch
) -> None:
    """``FASTAIAGENT_RESTORE_FROM_PLANE=0`` is the erasure escape hatch.

    Restoring resurrects a run whose local checkpoints were deliberately
    removed — right for disaster recovery, wrong for a right-to-be-forgotten
    request. An operator has to be able to say no.
    """
    state, url = plane
    await _pause_and_replicate(tmp_path, url, "ex-erased")
    monkeypatch.setenv("FASTAIAGENT_RESTORE_FROM_PLANE", "0")

    store_b = _store(tmp_path, "b.db")
    before = len(state.latest_gets)
    with pytest.raises(Exception):
        await _chain(store_b).resume("ex-erased", resume_value=Resume(approved=True))
    assert state.latest_gets[before:] == [], "the opt-out did not suppress the restore"
