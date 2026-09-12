"""Poison-row quarantine for the checkpoint outbox (durability audit D2).

A checkpoint the plane's ingest door refuses on **payload** grounds is refused
identically every time. The drain sends the oldest un-acked rows as one batch and
used to stop at the first failure while leaving them buffered — so the next kick
re-fetched the same batch, got the same refusal, and stopped again. One bad row
stranded every later checkpoint of every run on that checkpointer, permanently.

NO MOCKS. A real ``SQLiteCheckpointer`` against a real ``ThreadingHTTPServer``
speaking the real ingest wire, drained by the real ``platform_replica`` code.
The plane is a stub only in the sense that it is small — every byte still crosses
a socket, which is the point: the defect lived in how the SDK read an HTTP status.

The tests that matter most here are the two **negative** ones. It is easy to write
a quarantine that unwedges the outbox by throwing away anything the plane dislikes;
``test_403_never_quarantines`` is what says we did not. An un-entitled domain must
keep its buffered checkpoints forever, because "forever" is the right answer when
entitlement can be granted tomorrow.
"""

from __future__ import annotations

import json
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from fastaiagent.chain.checkpoint import Checkpoint
from fastaiagent.checkpointers import platform_replica
from fastaiagent.checkpointers.sqlite import SQLiteCheckpointer

_API_KEY = "fa_k_quarantine_test"


class _CheckpointPlane:
    """A programmable ingest door: what it refuses, how, and what it saw."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        #: ids that make the door refuse the WHOLE batch (the pre-p54t1 shape,
        #: and what any proxy or older plane still does).
        self.batch_poison: set[str] = set()
        #: status the door answers with when it sees a poisoned id.
        self.refusal_code = 422
        #: ids the door names in a 201 partial-success body (the p54t1 shape).
        self.per_item_rejects: set[str] = set()
        #: every id the door actually stored.
        self.ingested: list[str] = []
        #: one entry per POST — the ids it carried. Lets a test assert on how
        #: the batch was split, not just on the end state.
        self.posts: list[list[str]] = []

    def snapshot(self) -> tuple[list[str], list[list[str]]]:
        with self.lock:
            return list(self.ingested), [list(p) for p in self.posts]


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a: Any, **k: Any) -> None:
        pass

    def _json(self, code: int, obj: dict) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/public/v1/auth/check":
            self._json(
                200,
                {
                    "ok": True,
                    "domain_id": "dom-quarantine",
                    "project_id": "proj-quarantine",
                    "scopes": ["run:write", "run:read"],
                },
            )
        else:
            self._json(404, {"detail": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        st: _CheckpointPlane = self.server.state  # type: ignore[attr-defined]
        n = int(self.headers.get("Content-Length", 0) or 0)
        body = json.loads(self.rfile.read(n) or b"{}")

        if not self.path.endswith("/checkpoints/ingest"):
            self._json(404, {"detail": "not found"})
            return

        ids = [c.get("checkpoint_id") for c in body.get("checkpoints") or []]
        with st.lock:
            st.posts.append(ids)
            poisoned = [i for i in ids if i in st.batch_poison]
            if poisoned:
                code = st.refusal_code
                st_reason = f"checkpoint {poisoned[0]} is not storable"
                self._json(code, {"detail": st_reason})
                return
            rejected = [i for i in ids if i in st.per_item_rejects]
            stored = [i for i in ids if i not in st.per_item_rejects]
            st.ingested.extend(stored)

        out: dict[str, Any] = {"ingested": len(stored)}
        if rejected:
            out["rejected"] = len(rejected)
            out["rejections"] = [
                {"checkpoint_id": i, "reason": "state_snapshot is too large"} for i in rejected
            ]
        self._json(201, out)


@pytest.fixture
def plane():
    state = _CheckpointPlane()
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
        ("policy_cache", None),
        ("export_checkpoints", True),
    ):
        setattr(_connection, attr, val)


def _store(tmp_path, name: str = "cp.db") -> SQLiteCheckpointer:
    cp = SQLiteCheckpointer(db_path=str(tmp_path / name))
    cp.setup()
    # ``setup()`` registers the store process-globally, and ``connect()`` fires a
    # backlog drain across that registry on a daemon thread. Leaving it in would
    # mean an unsynchronised second drainer racing every assertion below — and in
    # the "no quarantine support" test it would drain the *inner* store, which
    # does support it, and prove the opposite of what that test claims. Each test
    # here drives its drain explicitly via ``_drain``.
    platform_replica._REGISTRY.discard(cp)
    return cp


def _write(store: SQLiteCheckpointer, execution_id: str, node: str) -> str:
    """Write one real checkpoint and return its id."""
    cid = str(uuid.uuid4())
    store.put(
        Checkpoint(
            checkpoint_id=cid,
            chain_name="quarantine-probe",
            execution_id=execution_id,
            node_id=node,
            node_index=0,
            status="completed",
            state_snapshot={"node": node},
        )
    )
    return cid


def _rows(store: SQLiteCheckpointer) -> dict[str, dict[str, Any]]:
    out = store._conn().fetchall(
        "SELECT checkpoint_id, synced, sync_error FROM checkpoints ORDER BY rowid"
    )
    return {r["checkpoint_id"]: dict(r) for r in out}


def _drain(target: Any, times: int = 3) -> None:
    """Drain repeatedly — a wedged outbox must stay wedged across kicks, which is
    exactly the property the pre-change code failed and a single drain cannot see.

    ⚠ Takes the checkpointer's drain lock first. ``connect()`` fires its own
    backlog drain on a daemon thread, and ``_drain_guarded`` acquires that lock
    **non-blocking** — so calling it here would silently no-op while the
    background drain held it, and the test would assert against a store nothing
    had drained. Holding the lock waits that thread out instead of racing it.
    """
    from fastaiagent.client import _connection

    lock = platform_replica._lock_for(target)
    for _ in range(times):
        with lock:
            platform_replica._drain_checkpointer(target, _connection)


# --------------------------------------------------------------------------


def test_poison_row_is_quarantined_and_the_rest_replicate(
    plane, _clean_platform, isolated_local_db, tmp_path
) -> None:
    """One row the door refuses must cost one row, not the whole outbox."""
    import fastaiagent

    state, url = plane
    store = _store(tmp_path)
    poison = _write(store, "exec-poison", "bad-node")
    good1 = _write(store, "exec-healthy", "n1")
    good2 = _write(store, "exec-healthy", "n2")
    state.batch_poison.add(poison)

    fastaiagent.connect(api_key=_API_KEY, target=url)
    _drain(store)

    rows = _rows(store)
    assert rows[poison]["synced"] == 1, "the poison row must stop being a re-send candidate"
    assert rows[poison]["sync_error"], "a quarantined row must record why it was given up on"
    assert "422" in rows[poison]["sync_error"]

    for cid in (good1, good2):
        assert rows[cid]["synced"] == 1, "a healthy row was stranded behind the poison row"
        assert rows[cid]["sync_error"] is None, "a row that reached the plane has no error"

    ingested, _posts = state.snapshot()
    assert set(ingested) == {good1, good2}
    assert poison not in ingested, "the door never stored it — the replica is incomplete, loudly"


def test_403_never_quarantines(plane, _clean_platform, isolated_local_db, tmp_path) -> None:
    """An un-entitled domain keeps its outbox. This is the guard on the fix.

    403 is not a property of any row — it applies to every checkpoint this SDK
    will ever send. Treating it as poison would quietly delete a whole tenant's
    durability replica the first time someone's plan lapsed.
    """
    import fastaiagent

    state, url = plane
    store = _store(tmp_path)
    ids = [_write(store, "exec-403", f"n{i}") for i in range(3)]
    state.batch_poison.update(ids)
    state.refusal_code = 403

    fastaiagent.connect(api_key=_API_KEY, target=url)
    _drain(store)

    rows = _rows(store)
    for cid in ids:
        assert rows[cid]["synced"] == 0, "a 403 must leave the rows buffered for a later attempt"
        assert rows[cid]["sync_error"] is None, "a 403 must never be recorded as giving up"


def test_404_from_an_older_plane_never_quarantines(
    plane, _clean_platform, isolated_local_db, tmp_path
) -> None:
    """A plane or proxy that does not route this path yet must not cost rows."""
    import fastaiagent

    state, url = plane
    store = _store(tmp_path)
    ids = [_write(store, "exec-404", f"n{i}") for i in range(2)]
    state.batch_poison.update(ids)
    state.refusal_code = 404

    fastaiagent.connect(api_key=_API_KEY, target=url)
    _drain(store)

    rows = _rows(store)
    assert all(rows[c]["synced"] == 0 and rows[c]["sync_error"] is None for c in ids)


def test_per_item_rejection_in_a_201_is_recorded_not_claimed_as_synced(
    plane, _clean_platform, isolated_local_db, tmp_path
) -> None:
    """A plane on p54t1 answers 201 and names what it dropped.

    The row never reached the replica, so marking it cleanly synced would tell the
    operator the run is fully replicated when it is not.
    """
    import fastaiagent

    state, url = plane
    store = _store(tmp_path)
    a = _write(store, "exec-mixed", "n1")
    b = _write(store, "exec-mixed", "n2")
    c = _write(store, "exec-mixed", "n3")
    state.per_item_rejects.add(b)

    fastaiagent.connect(api_key=_API_KEY, target=url)
    _drain(store)

    rows = _rows(store)
    assert rows[b]["synced"] == 1 and rows[b]["sync_error"]
    assert "plane rejected" in rows[b]["sync_error"]
    assert "too large" in rows[b]["sync_error"], "the plane's own reason must survive"
    for cid in (a, c):
        assert rows[cid]["synced"] == 1 and rows[cid]["sync_error"] is None

    ingested, posts = state.snapshot()
    assert set(ingested) == {a, c}
    assert len(posts) == 1, "a 201 needs no bisection — one post, done"


def test_two_poison_rows_are_both_isolated(
    plane, _clean_platform, isolated_local_db, tmp_path
) -> None:
    """Bisection has to terminate with more than one offender in the batch."""
    import fastaiagent

    state, url = plane
    store = _store(tmp_path)
    ids = [_write(store, "exec-many", f"n{i}") for i in range(8)]
    bad = {ids[1], ids[6]}
    state.batch_poison.update(bad)

    fastaiagent.connect(api_key=_API_KEY, target=url)
    _drain(store)

    rows = _rows(store)
    for cid in ids:
        assert rows[cid]["synced"] == 1, f"{cid} never advanced"
        if cid in bad:
            assert rows[cid]["sync_error"], "an offender must be recorded"
        else:
            assert rows[cid]["sync_error"] is None, "a healthy row must not be blamed"

    ingested, posts = state.snapshot()
    assert set(ingested) == set(ids) - bad
    # log2(8) splits, not 8 single-row posts: bisection, not a linear retry.
    assert len(posts) <= 12, f"bisection should be O(k log n), got {len(posts)} posts"


def test_checkpointer_without_quarantine_support_keeps_the_old_behaviour(
    plane, _clean_platform, isolated_local_db, tmp_path
) -> None:
    """``mark_quarantined`` is optional, so a third-party store must not break.

    It keeps the pre-change outcome — rows stay buffered — because losing a row we
    cannot record the loss of is strictly worse than stalling.
    """
    import fastaiagent

    state, url = plane
    store = _store(tmp_path)
    poison = _write(store, "exec-legacy", "bad")
    other = _write(store, "exec-legacy", "ok")
    state.batch_poison.add(poison)

    class _ReplicatedOnly:
        """Exactly the ``ReplicatedCheckpointer`` surface, nothing more."""

        def __init__(self, inner: SQLiteCheckpointer) -> None:
            self._inner = inner

        def fetch_unsynced(self, limit: int, project_id: str | None = None) -> list[dict]:
            return self._inner.fetch_unsynced(limit, project_id)

        def mark_synced(self, checkpoint_ids: list[str]) -> None:
            self._inner.mark_synced(checkpoint_ids)

    adapter = _ReplicatedOnly(store)
    fastaiagent.connect(api_key=_API_KEY, target=url)
    _drain(adapter)

    rows = _rows(store)
    assert rows[poison]["synced"] == 0
    assert rows[other]["synced"] == 0, "the old head-of-line behaviour, deliberately preserved"
    assert rows[poison]["sync_error"] is None
