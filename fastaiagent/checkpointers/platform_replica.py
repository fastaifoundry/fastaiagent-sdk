"""Non-blocking checkpoint replication to the FastAIAgent Platform (WS2 durability).

When the SDK is connected to an Enterprise plane, checkpoints written by a
:class:`SQLiteCheckpointer` / :class:`PostgresCheckpointer` are replicated to the
plane as a **managed durable copy**, so a run can be restored/resumed even if the
local store is lost. The plane is a passive replica + system-of-record: it
**serves** a checkpoint back; the **SDK resumes locally** (no execution on the
plane).

Design — like the trace outbox (:mod:`fastaiagent.trace.platform_export`) but
**write-driven, not span-driven**:

* The checkpointer writes locally first (the hot-path source of truth) with
  ``synced=0``, then — when connected — kicks a fire-and-forget background drain
  (:func:`kick`). Checkpoints are NOT tied to spans, and a ``trace=False`` run
  emits none, so a per-write daemon kick (not a ``BatchSpanProcessor``) is the
  reliable trigger.
* The drain POSTs un-acked rows to ``/public/v1/checkpoints/ingest`` (idempotent
  by ``checkpoint_id``) and marks them ``synced=1`` **only after a 2xx**.
* **Non-lossy:** un-acked rows are NEVER abandoned (unlike traces' age/count
  bound) — an active/paused run's durability must not be dropped. A transient
  failure just leaves the row buffered for the next kick / ``connect()`` flush.

Restore is :func:`restore_from_plane`: GET the latest checkpoint and write it back
into a local checkpointer so a normal ``resume`` proceeds.

Replication is **best-effort**: a no-op when not connected, and it never raises
into the agent hot path. Only checkpointers that satisfy the optional
:class:`~fastaiagent.checkpointers.protocol.ReplicatedCheckpointer` surface
replicate; others are silently skipped (non-breaking).
"""

from __future__ import annotations

import json
import logging
import threading
import time
import weakref
from typing import TYPE_CHECKING, Any

from fastaiagent.chain.checkpoint import Checkpoint
from fastaiagent.checkpointers.protocol import PendingInterrupt

if TYPE_CHECKING:
    from fastaiagent.client import _Connection

logger = logging.getLogger(__name__)

# Transient-only retry (connection / timeout / 5xx); 4xx (incl. 403) terminal.
_MAX_ATTEMPTS = 3
_BACKOFF_BASE = 0.5
_TIMEOUT = 10
_DRAIN_LIMIT = 200  # checkpoints per ingest batch; a backlog drains across loops

_VALID_STATUS = {"completed", "interrupted", "failed"}

# Live checkpointers that expose the replication surface. Weak so finished
# chains/agents are collected; the drain iterates whatever is still alive.
_REGISTRY: weakref.WeakSet[Any] = weakref.WeakSet()
# One drain at a time per checkpointer (coalesces write bursts — the in-flight
# drain re-queries fetch_unsynced and picks up rows written while it ran).
_LOCKS: weakref.WeakKeyDictionary[Any, threading.Lock] = weakref.WeakKeyDictionary()
_LOCKS_GUARD = threading.Lock()


def register_checkpointer(cp: Any) -> None:
    """Register a checkpointer so ``connect()``/``disconnect()`` can drain it.

    Called from each backend's ``setup()``. Cheap + idempotent; the drain no-ops
    when not connected, so registering an unused checkpointer costs nothing.
    """
    try:
        _REGISTRY.add(cp)
    except Exception:  # pragma: no cover - defensive
        logger.debug("checkpoint registry add failed", exc_info=True)


# --- wire mapping ---------------------------------------------------------


def _jload(value: Any) -> Any:
    """Parse a JSON column that may arrive as text (SQLite) or already-parsed
    (Postgres JSONB). NULL → ``{}``."""
    if value is None:
        return {}
    if isinstance(value, str):
        try:
            return json.loads(value) if value else {}
        except (ValueError, TypeError):
            return {}
    return value


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    iso = getattr(value, "isoformat", None)
    if callable(iso):
        result = iso()
        return result if isinstance(result, str) else str(result)
    return str(value)


def _resource_type(row: dict[str, Any]) -> str:
    """Best-effort agent-vs-chain discriminator from the checkpoint row.

    Agents stamp ``agent_path='agent:<name>/…'`` and ``node_id='turn:N/tool:…'``;
    plain chains do neither. Display/filtering only — restore is by execution_id,
    so an occasional misclassification never affects correctness.
    """
    ap = row.get("agent_path") or ""
    nid = row.get("node_id") or ""
    if ap.startswith("agent:") or nid.startswith("turn:"):
        return "agent"
    return "chain"


def _to_wire(row: dict[str, Any]) -> dict[str, Any]:
    """Map a local checkpoint row onto the ``/checkpoints/ingest`` wire shape.

    Fields not in the top-level wire (node_input/output, iteration[_counters],
    interrupt_*) ride in ``metadata`` so :func:`restore_from_plane` reconstructs a
    lossless :class:`Checkpoint`.
    """
    cid = row.get("checkpoint_id") or row.get("id") or ""
    chain_name = row.get("chain_name") or ""
    rtype = _resource_type(row)
    status = row.get("status") or "completed"
    if status not in _VALID_STATUS:
        status = "completed"

    wire: dict[str, Any] = {
        "checkpoint_id": cid,
        "execution_id": row.get("execution_id") or "",
        "resource_type": rtype,
        "agent_id": chain_name if rtype == "agent" else None,
        "chain_id": chain_name if rtype == "chain" else None,
        "node_id": row.get("node_id"),
        "step_index": row.get("node_index"),
        "status": status,
        "parent_checkpoint_id": row.get("parent_checkpoint_id"),
        "state_snapshot": _jload(row.get("state_snapshot")),
        "metadata": {
            "node_input": _jload(row.get("node_input")),
            "node_output": _jload(row.get("node_output")),
            "iteration": row.get("iteration") or 0,
            "iteration_counters": _jload(row.get("iteration_counters")),
            "interrupt_reason": row.get("interrupt_reason"),
            "interrupt_context": _jload(row.get("interrupt_context")),
            "agent_path": row.get("agent_path"),
        },
        "created_at": _iso(row.get("created_at")),
    }
    # Monotonic ordering hint for the plane's "latest" pick (SQLite rowid). When
    # absent (Postgres), the plane tie-breaks by server receive order and we
    # always drain oldest-first, so "latest received" stays correct.
    seq = row.get("_seq")
    if seq is not None:
        wire["sequence"] = int(seq)
    return wire


def _wire_to_checkpoint(data: dict[str, Any]) -> Checkpoint:
    """Inverse of :func:`_to_wire` — reconstruct a :class:`Checkpoint` from a
    ``CheckpointRead`` restore payload."""
    meta = data.get("metadata") or {}
    chain_name = data.get("agent_id") or data.get("chain_id") or ""
    return Checkpoint(
        checkpoint_id=data.get("checkpoint_id") or "",
        parent_checkpoint_id=data.get("parent_checkpoint_id"),
        chain_name=chain_name,
        execution_id=data.get("execution_id") or "",
        node_id=data.get("node_id") or "",
        node_index=data.get("step_index") or 0,
        status=data.get("status") or "completed",
        state_snapshot=data.get("state_snapshot") or {},
        node_input=meta.get("node_input") or {},
        node_output=meta.get("node_output") or {},
        iteration=meta.get("iteration") or 0,
        iteration_counters=meta.get("iteration_counters") or {},
        interrupt_reason=meta.get("interrupt_reason"),
        interrupt_context=meta.get("interrupt_context") or {},
        agent_path=meta.get("agent_path"),
        created_at=data.get("created_at") or "",
    )


# --- push / drain ---------------------------------------------------------


def _post_checkpoints(conn: _Connection, wire: list[dict[str, Any]]) -> bool:
    """POST ``wire`` to ``/public/v1/checkpoints/ingest``. Return True on 2xx.

    Retries connection errors, timeouts and 5xx with exponential backoff; a 4xx
    (e.g. 403 = domain not entitled to ``connected_state_plane``) is terminal and
    leaves the rows buffered for a later attempt (non-lossy).
    """
    import httpx

    url = f"{conn.target}/public/v1/checkpoints/ingest"
    payload = {"checkpoints": wire}

    for attempt in range(_MAX_ATTEMPTS):
        try:
            with httpx.Client(timeout=_TIMEOUT, verify=True) as client:
                resp = client.post(url, json=payload, headers=conn.headers)
            code = resp.status_code
            if 200 <= code < 300:
                return True
            if 400 <= code < 500:
                logger.warning(
                    "Plane rejected %d checkpoints with HTTP %d — not retrying; "
                    "left buffered (403 = domain not entitled to connected_state_plane).",
                    len(wire),
                    code,
                )
                return False
            logger.debug(
                "Checkpoint ingest HTTP %d (attempt %d/%d)", code, attempt + 1, _MAX_ATTEMPTS
            )
        except httpx.TransportError:
            logger.debug(
                "Checkpoint ingest transient error (attempt %d/%d)",
                attempt + 1,
                _MAX_ATTEMPTS,
                exc_info=True,
            )
        if attempt < _MAX_ATTEMPTS - 1:
            time.sleep(_BACKOFF_BASE * (2**attempt))
    return False


def _drain_checkpointer(cp: Any, conn: _Connection) -> None:
    """Drain one checkpointer's un-acked checkpoints to the plane until empty.

    Non-lossy: on a push failure we stop and leave rows ``synced=0`` for the next
    kick — we never abandon them.
    """
    from fastaiagent._internal.project import safe_get_project_id

    pid = safe_get_project_id()
    while True:
        try:
            rows = cp.fetch_unsynced(_DRAIN_LIMIT, pid)
        except Exception:
            logger.debug("checkpoint fetch_unsynced failed", exc_info=True)
            return
        if not rows:
            return
        wire = [_to_wire(r) for r in rows]
        if not _post_checkpoints(conn, wire):
            return  # transient/terminal — keep buffered (non-lossy)
        try:
            cp.mark_synced([w["checkpoint_id"] for w in wire])
        except Exception:
            logger.debug("checkpoint mark_synced failed", exc_info=True)
            return
        if len(rows) < _DRAIN_LIMIT:
            return


def _lock_for(cp: Any) -> threading.Lock:
    with _LOCKS_GUARD:
        lk = _LOCKS.get(cp)
        if lk is None:
            lk = threading.Lock()
            _LOCKS[cp] = lk
        return lk


def _drain_guarded(cp: Any) -> None:
    """Drain ``cp`` if connected and it has the replication surface; at most one
    drain runs per checkpointer at a time (bursts coalesce)."""
    from fastaiagent.client import _connection

    if not _connection.is_connected:
        return
    if not (hasattr(cp, "fetch_unsynced") and hasattr(cp, "mark_synced")):
        return
    lock = _lock_for(cp)
    if not lock.acquire(blocking=False):
        return  # a drain is already running; it will see the freshly-written rows
    try:
        _drain_checkpointer(cp, _connection)
    finally:
        lock.release()


def kick(cp: Any) -> None:
    """Fire-and-forget background drain of ``cp`` (call after a checkpoint write).

    Returns immediately — the POST + retry run on a daemon thread so the agent
    hot path never blocks. No-op when not connected.
    """
    try:
        from fastaiagent.client import _connection

        if not _connection.is_connected:
            return
        threading.Thread(target=_drain_guarded, args=(cp,), daemon=True).start()
    except Exception:
        logger.debug("checkpoint drain kick failed", exc_info=True)


def drain_all_async() -> None:
    """Drain every registered checkpointer on a daemon thread (called by
    ``connect()`` to flush any backlog written while disconnected)."""

    def _run() -> None:
        for cp in list(_REGISTRY):
            _drain_guarded(cp)

    try:
        threading.Thread(target=_run, daemon=True).start()
    except Exception:
        logger.debug("checkpoint drain_all_async failed", exc_info=True)


def drain_all_sync() -> None:
    """Best-effort synchronous drain of every registered checkpointer (called by
    ``disconnect()`` to flush before tearing the connection down)."""
    for cp in list(_REGISTRY):
        try:
            _drain_guarded(cp)
        except Exception:
            logger.debug("checkpoint drain_all_sync failed", exc_info=True)


# --- restore --------------------------------------------------------------


def fetch_latest_from_plane(execution_id: str, *, conn: Any | None = None) -> Checkpoint | None:
    """GET the latest checkpoint for ``execution_id`` from the plane, or None.

    Returns the highest-sequence checkpoint the plane holds (404 → None). The
    plane only **serves**; resuming happens locally — see :func:`restore_from_plane`.
    """
    if conn is None:
        from fastaiagent.client import _connection

        conn = _connection
    if not conn.is_connected:
        return None

    import httpx

    url = f"{conn.target}/public/v1/checkpoints/{execution_id}/latest"
    try:
        with httpx.Client(timeout=_TIMEOUT, verify=True) as client:
            resp = client.get(url, headers=conn.headers)
    except httpx.TransportError:
        logger.debug("checkpoint restore transient error", exc_info=True)
        return None
    if resp.status_code == 404:
        return None
    if not (200 <= resp.status_code < 300):
        logger.warning(
            "Checkpoint restore failed: HTTP %d for execution %s",
            resp.status_code,
            execution_id,
        )
        return None
    try:
        return _wire_to_checkpoint(resp.json())
    except Exception:
        logger.debug("checkpoint restore decode failed", exc_info=True)
        return None


def restore_from_plane(
    checkpointer: Any, execution_id: str, *, conn: Any | None = None
) -> Checkpoint | None:
    """Fetch the latest checkpoint from the plane and write it into ``checkpointer``.

    After this returns the restored :class:`Checkpoint`, a normal local resume
    (``chain.resume`` / ``agent.aresume``) proceeds against ``checkpointer`` —
    "resume from the plane copy". For an ``interrupted`` checkpoint the pending
    interrupt is re-created too, so a HITL resume can claim it. Returns None when
    not connected or the plane has no checkpoint for ``execution_id``.
    """
    ckpt = fetch_latest_from_plane(execution_id, conn=conn)
    if ckpt is None:
        return None
    if ckpt.status == "interrupted":
        pending = PendingInterrupt(
            execution_id=ckpt.execution_id,
            chain_name=ckpt.chain_name,
            node_id=ckpt.node_id,
            reason=ckpt.interrupt_reason or "",
            context=ckpt.interrupt_context,
            agent_path=ckpt.agent_path,
            created_at=ckpt.created_at,
        )
        checkpointer.record_interrupt(ckpt, pending)
    else:
        checkpointer.put(ckpt)
    return ckpt
