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
* **No age or count bound** (unlike the trace outbox): an active/paused run's
  durability must not be dropped just because it is old or the backlog is deep.
  A transient failure, a 5xx, an expired entitlement — all just leave the row
  buffered for the next kick / ``connect()`` flush.
* **One deliberate exception — the poison row** (durability audit D2). A
  checkpoint the door refuses on *payload* grounds is refused identically every
  time, so "leave it buffered" meant the next kick re-sent the same oldest batch,
  got the same refusal and stopped again — stranding every later checkpoint of
  every run on that checkpointer behind it, forever. Such a row is isolated by
  bisecting the batch, parked with its reason (``mark_quarantined``), and logged
  loudly. One checkpoint missing from the replica beats every later one stranded;
  it is the same trade the plane made when it moved to per-item refusals.

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
from typing import TYPE_CHECKING, Any, NamedTuple

from fastaiagent.chain.checkpoint import Checkpoint
from fastaiagent.checkpointers.protocol import PendingInterrupt

if TYPE_CHECKING:
    from fastaiagent.client import _Connection

logger = logging.getLogger(__name__)

# Transient-only retry (connection / timeout / 5xx); 4xx terminal for the batch.
_MAX_ATTEMPTS = 3
_BACKOFF_BASE = 0.5
_TIMEOUT = 10
_DRAIN_LIMIT = 200  # checkpoints per ingest batch; a backlog drains across loops

_VALID_STATUS = {"completed", "interrupted", "failed"}

# Outcomes of one POST to the ingest door.
_OK = "ok"
_RETRY_LATER = "retry_later"
_REFUSED = "refused"

#: 4xx codes that mean **this payload** is unacceptable, so re-sending it
#: unchanged can only fail again — the only codes that make a row a quarantine
#: candidate (durability audit D2).
#:
#: ⚠ **The exclusions are the important half.** 401 / 403 (no key, or the domain
#: is not entitled to ``connected_state_plane``), 404 / 405 (an older plane, or a
#: proxy that does not route this path), 408 and 429 are conditions of the
#: *connection*, not of any row. They apply identically to every checkpoint the
#: SDK will ever send, so quarantining on them would silently discard an entire
#: tenant's outbox — the one outcome worse than the stall this fix exists to
#: end. Those stay buffered forever, exactly as before, because for them
#: "forever" is the correct answer: entitlement gets granted, proxies get fixed.
_PAYLOAD_REFUSAL_CODES = frozenset({400, 409, 413, 422})

#: How much of a refusal body to keep as the quarantine reason. The plane names
#: the offending field and its limit in the first line or so; the rest is noise
#: in a database column an operator reads by eye.
_REASON_MAX_CHARS = 300


class _PostOutcome(NamedTuple):
    """What the ingest door said about one batch.

    ``rejections`` is the plane's **per-item** partial-success map (id → reason)
    parsed from a 2xx body. A plane on ``p54t1`` or later answers 201 and names
    what it dropped rather than failing the batch; those ids did *not* reach the
    replica, so recording them as cleanly synced would be a lie the operator
    cannot see.
    """

    outcome: str  # _OK | _RETRY_LATER | _REFUSED
    rejections: dict[str, str]
    reason: str  # human-readable, for the log and the quarantine record
    code: int | None


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
        # The plane has always had this key (capped at 40 chars) and the SDK
        # never sent it, which is why a finished run and one that died right
        # after its last step were indistinguishable there (audit D5).
        "step_type": row.get("step_type"),
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
    # A TIE-BREAK ONLY, and no longer the plane's primary ordering key.
    #
    # ⚠ This used to say the plane picked "latest" by sequence. It did, and that
    # was finding D1: ``sequence`` is the SQLite rowid of whichever local
    # database wrote the row, so a fresh store — the entire point of
    # restore-anywhere — starts at rowid 1 again and its NEW checkpoints carry
    # SMALLER sequences than the lost store's old ones. The plane kept serving
    # the stale pre-restore row, and a Postgres checkpointer (which sends no
    # sequence at all) was outranked by every sequenced SQLite row regardless of
    # recency.
    #
    # The plane now orders by the CLIENT clock — ``coalesce(created_at,
    # received_at)`` → ``received_at`` → ``sequence`` → ``id`` — mirroring the
    # SDK's own ``get_last``. The client clock leads deliberately: it is the only
    # term that stays right when a machine dies mid-run, the run is restored
    # elsewhere and finished, and the dead machine later drains its stale
    # backlog. We still send this because it remains a useful third-level
    # tie-break within one store.
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
        step_type=data.get("step_type"),
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


def _parse_rejections(resp: Any) -> dict[str, str]:
    """Per-item refusals named in a 2xx ingest body, as ``{checkpoint_id: reason}``.

    A plane on ``p54t1`` or later answers ``201`` with
    ``{"ingested", "rejected", "rejections": [{checkpoint_id, reason}]}`` instead
    of failing the whole batch. An older plane sends no such key and this is
    empty, which is the correct reading of its silence.
    """
    try:
        body = resp.json()
    except Exception:
        return {}
    if not isinstance(body, dict):
        return {}
    out: dict[str, str] = {}
    for item in body.get("rejections") or []:
        if not isinstance(item, dict):
            continue
        cid = item.get("checkpoint_id")
        if cid:
            out[str(cid)] = f"plane rejected: {str(item.get('reason') or 'no reason given')}"[
                :_REASON_MAX_CHARS
            ]
    return out


def _post_checkpoints(conn: _Connection, wire: list[dict[str, Any]]) -> _PostOutcome:
    """POST ``wire`` to ``/public/v1/checkpoints/ingest`` and classify the answer.

    Retries connection errors, timeouts and 5xx with exponential backoff. A 4xx
    is terminal for this batch either way, but *which* 4xx decides what the caller
    may do about it: a payload-shaped code (see :data:`_PAYLOAD_REFUSAL_CODES`)
    makes the rows quarantine candidates, while an auth / routing / throttle code
    leaves them buffered for a later attempt, unchanged from before.
    """
    import httpx

    url = f"{conn.target}/public/v1/checkpoints/ingest"
    payload = {"checkpoints": wire}
    reason = "no response from the plane"

    for attempt in range(_MAX_ATTEMPTS):
        try:
            with httpx.Client(timeout=_TIMEOUT, verify=True) as client:
                resp = client.post(url, json=payload, headers=conn.headers)
            code = resp.status_code
            if 200 <= code < 300:
                return _PostOutcome(_OK, _parse_rejections(resp), "", code)
            if 400 <= code < 500:
                detail = (resp.text or "").strip()[:_REASON_MAX_CHARS]
                reason = f"HTTP {code} from /checkpoints/ingest: {detail or '(no body)'}"
                if code in _PAYLOAD_REFUSAL_CODES:
                    return _PostOutcome(_REFUSED, {}, reason, code)
                logger.warning(
                    "Plane rejected %d checkpoints with HTTP %d — not retrying; left "
                    "buffered for a later attempt (403 = domain not entitled to "
                    "connected_state_plane). %s",
                    len(wire),
                    code,
                    detail,
                )
                return _PostOutcome(_RETRY_LATER, {}, reason, code)
            reason = f"HTTP {code} from /checkpoints/ingest"
            logger.debug(
                "Checkpoint ingest HTTP %d (attempt %d/%d)", code, attempt + 1, _MAX_ATTEMPTS
            )
        except httpx.TransportError as exc:
            reason = f"transport error: {type(exc).__name__}"
            logger.debug(
                "Checkpoint ingest transient error (attempt %d/%d)",
                attempt + 1,
                _MAX_ATTEMPTS,
                exc_info=True,
            )
        if attempt < _MAX_ATTEMPTS - 1:
            time.sleep(_BACKOFF_BASE * (2**attempt))
    return _PostOutcome(_RETRY_LATER, {}, reason, None)


def _isolate_refused_batch(
    conn: _Connection, wire: list[dict[str, Any]], refusal: _PostOutcome
) -> tuple[list[str], dict[str, str]]:
    """Bisect a payload-refused batch until each offender stands alone.

    Returns ``(synced_ids, quarantined)`` — the rows that got through on a
    re-send, and the ids that were still refused on their own, mapped to why.

    Why bisect rather than park the whole batch: the drain sends up to
    :data:`_DRAIN_LIMIT` rows at once, so a single bad checkpoint would otherwise
    cost 200 good ones — including healthy unrelated runs written after it. Two
    posts per split, ``O(k log n)`` for ``k`` offenders, and only ever on the
    failure path.

    A half that comes back transient mid-bisect is simply left buffered: not
    marked, not parked, retried on the next kick. Giving up on a row requires
    proof that the door refuses *it*, never an inference from a network blip.
    """
    if len(wire) == 1:
        return [], {wire[0]["checkpoint_id"]: refusal.reason[:_REASON_MAX_CHARS]}

    mid = len(wire) // 2
    synced: list[str] = []
    quarantined: dict[str, str] = {}
    for half in (wire[:mid], wire[mid:]):
        result = _post_checkpoints(conn, half)
        if result.outcome == _OK:
            quarantined.update(result.rejections)
            synced.extend(
                w["checkpoint_id"] for w in half if w["checkpoint_id"] not in result.rejections
            )
        elif result.outcome == _REFUSED:
            half_synced, half_bad = _isolate_refused_batch(conn, half, result)
            synced.extend(half_synced)
            quarantined.update(half_bad)
        # _RETRY_LATER: leave this half buffered for the next kick.
    return synced, quarantined


def _drain_checkpointer(cp: Any, conn: _Connection) -> None:
    """Drain one checkpointer's un-acked checkpoints to the plane until empty.

    Rows are marked ``synced`` only after a confirmed 2xx, and a transient
    failure just leaves them buffered for the next kick — the outbox has no age
    or count bound, unlike the trace exporter's.

    The **one** deliberate exception is a payload-shaped 4xx (durability audit
    D2). Such a refusal is deterministic, so leaving the rows buffered meant the
    next kick re-fetched the same oldest batch, got the same refusal and stopped
    again — permanently stranding every later checkpoint of every run on this
    checkpointer behind one bad row. The offenders are isolated and parked
    instead. The trade is the plane's own: one checkpoint missing from the
    replica beats every later one stranded. A checkpointer without
    ``mark_quarantined`` keeps the old behaviour rather than losing a row it
    cannot record the loss of.
    """
    from fastaiagent._internal.project import safe_get_project_id

    pid = safe_get_project_id()
    can_quarantine = hasattr(cp, "mark_quarantined")
    while True:
        try:
            rows = cp.fetch_unsynced(_DRAIN_LIMIT, pid)
        except Exception:
            logger.debug("checkpoint fetch_unsynced failed", exc_info=True)
            return
        if not rows:
            return
        wire = [_to_wire(r) for r in rows]
        result = _post_checkpoints(conn, wire)

        if result.outcome == _RETRY_LATER:
            return  # keep buffered — the condition is not this batch's fault
        if result.outcome == _REFUSED:
            if not can_quarantine:
                logger.warning(
                    "Plane refused %d checkpoints (%s) and this checkpointer cannot "
                    "quarantine a poison row — the outbox will stall behind it. "
                    "Implement mark_quarantined() to let the drain advance.",
                    len(wire),
                    result.reason,
                )
                return
            synced, quarantined = _isolate_refused_batch(conn, wire, result)
        else:
            quarantined = dict(result.rejections)
            if quarantined and not can_quarantine:
                # No way to record *why* the row never landed, so fall back to the
                # old reading of a 2xx: the batch is done. Better a lost reason
                # than a row re-sent forever to a door that keeps dropping it.
                quarantined = {}
            synced = [w["checkpoint_id"] for w in wire if w["checkpoint_id"] not in quarantined]

        for cid, why in quarantined.items():
            logger.warning(
                "Checkpoint %s will never replicate and has been quarantined: %s. "
                "The plane's copy of this run is incomplete.",
                cid,
                why,
            )
        try:
            if synced:
                cp.mark_synced(synced)
            if quarantined:
                cp.mark_quarantined(quarantined)
        except Exception:
            logger.debug("checkpoint mark_synced/mark_quarantined failed", exc_info=True)
            return
        if not synced and not quarantined:
            return  # nothing advanced (every half transient) — do not spin
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
    # security_audit_2 N7: honor the checkpoint-egress opt-out. Local durability
    # (SQLite/Postgres) is untouched — only replication of state to the plane is
    # suppressed. Single choke point for the write-kick, connect-drain, and
    # disconnect-drain paths.
    if not getattr(_connection, "export_checkpoints", True):
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

    "Latest" is by the **client clock** — ``coalesce(created_at, received_at)``,
    then ``received_at``, then ``sequence``, then ``id`` — which mirrors the
    SDK's own :meth:`get_last` so an operator reading the console and a resume
    reading the wire can never see different histories of the same run. It is
    NOT the highest ``sequence``; that was finding D1 and it broke restore across
    stores (see the ``_seq`` note in :func:`_to_wire`).

    404 → None. The plane only **serves**; resuming happens locally — see
    :func:`restore_from_plane`.
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


def restore_if_missing(checkpointer: Any, execution_id: str) -> Checkpoint | None:
    """Pull a run from the plane when the local store has never seen it (audit D4).

    Called at the top of every ``resume`` path. Until this existed
    :func:`restore_from_plane` was a helper nothing called: ``aresume``,
    ``Chain.resume``, the CLI and the local UI all consulted only local storage,
    so on a fresh machine a resume failed even though the plane was holding the
    state — which made the documented "restore anywhere" story false in exactly
    the situation it was written for.

    **Only when missing.** Never overwrite: ``SQLiteCheckpointer.put`` is a plain
    INSERT on its primary key (and since audit D3, so is Postgres), so restoring
    over a row the store already has would raise. Just as importantly, the local
    copy is the source of truth while it exists — the plane is a replica, and a
    resume must never prefer the replica to a run's own machine.

    No-ops when not connected, so a disconnected resume fails exactly as before.
    Set ``FASTAIAGENT_RESTORE_FROM_PLANE=0`` to keep it that way while connected:
    the restore resurrects a run whose local checkpoints were deliberately
    deleted, which is right for disaster recovery and wrong for an erasure
    request. That is a deployment-wide policy, which is why it is an environment
    switch and not a per-call argument.

    Returns the restored checkpoint, or None when nothing was restored.
    """
    import os

    if os.environ.get("FASTAIAGENT_RESTORE_FROM_PLANE") == "0":
        return None
    try:
        from fastaiagent.client import _connection

        if not _connection.is_connected:
            return None
        if checkpointer.get_last(execution_id) is not None:
            return None
    except Exception:
        logger.debug("restore-if-missing precheck failed", exc_info=True)
        return None

    restored = restore_from_plane(checkpointer, execution_id)
    if restored is not None:
        logger.info(
            "Restored execution %s from the plane (%s at %s) — the local store had "
            "no record of it.",
            execution_id,
            restored.status,
            restored.node_id,
        )
    return restored


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
