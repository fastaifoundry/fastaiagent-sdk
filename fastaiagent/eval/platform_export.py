"""Durable outbox that pushes Agent-CI verdicts to the control plane (Part D).

Contract — **metadata only**. A run's aggregates, gate verdict, thresholds, git
provenance and per-case scorer verdicts travel; ``input`` / ``expected_output`` /
``actual_output`` **never do**. The plane joins content via trace ingest using the
per-case ``trace_id``, so it corroborates verdicts against traces it received
independently rather than storing a second copy of potentially sensitive data.
This mirrors the HITL precedent (``trace/hitl_export.py`` sends ``context=None``).

Flow, identical in shape to the HITL and checkpoint outboxes:

* ``EvalResults.persist_local()`` writes the run ``synced=0`` (the durable source
  of truth) — but only when export is on; a disabled install writes ``synced=1``
  so it never grows an outbox that can't drain.
* ``record_gate_result()`` attaches the gate verdict and calls
  :func:`record_eval_run_for_export`, which kicks a fire-and-forget drain. A run
  has no ``gate_outcome`` before that point and the plane requires one, so the
  gate — not the persist — is what makes a run exportable.
* :meth:`EvalRunExporter.export` drains ``synced=0``, POSTs to
  ``/public/v1/eval/runs/ingest``, and marks ``synced=1`` **only after a
  confirmed 2xx**. The plane is idempotent by ``run_id``, so a re-send is free.

Egress posture: eval export is **egress**, not enforcement, so the local flag
always wins — ``connect(export_evals=False)`` is honored and the plane cannot
override it (exactly like ``export_traces``). What the plane does instead is
*mark* agents that produce no eval evidence; the SDK reports its posture at
enroll so "export disabled" stays distinct from "not running evals".
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from pydantic import BaseModel, Field

if TYPE_CHECKING:  # pragma: no cover
    from opentelemetry.sdk.trace import ReadableSpan

    from fastaiagent.client import _Connection

logger = logging.getLogger(__name__)

# One-shot transparency notice: state the contract the first time anything
# actually leaves the machine, then stay quiet. Repeated notices become wallpaper.
_contract_logged = False


def eval_export_enabled() -> bool:
    """Whether persisted eval runs should be queued for the plane.

    Resolution order: ``connect(export_evals=...)`` kwarg > ``FASTAIAGENT_EXPORT_EVALS``
    env > default on. Always False when disconnected — no plane, nothing to send.
    """
    try:
        from fastaiagent.client import _connection

        if not _connection.is_connected:
            return False
        flag = getattr(_connection, "export_evals", None)
        if flag is not None:
            return bool(flag)
        env = os.environ.get("FASTAIAGENT_EXPORT_EVALS")
        if env is not None:
            return env.lower() in ("1", "true")
        return True
    except Exception:  # pragma: no cover — posture check must never raise
        return False


class EvalCasePayload(BaseModel):
    """One case's verdict. Scorer outcomes + the trace join key — never content."""

    case_id: str | None = None
    ordinal: int | None = None
    per_scorer: dict[str, Any] | None = None
    trace_id: str | None = None
    error: str | None = None


class EvalRunPayload(BaseModel):
    """One Agent-CI run's verdict. Field names mirror the plane's ingest schema."""

    run_id: str
    run_name: str | None = None
    dataset_name: str | None = None
    agent_name: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    pass_count: int = 0
    fail_count: int = 0
    pass_rate: float | None = None
    errored_count: int = 0
    error_rate: float | None = None
    gate_outcome: str = "passed"
    thresholds: dict[str, Any] | None = None
    scorers: list[str] | None = None
    git_sha: str | None = None
    git_branch: str | None = None
    baseline: dict[str, Any] | None = None
    sdk_version: str | None = None
    instance_id: str | None = None
    cases: list[EvalCasePayload] = Field(default_factory=list)

    def to_wire(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def _jload(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


class EvalRunStore:
    """The ``eval_runs`` / ``eval_cases`` tables viewed as an outbox."""

    def __init__(self, db_path: str | Path | None = None):
        from fastaiagent._internal.config import get_config
        from fastaiagent.ui.db import init_local_db

        self.db_path = str(db_path) if db_path is not None else get_config().local_db_path
        # init_local_db runs the migration ladder, so ``synced`` exists even when
        # the store is the first thing to touch the DB.
        self._db = init_local_db(self.db_path)

    def _row_to_payload(self, row: dict[str, Any]) -> EvalRunPayload:
        """Build one wire payload from a persisted run + its cases.

        NULLs coerce to field defaults so a single partial row can never fail
        validation and abort the whole background drain.
        """
        from fastaiagent._version import __version__

        meta = _jload(row.get("metadata")) or {}
        if not isinstance(meta, dict):
            meta = {}
        run_id = row["run_id"]

        case_rows = self._db.fetchall(
            "SELECT case_id, ordinal, per_scorer, trace_id, error FROM eval_cases "
            "WHERE run_id = ? ORDER BY ordinal",
            (run_id,),
        )
        cases: list[EvalCasePayload] = []
        errored = 0
        for c in case_rows:
            if c.get("error"):
                errored += 1
            cases.append(
                EvalCasePayload(
                    case_id=c.get("case_id"),
                    ordinal=c.get("ordinal"),
                    # NOTE: per_scorer only — no input/expected/actual, ever.
                    per_scorer=_jload(c.get("per_scorer")) or None,
                    trace_id=c.get("trace_id"),
                    error=c.get("error"),
                )
            )
        total = len(cases)
        instance_id = None
        try:
            from fastaiagent._internal.instance import get_instance_id

            instance_id = get_instance_id()
        except Exception:  # pragma: no cover
            logger.debug("instance_id unavailable for eval export", exc_info=True)

        return EvalRunPayload(
            run_id=run_id,
            run_name=row.get("run_name"),
            dataset_name=row.get("dataset_name"),
            agent_name=row.get("agent_name"),
            started_at=row.get("started_at"),
            finished_at=row.get("finished_at"),
            pass_count=int(row.get("pass_count") or 0),
            fail_count=int(row.get("fail_count") or 0),
            pass_rate=row.get("pass_rate"),
            errored_count=errored,
            error_rate=round(errored / total, 4) if total else 0.0,
            gate_outcome=str(meta.get("gate_outcome") or "passed"),
            thresholds=meta.get("thresholds"),
            scorers=_jload(row.get("scorers")) or None,
            git_sha=meta.get("git_sha"),
            git_branch=meta.get("git_branch"),
            baseline=meta.get("baseline"),
            sdk_version=__version__,
            instance_id=instance_id,
            cases=cases,
        )

    def fetch_unsynced(self, limit: int, project_id: str | None = None) -> list[EvalRunPayload]:
        if project_id is None:
            rows = self._db.fetchall(
                "SELECT * FROM eval_runs WHERE synced = 0 ORDER BY started_at LIMIT ?",
                (limit,),
            )
        else:
            rows = self._db.fetchall(
                "SELECT * FROM eval_runs WHERE synced = 0 AND project_id = ? "
                "ORDER BY started_at LIMIT ?",
                (project_id, limit),
            )
        return [self._row_to_payload(dict(r)) for r in rows]

    def mark_synced(self, run_ids: list[str]) -> None:
        # Chunked to stay under SQLite's 999 bound-variable limit.
        for i in range(0, len(run_ids), 500):
            chunk = run_ids[i : i + 500]
            placeholders = ",".join("?" * len(chunk))
            self._db.execute(
                f"UPDATE eval_runs SET synced = 1 WHERE run_id IN ({placeholders})",
                tuple(chunk),
            )

    def count_unsynced(self, project_id: str | None = None) -> int:
        if project_id is None:
            row = self._db.fetchone("SELECT COUNT(*) AS n FROM eval_runs WHERE synced = 0")
        else:
            row = self._db.fetchone(
                "SELECT COUNT(*) AS n FROM eval_runs WHERE synced = 0 AND project_id = ?",
                (project_id,),
            )
        return int((row or {}).get("n") or 0)

    def enforce_buffer_bound(
        self,
        max_unsynced: int,
        max_age_days: int,
        project_id: str | None = None,
    ) -> int:
        """Abandon un-pushable runs so the buffer can't grow without bound.

        Abandon == ``mark_synced``, never DELETE: the rows stay in local.db and in
        the Local UI; they simply stop being re-send candidates.
        """
        if project_id is None:
            from fastaiagent._internal.project import safe_get_project_id

            project_id = safe_get_project_id()
        abandoned = 0
        cutoff = (datetime.now(tz=timezone.utc) - timedelta(days=max_age_days)).isoformat()
        aged = self._db.fetchall(
            "SELECT run_id FROM eval_runs WHERE synced = 0 AND project_id = ? AND started_at < ?",
            (project_id, cutoff),
        )
        if aged:
            ids = [r["run_id"] for r in aged]
            self.mark_synced(ids)
            abandoned += len(ids)

        excess = self.count_unsynced(project_id) - max_unsynced
        if excess > 0:
            oldest = self._db.fetchall(
                "SELECT run_id FROM eval_runs WHERE synced = 0 AND project_id = ? "
                "ORDER BY started_at ASC LIMIT ?",
                (project_id, excess),
            )
            ids = [r["run_id"] for r in oldest]
            if ids:
                self.mark_synced(ids)
                abandoned += len(ids)
        return abandoned

    def close(self) -> None:
        try:
            self._db.close()
        except Exception:  # pragma: no cover
            logger.debug("EvalRunStore close failed", exc_info=True)


class EvalRunExporter(SpanExporter):
    """Drains the eval outbox to the plane.

    Subclasses ``SpanExporter`` only so it can ride a ``BatchSpanProcessor`` for
    periodic flushes; the ``spans`` argument is ignored — it is a trigger, not data.
    """

    _MAX_ATTEMPTS = 3
    _BACKOFF_BASE = 0.5
    _TIMEOUT = 10
    _DRAIN_LIMIT = 200
    # Eval runs are low-volume and compliance-relevant: generous HITL-like bounds
    # rather than the tighter trace ones.
    _MAX_UNSYNCED = 50_000
    _MAX_AGE_DAYS = 30

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._store: EvalRunStore | None = None
        # Pin a specific DB when the caller persisted somewhere other than the
        # configured default (e.g. ``fastaiagent eval run --db ...``); otherwise
        # the drain would read the default file and silently export nothing.
        self._db_path = db_path

    def _get_store(self) -> EvalRunStore:
        if self._store is None:
            self._store = EvalRunStore(db_path=self._db_path)
        return self._store

    def export(self, spans: Sequence[ReadableSpan] = ()) -> SpanExportResult:
        from fastaiagent.client import _connection

        if not _connection.is_connected or not eval_export_enabled():
            return SpanExportResult.SUCCESS
        try:
            from fastaiagent._internal.project import safe_get_project_id

            store = self._get_store()
            pid = safe_get_project_id()
            pending = store.fetch_unsynced(limit=self._DRAIN_LIMIT, project_id=pid)
            if pending:
                wire = [p.to_wire() for p in pending]
                if self._post_with_retry(_connection, wire):
                    store.mark_synced([p.run_id for p in pending])
                    _log_contract_once(_connection)
            dropped = store.enforce_buffer_bound(
                self._MAX_UNSYNCED, self._MAX_AGE_DAYS, project_id=pid
            )
            if dropped:
                logger.warning(
                    "Eval export buffer bound hit: abandoned %d un-acked run(s). "
                    "They remain in local.db and the Local UI; they are no longer "
                    "re-send candidates.",
                    dropped,
                )
        except Exception:
            logger.debug("Eval run export drain failed", exc_info=True)
        # Always SUCCESS: the durable buffer owns retry, not the processor.
        return SpanExportResult.SUCCESS

    def _post_with_retry(self, conn: _Connection, wire: list[dict[str, Any]]) -> bool:
        """POST to ``/public/v1/eval/runs/ingest``. True on a confirmed 2xx."""
        import httpx

        url = f"{conn.target}/public/v1/eval/runs/ingest"
        payload = {"runs": wire}

        for attempt in range(self._MAX_ATTEMPTS):
            try:
                with httpx.Client(timeout=self._TIMEOUT, verify=True) as client:
                    resp = client.post(url, json=payload, headers=conn.headers)
                code = resp.status_code
                if 200 <= code < 300:
                    return True
                if 400 <= code < 500:
                    logger.warning(
                        "Platform rejected %d eval run(s) with HTTP %d — not retrying; "
                        "left buffered for the bound to age out. 403 = the API key "
                        "lacks the 'eval:execute' scope, or the domain is not entitled "
                        "to connected_state_plane. 404 = the plane predates the eval "
                        "ingest endpoint (SDK 1.49.0 needs wire v1.6).",
                        len(wire),
                        code,
                    )
                    return False
                logger.debug(
                    "Eval export HTTP %d (attempt %d/%d)",
                    code,
                    attempt + 1,
                    self._MAX_ATTEMPTS,
                )
            except httpx.TransportError:
                logger.debug(
                    "Eval export transient error (attempt %d/%d)",
                    attempt + 1,
                    self._MAX_ATTEMPTS,
                    exc_info=True,
                )
            if attempt < self._MAX_ATTEMPTS - 1:
                time.sleep(self._BACKOFF_BASE * (2**attempt))
        return False

    def shutdown(self) -> None:
        try:
            if self._store is not None:
                self._store.close()
                self._store = None
        except Exception:  # pragma: no cover
            logger.debug("Eval exporter shutdown failed", exc_info=True)


def _log_contract_once(conn: _Connection) -> None:
    """State what leaves the machine, once per process."""
    global _contract_logged
    if _contract_logged:
        return
    _contract_logged = True
    logger.info(
        "Eval export active -> %s: sending run aggregates, gate outcome, thresholds, "
        "git provenance and per-case scorer verdicts + trace_ids. Case inputs and "
        "outputs are NOT sent. Preview with `fastaiagent eval export --dry-run`; "
        "disable with connect(export_evals=False).",
        conn.target,
    )


_exporter: EvalRunExporter | None = None
_exporter_lock = threading.Lock()


def get_eval_exporter() -> EvalRunExporter:
    global _exporter
    if _exporter is None:
        with _exporter_lock:
            if _exporter is None:
                _exporter = EvalRunExporter()
    return _exporter


def _kick_drain(db_path: str | Path | None = None) -> None:
    try:
        if db_path is None:
            get_eval_exporter().export([])
            return
        # Custom DB: drain it with a transient exporter, then release the handle.
        exporter = EvalRunExporter(db_path=db_path)
        try:
            exporter.export([])
        finally:
            exporter.shutdown()
    except Exception:  # pragma: no cover
        logger.debug("Eval export drain kick failed", exc_info=True)


def record_eval_run_for_export(run_id: str, *, db_path: str | Path | None = None) -> None:
    """Offer a gated run to the plane. Strict no-op when disconnected or disabled.

    The row is already durable (``persist_local`` wrote it ``synced=0``); this only
    kicks a background drain. Never raises into the eval path.
    """
    try:
        from fastaiagent.client import _connection

        if not _connection.is_connected or not eval_export_enabled():
            return
        threading.Thread(target=_kick_drain, args=(db_path,), daemon=True).start()
    except Exception:  # pragma: no cover
        logger.debug("Eval export kick failed for run %s", run_id, exc_info=True)


def build_payloads(
    limit: int = 10,
    *,
    run_id: str | None = None,
    db_path: str | Path | None = None,
    unsynced_only: bool = False,
) -> list[dict[str, Any]]:
    """Build wire payloads without sending anything — powers ``eval export --dry-run``."""
    store = EvalRunStore(db_path=db_path)
    try:
        if run_id:
            rows = store._db.fetchall("SELECT * FROM eval_runs WHERE run_id = ?", (run_id,))
        elif unsynced_only:
            rows = store._db.fetchall(
                "SELECT * FROM eval_runs WHERE synced = 0 ORDER BY started_at DESC LIMIT ?",
                (limit,),
            )
        else:
            rows = store._db.fetchall(
                "SELECT * FROM eval_runs ORDER BY started_at DESC LIMIT ?", (limit,)
            )
        return [store._row_to_payload(dict(r)).to_wire() for r in rows]
    finally:
        store.close()
