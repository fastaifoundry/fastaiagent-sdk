"""Stable per-install SDK instance identity (WS4 governance enrollment).

A single UUID minted once and persisted in the local SQLite store (the
``sdk_instance`` table, migration v14 in :mod:`fastaiagent.ui.db`), then reused
across runs/processes. The connect-time governance enroll push sends it as
``instance_id``; the plane upserts on ``(domain_id, project_id, instance_id)``,
so stability per install is the whole point.

Never raises: a read-only filesystem or any storage error falls back to a
process-ephemeral UUID so ``connect()`` / enroll never break. (An ephemeral id
only means that one degraded run looks like a new install to the plane's
coverage view — acceptable; enrollment is best-effort attestation, not durable
state.) Mirrors the never-raises contract of
:func:`fastaiagent._internal.project.safe_get_project_id`.
"""

from __future__ import annotations

import logging
import threading
import uuid
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_CACHED: str | None = None


def get_instance_id() -> str:
    """Return the stable install ``instance_id``, creating + persisting it once."""
    global _CACHED
    with _LOCK:
        if _CACHED is not None:
            return _CACHED
        try:
            from fastaiagent.ui.db import init_local_db

            db = init_local_db()  # resolves FASTAIAGENT_LOCAL_DB; applies v14 first
            row = db.fetchone("SELECT instance_id FROM sdk_instance WHERE id = 1")
            if row and row.get("instance_id"):
                _CACHED = str(row["instance_id"])
                return _CACHED
            new_id = uuid.uuid4().hex
            db.execute(
                "INSERT OR IGNORE INTO sdk_instance (id, instance_id, created_at) "
                "VALUES (1, ?, ?)",
                (new_id, datetime.now(timezone.utc).isoformat()),
            )
            # Re-read in case a concurrent writer won the INSERT OR IGNORE race —
            # both processes then converge on whichever row landed.
            row = db.fetchone("SELECT instance_id FROM sdk_instance WHERE id = 1")
            _CACHED = str(row["instance_id"]) if row and row.get("instance_id") else new_id
            return _CACHED
        except Exception:
            logger.debug(
                "instance_id persistence unavailable; using ephemeral id", exc_info=True
            )
            _CACHED = uuid.uuid4().hex  # process-ephemeral fallback
            return _CACHED


def reset_for_testing() -> None:
    """Clear the cached id (test fixtures call this between tests)."""
    global _CACHED
    with _LOCK:
        _CACHED = None
