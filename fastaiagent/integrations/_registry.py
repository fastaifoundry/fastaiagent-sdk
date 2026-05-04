"""Shared infrastructure for the universal-harness integrations.

Holds module-private helpers used by all three integration modules
(``langchain``, ``crewai``, ``pydanticai``):

* ``GuardrailBlocked`` — the exception raised by ``with_guardrails(...)``
  wrappers when a blocking guardrail rejects an input or output. Defined
  here so ``except GuardrailBlocked`` catches across frameworks.

* ``upsert_agent`` / ``attach`` / ``fetch_agent`` / ``fetch_attachments``
  — the external-agent registry. Backed by the ``external_agents`` and
  ``external_agent_attachments`` tables added in v7 of
  ``fastaiagent/ui/db.py``. ``register_agent()`` (per integration)
  writes one row per external agent; harness helpers
  (``with_guardrails``, ``prompt_from_registry``, ``kb_as_retriever`` /
  ``kb_as_tool``) write attachment rows. The dependency-graph endpoint
  reads both and merges them with the in-memory ``ctx.runners`` lookup.

The implementations are best-effort — every helper swallows storage
errors so a misconfigured local DB never breaks the wrapped agent's
hot path. Tests assert the rows actually land via ``fetch_*``.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


class GuardrailBlocked(RuntimeError):  # noqa: N818  (public API; renaming breaks callers)
    """Raised by ``with_guardrails(...)`` wrappers when a blocking
    guardrail rejects an input or output.

    The wrapper logs a ``guardrail_events`` row before raising, so the
    Local UI's Guardrail Events page surfaces the rejection regardless
    of whether the caller catches the exception.
    """


def _open_db() -> Any:
    """Return a connection to the same ``local.db`` everything else uses.

    Honours ``FASTAIAGENT_LOCAL_DB`` via ``get_config()``. Returns
    ``None`` (and logs at debug) if init fails — callers are
    responsible for handling that case quietly so the harness doesn't
    fail the wrapped agent's hot path.
    """
    try:
        from fastaiagent._internal.config import get_config
        from fastaiagent.ui.db import init_local_db

        return init_local_db(get_config().local_db_path)
    except Exception:
        logger.debug("external-agent registry: open_db failed", exc_info=True)
        return None


def _now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _project_id() -> str:
    try:
        from fastaiagent._internal.project import safe_get_project_id

        return safe_get_project_id() or ""
    except Exception:
        return ""


def upsert_agent(
    name: str,
    framework: str,
    *,
    model: str | None = None,
    provider: str | None = None,
    system_prompt: str | None = None,
    topology: dict[str, Any] | None = None,
) -> None:
    """Insert or refresh an ``external_agents`` row.

    Idempotent — calling twice with the same ``name`` updates the
    existing row rather than creating a duplicate. The Phase 1
    ``register_agent()`` per-framework wrappers call this once per
    registration; the harness helpers (``with_guardrails`` / etc.)
    fall back to inserting a stub row tagged ``framework="unknown"``
    when no prior registration exists, which is later overwritten if
    the user calls ``register_agent``.
    """
    db = _open_db()
    if db is None:
        return
    try:
        topology_json = json.dumps(topology or {}, default=str)
        existing = db.fetchone(
            "SELECT framework FROM external_agents WHERE name = ?", (name,)
        )
        if existing is None:
            db.execute(
                """INSERT INTO external_agents
                   (name, framework, model, provider, system_prompt,
                    topology_json, metadata_json, created_at, updated_at,
                    project_id)
                   VALUES (?, ?, ?, ?, ?, ?, '{}', ?, ?, ?)""",
                (
                    name,
                    framework,
                    model,
                    provider,
                    (system_prompt or "")[:1_000] if system_prompt else None,
                    topology_json,
                    _now(),
                    _now(),
                    _project_id(),
                ),
            )
        else:
            # Don't downgrade a real framework tag back to ``unknown``.
            current = existing["framework"]
            new_framework = (
                current
                if framework == "unknown" and current != "unknown"
                else framework
            )
            db.execute(
                """UPDATE external_agents
                   SET framework = ?, model = COALESCE(?, model),
                       provider = COALESCE(?, provider),
                       system_prompt = COALESCE(?, system_prompt),
                       topology_json = ?, updated_at = ?
                   WHERE name = ?""",
                (
                    new_framework,
                    model,
                    provider,
                    (system_prompt or None),
                    topology_json,
                    _now(),
                    name,
                ),
            )
    except Exception:
        logger.debug("upsert_agent failed", exc_info=True)
    finally:
        try:
            db.close()
        except Exception:
            pass


def attach(
    agent_name: str,
    kind: str,
    ref_name: str,
    *,
    position: str | None = None,
    version: str | None = None,
) -> None:
    """Insert an ``external_agent_attachments`` row.

    The ``(agent_name, kind, ref_name, position)`` quadruple is unique
    in the schema, so re-attaching is a no-op (the conflict is ignored
    via ``INSERT OR IGNORE``). When the agent_name doesn't exist in
    ``external_agents`` yet, a stub row is created with
    ``framework="unknown"`` so the dependency-graph endpoint can still
    render the attachment.
    """
    db = _open_db()
    if db is None:
        return
    try:
        existing = db.fetchone(
            "SELECT name FROM external_agents WHERE name = ?", (agent_name,)
        )
        if existing is None:
            # Lazy-create the parent row so attachments still surface.
            db.execute(
                """INSERT INTO external_agents
                   (name, framework, model, provider, system_prompt,
                    topology_json, metadata_json, created_at, updated_at,
                    project_id)
                   VALUES (?, 'unknown', NULL, NULL, NULL, '{}', '{}',
                           ?, ?, ?)""",
                (agent_name, _now(), _now(), _project_id()),
            )
        db.execute(
            """INSERT OR IGNORE INTO external_agent_attachments
               (attachment_id, agent_name, kind, ref_name, position,
                version, metadata_json, created_at, project_id)
               VALUES (?, ?, ?, ?, ?, ?, '{}', ?, ?)""",
            (
                uuid.uuid4().hex,
                agent_name,
                kind,
                ref_name,
                position,
                version,
                _now(),
                _project_id(),
            ),
        )
    except Exception:
        logger.debug("attach failed", exc_info=True)
    finally:
        try:
            db.close()
        except Exception:
            pass


def fetch_agent(name: str) -> dict[str, Any] | None:
    """Read a single ``external_agents`` row, or ``None`` if absent."""
    db = _open_db()
    if db is None:
        return None
    try:
        row = db.fetchone(
            """SELECT name, framework, model, provider, system_prompt,
                      topology_json, metadata_json, created_at, updated_at
               FROM external_agents WHERE name = ?""",
            (name,),
        )
        if row is None:
            return None
        out = dict(row)
        for key in ("topology_json", "metadata_json"):
            try:
                out[key.replace("_json", "")] = json.loads(out.pop(key) or "{}")
            except Exception:
                out[key.replace("_json", "")] = {}
        return out
    finally:
        try:
            db.close()
        except Exception:
            pass


def fetch_attachments(
    name: str, *, kind: str | None = None
) -> list[dict[str, Any]]:
    """List all attachments for ``name``, optionally filtered by ``kind``."""
    db = _open_db()
    if db is None:
        return []
    try:
        if kind is None:
            rows = db.fetchall(
                """SELECT kind, ref_name, position, version, created_at
                   FROM external_agent_attachments WHERE agent_name = ?
                   ORDER BY created_at""",
                (name,),
            )
        else:
            rows = db.fetchall(
                """SELECT kind, ref_name, position, version, created_at
                   FROM external_agent_attachments
                   WHERE agent_name = ? AND kind = ?
                   ORDER BY created_at""",
                (name, kind),
            )
        return [dict(r) for r in rows]
    finally:
        try:
            db.close()
        except Exception:
            pass


__all__ = [
    "GuardrailBlocked",
    "upsert_agent",
    "attach",
    "fetch_agent",
    "fetch_attachments",
]
