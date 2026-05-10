"""Saved filter presets for the Traces page (Sprint 3).

Backed by the v1 ``saved_filters`` table, project-scoped via the v6
``project_id`` column. The ``filters`` column stores the filter object
verbatim as JSON so the frontend can round-trip without schema churn
when we add new filter fields.
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, ConfigDict, Field

from fastaiagent.ui.deps import get_context, project_filter, require_session


# security_review_1.md L3 — saved filter JSON used to flow through the
# API as an opaque ``dict[str, Any]``. Backend SQL was already
# parameterised, so server-side was safe, but the frontend was free to
# render whatever shape happened to be there. We now run reads through
# a *permissive* Pydantic model:
#
# * Documented top-level keys (``agent``, ``model``, …) are typed so
#   IDE / docs / future strictening have something to lean on.
# * ``extra="allow"`` keeps any extra keys in older saved presets
#   round-tripping unchanged — no migration, no break.
# * On any validation failure we fall back to the raw dict (still
#   safe — frontend treats it as untrusted), so a malformed historical
#   row never breaks the list endpoint.
class FilterValues(BaseModel):
    model_config = ConfigDict(extra="allow")

    agent: str | None = None
    model: str | None = None
    status: str | None = None
    project: str | None = None
    date_from: str | None = None
    date_to: str | None = None
    q: str | None = None
    has_error: bool | None = None

router = APIRouter(prefix="/api/filter-presets", tags=["filter-presets"])


# Looser than dataset names — presets are user-facing labels so spaces
# and basic punctuation are fine. Length-bounded so we don't store
# essays in the column.
_NAME_RE = re.compile(r"^[A-Za-z0-9 _\-.,'/()&]{1,80}$")


class FilterPreset(BaseModel):
    id: str
    name: str
    filters: dict[str, Any] = Field(default_factory=dict)
    created_at: str


class CreatePresetBody(BaseModel):
    name: str
    filters: dict[str, Any] = Field(default_factory=dict)


class UpdatePresetBody(BaseModel):
    """Either or both of name/filters can change."""

    name: str | None = None
    filters: dict[str, Any] | None = None


def _row_to_preset(row: dict[str, Any]) -> FilterPreset:
    raw_filters = row.get("filters") or "{}"
    try:
        filters = json.loads(raw_filters)
        if not isinstance(filters, dict):
            filters = {}
    except json.JSONDecodeError:
        filters = {}
    # L3 — validate against the permissive schema. On any failure we
    # fall through to the raw dict (forward-compat for filter shapes
    # added by future releases or hand-edited DB rows).
    try:
        validated = FilterValues.model_validate(filters).model_dump(
            exclude_none=False,
        )
    except Exception:
        validated = filters
    return FilterPreset(
        id=row["id"],
        name=row.get("name") or "",
        filters=validated if isinstance(validated, dict) else {},
        created_at=row.get("created_at") or "",
    )


def _validate_name(name: str) -> str:
    if not _NAME_RE.match(name):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "preset name must be 1-80 characters of letters/numbers/spaces "
            "and basic punctuation",
        )
    return name.strip()


@router.get("", response_model=list[FilterPreset])
def list_presets(
    request: Request, _user: str = Depends(require_session)
) -> list[FilterPreset]:
    """List the calling project's saved presets, newest first."""
    ctx = get_context(request)
    db = ctx.db()
    pid_clause, pid_params = project_filter(ctx)
    try:
        rows = db.fetchall(
            f"""SELECT id, name, filters, created_at
                FROM saved_filters
                WHERE 1=1 {pid_clause}
                ORDER BY created_at DESC""",
            tuple(pid_params),
        )
        return [_row_to_preset(r) for r in rows]
    finally:
        db.close()


@router.post("", response_model=FilterPreset, status_code=status.HTTP_201_CREATED)
def create_preset(
    request: Request,
    body: CreatePresetBody,
    _user: str = Depends(require_session),
) -> FilterPreset:
    name = _validate_name(body.name)
    ctx = get_context(request)
    db = ctx.db()
    try:
        preset_id = uuid.uuid4().hex
        created_at = datetime.now(tz=timezone.utc).isoformat()
        db.execute(
            """INSERT INTO saved_filters (id, name, filters, created_at, project_id)
               VALUES (?, ?, ?, ?, ?)""",
            (
                preset_id,
                name,
                json.dumps(body.filters, default=str),
                created_at,
                ctx.project_id or "",
            ),
        )
        return FilterPreset(
            id=preset_id,
            name=name,
            filters=body.filters,
            created_at=created_at,
        )
    finally:
        db.close()


@router.patch("/{preset_id}", response_model=FilterPreset)
def update_preset(
    request: Request,
    preset_id: str,
    body: UpdatePresetBody,
    _user: str = Depends(require_session),
) -> FilterPreset:
    ctx = get_context(request)
    db = ctx.db()
    pid_clause, pid_params = project_filter(ctx)
    try:
        existing = db.fetchone(
            f"SELECT id, name, filters, created_at FROM saved_filters "
            f"WHERE id = ? {pid_clause}",
            (preset_id, *pid_params),
        )
        if not existing:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "preset not found")

        new_name = _validate_name(body.name) if body.name is not None else existing.get("name")
        new_filters = (
            body.filters if body.filters is not None
            else json.loads(existing.get("filters") or "{}")
        )
        db.execute(
            f"UPDATE saved_filters SET name = ?, filters = ? "
            f"WHERE id = ? {pid_clause}",
            (
                new_name,
                json.dumps(new_filters, default=str),
                preset_id,
                *pid_params,
            ),
        )
        return FilterPreset(
            id=preset_id,
            name=new_name or "",
            filters=new_filters if isinstance(new_filters, dict) else {},
            created_at=existing.get("created_at") or "",
        )
    finally:
        db.close()


@router.delete("/{preset_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_preset(
    request: Request,
    preset_id: str,
    _user: str = Depends(require_session),
) -> Response:
    ctx = get_context(request)
    db = ctx.db()
    pid_clause, pid_params = project_filter(ctx)
    try:
        existing = db.fetchone(
            f"SELECT id FROM saved_filters WHERE id = ? {pid_clause}",
            (preset_id, *pid_params),
        )
        if not existing:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "preset not found")
        db.execute(
            f"DELETE FROM saved_filters WHERE id = ? {pid_clause}",
            (preset_id, *pid_params),
        )
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    finally:
        db.close()


__all__ = ["router"]
