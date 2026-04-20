"""Prompt registry browse + edit endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel

from fastaiagent._internal.errors import FragmentNotFoundError, PromptNotFoundError
from fastaiagent.prompt.registry import PromptRegistry
from fastaiagent.ui.deps import get_context, require_session

router = APIRouter(prefix="/api/prompts", tags=["prompts"])


def _registry(request: Request) -> PromptRegistry:
    ctx = get_context(request)
    return PromptRegistry(path=ctx.db_path)


@router.get("")
def list_prompts(
    request: Request, _user: str = Depends(require_session)
) -> dict[str, Any]:
    ctx = get_context(request)
    reg = _registry(request)
    prompts = reg.list()
    db = ctx.db()
    try:
        enriched: list[dict[str, Any]] = []
        for p in prompts:
            # Accept every prefix variant so this count works across traces
            # from older SDK releases too.
            trace_count = db.fetchone(
                """SELECT COUNT(DISTINCT trace_id) AS n
                   FROM spans
                   WHERE attributes LIKE ?
                      OR attributes LIKE ?
                      OR attributes LIKE ?""",
                (
                    f'%"fastaiagent.prompt.name": "{p["name"]}"%',
                    f'%"prompt.name": "{p["name"]}"%',
                    f'%"fastai.prompt.name": "{p["name"]}"%',
                ),
            )
            enriched.append(
                {
                    **p,
                    "linked_trace_count": int((trace_count or {}).get("n") or 0),
                    "registry_is_local": reg.is_local(),
                }
            )
        return {"rows": enriched, "registry_is_local": reg.is_local()}
    finally:
        db.close()


@router.get("/{slug}")
def get_prompt(
    request: Request,
    slug: str,
    _user: str = Depends(require_session),
) -> dict[str, Any]:
    reg = _registry(request)
    try:
        prompt = reg.load(slug)
    except PromptNotFoundError as e:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(e)) from e
    return {
        "slug": prompt.name,
        "latest_version": prompt.version,
        "template": prompt.template,
        "variables": prompt.variables,
        "metadata": prompt.metadata,
        "registry_is_local": reg.is_local(),
    }


@router.get("/{slug}/versions")
def list_versions(
    request: Request,
    slug: str,
    _user: str = Depends(require_session),
) -> dict[str, Any]:
    ctx = get_context(request)
    db = ctx.db()
    try:
        rows = db.fetchall(
            """SELECT slug, version, template, variables, created_at, created_by
               FROM prompt_versions
               WHERE slug = ?
               ORDER BY CAST(version AS INTEGER) ASC""",
            (slug,),
        )
        if not rows:
            raise HTTPException(
                status.HTTP_404_NOT_FOUND, f"Prompt '{slug}' not found"
            )
        return {"versions": rows}
    finally:
        db.close()


@router.get("/{slug}/versions/{version}")
def get_version(
    request: Request,
    slug: str,
    version: str,
    _user: str = Depends(require_session),
) -> dict[str, Any]:
    reg = _registry(request)
    try:
        prompt = reg.load(slug, version=int(version))
    except (PromptNotFoundError, ValueError) as e:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(e)) from e
    return {
        "slug": prompt.name,
        "version": prompt.version,
        "template": prompt.template,
        "variables": prompt.variables,
        "metadata": prompt.metadata,
    }


@router.get("/{slug}/diff")
def diff_versions(
    request: Request,
    slug: str,
    a: int,
    b: int,
    _user: str = Depends(require_session),
) -> dict[str, Any]:
    reg = _registry(request)
    try:
        return {"diff": reg.diff(slug, a, b)}
    except PromptNotFoundError as e:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(e)) from e


class EditRequest(BaseModel):
    template: str
    variables: list[str] | None = None
    metadata: dict[str, Any] | None = None


@router.put("/{slug}")
def update_prompt(
    request: Request,
    slug: str,
    body: EditRequest,
    _user: str = Depends(require_session),
) -> dict[str, Any]:
    reg = _registry(request)
    if not reg.is_local():
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            "This registry is external. Edit prompts via code or from the "
            "environment that owns this path.",
        )
    prompt = reg.register(
        name=slug,
        template=body.template,
        metadata=body.metadata or {},
    )
    return {
        "slug": prompt.name,
        "version": prompt.version,
        "template": prompt.template,
        "variables": prompt.variables,
    }


@router.get("/{slug}/lineage")
def lineage(
    request: Request,
    slug: str,
    _user: str = Depends(require_session),
) -> dict[str, Any]:
    ctx = get_context(request)
    db = ctx.db()
    try:
        likes = (
            f'%"fastaiagent.prompt.name": "{slug}"%',
            f'%"prompt.name": "{slug}"%',
            f'%"fastai.prompt.name": "{slug}"%',
        )
        trace_rows = db.fetchall(
            """SELECT DISTINCT trace_id FROM spans
               WHERE attributes LIKE ?
                  OR attributes LIKE ?
                  OR attributes LIKE ?""",
            likes,
        )
        eval_rows = db.fetchall(
            """SELECT DISTINCT run_id FROM eval_cases
               JOIN spans ON eval_cases.trace_id = spans.trace_id
               WHERE spans.attributes LIKE ?
                  OR spans.attributes LIKE ?
                  OR spans.attributes LIKE ?""",
            likes,
        )
        return {
            "trace_ids": [r["trace_id"] for r in trace_rows],
            "eval_run_ids": [r["run_id"] for r in eval_rows],
        }
    finally:
        db.close()


@router.get("/fragments/{name}")
def get_fragment(
    request: Request,
    name: str,
    _user: str = Depends(require_session),
) -> dict[str, Any]:
    reg = _registry(request)
    try:
        fragment = reg._storage.load_fragment(name)  # noqa: SLF001
    except FragmentNotFoundError as e:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(e)) from e
    return {"name": fragment.name, "content": fragment.content}
