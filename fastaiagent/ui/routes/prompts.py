"""Prompt registry browse + edit endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel

from fastaiagent._internal.errors import FragmentNotFoundError, PromptNotFoundError
from fastaiagent.prompt.registry import PromptRegistry
from fastaiagent.ui.deps import get_context, project_filter, require_session

router = APIRouter(prefix="/api/prompts", tags=["prompts"])


def _registry(request: Request) -> PromptRegistry:
    ctx = get_context(request)
    return PromptRegistry(path=ctx.db_path)


@router.get("")
def list_prompts(request: Request, _user: str = Depends(require_session)) -> dict[str, Any]:
    ctx = get_context(request)
    reg = _registry(request)
    prompts = reg.list()
    db = ctx.db()
    pid_clause, pid_params = project_filter(ctx)
    # Filter the file-registry list down to prompts that are actually
    # owned by this project. The ``prompts`` table has a ``project_id``
    # stamp; the on-disk registry doesn't, so we cross-reference.
    if ctx.project_id:
        own_slugs = {
            r["slug"]
            for r in db.fetchall(
                "SELECT slug FROM prompts WHERE project_id = ?",
                (ctx.project_id,),
            )
        }
        prompts = [p for p in prompts if p.get("name") in own_slugs]
    try:
        enriched: list[dict[str, Any]] = []
        for p in prompts:
            # Accept every prefix variant so this count works across traces
            # from older SDK releases too.
            trace_count = db.fetchone(
                f"""SELECT COUNT(DISTINCT trace_id) AS n
                   FROM spans
                   WHERE (attributes LIKE ?
                      OR attributes LIKE ?
                      OR attributes LIKE ?) {pid_clause}""",
                (
                    f'%"fastaiagent.prompt.name": "{p["name"]}"%',
                    f'%"prompt.name": "{p["name"]}"%',
                    f'%"fastai.prompt.name": "{p["name"]}"%',
                    *pid_params,
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
    ctx = get_context(request)
    if ctx.project_id:
        # 404 cross-project lookups before opening the registry to avoid
        # confirming the slug exists in another project.
        db = ctx.db()
        try:
            row = db.fetchone(
                "SELECT 1 FROM prompts WHERE slug = ? AND project_id = ? LIMIT 1",
                (slug, ctx.project_id),
            )
        finally:
            db.close()
        if row is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"Prompt '{slug}' not found")
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
    pid_clause, pid_params = project_filter(ctx)
    try:
        rows = db.fetchall(
            f"""SELECT slug, version, template, variables, created_at, created_by
               FROM prompt_versions
               WHERE slug = ? {pid_clause}
               ORDER BY CAST(version AS INTEGER) ASC""",
            (slug, *pid_params),
        )
        if not rows:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"Prompt '{slug}' not found")
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
    ctx = get_context(request)
    reg = _registry(request)
    if not reg.is_local():
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            "This registry is external. Edit prompts via code or from the "
            "environment that owns this path.",
        )
    # Stamp the new version with the *active* project so the UI's
    # project-scoped reads see it. Without this, ``safe_get_project_id()``
    # fallback inside the storage layer can write a different project id
    # than the AppContext is filtering on, and the new version becomes
    # invisible to the editor that just saved it.
    prompt = reg.register(
        name=slug,
        template=body.template,
        metadata=body.metadata or {},
        project_id=ctx.project_id or None,
    )
    return {
        "slug": prompt.name,
        "version": prompt.version,
        "template": prompt.template,
        "variables": prompt.variables,
    }


@router.delete("/{slug}")
def delete_prompt(
    request: Request,
    slug: str,
    _user: str = Depends(require_session),
) -> dict[str, Any]:
    """Delete a prompt (all versions + aliases) from local storage.

    Project-scoped: when the UI is bound to a ``project_id``, only rows in
    that project are removed. Other projects' copies of the same slug are
    left intact.

    Returns ``404`` when the prompt doesn't exist in this project, ``403``
    when the registry is external (matches the PUT semantics — UI mutates
    only what's clearly local and personal).
    """
    ctx = get_context(request)
    reg = _registry(request)
    if not reg.is_local():
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            "This registry is external. Delete prompts via code or from the "
            "environment that owns this path.",
        )
    deleted = reg.delete(slug, project_id=ctx.project_id or None)
    if deleted == 0:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, f"Prompt '{slug}' not found"
        )
    return {"slug": slug, "versions_deleted": deleted}


@router.get("/{slug}/lineage")
def lineage(
    request: Request,
    slug: str,
    _user: str = Depends(require_session),
) -> dict[str, Any]:
    ctx = get_context(request)
    db = ctx.db()
    pid_clause, pid_params = project_filter(ctx)
    pid_clause_spans, _ = project_filter(ctx, alias="spans")
    try:
        likes = (
            f'%"fastaiagent.prompt.name": "{slug}"%',
            f'%"prompt.name": "{slug}"%',
            f'%"fastai.prompt.name": "{slug}"%',
        )
        trace_rows = db.fetchall(
            f"""SELECT DISTINCT trace_id FROM spans
               WHERE (attributes LIKE ?
                  OR attributes LIKE ?
                  OR attributes LIKE ?) {pid_clause}""",
            (*likes, *pid_params),
        )
        eval_rows = db.fetchall(
            f"""SELECT DISTINCT run_id FROM eval_cases
               JOIN spans ON eval_cases.trace_id = spans.trace_id
               WHERE (spans.attributes LIKE ?
                  OR spans.attributes LIKE ?
                  OR spans.attributes LIKE ?) {pid_clause_spans}""",
            (*likes, *pid_params),
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
