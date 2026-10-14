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
            # security_review_1.md L1: prompt names are user-supplied
            # and may legitimately contain ``%`` / ``_`` / ``\`` — the
            # LIKE wildcards. Without escaping, ``my_prompt`` would
            # over-match every name with the same prefix-and-trailer.
            # Escape those three chars and tell SQLite about our escape
            # character via ``ESCAPE '\\'`` on every clause.
            esc = (
                p["name"]
                .replace("\\", "\\\\")
                .replace("%", "\\%")
                .replace("_", "\\_")
            )
            # Accept every prefix variant so this count works across traces
            # from older SDK releases too.
            trace_count = db.fetchone(
                f"""SELECT COUNT(DISTINCT trace_id) AS n
                   FROM spans
                   WHERE (attributes LIKE ? ESCAPE '\\'
                      OR attributes LIKE ? ESCAPE '\\'
                      OR attributes LIKE ? ESCAPE '\\') {pid_clause}""",
                (
                    f'%"fastaiagent.prompt.name": "{esc}"%',
                    f'%"prompt.name": "{esc}"%',
                    f'%"fastai.prompt.name": "{esc}"%',
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
        payload = _version_payload(reg, slug, None)
    except PromptNotFoundError as e:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(e)) from e
    return {
        **payload,
        "latest_version": payload["version"],
        "registry_is_local": reg.is_local(),
        "aliases": reg.aliases(slug),
    }


def _ordered_variables(template: str) -> list[str]:
    import re

    return list(dict.fromkeys(re.findall(r"\{\{(\w+)\}\}", template)))


def _version_payload(reg: PromptRegistry, slug: str, version: int | None) -> dict[str, Any]:
    """One shape for a prompt version, used by every prompt endpoint (1.89.0).

    ``template`` is resolved (what a run sends: the Playground needs it);
    ``template_raw`` is the stored text with its ``{{@fragment}}`` references,
    which is what the editor must edit. Saving the resolved text back used to
    replace each reference with a frozen copy of the fragment.

    The raw version is loaded first and resolved at *that* version, so a
    concurrent save can't mix two versions in one response.
    """
    from fastaiagent.prompt.registry import fragment_refs

    raw = reg.load(slug, version=version, resolve_fragments=False)
    resolved = reg.resolve_fragments(raw.template)
    return {
        "slug": raw.name,
        "version": raw.version,
        "template": resolved,
        "template_raw": raw.template,
        "fragments": fragment_refs(raw.template),
        "variables": _ordered_variables(resolved),
        "metadata": raw.metadata,
    }


@router.get("/{slug}/versions")
def list_versions(
    request: Request,
    slug: str,
    _user: str = Depends(require_session),
) -> dict[str, Any]:
    import json

    from fastaiagent.prompt.registry import fragment_refs

    ctx = get_context(request)
    db = ctx.db()
    pid_clause, pid_params = project_filter(ctx)
    try:
        rows = db.fetchall(
            f"""SELECT slug, version, template, metadata, created_at, created_by
               FROM prompt_versions
               WHERE slug = ? {pid_clause}
               ORDER BY CAST(version AS INTEGER) ASC""",
            (slug, *pid_params),
        )
        if not rows:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"Prompt '{slug}' not found")
    finally:
        db.close()
    # The same shape as GET /{slug} and /versions/{v} (1.89.0): ``variables``
    # used to be a JSON *string* here and ``template`` the raw text, while the
    # other two endpoints returned a list and the resolved text.
    reg = _registry(request)
    by_version: dict[int, list[str]] = {}
    for alias, v in reg.aliases(slug).items():
        by_version.setdefault(v, []).append(alias)
    versions = []
    for row in rows:
        raw = row["template"] or ""
        resolved = reg.resolve_fragments(raw)
        try:
            metadata = json.loads(row.get("metadata") or "{}")
        except (TypeError, ValueError):
            metadata = {}
        versions.append(
            {
                "slug": row["slug"],
                # Stays the stored string this endpoint has always returned;
                # only ``variables`` and ``template`` changed shape (D5).
                "version": row["version"],
                "template": resolved,
                "template_raw": raw,
                "fragments": fragment_refs(raw),
                "variables": _ordered_variables(resolved),
                "metadata": metadata,
                "created_at": row["created_at"],
                "created_by": row["created_by"],
                "aliases": sorted(by_version.get(int(row["version"]), [])),
            }
        )
    return {"versions": versions}


@router.get("/{slug}/versions/{version}")
def get_version(
    request: Request,
    slug: str,
    version: str,
    _user: str = Depends(require_session),
) -> dict[str, Any]:
    reg = _registry(request)
    try:
        return _version_payload(reg, slug, int(version))
    except (PromptNotFoundError, ValueError) as e:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(e)) from e


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
        side_a = reg.load(slug, version=a, resolve_fragments=False)
        side_b = reg.load(slug, version=b, resolve_fragments=False)
        return {
            "diff": reg.diff(slug, a, b),
            # The stored templates (fragment references intact), for a
            # side-by-side view (1.89.0).
            "a": _side(side_a),
            "b": _side(side_b),
        }
    except PromptNotFoundError as e:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(e)) from e


def _side(prompt: Any) -> dict[str, Any]:
    return {"version": prompt.version, "template": prompt.template, "metadata": prompt.metadata}


_ALIAS_RE = r"^[A-Za-z0-9_.-]{1,64}$"


class AliasBody(BaseModel):
    version: int


@router.put("/{slug}/aliases/{alias}")
def set_alias(
    request: Request,
    slug: str,
    alias: str,
    body: AliasBody,
    _user: str = Depends(require_session),
) -> dict[str, Any]:
    """Point an alias (``production``, ``staging``) at a version (1.89.0).

    Agents load ``reg.load(slug, alias="production")``; moving the alias
    promotes or rolls back a prompt without a code change.
    """
    import re

    if not re.match(_ALIAS_RE, alias):
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "alias must be 1-64 characters of letters, digits, '_', '.' or '-'",
        )
    reg = _registry(request)
    if not reg.is_local():
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            "This registry is external. Set aliases from the environment that owns it.",
        )
    try:
        reg.set_alias(slug, body.version, alias)
    except PromptNotFoundError as e:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(e)) from e
    return {"slug": slug, "aliases": reg.aliases(slug)}


@router.delete("/{slug}/aliases/{alias}")
def delete_alias(
    request: Request,
    slug: str,
    alias: str,
    _user: str = Depends(require_session),
) -> dict[str, Any]:
    reg = _registry(request)
    if not reg.is_local():
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            "This registry is external. Remove aliases from the environment that owns it.",
        )
    if not reg.delete_alias(slug, alias):
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Alias '{alias}' not found for '{slug}'")
    return {"slug": slug, "aliases": reg.aliases(slug)}


class EditRequest(BaseModel):
    template: str
    variables: list[str] | None = None
    """Ignored: variables are derived from the template."""
    metadata: dict[str, Any] | None = None
    """``None`` (omitted) inherits the base version's metadata; ``{}`` clears it."""
    base_version: int | None = None
    """The version this edit started from. Defaults to the latest."""


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
    #
    # Metadata (1.89.0): an omitted ``metadata`` inherits the base version's.
    # Every editor save used to send ``{}``, which this route wrote verbatim,
    # so each new version silently lost its owner, tags and the rest.
    base = None
    try:
        base = reg.load(slug, version=body.base_version, resolve_fragments=False)
    except PromptNotFoundError as e:
        if body.base_version is not None:
            raise HTTPException(
                status.HTTP_404_NOT_FOUND,
                f"Prompt '{slug}' has no version {body.base_version} to edit from",
            ) from e
    metadata = body.metadata if body.metadata is not None else (base.metadata if base else {})
    prompt = reg.register(
        name=slug,
        template=body.template,
        metadata=metadata,
        project_id=ctx.project_id or None,
        created_by="ui",
    )
    return _version_payload(reg, slug, prompt.version)


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
