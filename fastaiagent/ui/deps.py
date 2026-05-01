"""Shared FastAPI dependencies: auth session check + local.db handle."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import Request

from fastaiagent._internal.storage import SQLiteHelper
from fastaiagent.ui.auth import AuthFile, load_auth_file, read_session_cookie


class AppContext:
    """Per-app state shared across routes.

    Held on ``app.state.context`` so tests and ``fastaiagent ui start``
    can inject a custom DB path without touching the global config.

    ``runners`` is an optional registry of resumable objects (Chain / Agent /
    Swarm / Supervisor), keyed by their ``.name``. The ``/api/executions``
    POST resume endpoint looks up a runner by the checkpoint's ``chain_name``
    field. Empty by default — resume is a no-op until a runner is registered.
    """

    def __init__(
        self,
        *,
        db_path: str,
        auth_path: Path,
        no_auth: bool,
        runners: dict[str, Any] | None = None,
        project_id: str | None = None,
    ) -> None:
        self.db_path = db_path
        self.auth_path = auth_path
        self.no_auth = no_auth
        self.runners: dict[str, Any] = dict(runners) if runners else {}
        self._auth_cache: AuthFile | None = None
        # Project the UI is scoped to. Read-path SQL filters by this so the
        # same Postgres can host multiple projects without cross-contamination.
        # Resolved once at build_app() time from ProjectConfig.
        self.project_id: str = project_id or ""

    def auth(self) -> AuthFile | None:
        if self.no_auth:
            return None
        if self._auth_cache is None:
            self._auth_cache = load_auth_file(self.auth_path)
        return self._auth_cache

    def reload_auth(self) -> None:
        self._auth_cache = None

    def db(self) -> SQLiteHelper:
        from fastaiagent.ui.db import init_local_db

        return init_local_db(self.db_path)


def get_context(request: Request) -> AppContext:
    ctx = request.app.state.context
    assert isinstance(ctx, AppContext)
    return ctx


def current_project_id(request: Request) -> str:
    """Return the project_id the UI is scoped to.

    Returns ``""`` when scoping is disabled (legacy DB or test fixtures
    that didn't seed a project). Endpoints that filter by project_id
    should treat ``""`` as "show everything for backwards-compat" so
    pre-v4 traces (project_id='') still surface.
    """
    return get_context(request).project_id


def project_filter(ctx: AppContext, *, alias: str | None = None) -> tuple[str, tuple[Any, ...]]:
    """Return the SQL clause + bind tuple for a project_id filter.

    Use to mechanically scope every project-owned SELECT::

        clause, params_extra = project_filter(ctx)
        rows = db.fetchall(f"SELECT * FROM spans WHERE x = ? {clause}", (x, *params_extra))

    When the AppContext has no ``project_id`` set (legacy / test
    fixtures), returns ``("", ())`` so the caller's SQL is unchanged.
    Always emits ``AND project_id = ?`` (with the optional table alias)
    so it composes cleanly inside an existing WHERE clause.
    """
    if not ctx.project_id:
        return "", ()
    column = f"{alias}.project_id" if alias else "project_id"
    return f"AND {column} = ?", (ctx.project_id,)


def require_session(request: Request) -> str:
    """FastAPI dependency that returns the logged-in username.

    Raises 401 if no valid session cookie is present. Bypassed when the
    app was started with ``--no-auth``.
    """
    from fastapi import HTTPException, status

    ctx = get_context(request)
    if ctx.no_auth:
        return "anonymous"
    auth = ctx.auth()
    if auth is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Not authenticated")
    payload = read_session_cookie(request, auth)
    if not payload:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Not authenticated")
    username = payload.get("username", auth.username)
    return str(username)
