"""FastAPI app factory for ``fastaiagent ui``.

The app is built in :func:`build_app` so the CLI, tests, and embedding
environments can all construct their own instance with their own DB path,
auth file, and ``--no-auth`` flag. ``uvicorn`` then serves it.
"""

from __future__ import annotations

import importlib.resources as resources
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import hmac
import secrets

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from fastaiagent._internal.config import get_config
from fastaiagent.ui.auth import (
    SESSION_COOKIE_NAME,
    _request_is_secure,
    default_auth_path,
)
from fastaiagent.ui.db import init_local_db
from fastaiagent.ui.deps import AppContext
from fastaiagent.ui.routes import (
    agents,
    analytics,
    auth,
    datasets,
    evals,
    executions,
    filter_presets,
    guardrails,
    kb,
    learned_memory,
    overview,
    playground,
    prompts,
    providers,
    replay,
    traces,
    workflows,
)

logger = logging.getLogger(__name__)


# security_review_1.md M3 — Content-Security-Policy for the Local UI.
#
# Single-user same-origin app, so the policy is tight:
#   default-src   'self'    — everything must come from us
#   script-src    'self'    — no inline JS, no eval
#   style-src     allows inline because Tailwind utility classes ship
#                 small data: backgrounds and React injects ``style=""``
#                 attrs in places (these are static, not from user input)
#   img-src       'self' data: blob: — base64 thumbnails + Object URLs
#                 from upload previews
#   connect-src   'self'    — every API call goes to /api on the same
#                 origin; no cross-origin XHR
#   frame-src     'self'    — only the inline attachment iframe
#   frame-ancestors 'none'  — refuse to be embedded anywhere (clickjack
#                 defence; X-Frame-Options is the legacy mirror)
#   base-uri      'self'    — protect against ``<base>`` injection
#   form-action   'self'    — refuse to post forms to other origins
_CSP = (
    "default-src 'self'; "
    "script-src 'self'; "
    "style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data: blob:; "
    "font-src 'self' data:; "
    "connect-src 'self'; "
    "frame-src 'self'; "
    "frame-ancestors 'none'; "
    "base-uri 'self'; "
    "form-action 'self'"
)


class _SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Inject defence-in-depth security headers on every response."""

    async def dispatch(self, request: Request, call_next: Any) -> Response:  # type: ignore[override]
        response: Response = await call_next(request)
        # Don't overwrite a header the route deliberately set (e.g.
        # ``Cache-Control: no-store`` on the SSE stream).
        response.headers.setdefault("Content-Security-Policy", _CSP)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        response.headers.setdefault(
            "Permissions-Policy",
            "camera=(), microphone=(), geolocation=(), payment=(), usb=()",
        )
        return response


# security_review_1.md M4 — Double-submit-token CSRF defence.
#
# SameSite=Strict on the session cookie already prevents the classic
# cross-origin form-post attack in modern browsers. This middleware adds a
# belt: a per-session ``fastaiagent_csrf`` cookie (NOT httpOnly, so the
# bundled React UI can read it) plus an ``X-CSRF-Token`` request-header
# requirement on POST/PUT/PATCH/DELETE. Validation is constant-time.
#
# Skipped when:
# * The app was built with ``no_auth=True`` (developer "throwaway" mode).
# * The request method is safe (GET/HEAD/OPTIONS).
# * There is no session cookie — the request is anonymous, so there's
#   nothing to "ride".
# * The path is ``/api/auth/login`` (login itself replaces the session;
#   SameSite=Strict already blocks cross-origin login submission).
_CSRF_COOKIE_NAME = "fastaiagent_csrf"
_CSRF_HEADER_NAME = "x-csrf-token"
_CSRF_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})


class _CSRFMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Any) -> Response:  # type: ignore[override]
        ctx = getattr(request.app.state, "context", None)
        no_auth = getattr(ctx, "no_auth", False) if ctx is not None else False
        method = request.method.upper()
        path = request.url.path
        has_session = SESSION_COOKIE_NAME in request.cookies

        # Decide whether to enforce.
        enforce = (
            not no_auth
            and method not in _CSRF_SAFE_METHODS
            and has_session
            and path != "/api/auth/login"
        )
        if enforce:
            cookie = request.cookies.get(_CSRF_COOKIE_NAME, "")
            header = request.headers.get(_CSRF_HEADER_NAME, "")
            if not cookie or not header or not hmac.compare_digest(cookie, header):
                return JSONResponse(
                    {"detail": "CSRF token missing or invalid."},
                    status_code=403,
                )

        response: Response = await call_next(request)
        # Issue the cookie if missing so the React client has a value to
        # echo back. We set it on every response that doesn't already
        # carry it — this is harmless and self-healing.
        if _CSRF_COOKIE_NAME not in request.cookies:
            response.set_cookie(
                _CSRF_COOKIE_NAME,
                secrets.token_urlsafe(32),
                httponly=False,  # the React client MUST read it
                samesite="strict",
                secure=_request_is_secure(request),
                path="/",
            )
        return response


def _static_dir() -> Path | None:
    """Locate the bundled frontend static assets, if present."""
    # First: packaged location inside the wheel (fastaiagent/ui/static).
    try:
        packaged = resources.files("fastaiagent.ui").joinpath("static")
        if packaged.is_dir():
            return Path(str(packaged))
    except (ModuleNotFoundError, AttributeError, FileNotFoundError):
        logger.debug("Packaged static assets not found, falling back to sibling dir", exc_info=True)
    # Fall back to the sibling of this file (editable installs).
    candidate = Path(__file__).parent / "static"
    return candidate if candidate.exists() else None


def build_app(
    *,
    db_path: str | None = None,
    auth_path: Path | None = None,
    no_auth: bool = False,
    runners: Iterable[Any] | None = None,
    project_id: str | None = None,
) -> FastAPI:
    """Create a FastAPI app bound to a specific local.db and auth.json.

    ``runners`` (optional) is an iterable of resumable objects (Chain,
    Agent, Swarm, Supervisor) the server can call ``aresume(...)`` on.
    Each must expose ``.name`` and ``.aresume(...)``. The
    ``POST /api/executions/{id}/resume`` endpoint looks one up by the
    checkpoint's ``chain_name`` field and returns 503 if no match.

    ``project_id`` (optional) overrides the project the UI scopes to.
    When omitted, ``ProjectConfig.get_project_id()`` is used (which
    resolves ``./.fastaiagent/config.toml`` or the directory name).
    Endpoints filter SQL by this id so multiple projects can share the
    same DB (Postgres) without cross-contamination.
    """
    resolved_db = db_path or get_config().local_db_path
    resolved_auth = auth_path or default_auth_path()
    # Default to unscoped (project_id="") so test fixtures that don't seed
    # project_id keep working. The ``fastaiagent ui`` CLI explicitly sets
    # this via ProjectConfig so real users get isolation by default.
    resolved_project_id = project_id if project_id is not None else ""

    # Eagerly ensure the schema exists so every route can assume it's there.
    init_local_db(resolved_db).close()

    runner_map: dict[str, Any] = {}
    if runners is not None:
        for r in runners:
            name = getattr(r, "name", None)
            if not name or not hasattr(r, "aresume"):
                raise ValueError(
                    "build_app(runners=...) entries must have a .name and "
                    "an .aresume() method (Chain/Agent/Swarm/Supervisor). "
                    f"Got: {r!r}"
                )
            runner_map[str(name)] = r

    app = FastAPI(title="FastAIAgent", version="0.1", docs_url=None, redoc_url=None)
    app.state.context = AppContext(
        db_path=resolved_db,
        auth_path=resolved_auth,
        no_auth=no_auth,
        runners=runner_map,
        project_id=resolved_project_id,
    )
    # security_review_1.md M3 — defence-in-depth security headers.
    # The Local UI is single-user same-origin, so the policy is tight:
    # no cross-origin frames, no MIME sniffing, no referrer leakage,
    # no permissions for camera/mic/geolocation.
    app.add_middleware(_SecurityHeadersMiddleware)
    # security_review_1.md M4 — CSRF double-submit token on top of the
    # existing SameSite=Strict cookie. Issues ``fastaiagent_csrf`` on
    # safe responses and validates the matching ``X-CSRF-Token`` header
    # on every state-changing call from an authenticated session.
    app.add_middleware(_CSRFMiddleware)

    for r in (
        auth.router,
        overview.router,
        traces.router,
        replay.router,
        evals.router,
        prompts.router,
        guardrails.router,
        agents.router,
        analytics.router,
        kb.router,
        workflows.router,
        executions.router,
        playground.router,
        datasets.router,
        filter_presets.router,
        learned_memory.router,
        providers.router,
    ):
        app.include_router(r)

    static = _static_dir()
    if static is not None:
        assets_dir = static / "assets"
        if assets_dir.exists():
            app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

        index_file = static / "index.html"

        static_resolved = static.resolve()

        @app.get("/{path:path}", include_in_schema=False, response_model=None)
        async def spa_fallback(request: Request, path: str) -> FileResponse | JSONResponse:
            if path.startswith("api/"):
                return JSONResponse({"detail": "Not found"}, status_code=404)
            # Reject path-traversal attempts: the resolved candidate must stay
            # inside the static dir. Without this check, ``static / "../../etc/passwd"``
            # resolves outside the bundle and FileResponse would happily serve it.
            try:
                candidate = (static / path).resolve()
            except (OSError, RuntimeError):
                candidate = None
            if (
                candidate is not None
                and candidate.is_relative_to(static_resolved)
                and candidate.is_file()
            ):
                return FileResponse(candidate)
            if index_file.exists():
                return FileResponse(index_file)
            return JSONResponse(
                {
                    "detail": (
                        "Frontend bundle not found — the Python wheel was "
                        "built without it. Run `cd ui-frontend && pnpm build` "
                        "from source."
                    )
                },
                status_code=503,
            )

    return app


__all__ = ["build_app"]
