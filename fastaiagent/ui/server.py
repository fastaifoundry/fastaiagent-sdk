"""FastAPI app factory for ``fastaiagent ui``.

The app is built in :func:`build_app` so the CLI, tests, and embedding
environments can all construct their own instance with their own DB path,
auth file, and ``--no-auth`` flag. ``uvicorn`` then serves it.
"""

from __future__ import annotations

import hmac
import importlib.resources as resources
import logging
import os
import secrets
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

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
    optimizes,
    overview,
    playground,
    prompts,
    providers,
    replay,
    simulations,
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

        # security_audit_2 N16: reject cross-origin state-changing requests up
        # front, in BOTH auth modes. Browsers always send ``Origin`` on a
        # cross-origin POST/PUT/PATCH/DELETE; a value whose host isn't an allowed
        # UI host means a page on another site is driving the request, so refuse
        # it. Non-browser clients (curl, scripts) send no ``Origin`` and are
        # unaffected. This is what protects the ``--no-auth`` API (which has no
        # session and thus no double-submit token) from CORS-simple writes.
        if method not in _CSRF_SAFE_METHODS and path != "/api/auth/login":
            origin = request.headers.get("origin")
            if origin:
                origin_host = _host_of(urlparse(origin).netloc)
                if origin_host and origin_host not in _allowed_ui_hosts():
                    return JSONResponse(
                        {"detail": "Cross-origin request rejected."},
                        status_code=403,
                    )

        # Double-submit CSRF token — enforced for authenticated sessions
        # (unchanged behavior; anonymous unsafe requests are rejected by auth).
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


# security_audit_2 N16 — Host-header allowlist to defeat DNS-rebinding.
#
# The Local UI is a loopback service, but without ``Host`` validation a
# malicious page can rebind its own domain to 127.0.0.1 and reach the UI as
# "same-origin", reading traces / driving state. A rebind request arrives with
# the attacker's ``Host`` (e.g. ``evil.attacker.com``); we reject anything whose
# host isn't loopback or explicitly allowed via ``FASTAIAGENT_UI_ALLOWED_HOSTS``
# (comma-separated) — needed when the UI runs behind a proxy on a real hostname.
_LOOPBACK_HOST_NAMES: frozenset[str] = frozenset({"localhost", "127.0.0.1", "::1"})


def _allowed_ui_hosts() -> frozenset[str]:
    extra = os.environ.get("FASTAIAGENT_UI_ALLOWED_HOSTS", "")
    names = {h.strip().lower() for h in extra.split(",") if h.strip()}
    return frozenset(_LOOPBACK_HOST_NAMES | names)


def _host_of(header_value: str) -> str:
    """Extract the bare hostname from a ``Host`` header (strip port, IPv6 []) ."""
    value = header_value.strip()
    if value.startswith("["):  # IPv6 literal: [::1]:7842
        return value[1 : value.find("]")].lower() if "]" in value else value.lower()
    return value.rsplit(":", 1)[0].lower() if ":" in value else value.lower()


class _HostValidationMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Any) -> Response:  # type: ignore[override]
        host_header = request.headers.get("host", "")
        # Empty Host (HTTP/1.0, some non-browser clients) is allowed — browsers
        # always send it, so it isn't a rebinding vector.
        if host_header:
            hostname = _host_of(host_header)
            if hostname and hostname not in _allowed_ui_hosts():
                return JSONResponse(
                    {
                        "detail": (
                            f"Host {hostname!r} is not allowed. The Local UI only "
                            "serves loopback hosts; set FASTAIAGENT_UI_ALLOWED_HOSTS "
                            "to permit a proxy hostname."
                        )
                    },
                    status_code=400,
                )
        return await call_next(request)


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
    ``fastaiagent ui --agent path/to/file.py:attr`` populates this from
    the command line; embedded callers pass the objects directly.

    ``project_id`` (optional) overrides the project the UI scopes to.
    When omitted this defaults to ``""`` (unscoped reads). Note that the
    ``fastaiagent ui`` CLI does not currently set it, so CLI-launched UIs
    run unscoped even though writers persist a resolved project id.
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
    # Added last → outermost → runs first: reject rebound Hosts before anything
    # else touches the request (security_audit_2 N16).
    app.add_middleware(_HostValidationMiddleware)

    for r in (
        auth.router,
        overview.router,
        traces.router,
        replay.router,
        evals.router,
        simulations.router,
        optimizes.router,
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
