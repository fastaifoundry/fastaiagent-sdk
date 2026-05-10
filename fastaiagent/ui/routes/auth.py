"""Auth endpoints: /api/auth/{login,logout,status}."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, Response, status
from pydantic import BaseModel

from fastaiagent.ui.auth import (
    clear_session_cookie,
    issue_session_cookie,
    verify_password,
)
from fastaiagent.ui.deps import get_context
from fastaiagent.ui.throttle import get_default_throttler

router = APIRouter(prefix="/api/auth", tags=["auth"])


class LoginRequest(BaseModel):
    username: str
    password: str


class StatusResponse(BaseModel):
    authenticated: bool
    username: str | None
    no_auth: bool
    project_id: str = ""


def _client_key(request: Request, username: str) -> str:
    """Build a throttle key from the client IP and submitted username.

    Honors ``X-Forwarded-For`` (first hop) so a proxy doesn't collapse
    every client into one bucket. Falls back to ``request.client.host``.
    """
    fwd = request.headers.get("x-forwarded-for", "")
    ip = fwd.split(",")[0].strip() if fwd else (
        request.client.host if request.client else "unknown"
    )
    return f"{ip}|{username}"


@router.post("/login")
def login(body: LoginRequest, request: Request, response: Response) -> dict[str, str]:
    ctx = get_context(request)
    if ctx.no_auth:
        return {"status": "ok", "username": "anonymous"}

    throttler = get_default_throttler()
    key = _client_key(request, body.username)
    remaining = throttler.check(key)
    if remaining > 0:
        retry = max(int(remaining) + 1, 1)
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            f"Too many failed login attempts. Try again in {retry}s.",
            headers={"Retry-After": str(retry)},
        )

    auth = ctx.auth()
    if auth is None or auth.username != body.username or not verify_password(body.password, auth):
        cooldown = throttler.record_failure(key)
        if cooldown > 0:
            retry = max(int(cooldown) + 1, 1)
            raise HTTPException(
                status.HTTP_429_TOO_MANY_REQUESTS,
                f"Too many failed login attempts. Locked for {retry}s.",
                headers={"Retry-After": str(retry)},
            )
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid username or password")
    throttler.record_success(key)
    issue_session_cookie(response, auth, request)
    return {"status": "ok", "username": auth.username}


@router.post("/logout")
def logout(response: Response) -> dict[str, str]:
    clear_session_cookie(response)
    return {"status": "ok"}


@router.get("/status", response_model=StatusResponse)
def status_endpoint(request: Request) -> StatusResponse:
    ctx = get_context(request)
    pid = ctx.project_id
    if ctx.no_auth:
        return StatusResponse(
            authenticated=True, username="anonymous", no_auth=True, project_id=pid
        )
    auth = ctx.auth()
    if auth is None:
        return StatusResponse(
            authenticated=False, username=None, no_auth=False, project_id=pid
        )
    from fastaiagent.ui.auth import read_session_cookie

    payload = read_session_cookie(request, auth)
    if payload:
        return StatusResponse(
            authenticated=True,
            username=payload.get("username", auth.username),
            no_auth=False,
            project_id=pid,
        )
    return StatusResponse(
        authenticated=False, username=None, no_auth=False, project_id=pid
    )
