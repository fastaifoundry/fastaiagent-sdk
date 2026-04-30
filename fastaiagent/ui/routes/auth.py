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

router = APIRouter(prefix="/api/auth", tags=["auth"])


class LoginRequest(BaseModel):
    username: str
    password: str


class StatusResponse(BaseModel):
    authenticated: bool
    username: str | None
    no_auth: bool


@router.post("/login")
def login(body: LoginRequest, request: Request, response: Response) -> dict[str, str]:
    ctx = get_context(request)
    if ctx.no_auth:
        return {"status": "ok", "username": "anonymous"}

    auth = ctx.auth()
    if auth is None or auth.username != body.username or not verify_password(body.password, auth):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid username or password")
    issue_session_cookie(response, auth)
    return {"status": "ok", "username": auth.username}


@router.post("/logout")
def logout(response: Response) -> dict[str, str]:
    clear_session_cookie(response)
    return {"status": "ok"}


@router.get("/status", response_model=StatusResponse)
def status_endpoint(request: Request) -> StatusResponse:
    ctx = get_context(request)
    if ctx.no_auth:
        return StatusResponse(authenticated=True, username="anonymous", no_auth=True)
    auth = ctx.auth()
    if auth is None:
        return StatusResponse(authenticated=False, username=None, no_auth=False)
    from fastaiagent.ui.auth import read_session_cookie

    payload = read_session_cookie(request, auth)
    if payload:
        return StatusResponse(
            authenticated=True,
            username=payload.get("username", auth.username),
            no_auth=False,
        )
    return StatusResponse(authenticated=False, username=None, no_auth=False)
