"""Local-only auth for the fastaiagent UI.

One file on disk (``./.fastaiagent/auth.json``) holds the single user's bcrypt
hash plus a signed-cookie secret. There's no cloud, no OAuth, no JWT, no email.
"Forgot password" means delete ``auth.json`` and restart — acceptable for a
single-user local tool.
"""

from __future__ import annotations

import json
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import Request, Response

SESSION_COOKIE_NAME = "fastaiagent_session"
SESSION_MAX_AGE_SECONDS = 7 * 24 * 60 * 60  # 7 days


@dataclass
class AuthFile:
    """Contents of ``auth.json`` on disk."""

    username: str
    password_hash: str
    session_secret: str
    created_at: str
    version: int = 1

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AuthFile:
        return cls(
            username=data["username"],
            password_hash=data["password_hash"],
            session_secret=data["session_secret"],
            created_at=data.get("created_at", ""),
            version=int(data.get("version", 1)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "username": self.username,
            "password_hash": self.password_hash,
            "session_secret": self.session_secret,
            "created_at": self.created_at,
        }


def default_auth_path() -> Path:
    """``./.fastaiagent/auth.json`` alongside the local DB."""
    return Path(".fastaiagent") / "auth.json"


def auth_file_exists(path: Path | None = None) -> bool:
    return (path or default_auth_path()).exists()


def create_auth_file(
    username: str, password: str, *, path: Path | None = None
) -> AuthFile:
    """Hash the password, mint a session secret, and write ``auth.json``.

    Raises ``FileExistsError`` if one is already present — use
    :func:`delete_auth_file` and retry (that's the documented
    forgot-password flow).
    """
    import bcrypt

    target = path or default_auth_path()
    if target.exists():
        raise FileExistsError(
            f"{target} already exists. Delete it to reset credentials."
        )
    target.parent.mkdir(parents=True, exist_ok=True)

    password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    file = AuthFile(
        username=username,
        password_hash=password_hash,
        session_secret=secrets.token_hex(32),
        created_at=datetime.now(tz=timezone.utc).isoformat(),
    )
    target.write_text(json.dumps(file.to_dict(), indent=2))
    target.chmod(0o600)
    return file


def load_auth_file(path: Path | None = None) -> AuthFile:
    target = path or default_auth_path()
    if not target.exists():
        raise FileNotFoundError(f"{target} not found — run `fastaiagent ui` to set up.")
    return AuthFile.from_dict(json.loads(target.read_text()))


def delete_auth_file(path: Path | None = None) -> bool:
    target = path or default_auth_path()
    if target.exists():
        target.unlink()
        return True
    return False


def verify_password(password: str, auth: AuthFile) -> bool:
    import bcrypt

    return bcrypt.checkpw(password.encode(), auth.password_hash.encode())


def _serializer(session_secret: str) -> Any:
    from itsdangerous import URLSafeTimedSerializer

    return URLSafeTimedSerializer(session_secret, salt="fastaiagent.ui.session")


def issue_session_cookie(response: Response, auth: AuthFile) -> None:
    """Sign a cookie that lives for ``SESSION_MAX_AGE_SECONDS`` days."""
    token = _serializer(auth.session_secret).dumps({"username": auth.username})
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=token,
        max_age=SESSION_MAX_AGE_SECONDS,
        httponly=True,
        samesite="strict",
        secure=False,
        path="/",
    )


def clear_session_cookie(response: Response) -> None:
    response.delete_cookie(SESSION_COOKIE_NAME, path="/")


def read_session_cookie(request: Request, auth: AuthFile) -> dict[str, Any] | None:
    """Validate the cookie; return payload on success, ``None`` otherwise."""
    from itsdangerous import BadSignature, SignatureExpired

    token = request.cookies.get(SESSION_COOKIE_NAME)
    if not token:
        return None
    try:
        payload = _serializer(auth.session_secret).loads(
            token, max_age=SESSION_MAX_AGE_SECONDS
        )
    except (BadSignature, SignatureExpired):
        return None
    return payload if isinstance(payload, dict) else None
