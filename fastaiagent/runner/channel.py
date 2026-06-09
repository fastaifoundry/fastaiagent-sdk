"""Runner channel — the §7.5 wire client (4 endpoints).

Register with ``X-API-Key`` (the ``--key``); every later call authenticates with
``Authorization: Bearer <runner_token>``. The commands endpoint long-polls.

  POST /public/v1/runners                 -> 201 {runner_id, runner_token}
  POST /public/v1/runners/{id}/heartbeat  -> 200 {ok, ttl_seconds}
  GET  /public/v1/runners/{id}/commands   -> 200 {commands:[Command]}   (long-poll)
  POST /public/v1/runners/{id}/results    -> 202 {accepted}

The daemon (:mod:`fastaiagent.runner.daemon`) drives this client; it re-registers
on :class:`RunnerAuthError` (401/404 — token reaped or unknown runner).
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class RunnerAuthError(Exception):
    """The runner_token was rejected (401/403/404) — the daemon re-registers."""


class RunnerChannel:
    """Async client for the runner-channel endpoints."""

    def __init__(self, base_url: str, api_key: str, *, timeout: float = 35.0) -> None:
        # ``base_url`` is the ``--connect`` URL; the endpoints live under
        # ``/public/v1``. Long-poll needs a client timeout > the server's <=30s
        # hold, hence the 35s default.
        self._base = base_url.rstrip("/") + "/public/v1"
        self._api_key = api_key
        self._timeout = timeout
        self.runner_id: str | None = None
        self._runner_token: str | None = None  # in-memory only; never persisted

    def _bearer(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._runner_token}",
            "Content-Type": "application/json",
        }

    async def register(
        self,
        *,
        labels: list[str] | None = None,
        capabilities: list[str],
        deployments: list[str] | None = None,
    ) -> str:
        """POST /runners with X-API-Key -> mint a fresh runner_id + runner_token."""
        body = {
            "labels": labels or [],
            "deployments": deployments or [],
            "capabilities": capabilities,
        }
        async with httpx.AsyncClient(timeout=self._timeout, verify=True) as client:
            resp = await client.post(
                f"{self._base}/runners", json=body, headers={"X-API-Key": self._api_key}
            )
        if resp.status_code in (401, 403):
            raise RunnerAuthError(f"runner register rejected ({resp.status_code})")
        resp.raise_for_status()
        data = resp.json()
        self.runner_id = data["runner_id"]
        self._runner_token = data["runner_token"]
        return self.runner_id

    async def heartbeat(self, *, status: str, active_jobs: int) -> dict[str, Any]:
        """POST /runners/{id}/heartbeat -> {ok, ttl_seconds}.

        ``status="stopping"`` is the graceful deregister signal (no dedicated
        endpoint): the server marks the runner offline and re-queues its
        in-flight commands.
        """
        async with httpx.AsyncClient(timeout=self._timeout, verify=True) as client:
            resp = await client.post(
                f"{self._base}/runners/{self.runner_id}/heartbeat",
                json={"status": status, "active_jobs": active_jobs},
                headers=self._bearer(),
            )
        if resp.status_code in (401, 404):
            raise RunnerAuthError(f"heartbeat rejected ({resp.status_code})")
        resp.raise_for_status()
        return resp.json()

    async def poll_commands(self) -> list[dict[str, Any]]:
        """GET /runners/{id}/commands (long-poll) -> [Command]."""
        async with httpx.AsyncClient(timeout=self._timeout, verify=True) as client:
            resp = await client.get(
                f"{self._base}/runners/{self.runner_id}/commands", headers=self._bearer()
            )
        if resp.status_code in (401, 404):
            raise RunnerAuthError(f"poll rejected ({resp.status_code})")
        resp.raise_for_status()
        commands: list[dict[str, Any]] = resp.json().get("commands", [])
        return commands

    async def report_result(
        self,
        *,
        command_id: str,
        status: str,
        result: Any = None,
        trace_id: str | None = None,
        error: str | None = None,
    ) -> None:
        """POST /runners/{id}/results -> 202 accepted."""
        body: dict[str, Any] = {"command_id": command_id, "status": status}
        if result is not None:
            body["result"] = result
        if trace_id is not None:
            body["trace_id"] = trace_id
        if error is not None:
            body["error"] = error
        async with httpx.AsyncClient(timeout=self._timeout, verify=True) as client:
            resp = await client.post(
                f"{self._base}/runners/{self.runner_id}/results",
                json=body,
                headers=self._bearer(),
            )
        if resp.status_code in (401, 404):
            raise RunnerAuthError(f"result rejected ({resp.status_code})")
        resp.raise_for_status()
