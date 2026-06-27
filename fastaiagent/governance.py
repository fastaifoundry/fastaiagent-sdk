"""Managed governance over the wire (Task C / §3.5).

When the SDK is ``connect()``-ed, it caches the platform's policy
(``GET /public/v1/policy``). Before a tool call whose name matches a cached
**approval policy**, the agent asks the platform (``POST /policy/decide``):

* ``allow``            → the tool runs.
* ``deny``             → the tool is refused; the model is told why and continues.
* ``require_approval`` → the SDK registers a pending run
  (``POST /runs/{run_id}/pending``) and **pauses** the agent via the existing
  ``interrupt()`` checkpoint machinery. A human approves on the console
  (which flips the pending run's status); the SDK observes that by polling
  ``GET /runs/{run_id}/pending`` and **resumes** (blocking by default).

The gate is a no-op unless the SDK is connected AND a cached approval policy's
``tool_pattern`` (fnmatch) matches the tool — so unmanaged / policy-less runs are
unaffected. ``/policy/decide`` is **fail-closed**: if the check can't be reached,
the high-stakes tool is refused rather than run ungoverned.
"""

from __future__ import annotations

import asyncio
import fnmatch
import logging
from typing import Any

logger = logging.getLogger(__name__)

_RESOLVED = frozenset({"approved", "rejected", "expired"})

# Poll cadence + ceiling for the blocking wait-for-approval.
_POLL_INTERVAL_SECONDS = 2.0
_POLL_TIMEOUT_SECONDS = 600.0


def policy_matches(tool_name: str) -> bool:
    """True if a cached approval policy's ``tool_pattern`` matches ``tool_name``."""
    from fastaiagent.client import _connection

    policy = getattr(_connection, "policy_cache", None)
    if not policy:
        return False
    for ap in policy.get("approval_policies", []) or []:
        pattern = ap.get("tool_pattern")
        if pattern and fnmatch.fnmatch(tool_name, pattern):
            return True
    return False


async def _post(path: str, body: dict[str, Any]) -> dict[str, Any]:
    import httpx

    from fastaiagent.client import _connection

    async with httpx.AsyncClient(timeout=15, verify=True) as client:
        resp = await client.post(
            f"{_connection.target}/public/v1{path}", json=body, headers=_connection.headers
        )
    resp.raise_for_status()
    data: dict[str, Any] = resp.json()
    return data


async def _get(path: str) -> dict[str, Any]:
    import httpx

    from fastaiagent.client import _connection

    async with httpx.AsyncClient(timeout=15, verify=True) as client:
        resp = await client.get(
            f"{_connection.target}/public/v1{path}", headers=_connection.headers
        )
    resp.raise_for_status()
    data: dict[str, Any] = resp.json()
    return data


def enroll() -> dict[str, Any] | None:
    """Fire-and-forget governance enrollment (WS4). SYNC + best-effort.

    POSTs a stable ``instance_id`` (+ posture metadata) to
    ``/public/v1/governance/enroll``. The plane UPSERTs on
    ``(domain_id, project_id, instance_id)`` — keeping ``first_seen_at``,
    refreshing ``last_seen_at`` + posture — so this is safe to re-POST on every
    connect.

    Designed to run on a daemon thread from :func:`fastaiagent.client.connect`:
    it never raises, a short timeout caps latency, a 4xx is terminal (drop &
    continue — incl. 403 unentitled and an older/mock plane's 404), and a
    transient error is ignored. NOT a durable outbox — enrollment is ephemeral
    attestation, so a dropped POST just means the plane refreshes posture on the
    next connect. Returns the parsed 200 body (so callers/tests can assert the
    round-trip) or ``None``.
    """
    import socket

    import httpx

    from fastaiagent._internal.instance import get_instance_id
    from fastaiagent._version import __version__
    from fastaiagent.client import _connection

    if not _connection.is_connected:
        return None

    body: dict[str, Any] = {
        "instance_id": get_instance_id(),
        "sdk_version": __version__,
        "fail_mode": getattr(_connection, "governance_fail_mode", "open"),
        # Protocol version travels in the BODY, not a header: _connection.headers
        # has no X-FAA-Protocol today (prior workstreams shipped without it) and the
        # enroll schema accepts protocol_version, so this stays additive/minimal.
        "protocol_version": "1",
    }
    try:
        body["hostname"] = socket.gethostname()
    except Exception:
        pass
    # governed_agent_ids / deployment_type / attributes: the SDK has no cheap
    # source for these today, so they are omitted (all optional in the schema).

    try:
        with httpx.Client(timeout=5, verify=True) as client:
            resp = client.post(
                f"{_connection.target}/public/v1/governance/enroll",
                json=body,
                headers=_connection.headers,
            )
    except Exception:
        logger.debug("governance enroll transient error (ignored)", exc_info=True)
        return None

    code = resp.status_code
    if 200 <= code < 300:
        try:
            data: dict[str, Any] = resp.json()
        except Exception:
            return None
        logger.info(
            "Governance enroll OK: instance_id=%s fail_mode=%s",
            data.get("instance_id"),
            data.get("fail_mode"),
        )
        return data
    if 400 <= code < 500:
        # Terminal (incl. 403 = domain not entitled to connected_state_plane, and
        # 404 = mock/older plane without the endpoint). Drop & continue silently.
        logger.debug("governance enroll rejected HTTP %d (terminal, ignored)", code)
        return None
    logger.debug("governance enroll HTTP %d (ignored)", code)
    return None


async def decide(tool_name: str, tool_input: dict[str, Any], agent_id: str) -> dict[str, Any]:
    """POST /policy/decide → ``{decision, approval_request_id?, reason?}``."""
    return await _post(
        "/policy/decide",
        {"tool_name": tool_name, "tool_input": tool_input, "agent_id": agent_id, "context": None},
    )


async def post_pending(
    run_id: str, *, reason: str, context: dict[str, Any], kind: str
) -> dict[str, Any]:
    """POST /runs/{run_id}/pending → ``{pending_id, status}``."""
    return await _post(
        f"/runs/{run_id}/pending", {"reason": reason, "context": context, "kind": kind}
    )


async def get_pending_status(run_id: str) -> str | None:
    """GET /runs/{run_id}/pending → status, or None if unavailable."""
    try:
        return (await _get(f"/runs/{run_id}/pending")).get("status")
    except Exception:
        logger.debug("get pending status failed for run %s", run_id, exc_info=True)
        return None


async def await_resolution(
    run_id: str,
    *,
    poll_interval: float = _POLL_INTERVAL_SECONDS,
    timeout: float = _POLL_TIMEOUT_SECONDS,
) -> str:
    """Poll ``GET /runs/{run_id}/pending`` until the status resolves.

    Returns ``approved`` / ``rejected`` / ``expired``. Returns ``expired`` if the
    timeout elapses without a console decision.
    """
    waited = 0.0
    while waited < timeout:
        status = await get_pending_status(run_id)
        if status in _RESOLVED:
            return status
        await asyncio.sleep(poll_interval)
        waited += poll_interval
    return "expired"


async def gate_tool_call(
    tool_name: str, tool_input: dict[str, Any], agent_id: str, run_id: str
) -> str | None:
    """Gate one tool call against managed policy.

    Returns ``None`` to allow the call, or a refusal **string** to feed back to
    the model (deny / denied-approval). For ``require_approval`` it calls
    ``interrupt()`` — which raises ``InterruptSignal`` (the executor checkpoints
    and pauses) on the first pass, and on resume returns the ``Resume`` value so
    we allow (approved) or refuse (rejected).
    """
    from fastaiagent.chain.interrupt import _resume_value, interrupt
    from fastaiagent.client import _connection

    # WS4 opt-in fail-closed: when the operator has opted in (fail_mode="closed")
    # AND this agent is governed (agent_id set) AND we're connected but the policy
    # cache is missing (the plane was unreachable at connect, so governance can't
    # be evaluated), refuse rather than run ungoverned. Default fail_mode="open"
    # skips this entirely => the existing fail-open early-return below is unchanged.
    # This does NOT weaken the decide()-error fail-closed path further down (that
    # stays as shipped). When the cache IS present, this is skipped and normal
    # policy_matches -> decide() gating runs.
    if (
        getattr(_connection, "governance_fail_mode", "open") == "closed"
        and agent_id
        and _connection.is_connected
        and getattr(_connection, "policy_cache", None) is None
    ):
        logger.warning(
            "fail-closed: governance unavailable (no cached policy); refusing %r", tool_name
        )
        return "Refused: fail-closed mode — governance unavailable for this run"

    # Governance is opt-in per agent: without a platform ``agent_id`` we can't make
    # a ``/policy/decide`` call the plane will accept (it FK-validates the agent),
    # so the agent isn't enrolled — no gating.
    if not agent_id or not _connection.is_connected or not policy_matches(tool_name):
        return None
    # On resume, ``interrupt()`` returns the human's decision instead of raising.
    # Skip a second /policy/decide (and a second pending-run) for the same call.
    if _resume_value.get() is not None:
        resume = interrupt(reason="policy_approval_required", context={"tool": tool_name})
        return None if resume.approved else f"Refused: governance approval denied for '{tool_name}'"

    try:
        decision = await decide(tool_name, tool_input, agent_id)
    except Exception:
        logger.warning("policy/decide unreachable; refusing %r (fail-closed)", tool_name)
        logger.debug("policy/decide error detail", exc_info=True)
        return "Refused: governance check unavailable"

    verdict = decision.get("decision")
    if verdict == "deny":
        return f"Refused by governance policy: {decision.get('reason') or 'not permitted'}"
    if verdict == "require_approval":
        try:
            await post_pending(
                run_id,
                reason=decision.get("reason") or "approval required",
                context={
                    "approval_request_id": decision.get("approval_request_id"),
                    "tool": tool_name,
                    "tool_input": tool_input,
                },
                kind="approval",
            )
        except Exception:
            logger.warning("pending-run registration failed for %r", tool_name, exc_info=True)
        # Pause for console approval (raises InterruptSignal on the first pass).
        resume = interrupt(
            reason="policy_approval_required",
            context={
                "tool": tool_name,
                "run_id": run_id,
                "approval_request_id": decision.get("approval_request_id"),
            },
        )
        if not resume.approved:
            return f"Refused: governance approval denied for '{tool_name}'"
    return None
