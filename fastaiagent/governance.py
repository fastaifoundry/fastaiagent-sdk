"""Managed governance over the wire (Task C / §3.5).

When the SDK is ``connect()``-ed, it caches the platform's policy
(``GET /public/v1/policy``). Before a tool call whose name matches a cached
**approval policy**, the agent asks the platform (``POST /policy/decide``):

* ``allow``            → the tool runs.
* ``deny``             → the tool is refused; the model is told why and continues.
* ``require_approval`` → the SDK registers a pending run
  (``POST /runs/{run_id}/pending``) and **pauses** the agent via the existing
  ``interrupt()`` checkpoint machinery. The **calling application** is the
  approver: ``arun()`` returns the paused result, whose
  ``pending_interrupt["context"]`` carries the tool and its arguments, and the
  app resumes with ``Resume(approved=…, metadata={"resolver": …})``. A rejection
  refuses the call — the model is told, the tool never runs. The plane records
  the pause and its resolution and flags one that outlives its timeout; it never
  decides one (plane decision, 2026-09-23). The resolution event names the
  pending run it resolves (``context.pending_id``, see :func:`resolution_context`)
  so the plane closes exactly that pause.

The gate is a no-op unless the SDK is connected AND a cached approval policy's
``tool_pattern`` (fnmatch) matches the tool — so unmanaged / policy-less runs are
unaffected. ``/policy/decide`` is **fail-closed**: if the check can't be reached,
the high-stakes tool is refused rather than run ungoverned.
"""

from __future__ import annotations

import fnmatch
import logging
from typing import Any

logger = logging.getLogger(__name__)

#: The ``interrupt()`` reason a managed approval policy pauses with. It is what
#: tells a governance pause apart from an ``interrupt()`` in user code — on
#: resume, where a rejected one must not run the tool, and on the HITL ledger.
#:
#: ⚠ **Wire-stable: never rename.** The plane matches a resolution event to its
#: pause by ``kind`` (see :func:`hitl_kind`) and, for SDKs up to 1.73.0, by this
#: exact string (enterprise PR #198). Both are pinned as literals in
#: ``tests/test_governance_approvals.py``.
APPROVAL_REASON = "policy_approval_required"


def hitl_kind(reason: str | None) -> str:
    """The HITL ledger ``kind`` for a pause with this ``reason``.

    ``approval`` has been in the wire schema since v1.1 and was never sent before
    1.74.0, so the plane recorded every policy approval as an ad-hoc
    ``interrupt`` by an unknown person — the wrong evidence for human oversight.
    """
    return "approval" if reason == APPROVAL_REASON else "interrupt"


def denied(tool_name: str) -> str:
    """What the model is told when a policy approval is rejected or expires."""
    return f"Refused: governance approval denied for '{tool_name}'"


def resolution_context(
    reason: str | None, pause_context: dict[str, Any] | None
) -> dict[str, Any] | None:
    """The ``context`` a HITL *resolved* event carries — ``{"pending_id": ...}`` or nothing.

    The plane matches a resolution to its pause (enterprise PR #199, wire v1.1):

    * ``{"pending_id": "<id>"}`` — closes exactly that pending run;
    * ``{"pending_id": None}`` — registration failed, so there is no pending run:
      closes nothing. The key must be **present**: its absence reads as an older
      SDK and falls back to matching by position within the run, which is the
      guess this replaces;
    * no ``context`` — position-based matching.

    Only a policy pause carries it. A pause recorded before 1.76.0 has no
    ``pending_id`` in its saved context at all, so its resolution sends no
    ``context`` and keeps the positional match it was paused under. Nothing but
    the id ever goes here: the pause context also holds ``tool_input``, which
    must not leave the process on this channel.
    """
    if hitl_kind(reason) != "approval" or not pause_context or "pending_id" not in pause_context:
        return None
    return {"pending_id": pause_context["pending_id"]}


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


def enroll(governed_agent_ids: list[str] | None = None) -> dict[str, Any] | None:
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
    # governed_agent_ids: once agents self-register (Gap 2), connect() flushes
    # their platform ids and passes them here so the plane's governance coverage
    # knows which agents this instance governs. Omitted when empty (optional in
    # the schema). deployment_type / attributes still omitted (no cheap source).
    if governed_agent_ids:
        body["governed_agent_ids"] = governed_agent_ids
    # Part D: attest the Agent-CI verdict egress posture. This is the *marking*
    # mechanism — without it the plane cannot tell "export disabled" (a deliberate
    # config choice) apart from "this team isn't running evals at all", which are
    # very different governance conversations. The plane merges it into
    # SdkEnrollment.attributes. Additive: an older plane just ignores the key.
    try:
        from fastaiagent.eval.platform_export import eval_export_enabled

        body["export_evals"] = eval_export_enabled()
    except Exception:  # pragma: no cover — never block enroll on posture lookup
        logger.debug("Could not resolve export_evals posture for enroll", exc_info=True)

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
    """POST /policy/decide → ``{decision, approval_request_id?, reason?}``.

    Data egress note (security_audit_2 N8): this sends the full ``tool_input``
    (the tool's arguments) to the plane. That is intentional and load-bearing —
    a value-based approval policy (e.g. "approve refunds over $100") cannot be
    evaluated without the arguments. It only happens for a tool whose name
    matches a cached approval policy (governance is otherwise a no-op), so
    unmanaged tools never egress their inputs. If some arguments are too
    sensitive to leave the machine, don't place those tools under a plane
    approval policy. See ``docs/security.md`` → Governance.
    """
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


async def gate_tool_call(
    tool_name: str, tool_input: dict[str, Any], agent_id: str, run_id: str
) -> str | None:
    """Gate one tool call against managed policy.

    Returns ``None`` to allow the call, or a refusal **string** to feed back to
    the model (deny / denied-approval). For ``require_approval`` it calls
    ``interrupt()`` — which raises ``InterruptSignal`` (the executor checkpoints
    and pauses) on the first pass, and on resume returns the ``Resume`` value so
    we allow (approved) or refuse (rejected).

    That resume branch is only reached when a parent **Chain** owns the
    suspension and re-runs this agent. :meth:`Agent.aresume` re-enters the saved
    tool directly and never comes back here, so it applies the same refusal
    itself — see :data:`APPROVAL_REASON`.
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
        resume = interrupt(reason=APPROVAL_REASON, context={"tool": tool_name})
        return None if resume.approved else denied(tool_name)

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
        # The plane's id for this pause. Kept in the pause so the resolution can
        # name it (``resolution_context``); ``None`` records that registration
        # failed, which the plane must be told rather than left to guess.
        pending_id: str | None = None
        try:
            registered = await post_pending(
                run_id,
                reason=decision.get("reason") or "approval required",
                context={
                    "approval_request_id": decision.get("approval_request_id"),
                    "tool": tool_name,
                    "tool_input": tool_input,
                },
                kind="approval",
            )
            pending_id = registered.get("pending_id")
        except Exception:
            logger.warning("pending-run registration failed for %r", tool_name, exc_info=True)
        # Pause (raises InterruptSignal on the first pass). The calling app is
        # the approver, so the pause carries what it needs to ask its user —
        # the tool AND its arguments ("transfer $500 to Bob?"). ``tool_input``
        # matches the pending-run context above; it stays local (the same
        # arguments are already in the checkpoint's ``node_input``).
        resume = interrupt(
            reason=APPROVAL_REASON,
            context={
                "tool": tool_name,
                "tool_input": tool_input,
                "run_id": run_id,
                "approval_request_id": decision.get("approval_request_id"),
                "pending_id": pending_id,
            },
        )
        if not resume.approved:
            return denied(tool_name)
    return None
