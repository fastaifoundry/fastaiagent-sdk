"""Connection management for the FastAIAgent platform."""

from __future__ import annotations

import logging
from typing import Any

from fastaiagent._version import __version__

logger = logging.getLogger(__name__)


class _Connection:
    """Singleton holding platform connection state."""

    def __init__(self) -> None:
        self.api_key: str | None = None
        self.target: str = "https://app.fastaiagent.net"
        self.project: str | None = None
        self.domain_id: str | None = None
        self.project_id: str | None = None
        self.scopes: list[str] = []
        self.policy_cache: dict[str, Any] | None = None
        # WS4 governance: client fail mode. Default "open" preserves today's
        # behavior (a missing policy cache => the tool gate is a no-op). "closed"
        # opts in to fail-closed at the gate when governance can't be confirmed.
        self.governance_fail_mode: str = "open"
        self._platform_processor: Any = None
        self._hitl_processor: Any = None

    @property
    def is_connected(self) -> bool:
        return self.api_key is not None

    @property
    def headers(self) -> dict[str, str]:
        return {
            "X-API-Key": self.api_key or "",
            "Content-Type": "application/json",
            "User-Agent": f"fastaiagent-sdk/{__version__}",
        }

    def __repr__(self) -> str:
        # security_review_1.md M8 — never let ``repr(_connection)`` leak
        # the live API key (which lands in tracebacks, REPL output,
        # debug prints, etc.). Show just enough to confirm wiring.
        key = self.api_key or ""
        suffix = key[-4:] if len(key) >= 4 else ""
        redacted = f"***{suffix} (len={len(key)})" if key else "<unset>"
        return (
            f"_Connection(target={self.target!r}, project={self.project!r}, "
            f"api_key={redacted})"
        )


_global_connection = _Connection()


class _ConnectionProxy:
    """Scope-aware stand-in for the process-global connection.

    Reads resolve to the active :func:`fastaiagent.runtime.job_scope`
    connection (when one is set via ContextVar), else the process-global.
    Writes ALWAYS target the process-global — ``connect()`` / ``disconnect()``
    set the tenant; per-job overrides are installed by ``job_scope()`` through
    the ContextVar, not attribute writes.

    This makes every existing ``_connection.<attr>`` read job-scope-aware with
    no read-site changes (so no site can leak). In a background exporter thread
    (no active scope) reads resolve to the global — correct for one tenant per
    runner.
    """

    def _resolve(self) -> _Connection:
        from fastaiagent._internal.scope import scoped_connection

        active = scoped_connection.get()
        return active if active is not None else _global_connection

    def __getattr__(self, name: str) -> Any:
        return getattr(self._resolve(), name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(_global_connection, name, value)

    def __repr__(self) -> str:
        return repr(self._resolve())


_connection: Any = _ConnectionProxy()


def get_connection() -> _Connection:
    """Return the active connection: the ``job_scope()`` override if set, else
    the process-global. Equivalent to reading ``_connection``; provided as an
    explicit accessor for new code."""
    from fastaiagent._internal.scope import scoped_connection

    active = scoped_connection.get()
    return active if active is not None else _global_connection


def _normalize_target(target: str) -> str:
    """Normalize a platform target URL.

    Strips trailing slashes and validates that the caller supplied an
    explicit ``http://`` or ``https://`` scheme. We used to silently
    add a scheme (``http`` for hosts that *looked* local, ``https``
    otherwise), but that hid two real risks (security_review_1.md M10):

    * A typo like ``localhost.attacker.com`` matched
      ``host.endswith(".localhost")`` and got an unencrypted ``http://``
      URL — a network attacker could intercept the API key.
    * Auto-promoting bare hostnames to ``https`` looked safe but masked
      configuration mistakes (the user *thought* they pointed at a
      private gateway and we silently rewrote it).

    Now we refuse silently-rewriting and require the caller to be
    explicit. ``connect()`` callers that only pass ``"localhost:7842"``
    will get a clear ValueError pointing at the fix.
    """
    t = (target or "").strip().rstrip("/")
    if not t:
        return t
    if "://" not in t:
        raise ValueError(
            f"target must include an explicit scheme (got {t!r}). "
            "Use 'http://...' for plain HTTP loopback dev or 'https://...' "
            "for the production platform."
        )
    scheme = t.split("://", 1)[0].lower()
    if scheme not in ("http", "https"):
        raise ValueError(
            f"target scheme must be 'http' or 'https' (got {scheme!r})."
        )
    return t


def connect(
    api_key: str,
    target: str = "https://app.fastaiagent.net",
    project: str | None = None,
    *,
    governance_fail_mode: str | None = None,
) -> None:
    """Connect the SDK to FastAIAgent Platform for observability,
    prompt management, and evaluation services.

    All SDK features work without connect(). This adds platform
    backends alongside local storage.

    The API key carries its own domain and project scope from the
    platform. The ``project`` parameter is an optional override
    for the trace export payload.

    ``governance_fail_mode`` (WS4) is an opt-in posture: ``"closed"`` makes a
    governed agent refuse tool calls when governance can't be confirmed (the
    plane was unreachable at connect, so no policy is cached); the default
    ``"open"`` preserves today's fail-open behavior. The resolved mode is
    attested to the plane on enrollment. Falls back to the
    ``FASTAIAGENT_GOVERNANCE_FAIL_MODE`` env var when the argument is omitted.
    """
    import os

    import httpx

    from fastaiagent._internal.errors import PlatformAuthError, PlatformConnectionError

    _connection.api_key = api_key
    _connection.target = _normalize_target(target)
    _connection.project = project
    # WS4 governance fail mode: explicit kwarg > env var > "open" (non-breaking
    # default = today's fail-open behavior). Only the literal "closed" hardens the
    # gate, so a typo can never silently fail-close.
    _mode = (
        governance_fail_mode
        if governance_fail_mode is not None
        else os.environ.get("FASTAIAGENT_GOVERNANCE_FAIL_MODE")
    )
    _connection.governance_fail_mode = "closed" if (_mode or "").lower() == "closed" else "open"

    # Lightweight auth check — also captures domain/project from the key
    try:
        with httpx.Client(timeout=10, verify=True) as client:
            resp = client.get(
                f"{_connection.target}/public/v1/auth/check",
                headers=_connection.headers,
            )
            if resp.status_code == 401:
                _connection.api_key = None
                _connection.project = None
                raise PlatformAuthError(
                    "Invalid API key. Check your key at "
                    "https://app.fastaiagent.net/settings/api-keys"
                )
            if resp.status_code == 403:
                _connection.api_key = None
                _connection.project = None
                raise PlatformAuthError(f"Forbidden: {resp.text}")
            if resp.status_code == 200:
                data = resp.json()
                _connection.domain_id = data.get("domain_id")
                _connection.project_id = data.get("project_id")
                _connection.scopes = data.get("scopes", [])
                logger.info(
                    "Connected to platform: domain=%s project=%s scopes=%s",
                    _connection.domain_id,
                    _connection.project_id,
                    _connection.scopes,
                )
    except httpx.ConnectError:
        # Allow connecting even if platform is unreachable — traces will
        # queue locally and export when the platform becomes available.
        logger.warning(
            "Could not reach platform at %s. "
            "Connection stored — traces will export when platform is reachable.",
            _connection.target,
        )
    except (PlatformAuthError, PlatformConnectionError):
        raise

    # Pull + cache the governance policy (best-effort). Tool calls gate against
    # this cache (see fastaiagent.governance); on a pull failure we keep the
    # last-known cache rather than clearing it.
    try:
        with httpx.Client(timeout=10, verify=True) as client:
            presp = client.get(
                f"{_connection.target}/public/v1/policy", headers=_connection.headers
            )
        if presp.status_code == 200:
            _connection.policy_cache = presp.json()
            logger.info(
                "Cached governance policy: version=%s approval_policies=%d guardrail_rules=%d",
                _connection.policy_cache.get("version"),
                len(_connection.policy_cache.get("approval_policies", []) or []),
                len(_connection.policy_cache.get("guardrail_rules", []) or []),
            )
    except Exception:
        logger.debug("Could not pull governance policy (keeping last-known)", exc_info=True)

    # Register platform trace exporter
    try:
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        from fastaiagent.trace.otel import get_tracer_provider
        from fastaiagent.trace.platform_export import PlatformSpanExporter

        exporter = PlatformSpanExporter()
        processor = BatchSpanProcessor(exporter)
        get_tracer_provider().add_span_processor(processor)
        _connection._platform_processor = processor
    except Exception:
        logger.debug("Could not register platform trace exporter", exc_info=True)

    # Register the connected-HITL event exporter. This is secondary/opportunistic
    # insurance — the primary drain trigger is a per-emit daemon thread (see
    # trace/hitl_export.py), because tracing is trace=-gated and a trace=False
    # paused run produces no spans for the processor to flush on. Registering it
    # here still gives a backlog flush on the next traced span + a clean drain on
    # disconnect()'s force_flush.
    try:
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        from fastaiagent.trace.hitl_export import get_hitl_exporter
        from fastaiagent.trace.otel import get_tracer_provider

        hitl_processor = BatchSpanProcessor(get_hitl_exporter())
        get_tracer_provider().add_span_processor(hitl_processor)
        _connection._hitl_processor = hitl_processor
    except Exception:
        logger.debug("Could not register HITL event exporter", exc_info=True)

    # Flush any checkpoint backlog written while disconnected (WS2 durability).
    # Checkpoints aren't span-driven, so replication is write-kicked by the
    # checkpointers; this one-shot drain catches a backlog from before connect().
    # Best-effort, on a daemon thread — connect() never blocks on the plane.
    try:
        from fastaiagent.checkpointers import platform_replica

        platform_replica.drain_all_async()
    except Exception:
        logger.debug("Could not kick checkpoint drain on connect", exc_info=True)

    # WS4 governance enrollment (fire-and-forget attestation). One-shot best-effort
    # POST to /public/v1/governance/enroll on a daemon thread so connect() never
    # blocks on or raises from the plane — same non-blocking model as the checkpoint
    # drain above. NOT a durable outbox (enrollment is ephemeral attestation): a 4xx
    # is terminal, transient failures are dropped. Runs after the auth-check +
    # policy-pull so domain/project/policy_cache/governance_fail_mode are populated.
    try:
        import threading

        from fastaiagent import governance

        threading.Thread(target=governance.enroll, daemon=True).start()
    except Exception:
        logger.debug("Could not kick governance enroll on connect", exc_info=True)


def disconnect() -> None:
    """Disconnect from platform. Revert to local-only mode.

    Forces a flush of any pending trace spans before disconnecting.
    """
    if _connection._platform_processor is not None:
        try:
            _connection._platform_processor.force_flush(timeout_millis=5000)
            _connection._platform_processor.shutdown()
        except Exception:
            logger.debug("Failed to flush/shutdown platform processor on disconnect", exc_info=True)
        _connection._platform_processor = None
    if _connection._hitl_processor is not None:
        try:
            _connection._hitl_processor.force_flush(timeout_millis=5000)
            _connection._hitl_processor.shutdown()
        except Exception:
            logger.debug("Failed to flush/shutdown HITL processor on disconnect", exc_info=True)
        _connection._hitl_processor = None
    # Flush any pending checkpoint replications before tearing the connection
    # down (WS2 durability — non-lossy). Runs while still connected.
    try:
        from fastaiagent.checkpointers import platform_replica

        platform_replica.drain_all_sync()
    except Exception:
        logger.debug("Could not flush checkpoint outbox on disconnect", exc_info=True)
    _connection.api_key = None
    _connection.project = None
    _connection.domain_id = None
    _connection.project_id = None
    _connection.scopes = []
    _connection.policy_cache = None
    _connection.governance_fail_mode = "open"  # WS4: revert to non-breaking default
