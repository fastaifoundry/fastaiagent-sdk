"""Non-blocking trace export to the FastAIAgent Platform with durable buffering.

``connect()`` wires this exporter into an OTel ``BatchSpanProcessor`` that calls
:meth:`PlatformSpanExporter.export` from a background thread. The data flow is
**local-first, then drain-to-platform**:

* ``LocalStorageProcessor`` (always registered first) writes every span to the
  local SQLite store with ``synced=0`` — the durable source of truth.
* ``export()`` drains a bounded batch of ``synced=0`` rows, POSTs them to
  ``/public/v1/traces/ingest`` with bounded retry, and marks them ``synced=1``
  **only after a confirmed 2xx**. Un-acked spans stay buffered and re-drain on
  the next ``export()`` — no separate reconnect hook.

Because the spans are already in SQLite (the trigger batch included), ``export``
ignores the *content* of its ``spans`` argument and drains the store instead;
the argument just signals "there is work to flush". The platform therefore
receives exactly what the local store holds (same redaction/normalization).

The ``/public/v1/traces/ingest`` wire shape is **frozen** and the server is
idempotent by ``span_id`` (re-sending a stored span returns ``{"ingested": 0}``),
so at-least-once re-send is safe and never double-counts.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

if TYPE_CHECKING:
    from fastaiagent.client import _Connection
    from fastaiagent.trace.storage import TraceStore

logger = logging.getLogger(__name__)


# Process-cumulative meter: total un-acked spans abandoned from the re-send
# queue this process (oldest-first eviction). Surfaced in the warning and
# readable by tests / diagnostics via ``get_abandoned_total()``.
_abandoned_total = 0


def get_abandoned_total() -> int:
    """Total un-acked spans abandoned from the re-send queue this process."""
    return _abandoned_total


class PlatformSpanExporter(SpanExporter):
    """Exports buffered spans to the FastAIAgent Platform with retry + re-send.

    Runs inside the ``BatchSpanProcessor`` background thread, so the retry
    backoff sleeps here never block agent execution. ``export()`` always returns
    ``SUCCESS`` — the durable SQLite buffer (not the OTel processor) owns retry.
    """

    # Retry: transient failures (connection/timeout/5xx) only. ~3 attempts with
    # exponential backoff 0.5s, 1.0s between them. 4xx is not retried.
    _MAX_ATTEMPTS = 3
    _BACKOFF_BASE = 0.5
    _TIMEOUT = 10

    # Drain at most this many un-acked spans per export() call (bounded work in
    # the bg thread; a large backlog drains across successive exports).
    _DRAIN_LIMIT = 500

    # Buffer bound: keep at most this many un-acked spans / this many days. Older
    # / excess spans are abandoned (kept locally, dropped from the re-send queue).
    _MAX_UNSYNCED = 10_000
    _MAX_AGE_DAYS = 7

    def __init__(self) -> None:
        # Lazily opened in the bg thread on first export(); cached thereafter.
        self._store: TraceStore | None = None

    def _get_store(self) -> TraceStore:
        if self._store is None:
            from fastaiagent.trace.storage import TraceStore

            self._store = TraceStore()
        return self._store

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """Drain buffered spans from local SQLite and push them to the platform.

        The ``spans`` argument is the flush trigger only — the actual payload is
        drained from the store (the trigger batch is already persisted there).
        """
        from fastaiagent.client import _connection

        if not _connection.is_connected:
            return SpanExportResult.SUCCESS

        try:
            from fastaiagent._internal.project import safe_get_project_id

            store = self._get_store()
            pid = safe_get_project_id()

            pending = store.fetch_unsynced(limit=self._DRAIN_LIMIT, project_id=pid)
            if pending:
                wire = [s.model_dump() for s in pending]
                if self._post_with_retry(_connection, wire):
                    store.mark_synced([s.span_id for s in pending])

            dropped = store.enforce_buffer_bound(
                self._MAX_UNSYNCED, self._MAX_AGE_DAYS, project_id=pid
            )
            if dropped:
                global _abandoned_total
                _abandoned_total += dropped
                logger.warning(
                    "Trace platform buffer bound hit: abandoned %d oldest un-acked "
                    "span(s) from the re-send queue (%d this process). They remain in "
                    "local.db — run `fastaiagent traces prune` to reclaim space.",
                    dropped,
                    _abandoned_total,
                )
        except Exception:
            # Never let export crash the bg thread; local SQLite still has the data.
            logger.debug("Platform export drain failed", exc_info=True)

        # Always SUCCESS — the durable buffer owns retry, not the OTel processor.
        return SpanExportResult.SUCCESS

    def _post_with_retry(self, conn: _Connection, wire: list[dict[str, Any]]) -> bool:
        """POST ``wire`` to ``/public/v1/traces/ingest``. Return True on 2xx.

        Retries connection errors, timeouts, and HTTP 5xx with exponential
        backoff. Does **not** retry 4xx (auth/bad-request won't fix on retry) —
        the spans stay buffered and the buffer bound eventually ages them out.
        """
        import httpx

        url = f"{conn.target}/public/v1/traces/ingest"
        payload = {"project": conn.project_id or conn.project, "spans": wire}

        for attempt in range(self._MAX_ATTEMPTS):
            try:
                with httpx.Client(timeout=self._TIMEOUT, verify=True) as client:
                    resp = client.post(url, json=payload, headers=conn.headers)
                code = resp.status_code
                if 200 <= code < 300:
                    return True
                if 400 <= code < 500:
                    logger.warning(
                        "Platform rejected %d spans with HTTP %d — not retrying; "
                        "left buffered for the bound to age out.",
                        len(wire),
                        code,
                    )
                    return False
                # 5xx → transient; fall through to backoff + retry.
                logger.debug(
                    "Platform export HTTP %d (attempt %d/%d)",
                    code,
                    attempt + 1,
                    self._MAX_ATTEMPTS,
                )
            except httpx.TransportError:
                # Covers ConnectError, ConnectTimeout, ReadTimeout, PoolTimeout…
                logger.debug(
                    "Platform export transient error (attempt %d/%d)",
                    attempt + 1,
                    self._MAX_ATTEMPTS,
                    exc_info=True,
                )

            if attempt < self._MAX_ATTEMPTS - 1:
                time.sleep(self._BACKOFF_BASE * (2**attempt))

        return False

    def shutdown(self) -> None:
        if self._store is not None:
            try:
                self._store.close()
            except Exception:
                logger.debug("Failed to close exporter trace store", exc_info=True)
            self._store = None
