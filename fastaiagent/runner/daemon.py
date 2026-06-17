"""Registered-runner daemon (§8).

Loop: ``register -> heartbeat(every ttl/3) -> long-poll commands -> execute
(bounded by a semaphore, one asyncio task per job) -> report result``. On
SIGINT/SIGTERM it drains in-flight jobs, then sends a final ``status="stopping"``
heartbeat — the graceful deregister (no dedicated endpoint). It re-registers on
auth loss (heartbeat-miss / 401 / 404), minting a fresh in-memory token.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

from fastaiagent.runner.channel import RunnerAuthError, RunnerChannel
from fastaiagent.runner.execute import CommandResult, execute_command

logger = logging.getLogger(__name__)

_MAX_BACKOFF = 30.0


class RunnerDaemon:
    """Drives a :class:`RunnerChannel` through the runner lifecycle."""

    def __init__(
        self,
        channel: RunnerChannel,
        *,
        max_concurrency: int = 4,
        labels: list[str] | None = None,
        capabilities: tuple[str, ...] = ("live_playground", "eval_run"),
        executor: Callable[[dict], Awaitable[CommandResult]] = execute_command,
    ) -> None:
        self._channel = channel
        self._sem = asyncio.Semaphore(max_concurrency)
        self._labels = labels or []
        self._capabilities = list(capabilities)
        # The per-command executor (defaults to live_playground execution). An
        # injection point — not a test stub: callers pass the real executor.
        self._executor = executor
        self._active = 0  # in-flight job count -> reported as heartbeat.active_jobs
        self._status = "active"
        self._stop = asyncio.Event()
        self._inflight: set[asyncio.Task] = set()
        self._ttl = 30.0

    def request_stop(self) -> None:
        """Signal a graceful shutdown (wired to SIGINT/SIGTERM by the CLI)."""
        self._stop.set()

    async def run(self, *, drain_timeout: float = 30.0) -> None:
        await self._register_with_retry()
        hb = asyncio.create_task(self._heartbeat_loop())
        try:
            await self._poll_loop()
        finally:
            # Graceful shutdown: stop pulling, finish in-flight, then deregister
            # via a final ``stopping`` heartbeat.
            self._status = "draining"
            await self._drain(drain_timeout)
            self._status = "stopping"
            try:
                await self._channel.heartbeat(status="stopping", active_jobs=self._active)
            except Exception:
                logger.debug("final 'stopping' heartbeat failed", exc_info=True)
            hb.cancel()

    # --- lifecycle steps ---------------------------------------------------

    async def _register_with_retry(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            try:
                rid = await self._channel.register(
                    labels=self._labels, capabilities=self._capabilities
                )
                logger.info("runner registered: %s", rid)
                return
            except Exception as e:  # noqa: BLE001 — keep retrying with backoff
                logger.warning("register failed (%s); retrying in %.1fs", e, backoff)
                await self._sleep(backoff)
                backoff = min(backoff * 2, _MAX_BACKOFF)

    async def _heartbeat_loop(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            try:
                resp = await self._channel.heartbeat(
                    status=self._status, active_jobs=self._active
                )
                self._ttl = float(resp.get("ttl_seconds", self._ttl))
                backoff = 1.0
                await self._sleep(self._ttl / 3)
            except RunnerAuthError:
                logger.warning("heartbeat auth lost — re-registering")
                await self._register_with_retry()
            except Exception as e:  # noqa: BLE001 — transient; back off and retry
                logger.warning("heartbeat failed (%s); retry in %.1fs", e, backoff)
                await self._sleep(backoff)
                backoff = min(backoff * 2, self._ttl)

    async def _poll_loop(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            try:
                commands = await self._channel.poll_commands()
                backoff = 1.0
                for cmd in commands:
                    # Bound concurrency: block here when at --max-concurrency.
                    await self._sem.acquire()
                    if self._stop.is_set():
                        self._sem.release()
                        return
                    task = asyncio.create_task(self._run_command(cmd))
                    self._inflight.add(task)
                    task.add_done_callback(self._on_command_done)
            except RunnerAuthError:
                logger.warning("poll auth lost — re-registering")
                await self._register_with_retry()
            except Exception as e:  # noqa: BLE001 — transient; back off and retry
                logger.warning("poll failed (%s); retry in %.1fs", e, backoff)
                await self._sleep(backoff)
                backoff = min(backoff * 2, _MAX_BACKOFF)

    def _on_command_done(self, task: asyncio.Task) -> None:
        self._inflight.discard(task)
        self._sem.release()

    async def _run_command(self, cmd: dict) -> None:
        self._active += 1
        cid = cmd.get("command_id", "")
        try:
            res = await self._executor(cmd)
            # Push this job's trace before reporting the result that references
            # its trace_id, so the console links them promptly. Best-effort.
            await self._flush_traces()
            await self._channel.report_result(
                command_id=cid,
                status=res.status,
                result=res.result,
                trace_id=res.trace_id,
                error=res.error,
            )
        except Exception:
            logger.exception("failed to run/report command %s", cid)
            try:
                await self._channel.report_result(
                    command_id=cid, status="failed", error="runner internal error"
                )
            except Exception:
                logger.debug("could not report failure for %s", cid, exc_info=True)
        finally:
            self._active -= 1

    async def _flush_traces(self) -> None:
        """Best-effort flush of the platform span exporter so a job's trace lands
        promptly. No-op when the runner isn't connected to a platform (e.g. the
        e2e channel stand-in). Offloaded to a thread and never raised so it can't
        stall or break the command loop."""
        from fastaiagent.client import _connection

        processor = getattr(_connection, "_platform_processor", None)
        if processor is None:
            return
        try:
            await asyncio.to_thread(processor.force_flush, 5000)
        except Exception:
            logger.debug("trace force_flush failed", exc_info=True)

    async def _drain(self, timeout: float) -> None:
        if not self._inflight:
            return
        logger.info("draining %d in-flight job(s)", len(self._inflight))
        try:
            await asyncio.wait_for(
                asyncio.gather(*self._inflight, return_exceptions=True), timeout=timeout
            )
        except asyncio.TimeoutError:
            logger.warning("drain timed out; %d job(s) still running", len(self._inflight))

    async def _sleep(self, seconds: float) -> None:
        """Sleep that wakes immediately when a stop is requested."""
        try:
            await asyncio.wait_for(self._stop.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            pass
