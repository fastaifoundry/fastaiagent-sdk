"""``fastaiagent runner`` — the registered-runner daemon (task 2.6)."""

from __future__ import annotations

import asyncio
import logging
import signal

import typer
from rich.console import Console

runner_app = typer.Typer()
console = Console()


@runner_app.callback(invoke_without_command=True)
def runner(
    connect: str = typer.Option(
        ..., "--connect", help="Platform base URL (e.g. https://app.fastaiagent.net)."
    ),
    key: str = typer.Option(
        ..., "--key", help="SDK API key — sent as X-API-Key to register the runner."
    ),
    labels: list[str] = typer.Option(
        None, "--labels", help="A k=v label for routing (repeatable)."
    ),
    max_concurrency: int = typer.Option(
        4, "--max-concurrency", help="Max concurrent jobs this runner executes."
    ),
) -> None:
    """Run a registered runner: pull and execute live jobs in this boundary.

    Registers with the platform, heartbeats, long-polls for commands, runs each
    job as its own task (bounded by ``--max-concurrency``) in a request-scoped
    ``job_scope``, and reports results. Ctrl-C / SIGTERM drains in-flight jobs
    and deregisters gracefully.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from fastaiagent import connect as platform_connect
    from fastaiagent._internal.errors import PlatformAuthError
    from fastaiagent.runner.channel import RunnerChannel
    from fastaiagent.runner.daemon import RunnerDaemon

    # Connect to the platform with the SAME key used to register the runner. This
    # wires the PlatformSpanExporter so the traces of jobs this runner executes
    # are pushed to the plane (and routed by the key). A bad key fails fast (the
    # register call would reject it anyway); an unreachable plane is tolerated by
    # connect() — traces buffer locally and drain when it's reachable.
    try:
        platform_connect(api_key=key, target=connect)
    except PlatformAuthError as e:
        console.print(f"[red]runner: platform auth failed[/red] — {e}")
        raise typer.Exit(code=1) from e

    channel = RunnerChannel(base_url=connect, api_key=key)
    daemon = RunnerDaemon(
        channel, max_concurrency=max_concurrency, labels=list(labels or [])
    )

    async def _main() -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, daemon.request_stop)
            except NotImplementedError:
                # Windows has no add_signal_handler; Ctrl-C raises KeyboardInterrupt.
                pass
        await daemon.run()

    console.print(
        f"[green]runner[/green] -> {connect}  (max-concurrency={max_concurrency})"
    )
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        daemon.request_stop()
    console.print("[yellow]runner stopped[/yellow]")
