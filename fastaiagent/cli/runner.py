"""``fastaiagent runner`` — the registered-runner daemon (task 2.6)."""

from __future__ import annotations

import asyncio
import logging
import os
import signal
from urllib.parse import urlparse

import typer
from rich.console import Console

runner_app = typer.Typer()
console = Console()

_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def _require_secure_connect(connect: str) -> None:
    """Reject a non-loopback ``--connect`` plane over plaintext http (N5).

    The runner runs whatever the plane dispatches with the operator's creds, so
    the channel must be authenticated + confidential. https everywhere; loopback
    http is allowed for dev; ``FASTAIAGENT_RUNNER_ALLOW_INSECURE=1`` is the
    explicit opt-out for a trusted-network http plane.
    """
    parsed = urlparse(connect)
    scheme = parsed.scheme.lower()
    host = (parsed.hostname or "").lower()
    if scheme == "https":
        return
    if host in _LOOPBACK_HOSTS:
        return
    if os.environ.get("FASTAIAGENT_RUNNER_ALLOW_INSECURE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        console.print(
            f"[yellow]runner: connecting to {connect!r} over plaintext http "
            "(FASTAIAGENT_RUNNER_ALLOW_INSECURE set) — the control channel is not "
            "encrypted.[/yellow]"
        )
        return
    console.print(
        f"[red]runner: refusing to connect to {connect!r} over http.[/red]\n"
        "The runner executes plane-dispatched agents/tools with your credentials, "
        "so the channel must be https. Use an https:// URL, or set "
        "FASTAIAGENT_RUNNER_ALLOW_INSECURE=1 if the plane is on a trusted network."
    )
    raise typer.Exit(code=2)


def _load_tools(entrypoints: list[str]) -> None:
    """Import + call each ``module:callable`` to register this runner's LOCAL
    tools/connectors. They self-register in the ToolRegistry (e.g. by building
    ``FunctionTool``s with the operator's own creds); ``tool_exec`` then resolves
    them by their exposed name. Raises on a bad spec or import/exec error."""
    import importlib

    for ep in entrypoints:
        module_name, sep, attr = ep.partition(":")
        if not sep or not module_name or not attr:
            raise ValueError(f"--tools must be 'module:callable' (got {ep!r})")
        fn = getattr(importlib.import_module(module_name), attr)
        fn()


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
    tools: list[str] = typer.Option(
        None,
        "--tools",
        help="A 'module:callable' that registers this runner's LOCAL tools/connectors "
        "(repeatable). Providing it opts the runner into executing 'tool_exec' commands.",
    ),
) -> None:
    """Run a registered runner: pull and execute live jobs in this boundary.

    Registers with the platform, heartbeats, long-polls for commands, runs each
    job as its own task (bounded by ``--max-concurrency``) in a request-scoped
    ``job_scope``, and reports results. Ctrl-C / SIGTERM drains in-flight jobs
    and deregisters gracefully.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # security_audit_2 N5: the runner executes whatever the plane at ``--connect``
    # dispatches (agents, tool calls) with the operator's own credentials, so the
    # control channel must not be MITM-able. Require https for a non-loopback
    # plane; localhost/dev http is fine, and FASTAIAGENT_RUNNER_ALLOW_INSECURE=1
    # is an explicit escape hatch for a trusted-network http plane.
    _require_secure_connect(connect)

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

    # Opt-in: only advertise (and accept) tool_exec when the operator has loaded
    # local tools/connectors for it — otherwise the runner would claim a
    # capability it can't fulfil. The tools register in this process' ToolRegistry;
    # tool_exec resolves them by their exposed name.
    capabilities = ["live_playground", "eval_run"]
    if tools:
        try:
            _load_tools(list(tools))
        except Exception as e:  # noqa: BLE001 — a bad --tools spec is a fatal config error
            console.print(f"[red]runner: failed to load --tools[/red] — {e}")
            raise typer.Exit(code=1) from e
        capabilities.append("tool_exec")
        console.print(f"[green]tool_exec enabled[/green] (tools: {', '.join(tools)})")

    channel = RunnerChannel(base_url=connect, api_key=key)
    daemon = RunnerDaemon(
        channel,
        max_concurrency=max_concurrency,
        labels=list(labels or []),
        capabilities=tuple(capabilities),
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
