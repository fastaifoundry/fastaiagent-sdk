"""``fastaiagent resume / list-pending / inspect`` — CLI durability commands.

Three Typer commands that converge on the same internal resume path as the
Python API and the HTTP endpoint:

    fastaiagent resume <execution-id> --runner module:attr [--value JSON]
    fastaiagent list-pending [--db-path PATH] [--limit N]
    fastaiagent inspect <execution-id> [--db-path PATH]

``list-pending`` and ``inspect`` only read the local SQLite checkpoint
store; they don't need a Python entrypoint. ``resume`` needs the runner
because resuming a Chain / Agent / Swarm / Supervisor requires the
original Python definition to re-instantiate it.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import sys
from typing import Any

import typer

resume_app = typer.Typer(
    no_args_is_help=False,
    help="Resume durable executions from the CLI.",
)


def _open_store(db_path: str | None) -> Any:
    from fastaiagent.checkpointers import SQLiteCheckpointer

    cp = SQLiteCheckpointer(db_path=db_path)
    cp.setup()
    return cp


def _load_runner(spec: str) -> Any:
    """Import a runner from ``module:attr``.

    The attribute can be the runner instance directly, or a zero-arg
    callable that returns one (factory pattern).
    """
    if ":" not in spec:
        raise typer.BadParameter("--runner expects 'module:attr' (e.g. 'myapp.chains:my_chain')")
    module_name, attr = spec.split(":", 1)
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise typer.BadParameter(f"Cannot import module {module_name!r}: {e}") from e
    if not hasattr(module, attr):
        raise typer.BadParameter(f"Module {module_name!r} has no attribute {attr!r}")
    obj = getattr(module, attr)
    if callable(obj) and not hasattr(obj, "aresume"):
        # Zero-arg factory.
        obj = obj()
    if not hasattr(obj, "aresume"):
        raise typer.BadParameter(
            f"{spec!r} resolved to {obj!r}; expected a Chain / Agent / "
            "Swarm / Supervisor with .aresume()."
        )
    return obj


def _resume_command(
    execution_id: str = typer.Argument(..., help="The execution_id to resume."),
    runner: str = typer.Option(
        ...,
        "--runner",
        help=(
            "Python entrypoint for the Chain/Agent/Swarm/Supervisor to "
            "resume, in 'module:attr' form (e.g. 'myapp.chains:my_chain')."
        ),
    ),
    value: str = typer.Option(
        '{"approved": true}',
        "--value",
        help=(
            "JSON object passed to Resume(...). Defaults to "
            '{"approved": true}; pass {"approved": false, '
            '"metadata": {"reason": "..."}} to reject.'
        ),
    ),
) -> None:
    """Resume a suspended execution.

    Loads the runner via ``--runner module:attr`` and calls its
    ``aresume(execution_id, resume_value=Resume(...))``.
    """
    from fastaiagent import AlreadyResumed, Resume

    try:
        payload = json.loads(value)
    except json.JSONDecodeError as e:
        raise typer.BadParameter(f"--value is not valid JSON: {e}") from e
    if not isinstance(payload, dict):
        raise typer.BadParameter("--value must be a JSON object.")

    resume_value = Resume(
        approved=bool(payload.get("approved", True)),
        metadata=dict(payload.get("metadata") or {}),
    )

    runner_obj = _load_runner(runner)
    typer.echo(f"Resuming {execution_id} via {runner!r} …")

    try:
        result = asyncio.run(runner_obj.aresume(execution_id, resume_value=resume_value))
    except AlreadyResumed as e:
        typer.secho(
            f"AlreadyResumed: {e}",
            fg=typer.colors.YELLOW,
            err=True,
        )
        raise typer.Exit(code=2) from e

    status = getattr(result, "status", "completed")
    output = getattr(result, "output", "")
    typer.echo(f"status: {status}")
    if output:
        typer.echo(f"output: {output}")
    if status == "paused":
        pi = getattr(result, "pending_interrupt", None) or {}
        typer.echo(f"pending_interrupt: {json.dumps(pi, default=str)}")


def _list_pending_command(
    db_path: str | None = typer.Option(
        None,
        "--db-path",
        help="Path to local.db (defaults to the configured local_db_path).",
    ),
    limit: int = typer.Option(100, "--limit", help="Max rows to display."),
) -> None:
    """Show every pending interrupt as a Rich table."""
    from rich.console import Console
    from rich.table import Table

    store = _open_store(db_path)
    try:
        rows = store.list_pending_interrupts(limit=limit)
    finally:
        store.close()

    if not rows:
        typer.echo("No pending interrupts.")
        return

    table = Table(title=f"Pending Interrupts ({len(rows)})")
    table.add_column("execution_id", style="cyan", no_wrap=True)
    table.add_column("chain_name", style="white")
    table.add_column("node_id", style="white")
    table.add_column("reason", style="yellow")
    table.add_column("agent_path", style="magenta", overflow="fold")
    table.add_column("created_at", style="dim")

    for r in rows:
        table.add_row(
            r.execution_id,
            r.chain_name,
            r.node_id,
            r.reason,
            r.agent_path or "",
            r.created_at,
        )

    Console().print(table)


def _inspect_command(
    execution_id: str = typer.Argument(..., help="The execution_id to inspect."),
    db_path: str | None = typer.Option(
        None,
        "--db-path",
        help="Path to local.db (defaults to the configured local_db_path).",
    ),
    limit: int = typer.Option(100, "--limit", help="Max checkpoints to display."),
) -> None:
    """Show the checkpoint history for an execution."""
    from rich.console import Console
    from rich.table import Table

    store = _open_store(db_path)
    try:
        rows = store.list(execution_id, limit=limit)
        latest = store.get_last(execution_id)
    finally:
        store.close()

    if not rows or latest is None:
        typer.secho(
            f"No checkpoints found for execution {execution_id!r}.",
            fg=typer.colors.YELLOW,
            err=True,
        )
        raise typer.Exit(code=1)

    typer.echo(f"execution_id: {execution_id}")
    typer.echo(f"chain_name:   {latest.chain_name}")
    typer.echo(f"latest_status: {latest.status}")
    typer.echo(f"checkpoint_count: {len(rows)}")

    table = Table(title="Checkpoints (chronological)")
    table.add_column("#", style="dim", width=4)
    table.add_column("node_id", style="cyan", no_wrap=True)
    table.add_column("status", style="white")
    table.add_column("agent_path", style="magenta", overflow="fold")
    table.add_column("created_at", style="dim")

    for i, cp in enumerate(rows, 1):
        status_color = (
            "yellow" if cp.status == "interrupted" else "red" if cp.status == "failed" else "green"
        )
        table.add_row(
            str(i),
            cp.node_id,
            f"[{status_color}]{cp.status}[/{status_color}]",
            cp.agent_path or "",
            cp.created_at,
        )

    Console().print(table)


# Public entry points — registered as top-level commands by cli/main.py.
def register(app: typer.Typer) -> None:
    """Attach ``resume`` / ``list-pending`` / ``inspect`` to the parent CLI."""
    app.command(name="resume", help="Resume a suspended execution.")(_resume_command)
    app.command(name="list-pending", help="List pending interrupts.")(_list_pending_command)
    app.command(name="inspect", help="Show checkpoint history for an execution.")(_inspect_command)


# Also expose a standalone Typer instance so ``python -m fastaiagent.cli.resume``
# can be wired up if needed; not used by main.py but useful for tests.
def _build_standalone_app() -> typer.Typer:
    app = typer.Typer(no_args_is_help=True, help="Resume CLI")
    register(app)
    return app


__all__ = ["register"]


if __name__ == "__main__":  # pragma: no cover
    sys.exit(_build_standalone_app()())
