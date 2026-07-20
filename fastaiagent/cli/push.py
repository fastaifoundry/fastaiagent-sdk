"""``fastaiagent push`` — register agents defined in a module with the plane.

The deploy-time / CI counterpart to auto-registration: point it at a module
that *defines* agents (not one that runs them) and it pushes each as a governed
console object. Replaces hand-written ``httpx.post(...)`` registration.

    fastaiagent push --module my_app.agents
    fastaiagent push --module my_app.agents --dry-run   # preview payloads
"""

from __future__ import annotations

import importlib
import json
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

console = Console()


def _discover_agents(module: Any) -> list[Any]:
    """Collect distinct Agent instances defined at module scope."""
    from fastaiagent.agent.agent import Agent

    found: list[Any] = []
    seen: set[int] = set()
    for value in vars(module).values():
        if isinstance(value, Agent) and id(value) not in seen:
            seen.add(id(value))
            found.append(value)
    return found


def push(
    module: str = typer.Option(
        ...,
        "--module",
        "-m",
        help="Importable module that defines your agents, e.g. my_app.agents.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print the payloads that would be pushed, without contacting the plane.",
    ),
    api_key: str | None = typer.Option(
        None, "--api-key", help="Override the saved API key."
    ),
    target: str | None = typer.Option(
        None, "--target", help="Override the saved platform URL."
    ),
) -> None:
    """Register every agent defined in ``--module`` with the plane."""
    from fastaiagent.cli.auth import _load_saved_credentials

    # Import the module first so its agents exist. Do NOT rely on the module
    # calling connect() — we own the connection here.
    try:
        mod = importlib.import_module(module)
    except Exception as err:
        console.print(f"[red]Could not import module[/red] {module!r}: {err}")
        raise typer.Exit(code=1) from err

    agents = _discover_agents(mod)
    if not agents:
        console.print(
            f"[yellow]No agents found in[/yellow] {module!r}. "
            "Define module-level `fa.Agent(...)` objects there."
        )
        raise typer.Exit(code=1)

    if dry_run:
        console.print(f"[bold]Dry run[/bold] — {len(agents)} agent(s) in {module!r}:\n")
        for agent in agents:
            console.print(f"[cyan]{agent.name}[/cyan] → POST /public/v1/sdk/agents")
            console.print(json.dumps(agent.to_dict(), indent=2, default=str))
            console.print("")
        return

    # Connect with auto_register OFF so we push explicitly, once, per agent —
    # a deterministic result table rather than implicit flush side effects.
    creds = _load_saved_credentials()
    resolved_key = api_key or creds.get("api_key")
    resolved_target = target or creds.get("target")
    if not resolved_key:
        console.print(
            "[red]No API key.[/red] Run `fastaiagent connect` first or pass --api-key."
        )
        raise typer.Exit(code=1)

    from fastaiagent import client as _client

    try:
        _client.connect(
            api_key=resolved_key,
            target=resolved_target or "https://app.fastaiagent.net",
            project=creds.get("project"),
            auto_register=False,
        )
    except Exception as err:
        console.print(f"[red]Connection failed:[/red] {err}")
        raise typer.Exit(code=1) from err

    table = Table(title=f"Pushed {len(agents)} agent(s) from {module}")
    table.add_column("Agent")
    table.add_column("ID")
    table.add_column("Ver")
    table.add_column("Console URL")

    failures = 0
    for agent in agents:
        try:
            result = agent.push()
            table.add_row(
                result.name,
                result.agent_id or "—",
                str(result.version or "—"),
                result.url or "—",
            )
        except Exception as err:
            failures += 1
            table.add_row(agent.name, "[red]failed[/red]", "—", str(err))

    console.print(table)
    if failures:
        console.print(f"[red]{failures} agent(s) failed to register.[/red]")
        raise typer.Exit(code=1)
    console.print("[green]✓[/green] All agents registered.")


def register_push_command(app: typer.Typer) -> None:
    """Attach ``fastaiagent push`` to the root CLI app."""
    app.command(
        name="push",
        help="Register agents defined in a module with the plane (--module).",
    )(push)
