"""CLI commands for prompt management."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

prompts_app = typer.Typer()
console = Console()


def _default_db_path() -> str:
    from fastaiagent._internal.config import get_config

    return get_config().local_db_path


@prompts_app.command("list")
def list_prompts(
    path: str | None = typer.Option(
        None,
        "--path",
        help="Override local DB path (defaults to FASTAIAGENT_LOCAL_DB).",
    ),
) -> None:
    """List registered prompts."""
    from fastaiagent.prompt import PromptRegistry

    reg = PromptRegistry(path=path or _default_db_path())
    prompts = reg.list()

    if not prompts:
        console.print("[dim]No prompts found.[/dim]")
        return

    table = Table(title="Prompts")
    table.add_column("Name", style="cyan")
    table.add_column("Latest Version", justify="right")
    table.add_column("Total Versions", justify="right")

    for p in prompts:
        table.add_row(p["name"], str(p["latest_version"]), str(p["versions"]))

    console.print(table)


@prompts_app.command("diff")
def diff_prompts(
    name: str = typer.Argument(..., help="Prompt name"),
    v1: int = typer.Argument(..., help="First version"),
    v2: int = typer.Argument(..., help="Second version"),
    path: str | None = typer.Option(
        None,
        "--path",
        help="Override local DB path (defaults to FASTAIAGENT_LOCAL_DB).",
    ),
) -> None:
    """Show diff between two prompt versions."""
    from fastaiagent.prompt import PromptRegistry

    reg = PromptRegistry(path=path or _default_db_path())
    diff = reg.diff(name, v1, v2)
    console.print(diff)
