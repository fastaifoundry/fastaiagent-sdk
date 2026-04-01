"""CLI commands for prompt management."""

import typer
from rich.console import Console
from rich.table import Table

prompts_app = typer.Typer()
console = Console()


@prompts_app.command("list")
def list_prompts(path: str = typer.Option(".prompts/", help="Prompts directory")) -> None:
    """List registered prompts."""
    from fastaiagent.prompt import PromptRegistry

    reg = PromptRegistry(path=path)
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
    path: str = typer.Option(".prompts/", help="Prompts directory"),
) -> None:
    """Show diff between two prompt versions."""
    from fastaiagent.prompt import PromptRegistry

    reg = PromptRegistry(path=path)
    diff = reg.diff(name, v1, v2)
    console.print(diff)
