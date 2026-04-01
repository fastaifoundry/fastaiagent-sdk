"""CLI commands for knowledge base management."""

import typer
from rich.console import Console

kb_app = typer.Typer()
console = Console()


@kb_app.command("status")
def kb_status(name: str = typer.Option("default", help="KB name")) -> None:
    """Show knowledge base status."""
    from fastaiagent.kb import LocalKB

    kb = LocalKB(name=name)
    status = kb.status()
    console.print(f"KB: {status['name']}")
    console.print(f"Chunks: {status['chunk_count']}")
    console.print(f"Path: {status['path']}")


@kb_app.command("add")
def kb_add(
    file_path: str = typer.Argument(..., help="File to add"),
    name: str = typer.Option("default", help="KB name"),
) -> None:
    """Add a file to the knowledge base."""
    from fastaiagent.kb import LocalKB

    kb = LocalKB(name=name)
    count = kb.add(file_path)
    console.print(f"Added {count} chunks from {file_path}")
