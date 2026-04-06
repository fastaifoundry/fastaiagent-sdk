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
    console.print(f"Persist: {status['persist']}")
    console.print(f"Search type: {status['search_type']}")
    console.print(f"Index type: {status['index_type']}")
    kb.close()


@kb_app.command("add")
def kb_add(
    file_path: str = typer.Argument(..., help="File or directory to add"),
    name: str = typer.Option("default", help="KB name"),
) -> None:
    """Add a file or directory to the knowledge base."""
    from fastaiagent.kb import LocalKB

    kb = LocalKB(name=name)
    count = kb.add(file_path)
    console.print(f"Added {count} chunks from {file_path}")
    kb.close()


@kb_app.command("clear")
def kb_clear(name: str = typer.Option("default", help="KB name")) -> None:
    """Clear all data from a knowledge base."""
    from fastaiagent.kb import LocalKB

    kb = LocalKB(name=name)
    count = kb.status()["chunk_count"]
    kb.clear()
    console.print(f"Cleared {count} chunks from KB '{name}'")
    kb.close()


@kb_app.command("delete")
def kb_delete(
    source: str = typer.Argument(..., help="Source file path to remove"),
    name: str = typer.Option("default", help="KB name"),
) -> None:
    """Delete chunks from a source file."""
    from fastaiagent.kb import LocalKB

    kb = LocalKB(name=name)
    count = kb.delete_by_source(source)
    console.print(f"Deleted {count} chunks from source '{source}'")
    kb.close()
