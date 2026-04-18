"""CLI commands for knowledge base management."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

kb_app = typer.Typer()
console = Console()


@kb_app.command("list")
def kb_list(
    path: str = typer.Option(
        ".fastaiagent/kb/",
        "--path",
        help="Root directory to scan for KBs.",
    ),
) -> None:
    """List all knowledge bases under the given root directory."""
    root = Path(path)
    if not root.exists():
        console.print(f"No KBs found — {root} does not exist.")
        return

    from fastaiagent.kb import LocalKB

    rows: list[tuple[str, int, str]] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        sqlite = child / "kb.sqlite"
        if not sqlite.exists():
            continue
        try:
            kb = LocalKB(name=child.name, path=str(root))
            status = kb.status()
            rows.append((status["name"], status["chunk_count"], str(child)))
            kb.close()
        except Exception as err:  # pragma: no cover — surface odd KBs but keep scanning
            rows.append((child.name, -1, f"(error: {err})"))

    if not rows:
        console.print(f"No persistent KBs found under {root}.")
        return

    table = Table(title=f"Knowledge bases under {root}")
    table.add_column("Name")
    table.add_column("Chunks", justify="right")
    table.add_column("Path")
    for name, count, rel in rows:
        table.add_row(name, str(count) if count >= 0 else "?", rel)
    console.print(table)


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
