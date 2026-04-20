"""CLI for migrating legacy stores into the unified local.db."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

console = Console()


def migrate_command(
    db: Path | None = typer.Option(
        None,
        "--db",
        help="Target local.db path (defaults to FASTAIAGENT_LOCAL_DB).",
    ),
    trace_db: Path | None = typer.Option(
        None,
        "--trace-db",
        help="Override legacy traces.db path.",
    ),
    checkpoint_db: Path | None = typer.Option(
        None,
        "--checkpoint-db",
        help="Override legacy checkpoints.db path.",
    ),
    prompt_dir: Path | None = typer.Option(
        None,
        "--prompt-dir",
        help="Override legacy .prompts/ directory.",
    ),
) -> None:
    """Copy legacy storage (traces.db, checkpoints.db, .prompts/) into local.db.

    Idempotent — safe to run more than once. Legacy files are left in place;
    delete them yourself once you're satisfied with the migration report.
    """
    from fastaiagent.ui.migration import migrate_to_local_db

    report = migrate_to_local_db(
        target_db=db,
        legacy_trace_db=trace_db,
        legacy_checkpoint_db=checkpoint_db,
        legacy_prompt_dir=prompt_dir,
    )

    if report.nothing_to_do():
        console.print("[dim]Nothing to migrate — no legacy files detected.[/dim]")
        return

    table = Table(title="Migration report")
    table.add_column("Source", style="cyan")
    table.add_column("Count", justify="right")
    if report.legacy_trace_db:
        table.add_row(f"spans from {report.legacy_trace_db}", str(report.spans_migrated))
    if report.legacy_checkpoint_db:
        table.add_row(
            f"checkpoints from {report.legacy_checkpoint_db}",
            str(report.checkpoints_migrated),
        )
    if report.legacy_prompt_dir:
        table.add_row(
            f"prompts from {report.legacy_prompt_dir}",
            str(report.prompts_migrated),
        )
        table.add_row(
            "  prompt versions",
            str(report.prompt_versions_migrated),
        )
        table.add_row(
            "  fragments",
            str(report.fragments_migrated),
        )
        table.add_row(
            "  aliases",
            str(report.aliases_migrated),
        )
    console.print(table)
    console.print(
        "[green]Migration complete.[/green] Legacy files left in place; "
        "delete them manually once verified."
    )
