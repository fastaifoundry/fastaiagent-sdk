"""CLI commands for traces."""

import typer
from rich.console import Console
from rich.table import Table

traces_app = typer.Typer()
console = Console()


@traces_app.command("list")
def list_traces(last_hours: int = typer.Option(24, help="Show traces from last N hours")) -> None:
    """List recent traces."""
    from fastaiagent.trace.storage import TraceStore

    store = TraceStore()
    traces = store.list_traces(last_hours=last_hours)

    if not traces:
        console.print("[dim]No traces found.[/dim]")
        return

    table = Table(title="Recent Traces")
    table.add_column("Trace ID", style="cyan")
    table.add_column("Name")
    table.add_column("Status")
    table.add_column("Spans", justify="right")
    table.add_column("Time")

    for t in traces:
        table.add_row(
            t.trace_id[:12] + "...",
            t.name,
            t.status,
            str(t.span_count),
            t.start_time[:19],
        )

    console.print(table)
    store.close()


@traces_app.command("export")
def export_trace(
    trace_id: str = typer.Argument(..., help="Trace ID to export"),
    format: str = typer.Option("json", help="Export format (json)"),
) -> None:
    """Export a trace."""
    from fastaiagent.trace.storage import TraceStore

    store = TraceStore()
    exported = store.export(trace_id, format=format)
    console.print(exported)
    store.close()


@traces_app.command("purge")
def purge_traces(
    older_than_days: int | None = typer.Option(
        None,
        "--older-than-days",
        help="Only delete traces whose root span started this many days "
        "ago or more. Omit to delete all traces.",
    ),
    attachments: bool = typer.Option(
        False,
        "--attachments",
        help="Also delete attachment BLOBs (image/PDF bytes captured "
        "into local.db when ``trace_full_images`` was on). "
        "Without this flag, only span rows are removed.",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip the interactive confirmation prompt.",
    ),
) -> None:
    """Delete trace data from local.db.

    Closes security_review_1.md L6: traces capture user prompts and LLM
    output verbatim — including any PII the user typed — and image /
    PDF attachments live in the same DB. Operators occasionally need a
    way to scrub older history before sharing a project directory or
    cutting a backup. ``fastaiagent traces purge`` is that knob.

    Examples:
        # Delete every trace older than 30 days, including attachments.
        fastaiagent traces purge --older-than-days 30 --attachments

        # Wipe everything (with prompt).
        fastaiagent traces purge

        # Same, scriptable.
        fastaiagent traces purge -y
    """
    from datetime import datetime, timedelta, timezone

    from fastaiagent._internal.config import get_config
    from fastaiagent._internal.storage import SQLiteHelper

    db_path = get_config().resolved_trace_db_path
    cutoff: str | None = None
    if older_than_days is not None:
        if older_than_days < 0:
            console.print("[red]--older-than-days must be >= 0[/red]")
            raise typer.Exit(code=2)
        cutoff = (
            datetime.now(tz=timezone.utc) - timedelta(days=older_than_days)
        ).isoformat()

    with SQLiteHelper(db_path) as db:
        # Identify the affected trace_ids first so the user sees a
        # concrete count before we start deleting.
        if cutoff is None:
            scope_clause = ""
            scope_params: tuple = ()
        else:
            # A trace is "old" if its root span started before the
            # cutoff. Sub-spans inherit.
            scope_clause = (
                " WHERE trace_id IN ("
                "   SELECT trace_id FROM spans"
                "   WHERE parent_span_id IS NULL AND start_time < ?"
                " )"
            )
            scope_params = (cutoff,)
        affected = db.fetchall(
            f"SELECT DISTINCT trace_id FROM spans{scope_clause}",
            scope_params,
        )
        n_traces = len(affected)
        if n_traces == 0:
            console.print("[dim]No matching traces.[/dim]")
            return

        what = "traces" if cutoff is None else f"traces older than {older_than_days} day(s)"
        also = " plus attachment BLOBs" if attachments else ""
        if not yes:
            confirm = typer.confirm(
                f"Delete {n_traces} {what}{also} from {db_path}?"
            )
            if not confirm:
                console.print("[yellow]Aborted.[/yellow]")
                raise typer.Exit(code=1)

        # Delete in order: attachments → events → spans (FK-clean).
        if attachments:
            db.execute(
                f"DELETE FROM trace_attachments{scope_clause}",
                scope_params,
            )
        db.execute(
            f"DELETE FROM spans{scope_clause}",
            scope_params,
        )
        # Other per-trace tables that should track the same scope.
        for tbl in ("guardrail_events",):
            try:
                db.execute(
                    f"DELETE FROM {tbl}{scope_clause}",
                    scope_params,
                )
            except Exception:
                # Tables that don't exist on older schemas — skip.
                pass

    console.print(
        f"[green]\N{CHECK MARK} Deleted {n_traces} {what}{also}.[/green]"
    )
