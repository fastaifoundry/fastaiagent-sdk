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


@traces_app.command("prune")
def prune_acked(
    older_than_days: int | None = typer.Option(
        None,
        "--older-than-days",
        help="Only delete acked spans older than this many days. Omit to prune all acked spans.",
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt."),
) -> None:
    """Delete acked/abandoned spans from local.db to reclaim space.

    Removes only ``synced=1`` spans (already sent to the plane, or abandoned
    from the re-send queue by the buffer bound) — nothing waiting to be re-sent
    is ever touched. Complements ``purge`` (which deletes by trace scope).
    """
    from fastaiagent.trace.storage import TraceStore

    store = TraceStore()
    scope = (
        f"acked spans older than {older_than_days} day(s)"
        if older_than_days is not None
        else "all acked spans"
    )
    if not yes:
        confirm = typer.confirm(f"Delete {scope} from local.db?")
        if not confirm:
            console.print("[dim]Aborted.[/dim]")
            store.close()
            return
    deleted = store.prune_acked(older_than_days=older_than_days)
    console.print(f"[green]✓[/green] Pruned {deleted} acked span(s).")
    store.close()


@traces_app.command("scrub-secrets")
def scrub_secrets_cmd(
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Report what would change without writing anything."
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt."),
) -> None:
    """Remove credentials from spans an older SDK stored in local.db.

    Since 1.89.0 a guardrail's ``llm`` config never carries a key, and a
    provider error that echoes one is scrubbed before it is stored. Rows
    written earlier can still hold one, in ``agent.guardrails`` or in an
    exception event. This rewrites them in place.

    It cannot recall what was already sent to the plane. When the report
    counts already-sent spans, rotate the keys involved.
    """
    from fastaiagent._internal.config import get_config
    from fastaiagent.trace.secret_scrub import scrub_stored_spans

    db_path = get_config().resolved_trace_db_path
    preview = scrub_stored_spans(db_path, dry_run=True)
    if preview.spans_changed == 0:
        console.print(f"[green]✓[/green] No credentials found in {preview.spans_scanned} span(s).")
        return
    console.print(
        f"{preview.spans_changed} span(s) in {preview.traces_changed} trace(s) of {db_path} "
        f"hold a credential."
    )
    if preview.already_sent:
        console.print(
            f"[yellow]{preview.already_sent} of them were already sent to the control plane. "
            f"Scrubbing local.db cannot recall those copies: rotate the keys involved.[/yellow]"
        )
    if dry_run:
        console.print("[dim]Dry run: nothing written.[/dim]")
        return
    if not yes and not typer.confirm("Scrub them now?"):
        console.print("[dim]Aborted.[/dim]")
        raise typer.Exit(code=1)
    report = scrub_stored_spans(db_path)
    console.print(f"[green]✓[/green] Scrubbed {report.spans_changed} span(s).")


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

        # Delete in order: attachments → guardrail events → spans. The
        # guardrail rows go by the trace ids collected above: selecting them
        # through ``spans`` after the spans were gone removed none (1.89.0),
        # leaving judged payloads, which can hold PII, behind.
        from fastaiagent.ui.events import delete_guardrail_events

        if attachments:
            db.execute(
                f"DELETE FROM trace_attachments{scope_clause}",
                scope_params,
            )
        delete_guardrail_events(
            db,
            [r["trace_id"] for r in affected],
            orphans_before=cutoff,
            everything=cutoff is None,
        )
        db.execute(
            f"DELETE FROM spans{scope_clause}",
            scope_params,
        )

    console.print(
        f"[green]\N{CHECK MARK} Deleted {n_traces} {what}{also}.[/green]"
    )
