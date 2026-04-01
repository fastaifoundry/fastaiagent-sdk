"""CLI commands for traces."""

import typer
from rich.console import Console
from rich.table import Table

traces_app = typer.Typer()
console = Console()


@traces_app.command("list")
def list_traces(last_hours: int = typer.Option(24, help="Show traces from last N hours")):
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
):
    """Export a trace."""
    from fastaiagent.trace.storage import TraceStore

    store = TraceStore()
    exported = store.export(trace_id, format=format)
    console.print(exported)
    store.close()
