"""CLI commands for evaluation."""

import typer
from rich.console import Console
from rich.table import Table

eval_app = typer.Typer()
console = Console()


@eval_app.command("run")
def run_eval(
    dataset: str = typer.Option(..., help="Path to dataset (JSONL or CSV)"),
    agent: str = typer.Option(..., help="Agent module:function (e.g., myapp:agent.run)"),
    scorers: str = typer.Option("exact_match", help="Comma-separated scorer names"),
) -> None:
    """Run an evaluation."""
    console.print(f"Running evaluation with dataset={dataset}, agent={agent}, scorers={scorers}")
    console.print("[dim]Use the Python API for full evaluation features.[/dim]")


@eval_app.command("compare")
def compare_evals(
    run_a: str = typer.Argument(..., help="First eval run file"),
    run_b: str = typer.Argument(..., help="Second eval run file"),
) -> None:
    """Compare two evaluation runs."""
    console.print(f"Comparing {run_a} vs {run_b}")


@eval_app.command("curate")
def curate_cmd(
    out: str = typer.Option(..., "--out", "-o", help="Output JSONL path"),
    filter: str = typer.Option(
        "all", "--filter", "-f", help="all | favorites | noted | guardrail | failed"
    ),
    agent: str | None = typer.Option(None, "--agent", help="Only this agent's spans"),
    since: float | None = typer.Option(None, "--since", help="Only traces from the last N hours"),
    limit: int = typer.Option(200, "--limit", help="Max traces to read (most recent first)"),
    append: bool = typer.Option(False, "--append/--no-append", help="Append to the output file"),
    output_as_expected: bool | None = typer.Option(
        None,
        "--output-as-expected/--needs-review",
        help="Override the per-filter default for expected_output",
    ),
    dedup_by: str = typer.Option("none", "--dedup-by", help="none | input"),
    db: str | None = typer.Option(None, "--db", help="local.db path (default: from config)"),
) -> None:
    """Curate an eval dataset from captured agent traces.

    Each agent.<name> span (root, or nested inside a chain/supervisor/swarm)
    becomes one case. Good filters (all/favorites/noted) use the captured output
    as expected_output; failure filters (guardrail/failed) mark cases needs_review.
    """
    from fastaiagent.eval.curate import curate_from_traces
    from fastaiagent.eval.dataset import Dataset

    try:
        items = curate_from_traces(
            filter=filter,
            agent=agent,
            since_hours=since,
            limit=limit,
            mark_output_as_expected=output_as_expected,
            db_path=db,
            dedup_by=dedup_by,
        )
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    if not items:
        console.print(f"[yellow]No matching agent traces for filter '{filter}'.[/yellow]")
        raise typer.Exit(0)

    Dataset.from_list(items).to_jsonl(out, append=append)

    needs = sum(1 for it in items if it.get("needs_review"))
    table = Table(title=f"Curated {len(items)} case(s) -> {out}")
    table.add_column("metric")
    table.add_column("value", justify="right")
    table.add_row("filter", filter)
    table.add_row("cases", str(len(items)))
    table.add_row("ready to score", str(len(items) - needs))
    table.add_row("needs review", str(needs))
    console.print(table)
    if needs:
        console.print(
            '[dim]needs_review cases have expected_output="" - fill in the gold '
            "answer before evaluating.[/dim]"
        )
    console.print(
        f'[dim]Next: evaluate(agent_fn=..., dataset="{out}", scorers=[...]) in Python.[/dim]'
    )
