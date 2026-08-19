"""CLI commands for evaluation."""

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

eval_app = typer.Typer()
console = Console()


# Exit codes (documented contract; 2 belongs to Click for usage errors):
_EXIT_QUALITY_FAILED = 1
_EXIT_INFRA_INVALID = 3


@eval_app.command("run")
def run_eval(
    dataset: str = typer.Option(..., help="Path to dataset (JSONL or CSV)"),
    agent: str = typer.Option(
        ...,
        help="Agent target: 'path/to/file.py:agent' or 'pkg.module:agent' "
        "(an Agent instance or a callable)",
    ),
    scorers: str = typer.Option("exact_match", help="Comma-separated scorer names"),
    concurrency: int = typer.Option(4, help="Concurrent cases"),
    run_name: str | None = typer.Option(None, "--run-name", help="Name for the persisted run"),
    fail_under: list[str] = typer.Option(
        [],
        "--fail-under",
        help="Quality gate, repeatable: 'overall.pass_rate=0.9', "
        "'geval.avg_score=0.7', or 'exact_match=0.9' (bare = pass_rate)",
    ),
    max_error_rate: float | None = typer.Option(
        None,
        "--max-error-rate",
        help="Max fraction of infra-errored (unscored) cases before the run is INVALID (exit 3)",
    ),
    json_out: str | None = typer.Option(None, "--json", help="Write the report JSON here"),
    no_persist: bool = typer.Option(False, "--no-persist", help="Skip local.db persistence"),
    db: str | None = typer.Option(None, "--db", help="local.db path (default: from config)"),
) -> None:
    """Run an evaluation and gate on it.

    Exit codes: 0 gate passed · 1 quality gate failed · 3 run invalid
    (infra error rate exceeded / nothing scored).
    """
    import json as _json

    from fastaiagent._internal.errors import EvalError
    from fastaiagent._internal.target import resolve_agent_fn, resolve_target
    from fastaiagent.eval.evaluate import evaluate
    from fastaiagent.eval.gate import gate
    from fastaiagent.eval.results import Scorecard

    try:
        agent_fn = resolve_agent_fn(resolve_target(agent))
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    try:
        results = evaluate(
            agent_fn,
            dataset,
            scorers=[s.strip() for s in scorers.split(",") if s.strip()],
            concurrency=concurrency,
            persist=not no_persist and db is None,
            run_name=run_name,
        )
        if not no_persist and db is not None:
            results.run_id = results.persist_local(db_path=db, run_name=run_name)
        report = gate(results, fail_under=list(fail_under), max_error_rate=max_error_rate)
    except EvalError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(_EXIT_INFRA_INVALID) from exc

    scorecard = Scorecard.from_eval_results(results)
    table = Table(title=f"Eval run — gate {report.outcome.upper()}")
    table.add_column("metric")
    table.add_column("avg_score", justify="right")
    table.add_column("pass_rate", justify="right")
    table.add_column("n", justify="right")
    for m in scorecard.metrics:
        table.add_row(m.name, f"{m.avg_score:.4f}", f"{m.pass_rate:.4f}", str(m.n))
    table.add_row("overall", "-", f"{scorecard.overall_pass_rate:.4f}", str(report.scored))
    if report.errored:
        table.add_row("[yellow]errored (unscored)[/yellow]", "-", "-", str(report.errored))
    console.print(table)
    for line in report.describe():
        console.print(f"[dim]{line}[/dim]")
    if results.run_id:
        console.print(f"[dim]persisted run_id={results.run_id}[/dim]")

    if json_out:
        payload = {
            "schema_version": 1,
            "run_id": results.run_id,
            "run_name": run_name,
            "scorecard": scorecard.to_dict(),
            "gate": {
                "outcome": report.outcome,
                "checks": [c.describe() for c in report.checks],
                "scored": report.scored,
                "errored": report.errored,
                "error_rate": report.error_rate,
                "max_error_rate": report.max_error_rate,
            },
        }
        Path(json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(json_out).write_text(_json.dumps(payload, indent=2, default=str))
        console.print(f"[dim]report written to {json_out}[/dim]")

    if report.outcome == "invalid":
        console.print("[red]Run INVALID — infra failures, not an agent-quality verdict.[/red]")
        raise typer.Exit(_EXIT_INFRA_INVALID)
    if report.outcome == "failed":
        console.print("[red]Quality gate FAILED.[/red]")
        raise typer.Exit(_EXIT_QUALITY_FAILED)


@eval_app.command("compare")
def compare_evals(
    baseline: str = typer.Argument(..., help="Baseline run: run_id or run_name (latest wins)"),
    current: str = typer.Argument(..., help="Current run: run_id or run_name (latest wins)"),
    tolerance: float = typer.Option(
        0.0, "--tolerance", help="Allowed overall pass-rate drop before failing"
    ),
    fail_on_regression: bool = typer.Option(
        True,
        "--fail-on-regression/--no-fail-on-regression",
        help="Exit 1 when pass-rate drops more than --tolerance",
    ),
    db: str | None = typer.Option(None, "--db", help="local.db path (default: from config)"),
) -> None:
    """Compare two persisted eval runs (baseline first).

    Exit codes: 0 no regression · 1 regression beyond tolerance.
    """
    from fastaiagent._internal.errors import EvalError
    from fastaiagent.eval.compare import compare_runs

    try:
        comparison = compare_runs(baseline, current, db_path=db)
    except EvalError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(_EXIT_INFRA_INVALID) from exc

    table = Table(title="Eval run comparison")
    table.add_column("metric")
    table.add_column("value", justify="right")
    table.add_row("baseline pass_rate", f"{comparison.run_a.get('pass_rate') or 0.0:.4f}")
    table.add_row("current pass_rate", f"{comparison.run_b.get('pass_rate') or 0.0:.4f}")
    table.add_row("delta", f"{comparison.pass_rate_delta:+.4f}")
    table.add_row("regressed cases", str(len(comparison.regressed)))
    table.add_row("improved cases", str(len(comparison.improved)))
    table.add_row("unchanged pass/fail", f"{comparison.unchanged_pass}/{comparison.unchanged_fail}")
    console.print(table)
    for line in comparison.describe():
        console.print(f"[dim]{line}[/dim]")

    drop = -comparison.pass_rate_delta
    if fail_on_regression and drop > tolerance:
        console.print(
            f"[red]REGRESSION — pass-rate dropped {drop:.4f} (tolerance {tolerance}).[/red]"
        )
        raise typer.Exit(_EXIT_QUALITY_FAILED)


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
