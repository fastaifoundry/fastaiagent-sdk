"""CLI commands for evaluation."""

import typer
from rich.console import Console

eval_app = typer.Typer()
console = Console()


@eval_app.command("run")
def run_eval(
    dataset: str = typer.Option(..., help="Path to dataset (JSONL or CSV)"),
    agent: str = typer.Option(..., help="Agent module:function (e.g., myapp:agent.run)"),
    scorers: str = typer.Option("exact_match", help="Comma-separated scorer names"),
):
    """Run an evaluation."""
    console.print(f"Running evaluation with dataset={dataset}, agent={agent}, scorers={scorers}")
    console.print("[dim]Use the Python API for full evaluation features.[/dim]")


@eval_app.command("compare")
def compare_evals(
    run_a: str = typer.Argument(..., help="First eval run file"),
    run_b: str = typer.Argument(..., help="Second eval run file"),
):
    """Compare two evaluation runs."""
    console.print(f"Comparing {run_a} vs {run_b}")
