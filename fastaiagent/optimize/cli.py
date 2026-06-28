"""``fastaiagent optimize`` — eval-driven prompt optimization (P1).

Loads an ``Agent`` from a ``module:attr`` path, runs the optimization loop over a
dataset, prints the trajectory, and optionally writes the winning prompt to a
file. Mirrors the resolver used by ``fastaiagent agent`` / ``fastaiagent mcp``.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

optimize_app = typer.Typer()
console = Console()


def _resolve_target(spec: str) -> Any:
    """Resolve ``path/to/file.py:attr`` or ``pkg.module:attr`` into a live object."""
    if ":" not in spec:
        raise typer.BadParameter(
            f"Expected 'path/to/file.py:attr' or 'pkg.module:attr', got {spec!r}"
        )
    module_part, attr = spec.rsplit(":", 1)
    path = Path(module_part)
    if path.exists():
        module_name = path.stem
        spec_obj = importlib.util.spec_from_file_location(module_name, str(path))
        if spec_obj is None or spec_obj.loader is None:
            raise typer.BadParameter(f"Cannot load module from {path}")
        module = importlib.util.module_from_spec(spec_obj)
        sys.path.insert(0, str(path.parent.resolve()))
        spec_obj.loader.exec_module(module)
    else:
        module = importlib.import_module(module_part)
    if not hasattr(module, attr):
        raise typer.BadParameter(f"Module {module_part!r} has no attribute {attr!r}")
    return getattr(module, attr)


@optimize_app.callback(invoke_without_command=True)
def optimize_cmd(
    ctx: typer.Context,
    agent: str = typer.Option(
        ..., "--agent", help="path/to/file.py:agent or pkg.module:agent (an Agent instance)"
    ),
    dataset: str = typer.Option(..., "--dataset", help="Path to dataset (JSONL or CSV)"),
    scorers: str = typer.Option("exact_match", "--scorers", help="Comma-separated scorer names"),
    max_iterations: int = typer.Option(8, "--max-iterations", help="Max optimization rounds"),
    patience: int = typer.Option(3, "--patience", help="Stop after N non-improving rounds"),
    candidates: int = typer.Option(3, "--candidates", help="Prompt proposals per round"),
    seed: int = typer.Option(0, "--seed", help="Seeds the train/dev/holdout split"),
    primary_metric: str = typer.Option(
        None, "--primary-metric", help="Scorer name to select on (default: overall pass-rate)"
    ),
    judge: str = typer.Option(
        None, "--judge", help="Add an LLM judge (criteria string) as the selection scorer"
    ),
    proposer_model: str = typer.Option(
        None, "--proposer-model", help="Model id for the prompt proposer (default: env/config)"
    ),
    out: str = typer.Option(None, "--out", help="Write the winning system prompt to this file"),
    no_persist: bool = typer.Option(
        False, "--no-persist", help="Don't write per-candidate evals to local.db"
    ),
) -> None:
    """Optimize an agent's system prompt against a dataset (prompt-only, P1)."""
    if ctx.invoked_subcommand is not None:
        return

    from fastaiagent.eval.llm_judge import LLMJudge
    from fastaiagent.optimize import OptimizeConfig
    from fastaiagent.optimize import optimize as run_optimize

    target = _resolve_target(agent)
    scorer_list: list[Any] = [s.strip() for s in scorers.split(",") if s.strip()]
    selection_judge = LLMJudge(criteria=judge) if judge else None

    proposer_llm = None
    if proposer_model:
        from fastaiagent.llm import LLMClient

        proposer_llm = LLMClient(model=proposer_model)

    cfg = OptimizeConfig(
        max_iterations=max_iterations,
        patience=patience,
        candidates_per_iteration=candidates,
        seed=seed,
        primary_metric=primary_metric,
        selection_judge=selection_judge,
    )
    try:
        report = run_optimize(
            target,
            dataset,
            scorer_list,
            config=cfg,
            proposer_llm=proposer_llm,
            persist=not no_persist,
        )
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    console.print(report.summary())
    if out:
        winning = report.best_candidate.system_prompt or target.system_prompt
        Path(out).write_text(str(winning))
        console.print(f"[green]Wrote winning prompt → {out}[/green]")
