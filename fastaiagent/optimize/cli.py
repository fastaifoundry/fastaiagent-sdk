"""``fastaiagent optimize`` — eval-driven prompt optimization (P1).

Loads an ``Agent`` from a ``module:attr`` path, runs the optimization loop over a
dataset, prints the trajectory, and optionally writes the winning prompt to a
file. Mirrors the resolver used by ``fastaiagent agent`` / ``fastaiagent mcp``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer
from rich.console import Console

optimize_app = typer.Typer()
console = Console()


def _resolve_target(spec: str) -> Any:
    """Resolve ``path/to/file.py:attr`` or ``pkg.module:attr`` into a live object."""
    from fastaiagent._internal.target import resolve_target

    try:
        return resolve_target(spec)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e


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
    audit_judge: str = typer.Option(
        None,
        "--audit-judge",
        help="A distinct LLM judge (criteria string) for the holdout guard only",
    ),
    levers: str = typer.Option(
        "instructions",
        "--levers",
        help="Comma-separated levers to move: instructions, fewshot, memory",
    ),
    proposer_model: str = typer.Option(
        None, "--proposer-model", help="Model id for the prompt proposer (default: env/config)"
    ),
    out: str = typer.Option(None, "--out", help="Write the winning system prompt to this file"),
    no_persist: bool = typer.Option(
        False, "--no-persist", help="Don't write per-candidate evals to local.db"
    ),
) -> None:
    """Optimize an agent against a dataset: its system prompt, and optionally its
    few-shot examples and learned facts (``--levers``)."""
    if ctx.invoked_subcommand is not None:
        return

    from fastaiagent.eval.llm_judge import LLMJudge
    from fastaiagent.optimize import OptimizeConfig
    from fastaiagent.optimize import optimize as run_optimize

    target = _resolve_target(agent)
    scorer_list: list[Any] = [s.strip() for s in scorers.split(",") if s.strip()]

    proposer_llm = None
    if proposer_model:
        from fastaiagent.llm import LLMClient

        proposer_llm = LLMClient(model=proposer_model)

    try:
        cfg = OptimizeConfig(
            levers=tuple(lv.strip() for lv in levers.split(",") if lv.strip()),
            max_iterations=max_iterations,
            patience=patience,
            candidates_per_iteration=candidates,
            seed=seed,
            primary_metric=primary_metric,
            selection_judge=LLMJudge(criteria=judge) if judge else None,
            audit_judge=LLMJudge(criteria=audit_judge, name="audit_judge") if audit_judge else None,
        )
        report = run_optimize(
            target,
            dataset,
            scorer_list,
            config=cfg,
            proposer_llm=proposer_llm,
            persist=not no_persist,
        )
    except (ValueError, TypeError) as exc:
        console.print(str(exc), style="red", markup=False)
        raise typer.Exit(1) from exc

    # The winning prompt is written before anything is printed, and nothing is
    # printed as Rich markup: the summary holds model-written rationales and
    # "[lever]" tags, which markup ate — or raised on, after the paid run.
    if out:
        winning = report.best_candidate.system_prompt or target.system_prompt
        Path(out).write_text(str(winning))
    console.print(report.summary(), markup=False, highlight=False)
    if out:
        console.print(f"Wrote winning prompt → {out}", style="green", markup=False)
