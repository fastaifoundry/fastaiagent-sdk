"""Can this agent run on a cheaper model — or the one you're forced to move to?

A prompt is tuned to the model it was written for; move it to another model and
it usually gets worse. So don't move the prompt — re-tune it. For each model,
this runs AutoLLM on the same labelled dataset with the same seed, so every
model faces the **same holdout**, and reports:

    model         holdout with the live prompt → after AutoLLM    $ per 1k tickets

The cost is measured, not estimated: it is the spend of the traces behind each
model's holdout eval. Each model's winner is registered as its own
``ticket-triage`` version (``metadata.model``), so a model-specific prompt is
just another version — load it with ``load_agent(version=…)``.

    python switch_models.py                                   # 3 OpenAI sizes
    python switch_models.py --models gpt-4.1-nano --ollama llama3.1:8b
    python switch_models.py --models gpt-4o-2024-05-13,gpt-5-mini   # a forced migration

Run after step 3 (it needs the labelled dataset). ~5 minutes per OpenAI model; a
local model is far slower (an 8B model on a laptop: ~13 s a call, hours a re-tune).

UI: AutoLLM (one run per model) · Prompts → ticket-triage (one version per model).
"""

from __future__ import annotations

import argparse
import sqlite3
from typing import Any

from triage import (
    PROMPT,
    PROPOSER_MODEL,
    TriageMatch,
    current_version,
    dataset_path,
    load_agent,
)

import fastaiagent as fa
from fastaiagent._internal.config import get_config
from fastaiagent.eval import Dataset
from fastaiagent.eval.compare import load_run
from fastaiagent.prompt import PromptRegistry


def cost_per_1k(eval_run_id: str | None) -> float | None:
    """Spend per 1,000 tickets, from the llm spans of one eval run's traces."""
    if not eval_run_id:
        return None
    trace_ids = [c["trace_id"] for c in load_run(eval_run_id)["cases"] if c.get("trace_id")]
    if not trace_ids:
        return None
    marks = ",".join("?" * len(trace_ids))
    with sqlite3.connect(get_config().local_db_path) as db:
        (total,) = db.execute(
            f"""SELECT SUM(json_extract(attributes, '$."fastaiagent.cost.total_usd"'))
                FROM spans WHERE trace_id IN ({marks})""",
            trace_ids,
        ).fetchone()
    return None if total is None else total / len(trace_ids) * 1000


def tune(model: str, provider: str, dataset: Dataset) -> dict[str, Any]:
    agent = load_agent(alias="production")
    agent.llm = fa.LLMClient(provider=provider, model=model)
    report = fa.optimize(
        agent,
        dataset,
        [TriageMatch()],
        config=fa.OptimizeConfig(
            levers=("instructions", "fewshot"),
            max_iterations=6,
            patience=2,
            seed=0,  # the same split for every model
            max_eval_runs=40,
        ),
        proposer_llm=fa.LLMClient(provider="openai", model=PROPOSER_MODEL),
        run_name=f"ticket-triage on {model}",
    )
    before, after = report.holdout_baseline, report.holdout_best
    row: dict[str, Any] = {
        "model": model,
        "before": before.score if before else None,
        "after": after.score if after else None,
        "improved": report.improved,
        "cost": cost_per_1k(after.eval_run_id if after else None),
        "version": None,
    }
    if report.improved:
        winner = report.best_candidate
        base = current_version()
        row["version"] = (
            PromptRegistry()
            .register(
                PROMPT,
                winner.system_prompt or base.template,
                metadata={
                    "source": "autollm",
                    "model": model,
                    "optimize_run_id": report.run_id,
                    "from_version": base.version,
                    "holdout": {"baseline": row["before"], "best": row["after"]},
                    "fewshot_demos": winner.fewshot_demos or base.metadata.get("fewshot_demos"),
                },
            )
            .version
        )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", default="gpt-4.1,gpt-4.1-mini,gpt-4.1-nano")
    parser.add_argument("--ollama", help="also tune a local Ollama model, e.g. llama3.1:8b")
    args = parser.parse_args()

    targets = [("openai", m.strip()) for m in args.models.split(",") if m.strip()]
    if args.ollama:
        targets.append(("ollama", args.ollama))

    dataset = Dataset.from_jsonl(dataset_path())
    print(f"live prompt: {PROMPT} v{current_version().version} · {len(dataset)} labelled tickets\n")

    rows = []
    for provider, model in targets:
        print(f"── tuning for {model} ({provider}) …")
        rows.append(tune(model, provider, dataset))

    def fmt(x: float | None, spec: str) -> str:
        return "—" if x is None else format(x, spec)

    print(f"\n{'model':<22} {'holdout: live prompt → tuned':>30} {'$ / 1k tickets':>16}  version")
    for r in rows:
        moved = f"{fmt(r['before'], '.3f')} → {fmt(r['after'], '.3f')}"
        kept = f"v{r['version']}" if r["version"] else "kept the live prompt"
        print(f"{r['model']:<22} {moved:>30} {fmt(r['cost'], '>16.4f'):>16}  {kept}")


if __name__ == "__main__":
    main()
