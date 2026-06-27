"""The eval-driven optimization loop — greedy coordinate ascent.

P1 moves the system prompt; P2 adds the few-shot lever and cycles
``instructions → fewshot`` one lever per round.

CONTRACT 1: the loop body calls ``score_candidate(...)`` only — never
``aevaluate`` directly — so Replay-grounded scoring drops in later as a second
implementation of the same seam with no driver changes. (``bootstrap_demos`` is a
proposal-time teacher step, not the selection seam.)
"""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING, Any

from fastaiagent._internal.async_utils import run_sync
from fastaiagent.optimize.candidate import (
    Candidate,
    CandidateScore,
    _clone_memory_blocks,
    _resolve_memory_scope,
    apply_candidate,
    scorer_present,
)
from fastaiagent.optimize.config import OptimizeConfig
from fastaiagent.optimize.proposers import (
    bootstrap_demos,
    propose_fact_subsets,
    propose_prompt_rewrites,
)
from fastaiagent.optimize.report import OptimizationReport, TrajectoryPoint

if TYPE_CHECKING:
    from fastaiagent.agent.agent import Agent
    from fastaiagent.eval.dataset import Dataset
    from fastaiagent.eval.scorer import Scorer
    from fastaiagent.llm.client import LLMClient

logger = logging.getLogger(__name__)

# Demo-set sizes the few-shot lever tries as candidate variants per round.
_FEWSHOT_KS = (2, 4, 8)


def _to_items(dataset: Dataset | str | list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Coerce the accepted dataset forms into a plain list of case dicts."""
    from pathlib import Path

    from fastaiagent.eval.dataset import Dataset

    if isinstance(dataset, Dataset):
        return list(dataset)
    if isinstance(dataset, list):
        return list(dataset)
    if isinstance(dataset, str):
        p = Path(dataset)
        ds = Dataset.from_csv(p) if p.suffix == ".csv" else Dataset.from_jsonl(p)
        return list(ds)
    raise TypeError(f"unsupported dataset type: {type(dataset)!r}")


def _split(
    items: list[dict[str, Any]], splits: tuple[float, float, float], seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Seeded, deterministic train/dev/holdout partition.

    Guarantees at least one case in each split when ``len(items) >= 3``.
    """
    n = len(items)
    idx = list(range(n))
    random.Random(seed).shuffle(idx)

    n_train = int(round(splits[0] * n))
    n_train = max(1, min(n_train, n - 2)) if n >= 3 else max(1, n_train)
    rest = n - n_train
    denom = splits[1] + splits[2]
    n_dev = int(round(rest * (splits[1] / denom))) if denom > 0 else 0
    n_dev = max(1, min(n_dev, rest - 1)) if rest >= 2 else rest

    train = [items[i] for i in idx[:n_train]]
    dev = [items[i] for i in idx[n_train : n_train + n_dev]]
    holdout = [items[i] for i in idx[n_train + n_dev :]]
    return train, dev, holdout


def _candidate_for(
    best: Candidate,
    *,
    system_prompt: str | None = None,
    fewshot_demos: list[dict[str, Any]] | None = None,
    fact_ids: list[int] | None = None,
    origin: str,
    rationale: str,
) -> Candidate:
    """A new candidate = current best with one lever overridden (coordinate ascent).

    Carries best's other levers so each step builds on the accepted state.
    """
    return Candidate(
        system_prompt=system_prompt if system_prompt is not None else best.system_prompt,
        fewshot_demos=fewshot_demos if fewshot_demos is not None else best.fewshot_demos,
        fact_ids=fact_ids if fact_ids is not None else best.fact_ids,
        parent_id=best.id,
        origin=origin,
        rationale=rationale,
    )


async def aoptimize(
    agent: Agent,
    dataset: Dataset | str | list[dict[str, Any]],
    scorers: list[Scorer | str] | None = None,
    *,
    config: OptimizeConfig | None = None,
    proposer_llm: LLMClient | None = None,
    run_name: str | None = None,
    persist: bool = True,
) -> OptimizationReport:
    """Optimize an agent against ``dataset`` (async).

    Greedy coordinate ascent over the configured levers (``instructions`` and/or
    ``fewshot``), one lever per round: propose candidates on top of the current
    best, keep the best on dev when it beats the current best by ``min_delta``,
    stop on patience/budget/target, then a holdout guard reverts the winner if it
    regressed on selection-blind data.
    """
    cfg = config or OptimizeConfig()
    base_scorers: list[Any] = list(scorers or [])
    selection_judge = cfg.selection_judge
    audit_judge = cfg.resolve_audit_judge()

    # The instructions lever rewrites a static string prompt only.
    if "instructions" in cfg.levers and callable(agent.system_prompt):
        raise ValueError(
            "optimize: the agent's system_prompt is callable (dynamic); the "
            "instructions lever can only rewrite a static string prompt. Drop "
            "'instructions' from levers or pass a string system_prompt."
        )

    # Fail fast if memory can't be isolated per candidate (e.g. VectorBlock
    # without allow_writable_memory) — raises MemoryIsolationError up front rather
    # than mid-run. (Replaces P1's blanket writable-memory warning; memory-bearing
    # agents are now supported via block.isolated_copy().)
    _clone_memory_blocks(agent.memory, allow_writable_memory=cfg.allow_writable_memory)

    items = _to_items(dataset)
    if len(items) < 3:
        raise ValueError(
            f"optimize needs at least 3 cases to form train/dev/holdout splits; got {len(items)}."
        )
    if len(items) < 15:
        import warnings

        warnings.warn(
            f"optimize: only {len(items)} cases — splits will be small and scores noisy. "
            "~15+ cases recommended (or just run harden() once).",
            stacklevel=2,
        )
    train, dev, holdout = _split(items, cfg.splits, cfg.seed)
    if not dev or not holdout:
        raise ValueError(
            f"splits {cfg.splits} on {len(items)} cases left dev or holdout empty; "
            "use more cases or different split fractions."
        )
    splits_by_name = {"train": train, "dev": dev, "holdout": holdout}

    from fastaiagent.llm import LLMClient

    proposer = proposer_llm or LLMClient()

    # Memory lever: resolve the fact scope once (used by the lever + the skip check).
    mem_scope, mem_scope_id = _resolve_memory_scope(agent) if "memory" in cfg.levers else ("", "")

    eval_runs = 0
    judge_calls = 0

    async def score_candidate(
        candidate: Candidate, split: str, *, judge: Scorer | None
    ) -> CandidateScore:
        # CONTRACT 1: the loop calls only this; CONTRACT 3: judge composed + deduped.
        nonlocal eval_runs, judge_calls
        from fastaiagent.eval.dataset import Dataset
        from fastaiagent.eval.evaluate import aevaluate

        split_items = splits_by_name[split]
        run_scorers = list(base_scorers)
        if judge is not None and not scorer_present(run_scorers, judge):
            run_scorers.append(judge)
        if not run_scorers:
            run_scorers = ["exact_match"]
        cand_agent = apply_candidate(
            agent, candidate, allow_writable_memory=cfg.allow_writable_memory
        )
        results = await aevaluate(
            cand_agent.arun,
            Dataset.from_list(split_items),
            run_scorers,
            persist=persist,
            run_name=f"{run_name or 'optimize'}:{candidate.id[:8]}:{split}",
            agent_name=agent.name,
        )
        eval_runs += 1
        if judge is not None:
            judge_calls += len(split_items)
        return CandidateScore.from_eval(
            candidate.id, split, results, primary_metric=cfg.primary_metric
        )

    def _budget_exhausted() -> bool:
        if cfg.max_eval_runs is not None and eval_runs >= cfg.max_eval_runs:
            return True
        if cfg.max_judge_calls is not None and judge_calls >= cfg.max_judge_calls:
            return True
        return False

    async def _propose(lever: str, best: Candidate, best_train: CandidateScore) -> list[Candidate]:
        """Build candidate variants for the active lever, on top of ``best``."""
        if lever == "instructions":
            effective = (
                best.system_prompt if best.system_prompt is not None else agent.system_prompt
            )
            # Callable prompts are refused up front when the instructions lever is
            # active, so ``effective`` is always a concrete string here.
            assert isinstance(effective, str)
            rewrites = await propose_prompt_rewrites(
                current_prompt=effective,
                results=best_train.results,
                llm=proposer,
                n=cfg.candidates_per_iteration,
                agent_name=agent.name,
            )
            return [
                _candidate_for(best, system_prompt=p, origin="prompt:rewrite", rationale=r)
                for p, r in rewrites
            ]
        if lever == "fewshot":
            ks = sorted({k for k in _FEWSHOT_KS})[: cfg.candidates_per_iteration]
            best_agent = apply_candidate(
                agent, best, allow_writable_memory=cfg.allow_writable_memory
            )
            pool = await bootstrap_demos(
                agent=best_agent,
                train_items=train,
                scorers=base_scorers,
                judge=selection_judge,
                k=max(ks),
            )
            cands: list[Candidate] = []
            seen_sizes: set[int] = set()
            for k in ks:
                kk = min(k, len(pool))
                if kk == 0 or kk in seen_sizes:
                    continue
                seen_sizes.add(kk)
                cands.append(
                    _candidate_for(
                        best,
                        fewshot_demos=pool[:kk],
                        origin="fewshot:bootstrap",
                        rationale=f"few-shot k={kk}",
                    )
                )
            return cands
        if lever == "memory":
            subsets = propose_fact_subsets(
                scope=mem_scope,
                scope_id=mem_scope_id,
                n=cfg.candidates_per_iteration,
            )
            return [
                _candidate_for(
                    best, fact_ids=ids, origin="memory:subset", rationale=f"facts k={len(ids)}"
                )
                for ids in subsets
            ]
        return []

    # ── Baseline ──────────────────────────────────────────────────────────────
    # NOTE (deviation from spec §4): baseline-on-dev uses the SELECTION judge so the
    # dev deltas driving accept/reject are computed against a single judge; the audit
    # judge is used for the holdout guard. Identical when selection == audit (default).
    base_candidate = Candidate(origin="baseline")
    baseline_dev = await score_candidate(base_candidate, "dev", judge=selection_judge)
    best = base_candidate
    best_dev = baseline_dev
    trajectory = [
        TrajectoryPoint(0, "baseline", base_candidate.id, baseline_dev.score, True, "baseline")
    ]
    accepted: list[str] = []

    # Train results for the current best feed the instructions proposer; recomputed
    # whenever best changes (any accepted lever).
    best_train = await score_candidate(best, "train", judge=selection_judge)
    train_scored_for = best.id

    active_levers = list(cfg.levers)

    # Memory lever needs facts at the resolved scope; with none, skip it (don't
    # error, don't burn patience) and record the skip distinctly from a reject.
    if "memory" in active_levers:
        from fastaiagent.learn.store import MemoryStore

        if not MemoryStore().list_active(mem_scope, mem_scope_id):  # type: ignore[arg-type]
            active_levers = [lv for lv in active_levers if lv != "memory"]
            trajectory.append(
                TrajectoryPoint(
                    0,
                    "memory",
                    "",
                    baseline_dev.score,
                    accepted=False,
                    rationale=(
                        f"no learned facts at scope={mem_scope}:{mem_scope_id} — "
                        "run `fastaiagent learn` first"
                    ),
                    skipped=True,
                )
            )
            logger.info(
                "optimize: memory lever skipped — no learned facts at %s:%s",
                mem_scope,
                mem_scope_id,
            )

    no_improve = 0
    stopped_reason = ""
    iteration = 0

    while active_levers and iteration < cfg.max_iterations:
        iteration += 1
        if _budget_exhausted():
            stopped_reason = "budget"
            break

        lever = active_levers[(iteration - 1) % len(active_levers)]

        if lever == "instructions" and best.id != train_scored_for:
            best_train = await score_candidate(best, "train", judge=selection_judge)
            train_scored_for = best.id

        candidates = await _propose(lever, best, best_train)
        if not candidates:
            no_improve += 1
            if no_improve >= cfg.patience:
                stopped_reason = "patience"
                break
            continue

        scored: list[tuple[Candidate, CandidateScore]] = []
        for cand in candidates:
            if _budget_exhausted():
                break
            cs = await score_candidate(cand, "dev", judge=selection_judge)
            scored.append((cand, cs))
            trajectory.append(
                TrajectoryPoint(iteration, lever, cand.id, cs.score, False, cand.rationale)
            )

        if not scored:
            stopped_reason = "budget"
            break

        local_cand, local_cs = max(scored, key=lambda t: t[1].score)
        if local_cs.score - best_dev.score >= cfg.min_delta:
            best = local_cand
            best_dev = local_cs
            accepted.append(best.id)
            for tp in trajectory:
                if tp.candidate_id == local_cand.id:
                    tp.accepted = True
            no_improve = 0
        else:
            no_improve += 1

        if cfg.target_score is not None and best_dev.score >= cfg.target_score:
            stopped_reason = "target_score"
            break
        if no_improve >= cfg.patience:
            stopped_reason = "patience"
            break

    if not stopped_reason:
        stopped_reason = "max_iterations" if active_levers else "no_active_levers"

    # ── Holdout regression guard (selection-blind, audit judge) ─────────────────
    holdout_baseline = await score_candidate(base_candidate, "holdout", judge=audit_judge)
    reverted = False
    if best.id != base_candidate.id:
        holdout_best = await score_candidate(best, "holdout", judge=audit_judge)
        if holdout_best.score < holdout_baseline.score - cfg.holdout_regression_tol:
            reverted = True
            best = base_candidate
            best_dev = baseline_dev
    else:
        holdout_best = holdout_baseline

    return OptimizationReport(
        agent_name=agent.name,
        baseline=baseline_dev,
        best=best_dev,
        best_candidate=best,
        trajectory=trajectory,
        accepted=accepted,
        stopped_reason=stopped_reason + ("+reverted" if reverted else ""),
        holdout_baseline=holdout_baseline,
        holdout_best=holdout_best,
        reverted=reverted,
    )


def optimize(
    agent: Agent,
    dataset: Dataset | str | list[dict[str, Any]],
    scorers: list[Scorer | str] | None = None,
    *,
    config: OptimizeConfig | None = None,
    proposer_llm: LLMClient | None = None,
    run_name: str | None = None,
    persist: bool = True,
) -> OptimizationReport:
    """Synchronous wrapper around :func:`aoptimize` (for CLI / notebooks)."""
    return run_sync(
        aoptimize(
            agent,
            dataset,
            scorers,
            config=config,
            proposer_llm=proposer_llm,
            run_name=run_name,
            persist=persist,
        )
    )
