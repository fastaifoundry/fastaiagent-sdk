"""Prompt-rewrite proposer for the optimization loop (P1).

Per §H of the build spec, ``harden``/``aharden`` and the entire ``eval`` public
API stay 100% unchanged. The metaprompt that turns failing cases into a complete
revised system prompt lives here, so *optimize* (not *eval*) owns the
generation-for-apply step. We reuse only harden's internal failure-rendering
helper, ``_failures_text`` — a deliberate, documented private import.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

# DOCUMENTED PRIVATE REUSE (build spec §H): _failures_text renders failing cases
# from an EvalResults / SimulationResults into the same text block harden feeds
# its LLM. We depend on this internal rather than duplicating the logic; if
# harden's internals move, THIS import is the seam to update. harden's *public*
# API is untouched.
from fastaiagent.eval.harden import _failures_text

if TYPE_CHECKING:
    from fastaiagent.agent.agent import Agent
    from fastaiagent.eval.results import EvalResults
    from fastaiagent.eval.scorer import Scorer
    from fastaiagent.llm.client import LLMClient

_REWRITE_SYSTEM = (
    "You are an expert prompt engineer improving an AI agent's system prompt. "
    "You are given the current system prompt and cases where the agent failed. "
    "Produce complete, ready-to-use revised system prompts that fix the failures "
    "while preserving the agent's existing correct behavior. Make minimal, "
    "targeted edits; do not drop important instructions and do not pad length. "
    "Respond with JSON only."
)


def _strip_fences(raw: str) -> str:
    """Strip a ```` ```json ```` / ```` ``` ```` fence the same way harden does."""
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip()).strip()


async def propose_prompt_rewrites(
    *,
    current_prompt: str,
    results: EvalResults,
    llm: LLMClient,
    n: int,
    agent_name: str = "agent",
) -> list[tuple[str, str]]:
    """Return up to ``n`` ``(revised_system_prompt, rationale)`` proposals.

    Empty when there are no failing cases to learn from (the prompt already passes
    the split) or when the LLM response can't be parsed — both are non-fatal: the
    loop treats an empty proposal set as "no improvement this round".
    """
    failures, count = _failures_text(results)
    if count == 0:
        return []

    from fastaiagent.llm import SystemMessage, UserMessage

    user = (
        f"Agent name: {agent_name}\n\n"
        f'Current system prompt:\n"""\n{current_prompt}\n"""\n\n'
        f"Failing cases:\n{failures}\n\n"
        f"Propose {n} distinct revised system prompts that would make these cases pass. "
        "Vary the approach across proposals. Respond with JSON only:\n"
        '{"proposals": [{"system_prompt": "<full revised prompt>", '
        '"rationale": "<what changed and why>"}]}'
    )
    try:
        resp = await llm.acomplete([SystemMessage(_REWRITE_SYSTEM), UserMessage(user)])
        data = json.loads(_strip_fences(resp.content or ""))
    except Exception:
        return []

    out: list[tuple[str, str]] = []
    for item in data.get("proposals", [])[:n]:
        if not isinstance(item, dict):
            continue
        prompt = str(item.get("system_prompt", "")).strip()
        if not prompt:
            continue
        out.append((prompt, str(item.get("rationale", "")).strip()))
    return out


async def bootstrap_demos(
    *,
    agent: Agent,
    train_items: list[dict[str, Any]],
    scorers: list[Any],
    judge: Scorer | None,
    k: int,
    include_favorites: bool = True,
) -> list[dict[str, Any]]:
    """Build up to ``k`` few-shot demos from the TRAIN split (P2 KB lever).

    Gold-first: train items with a non-empty ``expected_output`` become free
    ``{"input", "output"}`` demos (no teacher call). Optionally augments with
    ``curate_from_traces(filter="favorites")`` (captured-good production runs). If
    still short of ``k``, teacher-bootstraps the no-gold items — runs the agent and
    metric-filters with the same scorers (+ judge), keeping passing
    ``(input, actual_output)`` pairs (DSPy ``BootstrapFewShot``).

    This is a **proposal-time** step (a teacher), analogous to
    :func:`propose_prompt_rewrites`. It uses ``aevaluate`` as a metric filter
    (spec §5) — it is *not* the selection seam; the loop's accept/reject still
    goes only through ``score_candidate``. Demos come from **train only** — never
    dev/holdout — so there's no test-answer leakage.
    """
    demos: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _add(inp: Any, out: Any) -> None:
        key = str(inp)
        if key in seen or not str(out).strip():
            return
        seen.add(key)
        demos.append({"input": inp, "output": out})

    # 1. gold demos from train (free — no teacher call).
    for it in train_items:
        if str(it.get("expected_output") or "").strip():
            _add(it["input"], it["expected_output"])

    # 2. favorites traces (captured-good outputs for this agent).
    if include_favorites:
        try:
            from fastaiagent.eval.curate import curate_from_traces

            for f in curate_from_traces(filter="favorites", agent=agent.name, limit=50):
                if str(f.get("expected_output") or "").strip():
                    _add(f["input"], f["expected_output"])
        except Exception:
            pass  # best-effort — no favorites is fine

    # 3. teacher-bootstrap the gap from no-gold train items.
    if len(demos) < k:
        no_gold = [it for it in train_items if not str(it.get("expected_output") or "").strip()]
        if no_gold:
            from fastaiagent.eval.dataset import Dataset
            from fastaiagent.eval.evaluate import aevaluate
            from fastaiagent.optimize.candidate import scorer_present

            run_scorers = list(scorers)
            if judge is not None and not scorer_present(run_scorers, judge):
                run_scorers.append(judge)
            if not run_scorers:
                run_scorers = ["exact_match"]
            # Proposal-time only (teacher metric-filter, spec §5) — NOT the
            # selection seam; the loop's accept/reject still goes solely through
            # score_candidate (CONTRACT 1).
            results = await aevaluate(
                agent.arun, Dataset.from_list(no_gold), run_scorers, persist=False
            )
            for case in results.cases:
                per = case.per_scorer or {}
                if per and all(d.get("passed") for d in per.values()):
                    _add(case.input, case.actual_output)

    return demos[:k]


def propose_fact_subsets(
    *,
    scope: str,
    scope_id: str,
    n: int,
    store: Any = None,
) -> list[list[int]]:
    """Propose up to ``n`` candidate fact-id subsets for the memory lever (P3).

    Pure **selection / ablation**: reads ``MemoryStore.list_active(scope, scope_id)``
    and returns subsets of the *existing* fact ids, ranked by confidence then
    recency (full set + progressively smaller high-confidence subsets). It never
    creates/edits/deletes/supersedes facts — the selection is run-local and leaves
    the audit chain untouched. Returns ``[]`` when there are no active facts (the
    caller skips the lever).
    """
    from fastaiagent.learn.store import MemoryStore

    facts = (store or MemoryStore()).list_active(scope, scope_id)  # type: ignore[arg-type]
    if not facts:
        return []
    ranked = sorted(
        facts,
        key=lambda f: (
            f.confidence if f.confidence is not None else 1.0,
            f.created_at or 0.0,
        ),
        reverse=True,
    )
    total = len(ranked)
    subsets: list[list[int]] = []
    seen_sizes: set[int] = set()
    for frac in (1.0, 0.66, 0.33):  # ablate down from "inject all"
        k = max(1, round(frac * total))
        if k in seen_sizes:
            continue
        seen_sizes.add(k)
        subsets.append([f.id for f in ranked[:k]])
        if len(subsets) >= n:
            break
    return subsets
