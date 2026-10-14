"""Agent hardening — turn eval / simulation failures into concrete fixes.

Reads the failures from a :class:`~fastaiagent.eval.simulate.SimulationResults`
or :class:`~fastaiagent.eval.results.EvalResults` run, inspects the agent's
configuration (system prompt, tools, guardrails), and asks an LLM to recommend
specific, actionable changes — to the instructions, model, tools, guardrails, or
memory — that would make the failing cases pass.

**v1 is recommend-only**: it never mutates the agent. Apply the recommendations
yourself, then re-run ``simulate()`` / ``evaluate()`` to confirm.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from fastaiagent._internal.async_utils import run_sync
from fastaiagent.eval.scorer import verdict_outcome

if TYPE_CHECKING:
    from fastaiagent.agent.agent import Agent
    from fastaiagent.llm.client import LLMClient

_VALID_TARGETS = ("instructions", "model", "tools", "guardrails", "memory")

# How many failing cases harden() and the optimize proposer show their model. An
# eval case renders to ~1.2k characters at most; unbounded, 2,000 failing cases
# made a ~335k-token request that no model accepts.
MAX_FAILURES_SHOWN = 40


@dataclass
class Recommendation:
    """One concrete, actionable change for the agent."""

    target: str  # instructions | model | tools | guardrails | memory
    recommendation: str
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "recommendation": self.recommendation,
            "rationale": self.rationale,
        }


@dataclass
class HardeningReport:
    """Structured output of :func:`harden` — fixes for the failing cases."""

    agent_name: str
    failure_count: int
    recommendations: list[Recommendation] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Hardening Report — {self.agent_name} ({self.failure_count} failing case(s))",
            "=" * 60,
        ]
        if not self.recommendations:
            lines.append("No recommendations (no failures, or the analysis returned nothing).")
        for i, r in enumerate(self.recommendations, 1):
            lines.append(f"{i}. [{r.target}] {r.recommendation}")
            if r.rationale:
                lines.append(f"     ↳ {r.rationale}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "failure_count": self.failure_count,
            "recommendations": [r.to_dict() for r in self.recommendations],
        }


def _failures_text(results: Any, *, max_cases: int | None = None) -> tuple[str, int]:
    """Render failing cases from SimulationResults or EvalResults → (text, count).

    ``max_cases`` caps how many failing cases or scenarios are rendered; ``count``
    is still the total. Unbounded, a large run builds a prompt no model can take.
    """
    # SimulationResults — has .results of items with .scenario_name / .verdicts.
    sim = getattr(results, "results", None)
    if sim and hasattr(sim[0], "scenario_name"):
        from fastaiagent.eval.simulate import _format_transcript

        blocks: list[str] = []
        count = 0
        for r in sim:
            if getattr(r, "passed", True):
                continue
            count += 1
            if max_cases is not None and len(blocks) >= max_cases:
                continue
            failed = [v.criterion for v in r.verdicts if not v.passed]
            convo = _format_transcript(r.transcript)
            blocks.append(
                f"Scenario '{r.scenario_name}' FAILED. Unmet/violated criteria: {failed}\n"
                f"Transcript:\n{convo}"
            )
        return "\n\n".join(blocks), count

    # EvalResults — prefer per-case detail, fall back to per-scorer aggregate.
    cases = getattr(results, "cases", None)
    if cases:
        blocks = []
        count = 0
        for c in cases:
            # An infrastructure-errored case is not an agent-quality failure — never
            # feed it to the proposer (it would chase a fault the agent can't fix).
            if getattr(c, "error", None):
                continue
            # Only real fails (1.89.0): an errored or skipped verdict is not a
            # quality failure for the proposer to chase.
            failed = [n for n, d in (c.per_scorer or {}).items() if verdict_outcome(d) == "failed"]
            if not failed:
                continue
            count += 1
            if max_cases is not None and len(blocks) >= max_cases:
                continue
            # Show the proposer/harden LLM what "correct" looks like, not just that
            # the case failed. Without the expected output — and the scorer's own
            # reason (e.g. "got 1120, expected 1120000") — a proposer can't recover
            # an output convention (scale, sign, format) it can only see by contrast.
            # Classification hides this gap (the label space is tiny); extraction and
            # structured-output tasks depend on it.
            parts = [f"Case input={str(c.input)[:600]!r}"]
            if getattr(c, "expected_output", None) is not None:
                parts.append(f"expected={str(c.expected_output)[:300]!r}")
            parts.append(f"output={str(c.actual_output)[:300]!r}")
            reasons = [
                d["reason"]
                for n, d in (c.per_scorer or {}).items()
                if verdict_outcome(d) == "failed" and d.get("reason")
            ]
            tail = f"; failed scorers: {failed}"
            if reasons:
                tail += f"; reasons: {reasons}"
            blocks.append("; ".join(parts) + tail)
        if blocks:
            return "\n".join(blocks), count

    scores = getattr(results, "scores", None)
    if scores:
        blocks = []
        count = 0
        for name, rlist in scores.items():
            for r in rlist:
                if verdict_outcome(r) == "failed":
                    count += 1
                    if max_cases is None or len(blocks) < max_cases:
                        blocks.append(f"scorer '{name}' failed: {getattr(r, 'reason', '')}")
        return "\n".join(blocks), count

    return str(results), 0


async def aharden(
    agent: Agent | Any,
    results: Any,
    *,
    llm: LLMClient | None = None,
    max_recommendations: int = 6,
    persist: bool = True,
) -> HardeningReport:
    """Analyse failures and recommend concrete fixes (async). See :func:`harden`."""
    report, model = await _analyse(agent, results, llm=llm, max_recommendations=max_recommendations)
    if persist:
        _persist_report(results, report, model)
    return report


def _persist_report(results: Any, report: HardeningReport, model: str | None) -> None:
    """Merge the report into the analysed run's ``metadata["hardening"]`` (D16).

    Only for a run that was persisted (``results.run_id``), into the database
    it was written to. Local only: the eval export never reads this key.
    Best-effort: a write failure never loses the report the caller holds.
    """
    run_id = getattr(results, "run_id", None)
    if not run_id:
        return
    from datetime import datetime, timezone
    from pathlib import Path

    from fastaiagent.eval.simulate import SimulationResults

    table = "sim_runs" if isinstance(results, SimulationResults) else "eval_runs"
    try:
        from fastaiagent._internal.config import get_config
        from fastaiagent._internal.scrub import known_credential_values, scrub_secrets
        from fastaiagent.ui.db import init_local_db

        path = getattr(results, "db_path", None) or get_config().local_db_path
        db = init_local_db(Path(path))
        try:
            row = db.fetchone(f"SELECT metadata FROM {table} WHERE run_id = ?", (run_id,))
            if row is None:
                return
            try:
                meta = json.loads(row["metadata"] or "{}")
            except json.JSONDecodeError:
                meta = {}
            secrets = known_credential_values()
            meta["hardening"] = {
                "created_at": datetime.now(tz=timezone.utc).isoformat(),
                "model": model,
                "failure_count": report.failure_count,
                "recommendations": [
                    {
                        "target": r.target,
                        "recommendation": scrub_secrets(r.recommendation, secrets),
                        "rationale": scrub_secrets(r.rationale, secrets),
                    }
                    for r in report.recommendations
                ],
            }
            db.execute(
                f"UPDATE {table} SET metadata = ? WHERE run_id = ?",
                (json.dumps(meta, default=str), run_id),
            )
        finally:
            db.close()
    except Exception:  # pragma: no cover - never lose the report over a write
        import logging

        logging.getLogger(__name__).debug("Could not persist the hardening report", exc_info=True)


async def _analyse(
    agent: Agent | Any,
    results: Any,
    *,
    llm: LLMClient | None,
    max_recommendations: int,
) -> tuple[HardeningReport, str | None]:
    agent_name = getattr(agent, "name", "agent")
    failures, count = _failures_text(results, max_cases=MAX_FAILURES_SHOWN)

    if count == 0:
        return HardeningReport(agent_name=agent_name, failure_count=0, recommendations=[]), None

    from fastaiagent.llm import LLMClient, SystemMessage, UserMessage

    client = llm or LLMClient()
    model = getattr(client, "model", None)
    sp = getattr(agent, "system_prompt", None)
    sp_text = sp if isinstance(sp, str) and sp else "(dynamic or no system prompt)"
    tools = [getattr(t, "name", "tool") for t in (getattr(agent, "tools", []) or [])]
    guards = [getattr(g, "name", "guardrail") for g in (getattr(agent, "guardrails", []) or [])]
    shown = min(count, MAX_FAILURES_SHOWN)
    heading = "Failing cases" if shown == count else f"Failing cases ({shown} of {count})"

    prompt = (
        f"An AI agent named '{agent_name}' failed some tests. Recommend specific, actionable "
        "changes that would make the failing cases pass.\n\n"
        f"Current system prompt:\n{sp_text}\n\n"
        f"Current tools: {tools or '(none)'}\n"
        f"Current guardrails: {guards or '(none)'}\n\n"
        f"{heading}:\n{failures}\n\n"
        f"Give up to {max_recommendations} recommendations. Each targets exactly one of: "
        f"{', '.join(_VALID_TARGETS)}.\n"
        'Respond with JSON only: {"recommendations": [{"target": "instructions|model|tools|'
        'guardrails|memory", "recommendation": "<what to change>", "rationale": "<why>"}]}'
    )
    try:
        # One root eval.harden span, stamped source=eval (D16): the analysis
        # call is an experiment, not a run of the agent.
        with _harden_span(results):
            resp = await client.acomplete(
                [
                    SystemMessage("You are an expert AI-agent debugger. Respond with JSON only."),
                    UserMessage(prompt),
                ]
            )
        raw = (resp.content or "").strip()
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw).strip()
        data = json.loads(raw)
    except Exception as e:
        return HardeningReport(
            agent_name=agent_name,
            failure_count=count,
            recommendations=[
                Recommendation(
                    target="instructions", recommendation="(analysis failed)", rationale=str(e)
                )
            ],
        ), model

    # The requested {"recommendations": [...]} object, or the bare list some models
    # send instead — reading .get() off a list raised out of harden() (1.86.0).
    items = data.get("recommendations") if isinstance(data, dict) else data
    if not isinstance(items, list):
        return HardeningReport(
            agent_name=agent_name,
            failure_count=count,
            recommendations=[
                Recommendation(
                    target="instructions",
                    recommendation="(analysis failed)",
                    rationale="the reply had no list of recommendations",
                )
            ],
        ), model
    recs: list[Recommendation] = []
    for item in items[:max_recommendations]:
        if not isinstance(item, dict):
            continue
        target = str(item.get("target", "instructions")).strip().lower()
        if target not in _VALID_TARGETS:
            target = "instructions"
        rec = str(item.get("recommendation", "")).strip()
        if not rec:
            continue
        recs.append(
            Recommendation(
                target=target, recommendation=rec, rationale=str(item.get("rationale", ""))
            )
        )
    return HardeningReport(agent_name=agent_name, failure_count=count, recommendations=recs), model


def _harden_span(results: Any) -> Any:
    import contextlib

    try:
        from opentelemetry import context as otel_context

        from fastaiagent.trace.otel import get_tracer
        from fastaiagent.trace.provenance import provenance_scope

        stack = contextlib.ExitStack()
        run_id = getattr(results, "run_id", None)
        attrs = {"fastaiagent.eval.run_id": run_id} if run_id else {}
        stack.enter_context(provenance_scope("eval", attributes=attrs))
        stack.enter_context(
            get_tracer("fastaiagent.eval").start_as_current_span(
                "eval.harden", context=otel_context.Context()
            )
        )
        return stack
    except Exception:  # pragma: no cover - tracing is bookkeeping
        return contextlib.nullcontext()


def harden(
    agent: Agent | Any,
    results: Any,
    *,
    llm: LLMClient | None = None,
    max_recommendations: int = 6,
    persist: bool = True,
) -> HardeningReport:
    """Analyse failures from a ``simulate()`` / ``evaluate()`` run and recommend fixes.

    Args:
        agent: The agent under test (introspected for system prompt / tools / guardrails).
        results: A :class:`~fastaiagent.eval.simulate.SimulationResults` or
            :class:`~fastaiagent.eval.results.EvalResults` from a prior run.
        llm: Optional ``LLMClient`` for the analysis (default constructed if omitted).
        max_recommendations: Cap on the number of recommendations.
        persist: When ``results`` was persisted (it has a ``run_id``), save the
            report onto that run as ``metadata["hardening"]`` so the Local UI
            shows it (1.89.0). Local only; never exported.

    Returns:
        A :class:`HardeningReport`. **v1 is recommend-only** — it never mutates the agent.

    Example::

        results = simulate(scenarios, agent)
        report = harden(agent, results)
        print(report.summary())
    """
    return run_sync(
        aharden(agent, results, llm=llm, max_recommendations=max_recommendations, persist=persist)
    )


__all__ = ["harden", "aharden", "HardeningReport", "Recommendation"]
