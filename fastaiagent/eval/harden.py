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

if TYPE_CHECKING:
    from fastaiagent.agent.agent import Agent
    from fastaiagent.llm.client import LLMClient

_VALID_TARGETS = ("instructions", "model", "tools", "guardrails", "memory")


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


def _failures_text(results: Any) -> tuple[str, int]:
    """Render failing cases from SimulationResults or EvalResults → (text, count)."""
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
            failed = [n for n, d in (c.per_scorer or {}).items() if not d.get("passed", True)]
            if not failed:
                continue
            count += 1
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
                if not d.get("passed", True) and d.get("reason")
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
                if not getattr(r, "passed", True):
                    count += 1
                    blocks.append(f"scorer '{name}' failed: {getattr(r, 'reason', '')}")
        return "\n".join(blocks), count

    return str(results), 0


async def aharden(
    agent: Agent | Any,
    results: Any,
    *,
    llm: LLMClient | None = None,
    max_recommendations: int = 6,
) -> HardeningReport:
    """Analyse failures and recommend concrete fixes (async). See :func:`harden`."""
    agent_name = getattr(agent, "name", "agent")
    failures, count = _failures_text(results)

    if count == 0:
        return HardeningReport(agent_name=agent_name, failure_count=0, recommendations=[])

    from fastaiagent.llm import LLMClient, SystemMessage, UserMessage

    client = llm or LLMClient()
    sp = getattr(agent, "system_prompt", None)
    sp_text = sp if isinstance(sp, str) and sp else "(dynamic or no system prompt)"
    tools = [getattr(t, "name", "tool") for t in (getattr(agent, "tools", []) or [])]
    guards = [getattr(g, "name", "guardrail") for g in (getattr(agent, "guardrails", []) or [])]

    prompt = (
        f"An AI agent named '{agent_name}' failed some tests. Recommend specific, actionable "
        "changes that would make the failing cases pass.\n\n"
        f"Current system prompt:\n{sp_text}\n\n"
        f"Current tools: {tools or '(none)'}\n"
        f"Current guardrails: {guards or '(none)'}\n\n"
        f"Failing cases:\n{failures}\n\n"
        f"Give up to {max_recommendations} recommendations. Each targets exactly one of: "
        f"{', '.join(_VALID_TARGETS)}.\n"
        'Respond with JSON only: {"recommendations": [{"target": "instructions|model|tools|'
        'guardrails|memory", "recommendation": "<what to change>", "rationale": "<why>"}]}'
    )
    try:
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
        )

    recs: list[Recommendation] = []
    for item in data.get("recommendations", [])[:max_recommendations]:
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
    return HardeningReport(agent_name=agent_name, failure_count=count, recommendations=recs)


def harden(
    agent: Agent | Any,
    results: Any,
    *,
    llm: LLMClient | None = None,
    max_recommendations: int = 6,
) -> HardeningReport:
    """Analyse failures from a ``simulate()`` / ``evaluate()`` run and recommend fixes.

    Args:
        agent: The agent under test (introspected for system prompt / tools / guardrails).
        results: A :class:`~fastaiagent.eval.simulate.SimulationResults` or
            :class:`~fastaiagent.eval.results.EvalResults` from a prior run.
        llm: Optional ``LLMClient`` for the analysis (default constructed if omitted).
        max_recommendations: Cap on the number of recommendations.

    Returns:
        A :class:`HardeningReport`. **v1 is recommend-only** — it never mutates the agent.

    Example::

        results = simulate(scenarios, agent)
        report = harden(agent, results)
        print(report.summary())
    """
    return run_sync(aharden(agent, results, llm=llm, max_recommendations=max_recommendations))


__all__ = ["harden", "aharden", "HardeningReport", "Recommendation"]
