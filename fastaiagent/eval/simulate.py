"""Agent simulation — multi-turn scenario testing.

A :class:`Scenario` drives a multi-turn conversation between a
:class:`SimulatedUser` (an LLM persona or a fixed script) and the agent under
test, then a judge scores the full transcript against natural-language success
criteria (and optional failure criteria).

This reuses existing primitives rather than inventing new infrastructure:

* the agent under test is a native :class:`~fastaiagent.agent.agent.Agent`
  (auto-wrapped via the additive ``messages=`` param) or any callable adapter
  ``(messages) -> str | AgentResult``;
* the judge is a thin use of :class:`~fastaiagent.eval.llm_judge.LLMJudge`
  (one call per criterion);
* concurrency mirrors :func:`~fastaiagent.eval.evaluate.aevaluate`
  (an ``asyncio.Semaphore`` + ``asyncio.gather``);
* persistence mirrors :class:`~fastaiagent.eval.results.EvalResults`
  (``sim_runs`` / ``sim_cases`` tables, the same local.db).

Relationship to neighbours: **Replay** debugs past traces; **evaluate()**
scores fixed input→output pairs; **simulate()** stress-tests multi-turn
behavior on synthetic conversations.

For deterministic tests, inject ``TestModel`` / ``FunctionModel`` (real
``LLMClient`` subclasses) as the ``llm`` for the agent, the simulated user, and
the judge — the whole multi-turn run then has no network and no mocks.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastaiagent._internal.async_utils import run_sync
from fastaiagent.llm.message import AssistantMessage, Message, UserMessage

if TYPE_CHECKING:
    from fastaiagent.agent.agent import Agent
    from fastaiagent.eval.llm_judge import LLMJudge
    from fastaiagent.llm.client import LLMClient


# An agent adapter receives the running transcript as ``list[Message]`` (prior
# turns + the latest user message last) and returns reply text, an object with
# an ``.output`` attribute (e.g. ``AgentResult``), or an awaitable of either.
AgentAdapter = Callable[[list[Message]], str | Any | Awaitable[Any]]


# --------------------------------------------------------------------------- #
# Public dataclasses
# --------------------------------------------------------------------------- #


@dataclass
class TranscriptTurn:
    """One turn in a simulated conversation."""

    turn_index: int
    role: str  # "user" | "assistant"
    content: str
    trace_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_index": self.turn_index,
            "role": self.role,
            "content": self.content,
            "trace_id": self.trace_id,
        }


@dataclass
class CriterionVerdict:
    """The judge's verdict on a single criterion."""

    criterion: str
    kind: str  # "success" | "failure"
    passed: bool  # True == the desired state holds (criterion met / failure absent)
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "criterion": self.criterion,
            "kind": self.kind,
            "passed": self.passed,
            "reason": self.reason,
        }


class SimulatedUser:
    """Produces the next user message in a simulated conversation.

    Exactly one of ``persona`` or ``script`` must be given:

    * ``persona`` — a natural-language description; an LLM role-plays the user,
      generating each turn from the transcript so far. Reply ``"END"`` (or the
      model deciding to stop) ends the conversation early.
    * ``script`` — a fixed list of user messages, returned one per turn; when
      exhausted, the conversation ends.

    ``llm`` is the model used for the persona path (ignored for scripts).
    """

    def __init__(
        self,
        *,
        persona: str | None = None,
        script: list[str] | None = None,
        llm: LLMClient | None = None,
    ) -> None:
        if (persona is None) == (script is None):
            raise ValueError("SimulatedUser requires exactly one of `persona` or `script`.")
        self.persona = persona
        self.script: list[str] | None = list(script) if script is not None else None
        self._llm = llm
        self._script_idx = 0

    async def anext(self, transcript: list[TranscriptTurn]) -> str | None:
        """Return the next user message, or ``None`` to end the conversation."""
        if self.script is not None:
            if self._script_idx >= len(self.script):
                return None
            msg = self.script[self._script_idx]
            self._script_idx += 1
            return msg
        return await self._generate(transcript)

    async def _generate(self, transcript: list[TranscriptTurn]) -> str | None:
        from fastaiagent.llm import LLMClient, SystemMessage
        from fastaiagent.llm import UserMessage as _UserMessage

        llm = self._llm or LLMClient()
        system = (
            "You are role-playing a human user in a conversation with an AI "
            f"agent. Your persona: {self.persona}\n\n"
            "Write ONLY the user's next message — no narration, no quotes. "
            "If the conversation has reached a natural end or your goal is "
            "satisfied, reply with exactly END."
        )
        convo = _format_transcript(transcript) or "(no messages yet — open the conversation)"
        response = await llm.acomplete(
            [SystemMessage(system), _UserMessage(convo)]
        )
        text = (response.content or "").strip()
        if not text or text.upper() == "END":
            return None
        return text


@dataclass
class Scenario:
    """A multi-turn test case for an agent."""

    name: str
    user: SimulatedUser
    success_criteria: list[str]
    failure_criteria: list[str] = field(default_factory=list)
    max_turns: int = 6  # hard cap on total user + assistant turns


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #


@dataclass
class SimulationResult:
    """The outcome of one scenario run."""

    scenario_name: str
    passed: bool
    transcript: list[TranscriptTurn]
    verdicts: list[CriterionVerdict]
    trace_id: str | None = None

    @property
    def success_criteria(self) -> list[str]:
        return [v.criterion for v in self.verdicts if v.kind == "success"]

    @property
    def failure_criteria(self) -> list[str]:
        return [v.criterion for v in self.verdicts if v.kind == "failure"]


class SimulationResults:
    """Results of a :func:`simulate` run — parallels ``EvalResults``."""

    def __init__(self, results: list[SimulationResult], *, agent_name: str | None = None):
        self.results = results
        self.agent_name = agent_name
        # Populated by ``persist_local()`` so callers can deep-link the UI.
        self.run_id: str | None = None

    @property
    def pass_count(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def fail_count(self) -> int:
        return sum(1 for r in self.results if not r.passed)

    @property
    def pass_rate(self) -> float:
        return (self.pass_count / len(self.results)) if self.results else 0.0

    def summary(self) -> str:
        lines = ["Simulation Results", "=" * 50]
        for r in self.results:
            status = "PASS" if r.passed else "FAIL"
            lines.append(f"[{status}] {r.scenario_name} ({len(r.transcript)} turns)")
            for v in r.verdicts:
                mark = "✓" if v.passed else "✗"
                lines.append(f"    {mark} ({v.kind}) {v.criterion}")
        lines.append("-" * 50)
        lines.append(
            f"{self.pass_count}/{len(self.results)} passed "
            f"(pass_rate={self.pass_rate:.0%})"
        )
        return "\n".join(lines)

    def export(self, path: str | Path, format: str = "json") -> None:
        """Export results to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [
            {
                "scenario_name": r.scenario_name,
                "passed": r.passed,
                "trace_id": r.trace_id,
                "transcript": [t.to_dict() for t in r.transcript],
                "verdicts": [v.to_dict() for v in r.verdicts],
            }
            for r in self.results
        ]
        path.write_text(json.dumps(data, indent=2))

    def persist_local(
        self,
        *,
        db_path: str | Path | None = None,
        run_name: str | None = None,
        agent_name: str | None = None,
    ) -> str:
        """Persist this run to the unified local.db.

        Writes one row to ``sim_runs`` and one per scenario to ``sim_cases``.
        Returns the generated ``run_id`` so callers can correlate / deep-link.
        """
        from fastaiagent._internal.config import get_config
        from fastaiagent._internal.project import safe_get_project_id
        from fastaiagent.ui.db import init_local_db

        resolved = Path(db_path) if db_path is not None else Path(get_config().local_db_path)
        run_id = uuid.uuid4().hex
        timestamp = datetime.now(tz=timezone.utc).isoformat()
        pid = safe_get_project_id()
        agent = agent_name or self.agent_name

        db = init_local_db(resolved)
        try:
            db.execute(
                """INSERT INTO sim_runs
                   (run_id, run_name, agent_name, scenario_count, pass_count,
                    fail_count, pass_rate, started_at, finished_at, metadata,
                    project_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_id,
                    run_name,
                    agent,
                    len(self.results),
                    self.pass_count,
                    self.fail_count,
                    self.pass_rate,
                    timestamp,
                    timestamp,
                    json.dumps({}),
                    pid,
                ),
            )
            for ordinal, r in enumerate(self.results):
                db.execute(
                    """INSERT INTO sim_cases
                       (case_id, run_id, ordinal, scenario_name, passed,
                        criteria, per_criterion, transcript, trace_id, project_id)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        uuid.uuid4().hex,
                        run_id,
                        ordinal,
                        r.scenario_name,
                        1 if r.passed else 0,
                        json.dumps(
                            {
                                "success": r.success_criteria,
                                "failure": r.failure_criteria,
                            }
                        ),
                        json.dumps([v.to_dict() for v in r.verdicts]),
                        json.dumps([t.to_dict() for t in r.transcript]),
                        r.trace_id,
                        pid,
                    ),
                )
        finally:
            db.close()
        self.run_id = run_id
        return run_id


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _format_transcript(transcript: list[TranscriptTurn]) -> str:
    """Render the transcript as readable role-prefixed lines."""
    return "\n".join(f"{t.role}: {t.content}" for t in transcript)


def _transcript_to_messages(transcript: list[TranscriptTurn]) -> list[Message]:
    """Convert transcript turns to LLM ``Message`` objects."""
    out: list[Message] = []
    for t in transcript:
        if t.role == "assistant":
            out.append(AssistantMessage(t.content))
        else:
            out.append(UserMessage(t.content))
    return out


def _make_agent_runner(
    agent: Agent | AgentAdapter,
) -> Callable[[list[TranscriptTurn]], Awaitable[tuple[str, str | None]]]:
    """Return an async runner that, given the transcript, returns
    ``(reply_text, trace_id)`` for the agent's next turn.

    Native ``Agent`` instances use the additive ``messages=`` param (the last
    user turn becomes ``input``, the rest becomes prior ``messages``). Any other
    callable is treated as an adapter receiving the full ``list[Message]``.
    """
    from fastaiagent.agent.agent import Agent

    if isinstance(agent, Agent):

        async def run_native(transcript: list[TranscriptTurn]) -> tuple[str, str | None]:
            last_user = transcript[-1].content
            prior = _transcript_to_messages(transcript[:-1])
            result = await agent.arun(input=last_user, messages=prior)
            return (result.output or ""), getattr(result, "trace_id", None)

        return run_native

    async def run_adapter(transcript: list[TranscriptTurn]) -> tuple[str, str | None]:
        messages = _transcript_to_messages(transcript)
        out: Any = agent(messages)
        if asyncio.iscoroutine(out):
            out = await out
        if hasattr(out, "output"):
            return (out.output or ""), getattr(out, "trace_id", None)
        return str(out), None

    return run_adapter


async def _judge_transcript(
    scenario: Scenario,
    transcript: list[TranscriptTurn],
    judge: LLMJudge | None,
) -> tuple[bool, list[CriterionVerdict]]:
    """Judge the full transcript against success / failure criteria.

    One :class:`LLMJudge` call per criterion (it returns one ``ScorerResult``).
    Overall pass = every success criterion holds AND no failure criterion holds.
    """
    from fastaiagent.eval.llm_judge import LLMJudge

    judge_llm = getattr(judge, "_llm", None) if judge is not None else None
    transcript_text = _format_transcript(transcript)

    verdicts: list[CriterionVerdict] = []
    overall = True

    for criterion in scenario.success_criteria:
        j = LLMJudge(criteria=criterion, llm=judge_llm)
        res = await j.ascore(input=scenario.name, output=transcript_text)
        verdicts.append(
            CriterionVerdict(
                criterion=criterion, kind="success", passed=res.passed, reason=res.reason
            )
        )
        if not res.passed:
            overall = False

    for criterion in scenario.failure_criteria:
        # Phrase so the judge scores "did this bad thing happen?": passed
        # (score>=0.5) means the failure occurred → desired state is the inverse.
        wrapped = (
            f"whether this undesirable condition occurred: {criterion} "
            "(score 1.0 if it clearly occurred, 0.0 if it did not)"
        )
        j = LLMJudge(criteria=wrapped, llm=judge_llm)
        res = await j.ascore(input=scenario.name, output=transcript_text)
        failure_happened = res.passed
        verdicts.append(
            CriterionVerdict(
                criterion=criterion,
                kind="failure",
                passed=not failure_happened,
                reason=res.reason,
            )
        )
        if failure_happened:
            overall = False

    return overall, verdicts


async def _run_scenario(
    scenario: Scenario,
    agent: Agent | AgentAdapter,
    judge: LLMJudge | None,
) -> SimulationResult:
    """Drive one scenario end-to-end and judge the transcript."""
    from fastaiagent.trace import trace_context
    from fastaiagent.trace.span import set_fastaiagent_attributes

    run_agent = _make_agent_runner(agent)

    with trace_context(f"simulation.{scenario.name}") as span:
        set_fastaiagent_attributes(
            span,
            **{
                "simulation.name": scenario.name,
                "simulation.max_turns": scenario.max_turns,
            },
        )

        transcript: list[TranscriptTurn] = []

        # 1. Simulated user opens the conversation.
        opening = await scenario.user.anext(transcript)
        if opening is not None:
            transcript.append(TranscriptTurn(turn_index=0, role="user", content=opening))

        # 2. Alternate agent / user turns until the cap or an early stop.
        while transcript and len(transcript) < scenario.max_turns:
            reply, trace_id = await run_agent(transcript)
            transcript.append(
                TranscriptTurn(
                    turn_index=len(transcript),
                    role="assistant",
                    content=reply,
                    trace_id=trace_id,
                )
            )
            if len(transcript) >= scenario.max_turns:
                break
            nxt = await scenario.user.anext(transcript)
            if nxt is None:
                break
            transcript.append(
                TranscriptTurn(turn_index=len(transcript), role="user", content=nxt)
            )

        # 3. Judge once over the full transcript.
        passed, verdicts = await _judge_transcript(scenario, transcript, judge)

        ctx = span.get_span_context()
        root_trace_id = format(ctx.trace_id, "032x")
        set_fastaiagent_attributes(span, **{"simulation.passed": passed})

    return SimulationResult(
        scenario_name=scenario.name,
        passed=passed,
        transcript=transcript,
        verdicts=verdicts,
        trace_id=root_trace_id,
    )


# --------------------------------------------------------------------------- #
# Public entrypoints
# --------------------------------------------------------------------------- #


async def asimulate(
    scenarios: Scenario | list[Scenario],
    agent: Agent | AgentAdapter,
    *,
    judge: LLMJudge | None = None,
    concurrency: int = 4,
    persist: bool = True,
    run_name: str | None = None,
) -> SimulationResults:
    """Run one or more :class:`Scenario` against ``agent`` (async).

    See :func:`simulate` for the full docstring.
    """
    scenario_list = [scenarios] if isinstance(scenarios, Scenario) else list(scenarios)

    sem = asyncio.Semaphore(concurrency)

    async def run_one(scenario: Scenario) -> SimulationResult:
        async with sem:
            return await _run_scenario(scenario, agent, judge)

    results = await asyncio.gather(*(run_one(s) for s in scenario_list))

    agent_name = getattr(agent, "name", None)
    sim_results = SimulationResults(list(results), agent_name=agent_name)

    if persist:
        try:
            sim_results.persist_local(run_name=run_name, agent_name=agent_name)
        except Exception:
            # Persistence is best-effort — never fail a simulation because the
            # local UI DB couldn't be written (mirrors evaluate()).
            import logging

            logging.getLogger(__name__).debug("Failed to persist simulation run", exc_info=True)

    return sim_results


def simulate(
    scenarios: Scenario | list[Scenario],
    agent: Agent | AgentAdapter,
    *,
    judge: LLMJudge | None = None,
    concurrency: int = 4,
    persist: bool = True,
    run_name: str | None = None,
) -> SimulationResults:
    """Run one or more :class:`Scenario` against ``agent`` and judge each.

    Args:
        scenarios: A single :class:`Scenario` or a list of them.
        agent: The agent under test — a native
            :class:`~fastaiagent.agent.agent.Agent` (driven via the additive
            ``messages=`` param) or any callable adapter
            ``(messages) -> str | AgentResult``.
        judge: Optional :class:`~fastaiagent.eval.llm_judge.LLMJudge`. Its
            ``llm`` is reused for every criterion; when ``None`` a default
            ``LLMClient`` is constructed per criterion.
        concurrency: Max scenarios run in parallel (``asyncio.Semaphore``).
        persist: When ``True`` (default), write a ``sim_runs`` / ``sim_cases``
            row set to the local UI DB.
        run_name: Optional label stored on the run.

    Returns:
        :class:`SimulationResults` — per-scenario transcripts, pass/fail, and
        per-criterion verdicts, with ``.summary()``, ``.export()``, and
        ``.persist_local()``.

    Example::

        from fastaiagent import Agent, Scenario, SimulatedUser, simulate

        agent = Agent(name="support", system_prompt="You are a support agent.")
        scenario = Scenario(
            name="refund-request",
            user=SimulatedUser(persona="A frustrated customer wanting a refund."),
            success_criteria=["The agent explains the refund policy."],
            failure_criteria=["The agent is rude or dismissive."],
        )
        results = simulate(scenario, agent)
        print(results.summary())
    """
    return run_sync(
        asimulate(
            scenarios,
            agent,
            judge=judge,
            concurrency=concurrency,
            persist=persist,
            run_name=run_name,
        )
    )


__all__ = [
    "Scenario",
    "SimulatedUser",
    "SimulationResult",
    "SimulationResults",
    "TranscriptTurn",
    "CriterionVerdict",
    "simulate",
    "asimulate",
]
