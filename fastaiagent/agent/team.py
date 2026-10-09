"""Supervisor/Worker team delegation pattern."""

from __future__ import annotations

import json as _json
import logging
import time
from collections.abc import AsyncGenerator, Callable, Sequence
from contextvars import ContextVar
from typing import Any

from pydantic import BaseModel, Field

from fastaiagent._internal.async_utils import run_sync
from fastaiagent._internal.pricing import run_cost, start_run_cost, stop_run_cost
from fastaiagent.agent.agent import Agent, AgentConfig, AgentResult
from fastaiagent.agent.context import RunContext
from fastaiagent.agent.executor import _AgentInterrupted
from fastaiagent.chain.checkpoint import latest_resumable
from fastaiagent.chain.interrupt import (
    Resume,
    _execution_id,
    _resume_value,
)
from fastaiagent.checkpointers import Checkpointer, SQLiteCheckpointer
from fastaiagent.guardrail.guardrail import (
    collected_firings,
    start_firing_collection,
    stop_firing_collection,
)
from fastaiagent.llm.client import LLMClient
from fastaiagent.llm.message import SystemMessage, UserMessage
from fastaiagent.llm.stream import HandoffEvent, StreamDone, StreamEvent, TextDelta, Usage
from fastaiagent.tool.base import Tool
from fastaiagent.tool.function import FunctionTool

_log = logging.getLogger(__name__)

# The Decisions-API review score of the most recent ``validation_mode="decisions"``
# check in this task, so a routed run can report it on ``SupervisorRoute.reviews``.
# A ContextVar, not instance state: one Supervisor may serve concurrent runs.
_last_review: ContextVar[float | None] = ContextVar("_fastaiagent_last_review", default=None)

ROUTING_MODES = ("tools", "decisions")
_DEFAULT_ROUTING_INSTRUCTIONS = "Which team member should handle this request?"


class SupervisorRoute(BaseModel):
    """How ``Supervisor(routing="decisions")`` routed one request.

    Attached to the run's :class:`~fastaiagent.agent.agent.AgentResult` as
    ``result.route``.
    """

    #: The worker role that handled the request.
    worker: str
    #: The routing ``Choice``'s confidence, or ``None`` when it was refused.
    confidence: float | None = None
    #: The role the model chose before any fallback (``None`` on a refusal).
    chosen: str | None = None
    #: True when the request went to ``fallback_worker`` — refused, or below
    #: ``routing_min_confidence``.
    fallback: bool = False
    #: The full Decisions result: the routing choice plus every
    #: ``routing_questions`` answer (``answers.predicates["urgent"]`` …).
    answers: Any = None
    latency_ms: int = 0
    cost_usd: float | None = None
    #: Decisions API review scores, one per attempt (``validation_mode="decisions"``).
    reviews: list[float | None] = Field(default_factory=list)


# Default validation prompt — kept minimal so it's cheap and reliable across
# providers. Users can override via ``Supervisor(validation_prompt=...)``.
_DEFAULT_VALIDATION_PROMPT = """You review a worker's answer for a delegated task.

Original task: {task}

Worker output:
{output}

Is the worker's output a complete, correct, and well-formed answer to the
task? Respond with strict JSON only, no prose:

  {{"approved": true}}                         — accept the answer
  {{"approved": false, "feedback": "..."}}     — reject; feedback tells the
                                                 worker how to improve

Use approved=false sparingly — only when the answer is clearly incomplete,
incorrect, or off-topic."""

# The criterion ``validation_mode="decisions"`` asks by default (1.84.0). A
# statement whose probability is "approve"; the task and output are sent as
# evidence. Deliberately judgeable from the two texts alone: a reviewer that has
# not seen the worker's tool results cannot confirm "correct", and asking it to
# scored good answers at 0.3-0.5 in live runs. Override with
# ``validation_criteria=`` when the reviewer can check more.
_DECISION_VALIDATION_PREDICATE = (
    "The worker output directly addresses what the original task asks, stays on "
    "topic, and is a complete reply rather than a partial or evasive one."
)


class Worker:
    """An agent assigned a specific role in a team.

    Example:
        researcher = Agent(name="researcher", system_prompt="You research topics.", ...)
        worker = Worker(agent=researcher, role="researcher", description="Searches for info")
    """

    def __init__(
        self,
        agent: Agent,
        role: str = "",
        description: str = "",
    ):
        self.agent = agent
        self.role = role or agent.name
        if description:
            self.description = description
        else:
            prompt = agent.system_prompt
            self.description = prompt[:200] if isinstance(prompt, str) else ""


class Supervisor:
    """Manages a team of Worker agents, delegating tasks.

    The supervisor LLM decides which worker(s) to invoke based on
    the task. Workers run as full agents with their own tools and
    system prompts.

    Example:
        supervisor = Supervisor(
            name="team-lead",
            llm=LLMClient(provider="openai", model="gpt-4o"),
            workers=[
                Worker(agent=researcher, role="researcher", description="Searches for info"),
                Worker(agent=writer, role="writer", description="Writes content"),
            ],
        )
        result = supervisor.run("Research and write a report on AI trends")
    """

    def __init__(
        self,
        name: str,
        llm: LLMClient | None = None,
        workers: list[Worker] | None = None,
        system_prompt: str | Callable[..., str] = "",
        max_delegation_rounds: int = 3,
        checkpointer: Checkpointer | None = None,
        validate_outputs: bool = False,
        validation_prompt: str | None = None,
        max_validation_retries_per_worker: int = 1,
        validation_mode: str = "chat",
        validation_llm: Any = None,
        validation_threshold: float = 0.5,
        validation_criteria: str | None = None,
        routing: str = "tools",
        router_llm: Any = None,
        fallback_worker: str | None = None,
        routing_min_confidence: float = 0.0,
        routing_questions: dict[str, Any] | None = None,
        routing_instructions: str | None = None,
    ):
        self.name = name
        self.llm = llm or LLMClient()
        self.workers = workers or []
        self.max_delegation_rounds = max_delegation_rounds
        self.system_prompt = system_prompt if system_prompt else self._build_supervisor_prompt()
        self._checkpointer: Checkpointer | None = checkpointer
        # Hierarchical-process additions (v1.9.0): when ``validate_outputs``
        # is True, the supervisor LLM checks each worker's output before
        # accepting it. On rejection, the worker is re-invoked once with the
        # manager's feedback. After ``max_validation_retries_per_worker``
        # rejections the supervisor proceeds with the last output and logs
        # a guardrail_events row tagged ``supervisor.validate`` /
        # outcome=warned so the failure is auditable in the local UI.
        self.validate_outputs = validate_outputs
        self.validation_prompt = validation_prompt or _DEFAULT_VALIDATION_PROMPT
        if max_validation_retries_per_worker < 0:
            raise ValueError("max_validation_retries_per_worker must be >= 0")
        self.max_validation_retries_per_worker = max_validation_retries_per_worker
        # 1.84.0: ``validation_mode="decisions"`` asks OpenAI's Decisions API one
        # predicate — "the output completes the task" — instead of a chat call,
        # and approves when its probability meets ``validation_threshold``.
        # ``validation_llm`` defaults to ``LLMClient(model="gpt-6-luna")``.
        if validation_mode not in ("chat", "decisions"):
            raise ValueError(
                f"validation_mode must be 'chat' or 'decisions', got {validation_mode!r}"
            )
        if not 0.0 <= validation_threshold <= 1.0:
            raise ValueError("validation_threshold must be in 0..1")
        self.validation_mode = validation_mode
        self.validation_llm = validation_llm
        self.validation_threshold = validation_threshold
        # The statement the decisions review asks about (task + output are the
        # evidence). ``validation_prompt`` is the chat-mode equivalent.
        if validation_criteria is not None and not validation_criteria.strip():
            raise ValueError("validation_criteria must be a non-empty statement")
        self.validation_criteria = validation_criteria or _DECISION_VALIDATION_PREDICATE

        # 1.84.0: two ways to pick a worker.
        #   "tools"     (default) — the supervisor is a chat model that delegates
        #               through ``delegate_to_<role>`` tool calls and writes the
        #               final answer. For multi-step work: decompose, call
        #               several workers, synthesise.
        #   "decisions" — the supervisor is OpenAI's Decisions API. One
        #               ``decide()`` call picks exactly ONE worker (a Choice
        #               built from the workers' roles and descriptions), and the
        #               worker's reply is the answer — no synthesis turn. For
        #               single-hop routing (support queues, triage).
        if routing not in ROUTING_MODES:
            raise ValueError(f"routing must be 'tools' or 'decisions', got {routing!r}")
        if not 0.0 <= routing_min_confidence <= 1.0:
            raise ValueError("routing_min_confidence must be in 0..1")
        self.routing = routing
        self.router_llm = router_llm
        self.fallback_worker = fallback_worker
        self.routing_min_confidence = routing_min_confidence
        self.routing_questions = dict(routing_questions or {})
        self.routing_instructions = routing_instructions or _DEFAULT_ROUTING_INSTRUCTIONS
        if routing == "decisions":
            self._check_decision_routing()

    # ------------------------------------------------------------------
    # routing="decisions"
    # ------------------------------------------------------------------

    def _check_decision_routing(self) -> None:
        """Refuse, at construction, a decision router that could not route."""
        from fastaiagent.llm.decisions import normalize_questions

        roles = [w.role for w in self.workers]
        if len(roles) < 2:
            raise ValueError("routing='decisions' needs at least 2 workers to choose between")
        if len(set(roles)) != len(roles):
            raise ValueError(f"routing='decisions' needs unique worker roles; got {roles}")
        if self.fallback_worker not in roles:
            # A refusal or a low-confidence route has to land somewhere, and
            # guessing a specialist would be worse than saying so up front.
            raise ValueError(
                f"routing='decisions' needs fallback_worker= set to one of {roles} — where "
                f"a refused or low-confidence route goes."
            )
        if "worker" in self.routing_questions:
            raise ValueError("routing_questions may not use the name 'worker' (it is the route)")
        if self.routing_questions:
            # Validates every extra question (raises on an unusable one).
            normalize_questions(self.routing_questions)
        self._routing_choice()

    def _routing_choice(self) -> Any:
        from fastaiagent.llm.decisions import Choice, Option

        return Choice(
            name="worker",
            instructions=self.routing_instructions,
            options=[Option(value=w.role, description=w.description or None) for w in self.workers],
        )

    async def _route(self, input: str) -> tuple[Worker, SupervisorRoute]:
        """One Decisions call: the routing Choice plus any ``routing_questions``."""
        from fastaiagent.llm.decisions import DEFAULT_DECISION_MODEL, ChoiceAnswer

        router = self.router_llm or LLMClient(model=DEFAULT_DECISION_MODEL)
        questions: dict[str, Any] = {"worker": self._routing_choice(), **self.routing_questions}
        start = time.monotonic()
        result = await router.adecide(input, questions)
        latency = int((time.monotonic() - start) * 1000)

        answer = result.get("worker")
        chosen = str(answer.choice) if isinstance(answer, ChoiceAnswer) else None
        confidence = answer.confidence if isinstance(answer, ChoiceAnswer) else None
        by_role = {w.role: w for w in self.workers}
        if chosen in by_role and confidence is not None and confidence >= self.routing_min_confidence:
            role, fallback = chosen, False
        else:
            role, fallback = str(self.fallback_worker), True
        return by_role[role], SupervisorRoute(
            worker=role,
            confidence=confidence,
            chosen=chosen,
            fallback=fallback,
            answers=result,
            latency_ms=latency,
            cost_usd=result.cost_usd,
        )

    def _routing_note(self, route: SupervisorRoute) -> str:
        """The extra answers, as a short note for the worker (empty without them)."""
        from fastaiagent.llm.decisions import ChoiceAnswer, PredicateAnswer, ScoreAnswer

        parts: list[str] = []
        for name in self.routing_questions:
            a = route.answers.get(name) if route.answers is not None else None
            if isinstance(a, PredicateAnswer):
                parts.append(f"{name}: {'yes' if a.probability >= 0.5 else 'no'} (p={a.probability:.2f})")
            elif isinstance(a, ChoiceAnswer):
                parts.append(f"{name}: {a.choice}")
            elif isinstance(a, ScoreAnswer):
                parts.append(f"{name}: {a.level}")
        return f"[Supervisor routing note] {'; '.join(parts)}" if parts else ""

    def _routed_clone(self, worker: Worker) -> Agent:
        """The worker clone a routed run executes.

        Its path label carries the whole ``supervisor:<s>/worker:<role>`` itself,
        and no parent path is set: a routed supervisor runs no agent of its own,
        so the worker is the outermost runner of the execution and is the one
        that writes the run-end checkpoint. Nested under a parent path it would
        not, and a finished run would look resumable.
        """
        clone = self._build_worker_clone(worker)
        clone._agent_path_label = f"supervisor:{self.name}/worker:{worker.role}"
        return clone

    async def _finish_routed(
        self,
        worker: Worker,
        task: str,
        first: AgentResult,
        route: SupervisorRoute,
        *,
        context: RunContext[Any] | None,
        execution_id: str | None,
    ) -> AgentResult:
        """Apply ``validate_outputs`` to a routed worker's result (same contract as tools)."""
        result = first
        if result.status == "paused" or not self.validate_outputs:
            return result
        current_task, last_feedback = task, ""
        for attempt in range(1 + self.max_validation_retries_per_worker):
            if attempt > 0:
                result = await self._routed_clone(worker).arun(
                    current_task, context=context, execution_id=execution_id
                )
                if result.status == "paused":
                    return result
            approved, feedback = await self._validate_worker_output(
                worker_role=worker.role, task=task, output=result.output
            )
            route.reviews.append(_last_review.get())
            if approved:
                return result
            last_feedback = feedback
            current_task = f"{task}\n\n[Supervisor feedback]: {feedback or 'please revise.'}"
        self._log_validation_warning(
            worker_role=worker.role, task=task, output=result.output, feedback=last_feedback
        )
        return result

    def _merge_route(self, result: AgentResult, route: SupervisorRoute, span: Any) -> AgentResult:
        """Stamp the route on the result and span, and add the decisions' spend."""
        from fastaiagent.llm.decisions import DecisionResult

        extra_cost = route.cost_usd
        extra_tokens = 0
        if isinstance(route.answers, DecisionResult):
            extra_tokens = int(route.answers.usage.get("total_tokens") or 0)
        result.route = route
        result.tokens_used += extra_tokens
        if extra_cost is None:
            result.cost_known = False
        else:
            result.cost += extra_cost
        span.set_attribute("supervisor.routing", "decisions")
        span.set_attribute("supervisor.route.worker", route.worker)
        span.set_attribute("supervisor.route.fallback", route.fallback)
        if route.confidence is not None:
            span.set_attribute("supervisor.route.confidence", route.confidence)
        return result

    async def _arun_routed(
        self,
        input: str,
        span: Any,
        *,
        context: RunContext[Any] | None,
        execution_id: str | None,
        **kwargs: Any,
    ) -> AgentResult:
        worker, route = await self._route(input)
        note = self._routing_note(route)
        task = f"{input}\n\n{note}" if note else input
        first = await self._routed_clone(worker).arun(
            task, context=context, execution_id=execution_id, **kwargs
        )
        result = await self._finish_routed(
            worker, task, first, route, context=context, execution_id=execution_id
        )
        return self._merge_route(result, route, span)

    def _build_supervisor_prompt(self) -> str:
        worker_desc = "\n".join(f"- {w.role}: {w.description}" for w in self.workers)
        return (
            f"You are a supervisor managing a team of workers.\n"
            f"Available workers:\n{worker_desc}\n\n"
            f"Delegate tasks by calling the appropriate worker tool. "
            f"Synthesize results into a final answer."
        )

    def _build_worker_clone(self, worker: Worker) -> Agent:
        """Build the agent that actually runs when ``delegate_to_<role>`` fires.

        The clone shares everything with the user's worker.agent but:
            - uses the supervisor's checkpointer (so its checkpoints land
              under the supervisor's execution_id);
            - uses ``"worker:<role>"`` as its agent_path label so paths nest
              as ``supervisor:<s>/worker:<role>/...``.
        """
        base = worker.agent
        return Agent(
            name=base.name,
            # The platform identity is what managed policy is enforced on
            # (``/policy/decide``) and what plane guardrails are scoped to. A
            # clone without it was never gated: a worker's approval-policy tool
            # ran with no approval at all (fixed 1.74.0).
            agent_id=base.agent_id,
            system_prompt=base.system_prompt,
            llm=base.llm,
            tools=list(base.tools),
            guardrails=list(base.guardrails),
            memory=base.memory,
            config=base.config,
            output_type=base.output_type,
            middleware=list(base.middleware),
            checkpointer=self._checkpointer,
            agent_path_label=f"worker:{worker.role}",
        )

    def _has_worker_state(self, execution_id: str, worker_role: str) -> bool:
        """True if any checkpoint exists under this worker's path prefix."""
        if self._checkpointer is None:
            return False
        prefix = f"supervisor:{self.name}/worker:{worker_role}"
        for cp in self._checkpointer.list(execution_id, limit=500):
            if (cp.agent_path or "").startswith(prefix):
                return True
        return False

    async def _validate_worker_output(
        self, *, worker_role: str, task: str, output: str
    ) -> tuple[bool, str]:
        """Ask the supervisor LLM to vet a worker's output.

        Returns ``(approved, feedback)``. On any LLM/parse failure the
        output is approved (fail-open) — the manager loop should not crash
        a working agent because the validator misbehaved.
        """
        _last_review.set(None)
        if self.validation_mode == "decisions":
            return await self._validate_with_decision(
                worker_role=worker_role, task=task, output=output
            )
        prompt = self.validation_prompt.format(task=task, output=output)
        try:
            resp = await self.llm.acomplete(
                [
                    SystemMessage("You are an output reviewer. Reply with strict JSON only."),
                    UserMessage(prompt),
                ]
            )
        except Exception as err:  # network, provider, etc.
            _log.warning(
                "supervisor.validate (%s): LLM call failed, accepting output: %s",
                worker_role,
                err,
            )
            return True, ""

        text = (resp.content or "").strip()
        # Strip markdown fences the validator may add despite the prompt.
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:].lstrip()
        try:
            data = _json.loads(text)
        except _json.JSONDecodeError:
            _log.warning(
                "supervisor.validate (%s): unparseable JSON, accepting: %r",
                worker_role,
                text[:200],
            )
            return True, ""

        approved = bool(data.get("approved", True))
        feedback = str(data.get("feedback") or "").strip()
        return approved, feedback

    async def _validate_with_decision(
        self, *, worker_role: str, task: str, output: str
    ) -> tuple[bool, str]:
        """``validation_mode="decisions"``: one predicate on the Decisions API.

        Same contract as the chat path, including fail-open: a failed call or a
        refusal approves the output with a warning, because a misbehaving
        validator must not stop a working team. The task and output travel as
        evidence; only the fixed criterion is the question.
        """
        from fastaiagent.llm.decisions import DEFAULT_DECISION_MODEL, Predicate, PredicateAnswer

        llm = self.validation_llm or LLMClient(model=DEFAULT_DECISION_MODEL)
        try:
            result = await llm.adecide(
                f"Original task:\n{task}\n\nWorker output:\n{output}",
                [Predicate(name="approved", instructions=self.validation_criteria)],
            )
        except Exception as err:
            _log.warning(
                "supervisor.validate (%s): Decisions call failed, accepting output: %s",
                worker_role,
                err,
            )
            return True, ""
        answer = result[0]
        if not isinstance(answer, PredicateAnswer):
            _log.warning(
                "supervisor.validate (%s): Decisions API refused, accepting output", worker_role
            )
            return True, ""
        _last_review.set(answer.probability)
        if answer.probability >= self.validation_threshold:
            return True, ""
        return False, (
            f"A reviewer judged this output unlikely to be a complete, correct answer to "
            f"the task (p={answer.probability:.2f}, needs {self.validation_threshold:.2f}). "
            f"Revise it so it fully and correctly answers: {task}"
        )

    def _log_validation_warning(
        self, *, worker_role: str, task: str, output: str, feedback: str
    ) -> None:
        """Emit a ``guardrail_events`` row for an exhausted-retry validation.

        The supervisor proceeded with the worker's last output despite
        repeated rejection. We surface that as an auditable event in the
        local UI's Guardrails page (``guardrail_name = supervisor.validate``,
        ``outcome = warned``).
        """
        try:
            from fastaiagent.guardrail.guardrail import (
                Guardrail,
                GuardrailPosition,
                GuardrailResult,
                GuardrailType,
            )
            from fastaiagent.ui.events import log_guardrail_event

            guard = Guardrail(
                name="supervisor.validate",
                guardrail_type=GuardrailType.llm_judge,
                position=GuardrailPosition.output,
                blocking=False,
                description=(
                    f"Supervisor '{self.name}' validation rejected worker "
                    f"'{worker_role}' output beyond max retries; proceeding."
                ),
            )
            result = GuardrailResult(
                passed=False,
                score=0.0,
                message=feedback or "rejected after max retries",
                metadata={
                    "supervisor": self.name,
                    "worker_role": worker_role,
                    "task": task,
                    "last_output": output[:2000],
                    "feedback": feedback,
                },
            )
            log_guardrail_event(guard, result, agent_name=self.name)
        except Exception:  # pragma: no cover — observability is best-effort
            _log.debug(
                "supervisor.validate: failed to log guardrail event",
                exc_info=True,
            )

    def _build_worker_tools(self, context: RunContext[Any] | None = None) -> Sequence[Tool]:
        """Create durability-aware ``delegate_to_<role>`` tools.

        Each tool, when fired:
          1. Reads the active execution_id from the ``_execution_id`` ContextVar.
          2. If a worker checkpoint exists for this (execution, role) — i.e.
             we're inside :meth:`Supervisor.aresume` and the worker is mid-flight
             — calls ``worker_clone.aresume(...)`` (with the supervisor's
             ``_resume_value`` if any). Otherwise calls ``worker_clone.arun(task)``.
          3. If the worker returns ``status="paused"``, raises
             :class:`_AgentInterrupted` so the supervisor's tool loop bubbles
             paused state up — the worker has already persisted its
             interrupted checkpoint and ``pending_interrupts`` row with the
             full nested ``agent_path``.
        """
        tools: list[Tool] = []
        for worker in self.workers:

            async def delegate(
                task: str,
                _worker: Worker = worker,
                _ctx: RunContext[Any] | None = context,
                _supervisor: Supervisor = self,
            ) -> str:
                async def _invoke_worker(current_task: str) -> AgentResult:
                    clone = _supervisor._build_worker_clone(_worker)
                    exec_id = _execution_id.get()
                    rv = _resume_value.get()
                    # Branch: resume vs. fresh run.
                    if exec_id is not None and _supervisor._has_worker_state(exec_id, _worker.role):
                        # Scope the resume to the worker's subtree so it
                        # doesn't accidentally pick up the supervisor's own
                        # pre-tool checkpoint as "latest".
                        worker_prefix = f"supervisor:{_supervisor.name}/worker:{_worker.role}"
                        return await clone.aresume(
                            exec_id,
                            resume_value=rv,
                            context=_ctx,
                            agent_path_prefix=worker_prefix,
                        )
                    return await clone.arun(current_task, context=_ctx, execution_id=exec_id)

                current_task = task
                last_output = ""
                last_feedback = ""

                # First attempt + up to N validation-driven retries. With
                # ``validate_outputs=False`` the loop runs exactly once.
                attempts = (
                    1 + _supervisor.max_validation_retries_per_worker
                    if _supervisor.validate_outputs
                    else 1
                )
                for attempt in range(attempts):
                    result = await _invoke_worker(current_task)
                    if result.status == "paused":
                        pi = result.pending_interrupt or {}
                        raise _AgentInterrupted(
                            reason=str(pi.get("reason", "")),
                            context=dict(pi.get("context", {})),
                            node_id=str(pi.get("node_id", "")),
                            agent_path=pi.get("agent_path"),
                        )
                    last_output = result.output

                    if not _supervisor.validate_outputs:
                        return last_output

                    approved, feedback = await _supervisor._validate_worker_output(
                        worker_role=_worker.role,
                        task=task,
                        output=last_output,
                    )
                    if approved:
                        return last_output

                    last_feedback = feedback
                    # Retry: append the manager's feedback to the task so
                    # the worker has explicit guidance for the next pass.
                    current_task = (
                        f"{task}\n\n[Supervisor feedback]: {feedback or 'please revise.'}"
                    )

                # Retries exhausted. Log a guardrail_events warning and
                # proceed with the last output so the supervisor's run
                # doesn't deadlock on a stubborn validator.
                _supervisor._log_validation_warning(
                    worker_role=_worker.role,
                    task=task,
                    output=last_output,
                    feedback=last_feedback,
                )
                return last_output

            tools.append(
                FunctionTool(
                    name=f"delegate_to_{worker.role}",
                    fn=delegate,
                    description=f"Delegate a task to {worker.role}: {worker.description}",
                    parameters={
                        "type": "object",
                        "properties": {
                            "task": {
                                "type": "string",
                                "description": "The task to delegate",
                            }
                        },
                        "required": ["task"],
                    },
                )
            )
        return tools

    def _build_inner_agent(self, context: RunContext[Any] | None = None) -> Agent:
        """Build the supervisor-as-agent with durability + custom path label."""
        return Agent(
            name=self.name,
            system_prompt=self.system_prompt,
            llm=self.llm,
            tools=self._build_worker_tools(context=context),
            config=AgentConfig(max_iterations=self.max_delegation_rounds * 2),
            checkpointer=self._checkpointer,
            agent_path_label=f"supervisor:{self.name}",
        )

    def run(
        self,
        input: str,
        *,
        context: RunContext[Any] | None = None,
        execution_id: str | None = None,
        **kwargs: Any,
    ) -> AgentResult:
        """Run the supervisor synchronously."""
        return run_sync(self.arun(input, context=context, execution_id=execution_id, **kwargs))

    async def arun(
        self,
        input: str,
        *,
        context: RunContext[Any] | None = None,
        execution_id: str | None = None,
        **kwargs: Any,
    ) -> AgentResult:
        """Run the supervisor — delegates to workers via tool calls.

        ``execution_id`` (optional) names this run for resume; pair with a
        ``checkpointer`` on the Supervisor to get crash- and interrupt-
        recovery via :meth:`resume`.
        """
        from fastaiagent.trace.otel import get_tracer

        tracer = get_tracer()
        # Root span wraps the supervisor run so the delegated worker spans
        # nest as children in the UI span tree.
        with tracer.start_as_current_span(f"supervisor.{self.name}") as span:
            span.set_attribute("supervisor.name", self.name)
            span.set_attribute(
                "supervisor.worker_count",
                len(getattr(self, "workers", []) or []),
            )
            span.set_attribute("fastaiagent.runner.type", "supervisor")
            span.set_attribute("fastaiagent.framework", "fastaiagent")
            span.set_attribute("supervisor.input", input)

            if self._checkpointer is not None:
                self._checkpointer.setup()

            if self.routing == "decisions":
                result = await self._arun_routed(
                    input, span, context=context, execution_id=execution_id, **kwargs
                )
                span.set_attribute("supervisor.output", result.output)
                return result

            agent = self._build_inner_agent(context=context)
            result = await agent.arun(input, context=context, execution_id=execution_id, **kwargs)
            span.set_attribute("supervisor.output", result.output)
            return result

    def resume(
        self,
        execution_id: str,
        *,
        resume_value: Resume | None = None,
        context: RunContext[Any] | None = None,
        **kwargs: Any,
    ) -> AgentResult:
        """Synchronous resume wrapper around :meth:`aresume`."""
        return run_sync(
            self.aresume(execution_id, resume_value=resume_value, context=context, **kwargs)
        )

    async def aresume(
        self,
        execution_id: str,
        *,
        resume_value: Resume | None = None,
        context: RunContext[Any] | None = None,
        **kwargs: Any,
    ) -> AgentResult:
        """Resume a paused or crashed supervisor run.

        Strategy: rebuild the supervisor's inner Agent, recover the original
        input from the supervisor's earliest checkpoint, and call
        ``inner_agent.arun(input, execution_id=...)`` again. The supervisor's
        LLM is re-issued; it will deterministically call the same
        ``delegate_to_<role>`` tools in order. Each delegate tool detects
        that worker state exists for this execution and calls
        ``worker_clone.aresume(...)`` (with the supervisor's ``resume_value``
        in scope) instead of running the worker fresh. So the resumed
        worker picks up exactly where it left off; subsequent workers run
        normally.
        """
        from fastaiagent._internal.errors import ChainCheckpointError

        store: Checkpointer = self._checkpointer or SQLiteCheckpointer()
        store.setup()

        # Restore-anywhere (audit D4): when this machine has never seen the run
        # but the plane is holding it, pull it down before deciding there is
        # nothing to resume. No-op when disconnected or already present.
        from fastaiagent.checkpointers.platform_replica import restore_if_missing

        restore_if_missing(store, execution_id)
        # Refuses a finished run and steps past a `failed` tombstone (audit D5).
        latest = latest_resumable(store, execution_id, runner="Supervisor execution")
        if latest is None:
            raise ChainCheckpointError(
                f"No checkpoint found for supervisor execution '{execution_id}'"
            )
        if resume_value is None and latest.status == "interrupted":
            raise ChainCheckpointError(
                f"Supervisor execution '{execution_id}' is suspended on interrupt(); "
                "pass resume_value=Resume(...) to supervisor.resume()."
            )

        if self.routing == "decisions":
            return await self._aresume_routed(
                store, execution_id, resume_value=resume_value, context=context, **kwargs
            )

        # Recover the original supervisor input by walking the supervisor's
        # own earliest checkpoint and pulling the first user message.
        input_str = self._hydrate_input(store, execution_id)
        if input_str is None:
            raise ChainCheckpointError(
                f"Cannot recover original input for supervisor execution '{execution_id}'."
            )

        # Bind the resume value so every worker the supervisor delegates to
        # during this resumed run can pick it up via _resume_value.
        rv_token = _resume_value.set(resume_value) if resume_value is not None else None
        try:
            return await self.arun(
                input_str,
                context=context,
                execution_id=execution_id,
                **kwargs,
            )
        finally:
            if rv_token is not None:
                _resume_value.reset(rv_token)

    async def _aresume_routed(
        self,
        store: Checkpointer,
        execution_id: str,
        *,
        resume_value: Resume | None,
        context: RunContext[Any] | None,
        **kwargs: Any,
    ) -> AgentResult:
        """Resume a routed run: the worker that holds state continues — no re-route.

        Asking the router again could pick a different worker than the one that
        paused, so the route is read back from the checkpoints instead. The
        resumed result is returned as-is (``validate_outputs`` reviews fresh
        attempts, not a resumed one).
        """
        from fastaiagent._internal.errors import ChainCheckpointError
        from fastaiagent.trace.otel import get_tracer

        paths = [cp.agent_path or "" for cp in store.list(execution_id, limit=500)]
        worker = next(
            (
                w
                for w in self.workers
                if any(p.startswith(f"supervisor:{self.name}/worker:{w.role}") for p in paths)
            ),
            None,
        )
        if worker is None:
            raise ChainCheckpointError(
                f"Supervisor execution '{execution_id}' has no routed worker state to resume."
            )
        prefix = f"supervisor:{self.name}/worker:{worker.role}"
        with get_tracer().start_as_current_span(f"supervisor.{self.name}") as span:
            span.set_attribute("supervisor.name", self.name)
            span.set_attribute("fastaiagent.runner.type", "supervisor")
            span.set_attribute("fastaiagent.framework", "fastaiagent")
            span.set_attribute("supervisor.routing", "decisions")
            span.set_attribute("supervisor.route.worker", worker.role)
            span.set_attribute("supervisor.resumed", True)
            result = await self._routed_clone(worker).aresume(
                execution_id,
                resume_value=resume_value,
                context=context,
                agent_path_prefix=prefix,
                **kwargs,
            )
            result.route = SupervisorRoute(worker=worker.role, chosen=worker.role)
            span.set_attribute("supervisor.output", result.output)
            return result

    def _hydrate_input(self, store: Checkpointer, execution_id: str) -> str | None:
        """Pull the original supervisor input from its earliest turn checkpoint."""
        prefix_exact = f"supervisor:{self.name}"
        # ``list`` returns checkpoints in chronological order — first match
        # wins. Filter to checkpoints written by the supervisor itself
        # (not by a worker nested under the supervisor).
        for cp in store.list(execution_id, limit=500):
            if (cp.agent_path or "") != prefix_exact:
                continue
            for raw in cp.state_snapshot.get("messages", []) or []:
                if (raw or {}).get("role") == "user" and raw.get("content"):
                    return str(raw["content"])
        return None

    async def astream(
        self, input: str, *, context: RunContext[Any] | None = None, **kwargs: Any
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream the supervisor — yields events as tokens arrive.

        Yields TextDelta for the supervisor's synthesis, and
        ToolCallStart/ToolCallEnd for worker delegations.
        Worker execution itself is not streamed.

        With ``routing="decisions"`` there is no synthesis: the stream opens with
        a :class:`~fastaiagent.llm.stream.HandoffEvent` naming the routed worker,
        then streams that worker's own tokens. With ``validate_outputs`` the
        reviewed reply arrives as one ``TextDelta`` (a rejected draft is never
        streamed to the caller).
        """
        if self.routing == "decisions":
            async for event in self._astream_routed(input, context=context, **kwargs):
                yield event
            return
        agent = Agent(
            name=self.name,
            system_prompt=self.system_prompt,
            llm=self.llm,
            tools=self._build_worker_tools(context=context),
            config=AgentConfig(max_iterations=self.max_delegation_rounds * 2),
        )
        async for event in agent.astream(input, context=context, **kwargs):
            yield event

    async def _astream_routed(
        self, input: str, *, context: RunContext[Any] | None = None, **kwargs: Any
    ) -> AsyncGenerator[StreamEvent, None]:
        worker, route = await self._route(input)
        yield HandoffEvent(
            from_agent=self.name,
            to_agent=worker.role,
            reason=(
                "fallback"
                if route.fallback
                else f"decisions: confidence {route.confidence:.2f}"
                if route.confidence is not None
                else "decisions"
            ),
        )
        note = self._routing_note(route)
        task = f"{input}\n\n{note}" if note else input
        if self.validate_outputs:
            first = await self._routed_clone(worker).arun(task, context=context, **kwargs)
            result = await self._finish_routed(
                worker, task, first, route, context=context, execution_id=None
            )
            if result.output:
                yield TextDelta(text=result.output)
            yield StreamDone()
            return
        async for event in self._routed_clone(worker).astream(task, context=context, **kwargs):
            yield event

    def stream(
        self, input: str, *, context: RunContext[Any] | None = None, **kwargs: Any
    ) -> AgentResult:
        """Synchronous streaming — collects stream into AgentResult.

        Opens a ``supervisor.<name>`` root span so a streamed supervisor run has
        the same identity ``run``/``arun`` do. ``run`` got its ``trace_id`` for
        free by inheriting the inner agent's; the stream path built its result
        by hand and inherited nothing.
        """
        from fastaiagent.trace.otel import get_tracer
        from fastaiagent.trace.span import trace_id_of

        async def _collect() -> AgentResult:
            start = time.monotonic()
            text_parts: list[str] = []
            gf_token = start_firing_collection()
            # A hand-assembled result reports what it is told to report: this
            # one was never told about tokens or cost, so a streamed supervisor
            # run looked free. ``Usage`` is the only event carrying a stream's
            # token counts.
            #
            # These are the SUPERVISOR's own turns. A delegated worker runs
            # through ``arun`` inside a delegate tool, which opens its own
            # accumulator and reports on its own result — so the worker's spend
            # is counted once, there, and not re-counted here. That is the same
            # boundary ``run``/``arun`` have, and the same one ``cost`` took in
            # 1.67.0.
            streamed_tokens = 0
            rcost_token = start_run_cost()
            try:
                with get_tracer().start_as_current_span(f"supervisor.{self.name}") as span:
                    span.set_attribute("supervisor.name", self.name)
                    span.set_attribute(
                        "supervisor.worker_count", len(getattr(self, "workers", []) or [])
                    )
                    span.set_attribute("fastaiagent.runner.type", "supervisor")
                    span.set_attribute("fastaiagent.framework", "fastaiagent")
                    span.set_attribute("supervisor.streamed", True)
                    async for event in self.astream(input, context=context, **kwargs):
                        if isinstance(event, TextDelta):
                            text_parts.append(event.text)
                        elif isinstance(event, Usage):
                            streamed_tokens += event.prompt_tokens + event.completion_tokens
                    output = "".join(text_parts)
                    span.set_attribute("supervisor.output", output)
                    span.set_attribute("supervisor.tokens_used", streamed_tokens)
                    trace_id = trace_id_of(span)
                latency = int((time.monotonic() - start) * 1000)
                cost, cost_known = run_cost()
                return AgentResult(
                    output=output,
                    tokens_used=streamed_tokens,
                    cost=cost,
                    cost_known=cost_known,
                    latency_ms=latency,
                    trace_id=trace_id,
                    guardrails=collected_firings(),
                )
            finally:
                stop_run_cost(rcost_token)
                stop_firing_collection(gf_token)

        return run_sync(_collect())

    def to_dict(self) -> dict[str, Any]:
        """Serialize the supervisor structure for the Local UI topology view.

        Worker agents are referenced by name + role; rebuilding requires the
        caller to pass the live :class:`Worker` instances back in.
        """
        return {
            "name": self.name,
            "supervisor_llm": {
                "provider": getattr(self.llm, "provider", ""),
                "model": getattr(self.llm, "model", ""),
            },
            "workers": [
                {
                    "role": w.role,
                    "agent_name": w.agent.name,
                    "description": w.description,
                    "model": getattr(w.agent.llm, "model", ""),
                    "tools": [t.name for t in (w.agent.tools or [])],
                }
                for w in self.workers
            ],
            "max_delegation_rounds": self.max_delegation_rounds,
            "validate_outputs": self.validate_outputs,
            "max_validation_retries_per_worker": self.max_validation_retries_per_worker,
            # Only off the default, so an existing topology dict is unchanged.
            **(
                {"validation_mode": self.validation_mode}
                if self.validation_mode != "chat"
                else {}
            ),
            **(
                {"routing": self.routing, "fallback_worker": self.fallback_worker}
                if self.routing != "tools"
                else {}
            ),
        }
