"""Auto-tracing for CrewAI 1.x.

CrewAI exposes two interception surfaces and we use both:

1. **Method patches** for the structural spans (root crew, per-agent,
   per-task). Monkey-patching ``Crew.kickoff``, ``Agent.execute_task``,
   and ``Task.execute_sync/_async`` lets us put a real ``with
   tracer.start_as_current_span(...)`` block around the call so OTel's
   current-context propagates to anything those calls invoke.

2. **Event-bus subscriptions** for LLM and tool spans. CrewAI's internal
   ``LLM.call`` does not always go through the public class method (some
   reasoning flows reach into private paths), but every call emits
   ``LLMCallStartedEvent`` / ``LLMCallCompletedEvent`` on the bus.
   Likewise tool execution emits ``ToolUsageStartedEvent`` /
   ``ToolUsageFinishedEvent`` / ``ToolUsageErrorEvent``.

By the time those events fire, the OTel current-span is already the
agent span we opened in step 1, so opening child spans via
``tracer.start_span(name, context=set_span_in_context(current))`` gives
us the right parent-child hierarchy without manual run-id threading.
"""

from __future__ import annotations

import functools
import json
from typing import Any

_INSTALL_HINT = 'CrewAI is required. Install with: pip install "fastaiagent[crewai]"'
_PAYLOAD_TRUNC = 10_000

_enabled = False
# Open spans, keyed by event call_id / event_id. Cleared on completion.
_llm_spans: dict[str, Any] = {}
_tool_spans: dict[str, Any] = {}


def _require() -> None:
    try:
        import crewai  # noqa: F401
    except ImportError as e:
        raise ImportError(_INSTALL_HINT) from e


def _crewai_version() -> str:
    try:
        import crewai

        return getattr(crewai, "__version__", "unknown")
    except ImportError:
        return "unknown"


def _safe_json(obj: Any, *, limit: int = _PAYLOAD_TRUNC) -> str:
    try:
        text = json.dumps(obj, default=_json_default, ensure_ascii=False)
    except Exception:
        text = str(obj)
    if len(text) > limit:
        return text[:limit] + f"…[+{len(text) - limit}B]"
    return text


def _json_default(o: Any) -> Any:
    if hasattr(o, "model_dump"):
        try:
            return o.model_dump()
        except Exception:
            pass
    if hasattr(o, "__dict__"):
        try:
            return {k: v for k, v in o.__dict__.items() if not k.startswith("_")}
        except Exception:
            pass
    return str(o)


def _truncate(text: Any, limit: int = 200) -> str:
    s = "" if text is None else str(text)
    return s if len(s) <= limit else s[:limit] + "…"


def _provider_from_model(model: str | None) -> str:
    """Best-effort provider inference from the litellm-style model id."""
    if not model:
        return "unknown"
    if "/" in model:
        return model.split("/", 1)[0].lower()
    lower = model.lower()
    if any(k in lower for k in ("gpt", "o1", "o3")):
        return "openai"
    if "claude" in lower or "anthropic" in lower:
        return "anthropic"
    if "gemini" in lower or "google" in lower:
        return "google"
    return "unknown"


def _bare_model(model: str | None) -> str:
    if not model:
        return "unknown"
    return model.split("/", 1)[1] if "/" in model else model


def _patched(fn: Any) -> bool:
    return bool(getattr(fn, "_fastaiagent_patched", False))


def _mark_patched(wrapper: Any, original: Any) -> Any:
    wrapper._fastaiagent_patched = True  # noqa: SLF001
    wrapper._fastaiagent_original = original  # noqa: SLF001
    return wrapper


def _install_method_patches() -> None:
    """Wrap Crew.kickoff, Agent.execute_task, Task.execute_* with span ctxs."""
    from crewai.agent import Agent
    from crewai.crew import Crew
    from crewai.task import Task

    from fastaiagent.trace.otel import get_tracer
    from fastaiagent.trace.span import (
        set_fastaiagent_attributes,
        trace_payloads_enabled,
    )

    tracer = get_tracer("fastaiagent.integrations.crewai")

    if not _patched(Crew.kickoff):
        original_kickoff = Crew.kickoff

        @functools.wraps(original_kickoff)
        def kickoff_wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            crew_name = getattr(self, "name", None) or "crew"
            with tracer.start_as_current_span(f"crewai.crew.{crew_name}") as span:
                set_fastaiagent_attributes(
                    span,
                    framework="crewai",
                    **{"framework.version": _crewai_version()},
                )
                span.set_attribute("crewai.crew.name", str(crew_name))
                process = getattr(self, "process", None)
                if process is not None:
                    span.set_attribute("crewai.crew.process", str(process))
                span.set_attribute(
                    "crewai.crew.agent_count", len(getattr(self, "agents", []) or [])
                )
                span.set_attribute(
                    "crewai.crew.task_count", len(getattr(self, "tasks", []) or [])
                )
                inputs = kwargs.get("inputs") or (args[0] if args else None)
                if inputs is not None and trace_payloads_enabled():
                    span.set_attribute("crewai.crew.inputs", _safe_json(inputs))
                try:
                    result = original_kickoff(self, *args, **kwargs)
                except BaseException as e:
                    span.record_exception(e)
                    raise
                if trace_payloads_enabled():
                    raw = getattr(result, "raw", None)
                    span.set_attribute("crewai.crew.output", _safe_json(raw or result))
                return result

        Crew.kickoff = _mark_patched(kickoff_wrapper, original_kickoff)  # type: ignore[method-assign]

    if not _patched(Crew.kickoff_async):
        original_kickoff_async = Crew.kickoff_async

        @functools.wraps(original_kickoff_async)
        async def kickoff_async_wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            crew_name = getattr(self, "name", None) or "crew"
            with tracer.start_as_current_span(f"crewai.crew.{crew_name}") as span:
                set_fastaiagent_attributes(
                    span,
                    framework="crewai",
                    **{"framework.version": _crewai_version()},
                )
                span.set_attribute("crewai.crew.name", str(crew_name))
                inputs = kwargs.get("inputs") or (args[0] if args else None)
                if inputs is not None and trace_payloads_enabled():
                    span.set_attribute("crewai.crew.inputs", _safe_json(inputs))
                try:
                    result = await original_kickoff_async(self, *args, **kwargs)
                except BaseException as e:
                    span.record_exception(e)
                    raise
                if trace_payloads_enabled():
                    raw = getattr(result, "raw", None)
                    span.set_attribute("crewai.crew.output", _safe_json(raw or result))
                return result

        Crew.kickoff_async = _mark_patched(kickoff_async_wrapper, original_kickoff_async)  # type: ignore[method-assign]

    if not _patched(Agent.execute_task):
        original_execute_task = Agent.execute_task

        @functools.wraps(original_execute_task)
        def execute_task_wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            role = getattr(self, "role", None) or "agent"
            with tracer.start_as_current_span(f"crewai.agent.{role}") as span:
                span.set_attribute("crewai.agent.role", str(role))
                model = getattr(getattr(self, "llm", None), "model", None)
                if model:
                    span.set_attribute("crewai.agent.model", str(model))
                if trace_payloads_enabled():
                    goal = getattr(self, "goal", None)
                    if goal:
                        span.set_attribute("crewai.agent.goal", _truncate(goal))
                    backstory = getattr(self, "backstory", None)
                    if backstory:
                        span.set_attribute("crewai.agent.backstory", _truncate(backstory))
                task = kwargs.get("task") or (args[0] if args else None)
                if task is not None and trace_payloads_enabled():
                    desc = getattr(task, "description", None)
                    if desc:
                        span.set_attribute(
                            "crewai.agent.task_description", _truncate(desc, 400)
                        )
                try:
                    result = original_execute_task(self, *args, **kwargs)
                except BaseException as e:
                    span.record_exception(e)
                    raise
                if trace_payloads_enabled():
                    span.set_attribute("crewai.agent.output", _safe_json(result))
                return result

        Agent.execute_task = _mark_patched(execute_task_wrapper, original_execute_task)  # type: ignore[method-assign]

    if not _patched(Task.execute_sync):
        original_task_sync = Task.execute_sync

        @functools.wraps(original_task_sync)
        def task_sync_wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            slug = _truncate(getattr(self, "description", None) or "task", 60)
            with tracer.start_as_current_span(f"crewai.task.{slug}") as span:
                if trace_payloads_enabled():
                    desc = getattr(self, "description", None)
                    if desc:
                        span.set_attribute("crewai.task.description", _truncate(desc, 1_000))
                    expected = getattr(self, "expected_output", None)
                    if expected:
                        span.set_attribute(
                            "crewai.task.expected_output", _truncate(expected, 1_000)
                        )
                assigned = getattr(getattr(self, "agent", None), "role", None)
                if assigned:
                    span.set_attribute("crewai.task.agent_role", str(assigned))
                try:
                    result = original_task_sync(self, *args, **kwargs)
                except BaseException as e:
                    span.record_exception(e)
                    raise
                if trace_payloads_enabled():
                    raw = getattr(result, "raw", None)
                    span.set_attribute("crewai.task.output", _safe_json(raw or result))
                return result

        Task.execute_sync = _mark_patched(task_sync_wrapper, original_task_sync)  # type: ignore[method-assign]

    if not _patched(Task.execute_async):
        original_task_async = Task.execute_async

        @functools.wraps(original_task_async)
        def task_async_wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            slug = _truncate(getattr(self, "description", None) or "task", 60)
            with tracer.start_as_current_span(f"crewai.task.{slug}") as span:
                span.set_attribute("crewai.task.async", True)
                if trace_payloads_enabled():
                    span.set_attribute(
                        "crewai.task.description",
                        _truncate(getattr(self, "description", "") or "", 1_000),
                    )
                try:
                    return original_task_async(self, *args, **kwargs)
                except BaseException as e:
                    span.record_exception(e)
                    raise

        Task.execute_async = _mark_patched(task_async_wrapper, original_task_async)  # type: ignore[method-assign]


def _install_event_listeners() -> None:
    """Subscribe to CrewAI's event bus for LLM + tool spans."""
    from crewai.events import crewai_event_bus
    from crewai.events.types.llm_events import (
        LLMCallCompletedEvent,
        LLMCallStartedEvent,
    )
    from crewai.events.types.tool_usage_events import (
        ToolUsageErrorEvent,
        ToolUsageFinishedEvent,
        ToolUsageStartedEvent,
    )
    from opentelemetry import trace as otel_trace

    from fastaiagent.trace.otel import get_tracer
    from fastaiagent.trace.span import (
        set_fastaiagent_attributes,
        set_genai_attributes,
        trace_payloads_enabled,
    )
    from fastaiagent.ui.pricing import compute_cost_usd

    tracer = get_tracer("fastaiagent.integrations.crewai")

    @crewai_event_bus.on(LLMCallStartedEvent)
    def _on_llm_started(_source: Any, event: Any) -> None:
        model = getattr(event, "model", None)
        provider = _provider_from_model(model)
        bare = _bare_model(model)
        # Nest under the current OTel span (agent / task).
        current = otel_trace.get_current_span()
        ctx = otel_trace.set_span_in_context(current) if current else None
        span = tracer.start_span(f"llm.{provider}.{bare}", context=ctx)
        set_genai_attributes(
            span,
            system=provider,
            model=bare,
            request_messages=(
                _safe_json(getattr(event, "messages", None))
                if trace_payloads_enabled()
                else None
            ),
        )
        call_id = str(getattr(event, "call_id", "") or getattr(event, "event_id", ""))
        if call_id:
            _llm_spans[call_id] = span
        else:
            # No correlation id — close immediately so we don't leak.
            try:
                span.end()
            except Exception:
                pass

    @crewai_event_bus.on(LLMCallCompletedEvent)
    def _on_llm_completed(_source: Any, event: Any) -> None:
        call_id = str(getattr(event, "call_id", "") or getattr(event, "event_id", ""))
        span = _llm_spans.pop(call_id, None)
        if span is None:
            return
        usage = getattr(event, "usage", None) or {}
        in_toks = (
            usage.get("prompt_tokens")
            or usage.get("input_tokens")
            or usage.get("prompt_token_count")
            if isinstance(usage, dict)
            else getattr(usage, "prompt_tokens", None)
            or getattr(usage, "input_tokens", None)
        )
        out_toks = (
            usage.get("completion_tokens")
            or usage.get("output_tokens")
            or usage.get("completion_token_count")
            if isinstance(usage, dict)
            else getattr(usage, "completion_tokens", None)
            or getattr(usage, "output_tokens", None)
        )
        bare = _bare_model(getattr(event, "model", None))
        response = getattr(event, "response", None)
        set_genai_attributes(
            span,
            input_tokens=int(in_toks) if in_toks else None,
            output_tokens=int(out_toks) if out_toks else None,
            response_content=(
                _safe_json(response)
                if response is not None and trace_payloads_enabled()
                else None
            ),
        )
        cost = compute_cost_usd(bare, in_toks, out_toks)
        if cost is not None:
            set_fastaiagent_attributes(span, **{"cost.total_usd": cost})
        try:
            span.end()
        except Exception:
            pass

    @crewai_event_bus.on(ToolUsageStartedEvent)
    def _on_tool_started(_source: Any, event: Any) -> None:
        tool_name = getattr(event, "tool_name", None) or "tool"
        current = otel_trace.get_current_span()
        ctx = otel_trace.set_span_in_context(current) if current else None
        span = tracer.start_span(f"tool.{tool_name}", context=ctx)
        set_fastaiagent_attributes(span, **{"tool.name": tool_name})
        if trace_payloads_enabled():
            span.set_attribute(
                "tool.input", _safe_json(getattr(event, "tool_args", None))
            )
        event_id = str(getattr(event, "event_id", "") or "")
        if event_id:
            _tool_spans[event_id] = span
        else:
            try:
                span.end()
            except Exception:
                pass

    @crewai_event_bus.on(ToolUsageFinishedEvent)
    def _on_tool_finished(_source: Any, event: Any) -> None:
        # Match by ``started_event_id`` (the started event's id) rather than the
        # finished event's own id.
        event_id = str(getattr(event, "started_event_id", "") or "")
        span = _tool_spans.pop(event_id, None)
        if span is None:
            return
        if trace_payloads_enabled():
            span.set_attribute("tool.output", _safe_json(getattr(event, "output", None)))
        try:
            span.end()
        except Exception:
            pass

    @crewai_event_bus.on(ToolUsageErrorEvent)
    def _on_tool_error(_source: Any, event: Any) -> None:
        event_id = str(getattr(event, "started_event_id", "") or "")
        span = _tool_spans.pop(event_id, None)
        if span is None:
            return
        err = getattr(event, "error", None) or getattr(event, "message", None)
        try:
            if isinstance(err, BaseException):
                span.record_exception(err)
            elif err:
                span.add_event("tool.error", attributes={"message": str(err)[:500]})
            span.end()
        except Exception:
            pass


class _EvaluableResult:
    """Duck-typed object so ``fa.evaluate()`` reads ``.output`` and
    ``.trace_id`` (see ``evaluate.py:122-126``)."""

    __slots__ = ("output", "trace_id")

    def __init__(self, output: str, trace_id: str | None = None) -> None:
        self.output = output
        self.trace_id = trace_id


def _current_trace_id() -> str | None:
    from opentelemetry import trace as otel_trace

    span = otel_trace.get_current_span()
    if span is None:
        return None
    ctx = span.get_span_context()
    if ctx is None or not ctx.trace_id:
        return None
    return format(ctx.trace_id, "032x")


def as_evaluable(
    crew: Any,
    *,
    input_mapper: Any = None,
    output_mapper: Any = None,
) -> Any:
    """Adapt a CrewAI ``Crew`` for ``fastaiagent.evaluate(...)``.

    Opens an outer ``eval.case`` span so we can capture ``trace_id``
    *while* it is still active — the patched ``crewai.crew.*`` span has
    already closed by the time ``kickoff`` returns.
    """
    _require()
    from fastaiagent.trace.otel import get_tracer

    in_map = input_mapper or (lambda s: {"input": s})
    out_map = output_mapper or (lambda r: getattr(r, "raw", None) or str(r))
    tracer = get_tracer("fastaiagent.integrations.crewai")

    def _evaluable(text: str) -> _EvaluableResult:
        inputs = in_map(text)
        with tracer.start_as_current_span("eval.case"):
            result = crew.kickoff(
                inputs=dict(inputs) if isinstance(inputs, dict) else inputs
            )
            return _EvaluableResult(
                output=str(out_map(result)), trace_id=_current_trace_id()
            )

    return _evaluable


def enable() -> None:
    """Enable CrewAI auto-tracing.

    Idempotent — repeated calls are a no-op (each patched method carries
    a ``_fastaiagent_patched`` sentinel; event-bus subscriptions are
    guarded by the module-level ``_enabled`` flag).
    """
    global _enabled
    if _enabled:
        return
    _require()
    _install_method_patches()
    _install_event_listeners()
    _enabled = True


def disable() -> None:
    """Disable CrewAI auto-tracing — restores method patches.

    Event-bus subscriptions remain registered (CrewAI's bus offers no
    public unsubscribe), but they no-op when there's no current OTel
    parent span anyway. Use process restart for a clean reset.
    """
    global _enabled
    _enabled = False
    try:
        from crewai.agent import Agent
        from crewai.crew import Crew
        from crewai.task import Task
    except ImportError:
        return
    for cls, name in (
        (Crew, "kickoff"),
        (Crew, "kickoff_async"),
        (Agent, "execute_task"),
        (Task, "execute_sync"),
        (Task, "execute_async"),
    ):
        fn = getattr(cls, name, None)
        original = getattr(fn, "_fastaiagent_original", None)
        if original is not None:
            setattr(cls, name, original)


def _extract_input_text(crew_input: Any) -> str:
    """Read the user-input string from a CrewAI ``kickoff`` payload."""
    if isinstance(crew_input, str):
        return crew_input
    if isinstance(crew_input, dict):
        for key in ("input", "query", "question", "text"):
            if key in crew_input and crew_input[key] is not None:
                v = crew_input[key]
                return v if isinstance(v, str) else str(v)
    return str(crew_input)


def _extract_output_text(result: Any) -> str:
    raw = getattr(result, "raw", None)
    return raw if isinstance(raw, str) else str(raw if raw is not None else result)


def _run_guardrails(
    text: str,
    guardrails: list[Any] | None,
    *,
    side: str,
    agent_name: str | None,
) -> None:
    if not guardrails:
        return
    from fastaiagent.integrations._registry import GuardrailBlocked

    for g in guardrails:
        result = g.execute(text)
        if not result.passed and getattr(g, "blocking", True):
            try:
                from fastaiagent.ui.events import log_guardrail_event

                merged = dict(result.metadata or {})
                merged.setdefault("framework", "crewai")
                merged.setdefault("side", side)
                result.metadata = merged
                log_guardrail_event(g, result, agent_name=agent_name)
            except Exception:
                pass
            raise GuardrailBlocked(
                f"{side} blocked by {g.name}: {result.message or ''}"
            )


class _GuardedCrew:
    """Proxy around a CrewAI ``Crew`` exposing ``kickoff`` /
    ``kickoff_async`` with input + output guardrails layered on."""

    def __init__(
        self,
        wrapped: Any,
        *,
        name: str | None = None,
        input_guardrails: list[Any] | None = None,
        output_guardrails: list[Any] | None = None,
    ) -> None:
        self._wrapped = wrapped
        self._fastaiagent_name = name
        self._input_guardrails = list(input_guardrails or [])
        self._output_guardrails = list(output_guardrails or [])

    def __getattr__(self, item: str) -> Any:
        return getattr(self._wrapped, item)

    def kickoff(self, inputs: Any = None, **kwargs: Any) -> Any:
        _run_guardrails(
            _extract_input_text(inputs) if inputs is not None else "",
            self._input_guardrails,
            side="input",
            agent_name=self._fastaiagent_name,
        )
        result = self._wrapped.kickoff(inputs=inputs, **kwargs)
        _run_guardrails(
            _extract_output_text(result),
            self._output_guardrails,
            side="output",
            agent_name=self._fastaiagent_name,
        )
        return result

    async def kickoff_async(self, inputs: Any = None, **kwargs: Any) -> Any:
        _run_guardrails(
            _extract_input_text(inputs) if inputs is not None else "",
            self._input_guardrails,
            side="input",
            agent_name=self._fastaiagent_name,
        )
        result = await self._wrapped.kickoff_async(inputs=inputs, **kwargs)
        _run_guardrails(
            _extract_output_text(result),
            self._output_guardrails,
            side="output",
            agent_name=self._fastaiagent_name,
        )
        return result


def with_guardrails(
    crew: Any,
    *,
    name: str | None = None,
    input_guardrails: list[Any] | None = None,
    output_guardrails: list[Any] | None = None,
) -> _GuardedCrew:
    """Wrap a CrewAI ``Crew`` with input/output guardrails.

    Block-only semantics: a failing blocking guardrail logs a
    ``guardrail_events`` row tagged ``framework=crewai`` and raises
    :class:`fastaiagent.integrations._registry.GuardrailBlocked`.
    """
    _require()
    return _GuardedCrew(
        crew,
        name=name,
        input_guardrails=input_guardrails,
        output_guardrails=output_guardrails,
    )


def prompt_from_registry(
    slug: str,
    *,
    version: Any = "latest",
    agent: str | None = None,
) -> str:
    """Return a CrewAI-friendly prompt string from the FastAIAgent registry.

    CrewAI takes plain strings for ``role`` / ``goal`` / ``backstory`` /
    ``Task.description``, so we hand back the raw template. Callers that
    want variable substitution can pass the result through
    ``Prompt.format(**kw)`` themselves (the underlying ``Prompt`` is
    available via ``PromptRegistry().get(slug)`` directly).
    """
    _require()
    from fastaiagent.prompt import PromptRegistry

    resolved_version: int | None = None if version in (None, "latest") else int(version)
    prompt = PromptRegistry().get(slug, version=resolved_version)

    if agent:
        try:
            from fastaiagent.integrations._registry import attach as _attach

            _attach(agent, "prompt", slug, version=str(prompt.version))
        except Exception:
            pass

    return prompt.template


def _crew_topology(crew: Any) -> dict[str, Any]:
    """Build a ``{nodes, edges}`` topology from a CrewAI ``Crew``.

    For ``Process.sequential`` we wire tasks in execution order; for
    ``Process.hierarchical`` we wire the manager agent to each worker.
    Agents and tasks both become nodes; edges express the runtime
    relationship the dependency-graph UI cares about.
    """
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []

    agents = getattr(crew, "agents", None) or []
    tasks = getattr(crew, "tasks", None) or []

    for agent in agents:
        nodes.append(
            {
                "id": getattr(agent, "role", None) or f"agent-{id(agent)}",
                "type": "agent",
                "model": str(getattr(getattr(agent, "llm", None), "model", "")) or None,
            }
        )
    for idx, task in enumerate(tasks):
        node_id = f"task-{idx}"
        nodes.append(
            {
                "id": node_id,
                "type": "task",
                "description": (
                    str(getattr(task, "description", ""))[:200] if task else ""
                ),
                "agent_role": getattr(getattr(task, "agent", None), "role", None),
            }
        )

    process = str(getattr(crew, "process", "")).lower()
    if "sequential" in process:
        # Each task feeds the next.
        for i in range(len(tasks) - 1):
            edges.append({"source": f"task-{i}", "target": f"task-{i + 1}", "kind": "next"})
    elif "hierarchical" in process:
        manager = getattr(crew, "manager_agent", None)
        manager_role = getattr(manager, "role", None) if manager else None
        if manager_role:
            for agent in agents:
                role = getattr(agent, "role", None)
                if role and role != manager_role:
                    edges.append({"source": manager_role, "target": role, "kind": "delegates"})

    # Each task is owned by its agent (visualised as a containment edge).
    for idx, task in enumerate(tasks):
        agent_role = getattr(getattr(task, "agent", None), "role", None)
        if agent_role:
            edges.append({"source": agent_role, "target": f"task-{idx}", "kind": "owns"})

    return {"nodes": nodes, "edges": edges, "process": process or None}


def register_agent(crew: Any, *, name: str) -> None:
    """Persist a CrewAI ``Crew`` to the external-agent registry."""
    _require()
    from fastaiagent.integrations._registry import upsert_agent

    agents = getattr(crew, "agents", None) or []
    first_model = None
    if agents:
        first_model = getattr(getattr(agents[0], "llm", None), "model", None)
    provider = _provider_from_model(first_model) if first_model else None
    upsert_agent(
        name,
        framework="crewai",
        model=_bare_model(first_model) if first_model else None,
        provider=provider,
        topology=_crew_topology(crew),
    )


def kb_as_tool(
    kb_name: str,
    *,
    top_k: int = 5,
    description: str | None = None,
    agent: str | None = None,
) -> Any:
    """Wrap a FastAIAgent ``LocalKB`` as a CrewAI ``BaseTool``.

    The returned tool can be passed straight into ``Agent(tools=[...])``;
    invocations route through the named LocalKB's ``search()`` and
    return a Markdown-ish concatenation of chunk content + scores so
    the LLM has something useful to read.
    """
    _require()
    from crewai.tools import BaseTool

    from fastaiagent.kb import LocalKB

    kb = LocalKB(name=kb_name)
    desc = description or (
        f"Search the {kb_name!r} knowledge base for documents relevant to "
        "the input query. Returns up to "
        f"{top_k} chunks with their similarity scores."
    )

    class _FastAIAgentLocalKBTool(BaseTool):
        name: str = f"search_{kb_name}"
        description: str = desc

        def _run(self, query: str) -> str:
            results = kb.search(query, top_k=top_k)
            if not results:
                return f"(no documents found for {query!r})"
            lines: list[str] = []
            for i, r in enumerate(results, start=1):
                content = getattr(r.chunk, "content", "") or ""
                score = float(getattr(r, "score", 0.0))
                lines.append(f"[{i}] (score={score:.3f}) {content}")
            return "\n\n".join(lines)

    if agent:
        try:
            from fastaiagent.integrations._registry import attach as _attach

            _attach(agent, "kb", kb_name)
        except Exception:
            pass

    return _FastAIAgentLocalKBTool()


__all__ = [
    "enable",
    "disable",
    "as_evaluable",
    "with_guardrails",
    "prompt_from_registry",
    "register_agent",
    "kb_as_tool",
]
