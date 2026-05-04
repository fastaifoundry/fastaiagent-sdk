"""Auto-tracing for PydanticAI.

PydanticAI 0.1+ (tested against 1.x) ships its own OpenTelemetry
instrumentation gated by ``Agent.instrument_all()``, and those spans
already use GenAI semantic conventions. So the integration here is
deliberately thin:

1. ``enable()`` calls ``Agent.instrument_all()`` so every PydanticAI
   ``run`` / ``run_sync`` / ``run_stream`` emits its native ``agent run``
   + ``chat <model>`` spans into our ``LocalStorageProcessor``.

2. We wrap ``run`` / ``run_sync`` / ``run_stream`` with a thin OTel
   parent span so the root ``pydanticai.agent.{name}`` span carries the
   ``fastaiagent.framework=pydanticai`` attribute the UI badge / filter
   reads. PydanticAI's native spans nest as children of that root.

3. After the call returns we read ``AgentRunResult.usage()`` and stamp
   ``gen_ai.usage.input_tokens`` / ``output_tokens`` and the computed
   cost on the root for the analytics rollup.

Idempotent: ``Agent.instrument_all()`` is itself idempotent in
PydanticAI; we additionally guard the run-method wrap with the
``_fastaiagent_patched`` sentinel.
"""

from __future__ import annotations

import functools
from typing import Any

_INSTALL_HINT = 'pydantic-ai is required. Install with: pip install "fastaiagent[pydanticai]"'

_enabled = False


def _require() -> None:
    try:
        import pydantic_ai  # noqa: F401
    except ImportError as e:
        raise ImportError(_INSTALL_HINT) from e


def _pa_version() -> str:
    try:
        import pydantic_ai

        return getattr(pydantic_ai, "__version__", "unknown")
    except ImportError:
        return "unknown"


def _patched(fn: Any) -> bool:
    return bool(getattr(fn, "_fastaiagent_patched", False))


def _mark_patched(wrapper: Any, original: Any) -> Any:
    wrapper._fastaiagent_patched = True  # noqa: SLF001
    wrapper._fastaiagent_original = original  # noqa: SLF001
    return wrapper


def _provider_from_model(model_name: str | None) -> str:
    """PydanticAI's ``KnownModelName`` is ``provider:model`` (e.g.
    ``openai:gpt-4o-mini``, ``anthropic:claude-haiku-4-5``)."""
    if not model_name:
        return "unknown"
    if ":" in model_name:
        return model_name.split(":", 1)[0].lower()
    lower = model_name.lower()
    if any(k in lower for k in ("gpt", "o1", "o3")):
        return "openai"
    if "claude" in lower:
        return "anthropic"
    if "gemini" in lower:
        return "google"
    return "unknown"


def _bare_model(model_name: str | None) -> str:
    if not model_name:
        return "unknown"
    return model_name.split(":", 1)[1] if ":" in model_name else model_name


def _model_name(agent: Any) -> str:
    """Best-effort model-name extraction.

    PydanticAI's ``Agent.model`` may be a string, ``KnownModelName``, or
    a ``Model`` instance. Try the common attribute names.
    """
    m = getattr(agent, "model", None)
    if m is None:
        return "unknown"
    if isinstance(m, str):
        return m
    return (
        getattr(m, "model_name", None)
        or getattr(m, "name", None)
        or str(m)
    )


def _agent_name(agent: Any) -> str:
    """Use the user-supplied name if present, otherwise the model id."""
    return (
        getattr(agent, "name", None)
        or getattr(agent, "_name", None)
        or _bare_model(_model_name(agent))
    )


def _stamp_usage_and_cost(span: Any, result: Any) -> None:
    """Pull token counts off ``AgentRunResult.usage()`` and compute cost."""
    from fastaiagent.trace.span import (
        set_fastaiagent_attributes,
        set_genai_attributes,
    )
    from fastaiagent.ui.pricing import compute_cost_usd

    if result is None:
        return
    usage_fn = getattr(result, "usage", None)
    if not callable(usage_fn):
        return
    try:
        usage = usage_fn()
    except Exception:
        return
    if usage is None:
        return

    in_toks = (
        getattr(usage, "input_tokens", None)
        or getattr(usage, "request_tokens", None)
        or getattr(usage, "prompt_tokens", None)
    )
    out_toks = (
        getattr(usage, "output_tokens", None)
        or getattr(usage, "response_tokens", None)
        or getattr(usage, "completion_tokens", None)
    )
    set_genai_attributes(
        span,
        input_tokens=int(in_toks) if in_toks else None,
        output_tokens=int(out_toks) if out_toks else None,
    )
    bare = _bare_model(getattr(span, "_model_name", None) or "")
    cost = compute_cost_usd(bare, in_toks, out_toks)
    if cost is not None:
        set_fastaiagent_attributes(span, **{"cost.total_usd": cost})


def _install_method_patches() -> None:
    from pydantic_ai import Agent

    from fastaiagent.trace.otel import get_tracer
    from fastaiagent.trace.span import (
        set_fastaiagent_attributes,
        set_genai_attributes,
        trace_payloads_enabled,
    )

    tracer = get_tracer("fastaiagent.integrations.pydanticai")

    def _open_root(agent: Any) -> Any:
        name = _agent_name(agent)
        bare = _bare_model(_model_name(agent))
        span = tracer.start_span(f"pydanticai.agent.{name}")
        set_fastaiagent_attributes(
            span,
            framework="pydanticai",
            **{"framework.version": _pa_version()},
        )
        # Stash for cost-calc lookup post-run.
        span._model_name = bare  # noqa: SLF001
        provider = _provider_from_model(_model_name(agent))
        set_genai_attributes(span, system=provider, model=bare)
        if trace_payloads_enabled():
            sysprompts = getattr(agent, "_system_prompts", None) or getattr(
                agent, "system_prompt", None
            )
            if sysprompts:
                preview = (
                    sysprompts
                    if isinstance(sysprompts, str)
                    else " | ".join(str(s) for s in sysprompts)
                )[:200]
                span.set_attribute("pydanticai.agent.system_prompt", preview)
        return span

    if not _patched(Agent.run_sync):
        original_run_sync = Agent.run_sync

        @functools.wraps(original_run_sync)
        def run_sync_wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            from opentelemetry import trace as otel_trace

            span = _open_root(self)
            with otel_trace.use_span(span, end_on_exit=True):
                if trace_payloads_enabled() and args:
                    span.set_attribute(
                        "pydanticai.agent.input", str(args[0])[:1_000]
                    )
                try:
                    result = original_run_sync(self, *args, **kwargs)
                except BaseException as e:
                    span.record_exception(e)
                    raise
                if trace_payloads_enabled():
                    output = getattr(result, "output", None) or getattr(
                        result, "data", None
                    )
                    if output is not None:
                        span.set_attribute(
                            "pydanticai.agent.output", str(output)[:1_000]
                        )
                _stamp_usage_and_cost(span, result)
                return result

        Agent.run_sync = _mark_patched(run_sync_wrapper, original_run_sync)  # type: ignore[method-assign]

    if not _patched(Agent.run):
        original_run = Agent.run

        @functools.wraps(original_run)
        async def run_wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            from opentelemetry import trace as otel_trace

            span = _open_root(self)
            with otel_trace.use_span(span, end_on_exit=True):
                if trace_payloads_enabled() and args:
                    span.set_attribute(
                        "pydanticai.agent.input", str(args[0])[:1_000]
                    )
                try:
                    result = await original_run(self, *args, **kwargs)
                except BaseException as e:
                    span.record_exception(e)
                    raise
                if trace_payloads_enabled():
                    output = getattr(result, "output", None) or getattr(
                        result, "data", None
                    )
                    if output is not None:
                        span.set_attribute(
                            "pydanticai.agent.output", str(output)[:1_000]
                        )
                _stamp_usage_and_cost(span, result)
                return result

        Agent.run = _mark_patched(run_wrapper, original_run)  # type: ignore[method-assign]

    if not _patched(Agent.run_stream):
        original_run_stream = Agent.run_stream

        class _StreamSpanCM:
            """Async-context-manager that wraps PydanticAI's ``run_stream``
            with a parent OTel span tagged ``framework=pydanticai``.

            The wrapped object is what ``Agent.run_stream`` returns —
            itself an async cm — so we open our span on ``__aenter__``,
            delegate the actual streaming to the inner cm, and close the
            span on ``__aexit__`` after stamping usage / cost.
            """

            def __init__(self, agent: Any, args: tuple[Any, ...], kwargs: dict[str, Any]):
                self._agent = agent
                self._args = args
                self._kwargs = kwargs
                self._span: Any | None = None
                self._token: Any | None = None
                self._inner: Any | None = None
                self._run_result: Any | None = None

            async def __aenter__(self) -> Any:
                from opentelemetry import trace as otel_trace

                span = _open_root(self._agent)
                token = otel_trace.use_span(span, end_on_exit=False)
                token.__enter__()
                self._span = span
                self._token = token
                if trace_payloads_enabled() and self._args:
                    span.set_attribute(
                        "pydanticai.agent.input", str(self._args[0])[:1_000]
                    )
                inner = original_run_stream(self._agent, *self._args, **self._kwargs)
                self._inner = inner
                self._run_result = await inner.__aenter__()
                return self._run_result

            async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> Any:
                try:
                    if self._inner is not None:
                        await self._inner.__aexit__(exc_type, exc, tb)
                finally:
                    if self._span is not None:
                        try:
                            if exc is not None:
                                self._span.record_exception(exc)
                            else:
                                _stamp_usage_and_cost(self._span, self._run_result)
                        except Exception:
                            pass
                    if self._token is not None:
                        try:
                            self._token.__exit__(None, None, None)
                        except Exception:
                            pass
                    if self._span is not None:
                        try:
                            self._span.end()
                        except Exception:
                            pass

        @functools.wraps(original_run_stream)
        def run_stream_wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            return _StreamSpanCM(self, args, kwargs)

        Agent.run_stream = _mark_patched(run_stream_wrapper, original_run_stream)  # type: ignore[method-assign]


class _EvaluableResult:
    """Duck-typed object so ``fa.evaluate()`` reads ``.output`` /
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
    agent: Any,
    *,
    input_mapper: Any = None,
    output_mapper: Any = None,
) -> Any:
    """Adapt a PydanticAI ``Agent`` for ``fastaiagent.evaluate(...)``.

    Returns an *async* callable so ``fa.evaluate(...)`` can ``await``
    it — PydanticAI's ``run_sync`` cannot be invoked from inside an
    already-running event loop, and ``evaluate()`` runs its cases
    under ``asyncio.gather``.

    Opens an outer ``eval.case`` span so we can capture ``trace_id``
    *while* it is still active — PydanticAI's per-run span has already
    closed by the time ``run`` returns.
    """
    _require()
    from fastaiagent.trace.otel import get_tracer

    in_map = input_mapper or (lambda s: s)
    out_map = output_mapper or (
        lambda r: getattr(r, "output", None) or getattr(r, "data", None) or str(r)
    )
    tracer = get_tracer("fastaiagent.integrations.pydanticai")

    async def _evaluable(text: str) -> _EvaluableResult:
        with tracer.start_as_current_span("eval.case"):
            result = await agent.run(in_map(text))
            return _EvaluableResult(
                output=str(out_map(result)), trace_id=_current_trace_id()
            )

    return _evaluable


def enable() -> None:
    """Enable PydanticAI auto-tracing.

    Idempotent — calling twice is a no-op (each patched method carries
    a ``_fastaiagent_patched`` sentinel; ``Agent.instrument_all`` is
    itself idempotent).
    """
    global _enabled
    if _enabled:
        return
    _require()
    from pydantic_ai import Agent

    try:
        Agent.instrument_all()
    except Exception:
        # If the instrument-all surface ever changes, fall back to our
        # wrapper-only spans — they still carry framework + cost; just
        # without PydanticAI's per-message detail.
        pass
    _install_method_patches()
    _enabled = True


def disable() -> None:
    """Disable PydanticAI auto-tracing — restores the run-method patches.

    PydanticAI's internal ``instrument_all`` registration cannot be
    undone (no public unsubscribe), but with our wrapper gone the
    spans no longer get the ``fastaiagent.framework`` tag.
    """
    global _enabled
    _enabled = False
    try:
        from pydantic_ai import Agent
    except ImportError:
        return
    for name in ("run", "run_sync", "run_stream"):
        fn = getattr(Agent, name, None)
        original = getattr(fn, "_fastaiagent_original", None)
        if original is not None:
            setattr(Agent, name, original)


def _extract_output_text(result: Any) -> str:
    """Pull text out of a PydanticAI ``AgentRunResult``."""
    return str(
        getattr(result, "output", None)
        or getattr(result, "data", None)
        or result
    )


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
                merged.setdefault("framework", "pydanticai")
                merged.setdefault("side", side)
                result.metadata = merged
                log_guardrail_event(g, result, agent_name=agent_name)
            except Exception:
                pass
            raise GuardrailBlocked(
                f"{side} blocked by {g.name}: {result.message or ''}"
            )


class _GuardedAgent:
    """Proxy around a PydanticAI ``Agent`` with input/output guardrails
    layered on top of ``run`` / ``run_sync`` / ``run_stream``."""

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

    def _check_input(self, user_prompt: Any) -> None:
        text = user_prompt if isinstance(user_prompt, str) else str(user_prompt)
        _run_guardrails(
            text,
            self._input_guardrails,
            side="input",
            agent_name=self._fastaiagent_name,
        )

    def _check_output(self, result: Any) -> None:
        _run_guardrails(
            _extract_output_text(result),
            self._output_guardrails,
            side="output",
            agent_name=self._fastaiagent_name,
        )

    def run_sync(self, user_prompt: Any = None, **kwargs: Any) -> Any:
        self._check_input(user_prompt)
        result = self._wrapped.run_sync(user_prompt, **kwargs)
        self._check_output(result)
        return result

    async def run(self, user_prompt: Any = None, **kwargs: Any) -> Any:
        self._check_input(user_prompt)
        result = await self._wrapped.run(user_prompt, **kwargs)
        self._check_output(result)
        return result

    def run_stream(self, user_prompt: Any = None, **kwargs: Any) -> Any:
        # Output guardrails for streaming have to wait for full output;
        # input guardrails fire before the inner cm opens, matching the
        # spec's "streaming with input guardrails only is the no-latency
        # path" guidance.
        self._check_input(user_prompt)
        return self._wrapped.run_stream(user_prompt, **kwargs)


def with_guardrails(
    agent: Any,
    *,
    name: str | None = None,
    input_guardrails: list[Any] | None = None,
    output_guardrails: list[Any] | None = None,
) -> _GuardedAgent:
    """Wrap a PydanticAI ``Agent`` with input/output guardrails.

    Block-only semantics: a failing blocking guardrail logs a
    ``guardrail_events`` row tagged ``framework=pydanticai`` and raises
    :class:`fastaiagent.integrations._registry.GuardrailBlocked`.
    """
    _require()
    return _GuardedAgent(
        agent,
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
    """Return a PydanticAI-friendly prompt string from the FastAIAgent
    registry, ready to pass as ``Agent(..., system_prompt=...)``.

    The raw template is returned so the caller can either use it
    verbatim (no placeholders) or substitute variables themselves via
    ``PromptRegistry().get(slug).format(**kw)``.
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


def register_agent(agent: Any, *, name: str) -> None:
    """Persist a PydanticAI ``Agent`` to the external-agent registry.

    PydanticAI agents are single-agent (no graph topology), so we
    capture model + tools only — the dependency-graph endpoint shows
    the harness layers (guardrails / KBs / prompts) plus the model
    badge, no workflow visualization.
    """
    _require()
    from fastaiagent.integrations._registry import upsert_agent

    model = _model_name(agent)
    provider = _provider_from_model(model)
    bare = _bare_model(model)

    tools_meta: list[dict[str, Any]] = []
    raw_tools = getattr(agent, "_function_tools", None) or {}
    if isinstance(raw_tools, dict):
        for tool_name, tool in raw_tools.items():
            tools_meta.append(
                {
                    "id": str(tool_name),
                    "type": "tool",
                    "description": str(getattr(tool, "description", ""))[:200],
                }
            )

    sysprompts = (
        getattr(agent, "_system_prompts", None)
        or getattr(agent, "system_prompt", None)
        or None
    )
    sysprompt_str: str | None = None
    if sysprompts:
        sysprompt_str = (
            sysprompts
            if isinstance(sysprompts, str)
            else " | ".join(str(s) for s in sysprompts)
        )[:1_000]

    upsert_agent(
        name,
        framework="pydanticai",
        model=bare,
        provider=provider,
        system_prompt=sysprompt_str,
        topology={"nodes": tools_meta, "edges": []},
    )


def kb_as_tool(
    kb_name: str,
    *,
    top_k: int = 5,
    agent: str | None = None,
) -> Any:
    """Return a plain function that can be passed into PydanticAI's
    ``Agent(tools=[...])`` (or used with ``@agent.tool_plain``).

    Signature: ``search_<kb_name>(query: str) -> str``. The function
    wraps a ``LocalKB.search()`` call and returns a Markdown-ish
    response so the LLM can quote / reason over the retrieved chunks.
    """
    _require()
    from fastaiagent.kb import LocalKB

    kb = LocalKB(name=kb_name)
    fn_name = f"search_{kb_name}"

    def _search_fn(query: str) -> str:
        """Search the named knowledge base and return up to top_k chunks."""
        results = kb.search(query, top_k=top_k)
        if not results:
            return f"(no documents found for {query!r})"
        lines: list[str] = []
        for i, r in enumerate(results, start=1):
            content = getattr(r.chunk, "content", "") or ""
            score = float(getattr(r, "score", 0.0))
            lines.append(f"[{i}] (score={score:.3f}) {content}")
        return "\n\n".join(lines)

    _search_fn.__name__ = fn_name
    _search_fn.__doc__ = (
        f"Search the {kb_name!r} knowledge base for documents relevant to "
        "the input query."
    )

    if agent:
        try:
            from fastaiagent.integrations._registry import attach as _attach

            _attach(agent, "kb", kb_name)
        except Exception:
            pass

    return _search_fn


__all__ = [
    "enable",
    "disable",
    "as_evaluable",
    "with_guardrails",
    "prompt_from_registry",
    "register_agent",
    "kb_as_tool",
]
