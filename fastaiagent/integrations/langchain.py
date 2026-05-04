"""Auto-tracing for LangChain / LangGraph via callback handler.

Public API
----------
``enable()`` registers a global ``BaseCallbackHandler`` that translates
LangChain's callback events into FastAIAgent OTel spans. Every chain /
graph / LLM / tool / retriever event becomes a span in the same local
trace store the native Local UI already reads from, with full input /
output payloads, token usage, computed cost, and a
``fastaiagent.framework=langchain`` attribute on the root span so the
UI can render the LC badge and filter on it.

The handler's design keeps a ``run_id -> Span`` map so that LangChain's
``parent_run_id`` / ``run_id`` UUIDs round-trip properly into our span
parent-child relationship — fixing the prior implementation's bug where
spans were stashed in ``**kwargs`` (which Python does not pass back) and
therefore leaked open.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Any
from uuid import UUID

if TYPE_CHECKING:  # pragma: no cover - hint-only imports
    from langchain_core.callbacks import BaseCallbackHandler

_INSTALL_HINT = (
    'langchain-core / langgraph are required. Install with: pip install "fastaiagent[langchain]"'
)
_PAYLOAD_TRUNC = 10_000  # 10KB cap per spec for tool inputs/outputs

# Module-level state — all guarded by the idempotency check in ``enable()``.
_enabled = False
_handler_singleton: BaseCallbackHandler | None = None

# Lineage tracking: ``_TrackedTemplate.format_messages`` pushes
# ``(slug, version)`` onto a per-thread LIFO stack right before
# LangChain dispatches to the LLM, and the callback handler's
# ``on_*_start`` methods pop the entry to stamp lineage attributes on
# the LLM span. We use a thread-local stack rather than a ContextVar
# because LCEL isolates each step with ``copy_context().run(...)`` —
# values set inside ``format_messages`` don't survive into
# ``on_chat_model_start``. Concurrency caveat: parallel chains in the
# same thread can race; documented in the integration guide.
_lineage_state = threading.local()


def _push_prompt_lineage(slug: str, version: int) -> None:
    stack: list[tuple[str, int]] = getattr(_lineage_state, "stack", [])
    stack.append((slug, version))
    _lineage_state.stack = stack


def _pop_prompt_lineage() -> tuple[str, int] | None:
    stack: list[tuple[str, int]] = getattr(_lineage_state, "stack", [])
    if not stack:
        return None
    value = stack.pop()
    _lineage_state.stack = stack
    return value


def _require() -> None:
    try:
        import langchain_core.callbacks  # noqa: F401
    except ImportError as e:
        raise ImportError(_INSTALL_HINT) from e


def _lc_version() -> str:
    try:
        import langchain_core

        return getattr(langchain_core, "__version__", "unknown")
    except ImportError:
        return "unknown"


def _safe_json(obj: Any, *, limit: int = _PAYLOAD_TRUNC) -> str:
    """JSON-serialize ``obj`` with a size cap.

    LangChain inputs/outputs include domain objects (``HumanMessage``,
    ``AIMessage``, ``Document``) that are not JSON-native. Falls back to
    ``str()`` per object via ``default=`` and trims the result so a
    pathological 50MB blob never lands in a span attribute.
    """
    try:
        text = json.dumps(obj, default=_json_default, ensure_ascii=False)
    except Exception:
        text = str(obj)
    if len(text) > limit:
        return text[:limit] + f"…[+{len(text) - limit}B]"
    return text


def _json_default(o: Any) -> Any:
    # LangChain message objects expose ``.content`` and ``.type``; surface them
    # rather than the opaque ``BaseMessage(content=…)`` repr.
    if hasattr(o, "content") and hasattr(o, "type"):
        return {"type": o.type, "content": o.content}
    if hasattr(o, "page_content"):
        return {"page_content": o.page_content, "metadata": getattr(o, "metadata", {})}
    if hasattr(o, "model_dump"):
        try:
            return o.model_dump()
        except Exception:
            return str(o)
    return str(o)


def _is_langgraph(serialized: dict[str, Any] | None) -> bool:
    """Heuristic: LangGraph compiles its graphs into LCEL chains whose
    serialized id path includes ``langgraph``. We use this to pick the
    span-name prefix (``langgraph.{name}`` vs ``langchain.{name}``).
    """
    if not serialized:
        return False
    ids = serialized.get("id") or []
    return any("langgraph" in str(part).lower() for part in ids)


def _extract_messages_payload(prompts: list[Any] | None, messages: Any) -> Any:
    """Normalize whatever LangChain hands us into a JSON-serializable
    structure for the ``gen_ai.request.messages`` attribute.

    ``on_llm_start`` passes ``prompts: list[str]``; ``on_chat_model_start``
    passes ``messages: list[list[BaseMessage]]``. We return the richer
    of the two when both are present.
    """
    if messages:
        return messages
    return prompts or []


def _extract_token_usage(response: Any) -> tuple[int | None, int | None]:
    """Best-effort token extraction from an LLMResult.

    LangChain providers populate token usage in different places —
    ``llm_output['token_usage']`` for OpenAI / many wrappers, and
    ``generation.generation_info['usage_metadata']`` (or ``.usage_metadata``
    on the message itself) for newer chat-model providers including
    Anthropic. We try all of them and return the first hit.
    """
    in_toks: int | None = None
    out_toks: int | None = None

    llm_output = getattr(response, "llm_output", None) or {}
    usage = llm_output.get("token_usage") or llm_output.get("usage") or {}
    in_toks = (
        usage.get("prompt_tokens")
        or usage.get("input_tokens")
        or usage.get("prompt_token_count")
    )
    out_toks = (
        usage.get("completion_tokens")
        or usage.get("output_tokens")
        or usage.get("completion_token_count")
    )

    if in_toks is None or out_toks is None:
        # Newer LC chat models attach usage to the AIMessage directly.
        for gen_list in getattr(response, "generations", []) or []:
            for gen in gen_list:
                msg = getattr(gen, "message", None)
                meta = getattr(msg, "usage_metadata", None) if msg else None
                if meta:
                    in_toks = in_toks or meta.get("input_tokens")
                    out_toks = out_toks or meta.get("output_tokens")
                gi = getattr(gen, "generation_info", None) or {}
                gi_usage = (
                    gi.get("usage_metadata")
                    or gi.get("token_usage")
                    or gi.get("usage")
                    or {}
                )
                in_toks = in_toks or gi_usage.get("input_tokens") or gi_usage.get("prompt_tokens")
                out_toks = (
                    out_toks
                    or gi_usage.get("output_tokens")
                    or gi_usage.get("completion_tokens")
                )
                if in_toks and out_toks:
                    break
            if in_toks and out_toks:
                break

    return (int(in_toks) if in_toks else None, int(out_toks) if out_toks else None)


def _extract_response_text(response: Any) -> str:
    """Pull a flat text string out of an LLMResult."""
    chunks: list[str] = []
    for gen_list in getattr(response, "generations", []) or []:
        for gen in gen_list:
            text = getattr(gen, "text", None)
            if text:
                chunks.append(text)
                continue
            msg = getattr(gen, "message", None)
            if msg is not None and hasattr(msg, "content"):
                content = msg.content
                chunks.append(content if isinstance(content, str) else str(content))
    return "\n".join(chunks)


def _model_name_from_serialized(serialized: dict[str, Any] | None) -> str:
    if not serialized:
        return "unknown"
    kwargs = serialized.get("kwargs") or {}
    return (
        kwargs.get("model")
        or kwargs.get("model_name")
        or kwargs.get("deployment_name")
        or serialized.get("name")
        or "unknown"
    )


def _provider_from_serialized(serialized: dict[str, Any] | None) -> str:
    """Best-effort provider inference from the serialized ``id`` path.

    LangChain stores ``id: ['langchain', 'chat_models', 'openai', 'ChatOpenAI']``
    or similar. The third element is typically the provider name.
    """
    if not serialized:
        return "unknown"
    ids = serialized.get("id") or []
    parts = [str(p).lower() for p in ids]
    for known in ("openai", "anthropic", "azure", "google", "mistral", "cohere", "ollama"):
        if any(known in p for p in parts):
            return known
    return parts[2] if len(parts) >= 3 else "unknown"


def _build_handler() -> BaseCallbackHandler:
    """Build the actual callback class. Done lazily so importing this
    module does not require ``langchain-core``."""
    _require()
    from langchain_core.callbacks import BaseCallbackHandler

    from fastaiagent.trace.otel import get_tracer
    from fastaiagent.trace.span import (
        set_fastaiagent_attributes,
        set_genai_attributes,
        trace_payloads_enabled,
    )
    from fastaiagent.ui.pricing import compute_cost_usd

    class FastAIAgentCallbackHandler(BaseCallbackHandler):
        """Translates LangChain/LangGraph callback events to OTel spans.

        Per-instance state holds open spans keyed by LangChain ``run_id``,
        so that nested chain/LLM/tool events nest under the right parent
        in the trace tree. ``BaseCallbackHandler``'s ``ignore_*`` flags
        already default to ``False``, so we receive every event without
        further configuration.
        """

        def __init__(self) -> None:
            super().__init__()
            self._runs: dict[UUID, Any] = {}

        # -- lifecycle helpers ------------------------------------------------
        def _start(
            self, run_id: UUID, name: str, parent_run_id: UUID | None = None
        ) -> Any:
            from opentelemetry import trace as otel_trace

            tracer = get_tracer("fastaiagent.integrations.langchain")
            ctx = None
            if parent_run_id is not None:
                parent_span = self._runs.get(parent_run_id)
                if parent_span is not None:
                    ctx = otel_trace.set_span_in_context(parent_span)
            span = tracer.start_span(name, context=ctx)
            self._runs[run_id] = span
            return span

        def _end(self, run_id: UUID) -> None:
            span = self._runs.pop(run_id, None)
            if span is None:
                return
            try:
                span.end()
            except Exception:
                pass

        def _record_error(self, run_id: UUID, error: BaseException) -> None:
            from opentelemetry.trace import Status, StatusCode

            span = self._runs.get(run_id)
            if span is None:
                return
            try:
                span.record_exception(error)
                span.set_status(Status(StatusCode.ERROR, str(error)))
            except Exception:
                pass

        # -- chain / graph ----------------------------------------------------
        def on_chain_start(
            self,
            serialized: dict[str, Any] | None,
            inputs: dict[str, Any] | Any,
            *,
            run_id: UUID,
            parent_run_id: UUID | None = None,
            **kwargs: Any,
        ) -> None:
            is_root = parent_run_id is None
            prefix = "langgraph" if _is_langgraph(serialized) else "langchain"
            name_part = (
                (serialized or {}).get("name")
                or ((serialized or {}).get("id") or ["chain"])[-1]
            )
            span_name = (
                f"{prefix}.{name_part}" if is_root else f"node.{name_part}"
            )
            span = self._start(run_id, span_name, parent_run_id=parent_run_id)
            if is_root:
                set_fastaiagent_attributes(
                    span,
                    framework="langchain",
                    **{"framework.version": _lc_version()},
                )
            if trace_payloads_enabled():
                span.set_attribute("input", _safe_json(inputs))

        def on_chain_end(
            self,
            outputs: dict[str, Any] | Any,
            *,
            run_id: UUID,
            **kwargs: Any,
        ) -> None:
            span = self._runs.get(run_id)
            if span is not None and trace_payloads_enabled():
                span.set_attribute("output", _safe_json(outputs))
            self._end(run_id)

        def on_chain_error(
            self,
            error: BaseException,
            *,
            run_id: UUID,
            **kwargs: Any,
        ) -> None:
            self._record_error(run_id, error)
            self._end(run_id)

        # -- LLM / chat model -------------------------------------------------
        def _llm_start(
            self,
            serialized: dict[str, Any] | None,
            payload: Any,
            *,
            run_id: UUID,
            parent_run_id: UUID | None,
            invocation_params: dict[str, Any] | None,
        ) -> None:
            model = _model_name_from_serialized(serialized)
            provider = _provider_from_serialized(serialized)
            span = self._start(
                run_id, f"llm.{provider}.{model}", parent_run_id=parent_run_id
            )
            inv = invocation_params or {}
            set_genai_attributes(
                span,
                system=provider,
                model=model,
                temperature=inv.get("temperature"),
                max_tokens=inv.get("max_tokens"),
                request_messages=_safe_json(payload),
            )
            # Lineage: if a registry-backed template was just rendered,
            # tag the LLM span so the Prompt detail page can find it.
            current = _pop_prompt_lineage()
            if current is not None:
                slug, version = current
                set_fastaiagent_attributes(
                    span,
                    **{"prompt.slug": slug, "prompt.version": int(version)},
                )

        def on_llm_start(
            self,
            serialized: dict[str, Any] | None,
            prompts: list[str],
            *,
            run_id: UUID,
            parent_run_id: UUID | None = None,
            invocation_params: dict[str, Any] | None = None,
            **kwargs: Any,
        ) -> None:
            self._llm_start(
                serialized,
                _extract_messages_payload(prompts, None),
                run_id=run_id,
                parent_run_id=parent_run_id,
                invocation_params=invocation_params,
            )

        def on_chat_model_start(
            self,
            serialized: dict[str, Any] | None,
            messages: list[list[Any]],
            *,
            run_id: UUID,
            parent_run_id: UUID | None = None,
            invocation_params: dict[str, Any] | None = None,
            **kwargs: Any,
        ) -> None:
            self._llm_start(
                serialized,
                _extract_messages_payload(None, messages),
                run_id=run_id,
                parent_run_id=parent_run_id,
                invocation_params=invocation_params,
            )

        def on_llm_end(
            self,
            response: Any,
            *,
            run_id: UUID,
            **kwargs: Any,
        ) -> None:
            span = self._runs.get(run_id)
            if span is None:
                return
            in_toks, out_toks = _extract_token_usage(response)
            response_text = _extract_response_text(response)
            # Pull model from response if the start callback couldn't.
            llm_output = getattr(response, "llm_output", None) or {}
            model = llm_output.get("model_name") or llm_output.get("model")
            set_genai_attributes(
                span,
                model=model,
                input_tokens=in_toks,
                output_tokens=out_toks,
                response_content=response_text,
            )
            cost = compute_cost_usd(model, in_toks, out_toks)
            if cost is not None:
                set_fastaiagent_attributes(span, **{"cost.total_usd": cost})
            self._end(run_id)

        def on_llm_error(
            self,
            error: BaseException,
            *,
            run_id: UUID,
            **kwargs: Any,
        ) -> None:
            self._record_error(run_id, error)
            self._end(run_id)

        # -- tool -------------------------------------------------------------
        def on_tool_start(
            self,
            serialized: dict[str, Any] | None,
            input_str: str,
            *,
            run_id: UUID,
            parent_run_id: UUID | None = None,
            inputs: dict[str, Any] | None = None,
            **kwargs: Any,
        ) -> None:
            tool_name = (serialized or {}).get("name", "tool")
            span = self._start(run_id, f"tool.{tool_name}", parent_run_id=parent_run_id)
            set_fastaiagent_attributes(span, **{"tool.name": tool_name})
            if trace_payloads_enabled():
                payload = inputs if inputs is not None else input_str
                span.set_attribute("tool.input", _safe_json(payload))

        def on_tool_end(
            self,
            output: Any,
            *,
            run_id: UUID,
            **kwargs: Any,
        ) -> None:
            span = self._runs.get(run_id)
            if span is not None and trace_payloads_enabled():
                span.set_attribute("tool.output", _safe_json(output))
            self._end(run_id)

        def on_tool_error(
            self,
            error: BaseException,
            *,
            run_id: UUID,
            **kwargs: Any,
        ) -> None:
            self._record_error(run_id, error)
            self._end(run_id)

        # -- retriever --------------------------------------------------------
        def on_retriever_start(
            self,
            serialized: dict[str, Any] | None,
            query: str,
            *,
            run_id: UUID,
            parent_run_id: UUID | None = None,
            **kwargs: Any,
        ) -> None:
            retriever_name = (serialized or {}).get("name", "retriever")
            span = self._start(
                run_id, f"retrieval.{retriever_name}", parent_run_id=parent_run_id
            )
            top_k = (serialized or {}).get("kwargs", {}).get("k") or (
                serialized or {}
            ).get("kwargs", {}).get("top_k")
            if top_k is not None:
                span.set_attribute("retrieval.top_k", int(top_k))
            if trace_payloads_enabled():
                span.set_attribute("retrieval.query", str(query))

        def on_retriever_end(
            self,
            documents: Any,
            *,
            run_id: UUID,
            **kwargs: Any,
        ) -> None:
            span = self._runs.get(run_id)
            if span is not None:
                span.set_attribute("retrieval.document_count", len(documents or []))
                if trace_payloads_enabled():
                    preview = [
                        {
                            "content": (getattr(d, "page_content", "") or "")[:200],
                            "metadata": getattr(d, "metadata", {}) or {},
                        }
                        for d in (documents or [])[:5]
                    ]
                    span.set_attribute("retrieval.documents", _safe_json(preview))
            self._end(run_id)

        def on_retriever_error(
            self,
            error: BaseException,
            *,
            run_id: UUID,
            **kwargs: Any,
        ) -> None:
            self._record_error(run_id, error)
            self._end(run_id)

    return FastAIAgentCallbackHandler()


def get_callback_handler() -> Any:
    """Return the FastAIAgent LangChain callback handler singleton.

    Build-once-per-process — repeated calls return the same instance so
    that ``enable()`` is idempotent and callers passing the handler
    explicitly into a ``RunnableConfig`` see the same span-state map as
    the globally registered one.
    """
    global _handler_singleton
    if _handler_singleton is None:
        _handler_singleton = _build_handler()
    return _handler_singleton


class _EvaluableResult:
    """Tiny wrapper so ``fa.evaluate()`` picks up the trace_id linkage.

    ``evaluate.py:122-126`` already reads ``.output`` and ``.trace_id``
    from the agent function's return — so all we need is a duck-typed
    object that exposes both.
    """

    __slots__ = ("output", "trace_id")

    def __init__(self, output: str, trace_id: str | None = None) -> None:
        self.output = output
        self.trace_id = trace_id


def _current_trace_id() -> str | None:
    """Format the current OTel trace id as 32-hex, or ``None`` if no span."""
    from opentelemetry import trace as otel_trace

    span = otel_trace.get_current_span()
    if span is None:
        return None
    ctx = span.get_span_context()
    if ctx is None or not ctx.trace_id:
        return None
    return format(ctx.trace_id, "032x")


def _default_input_mapper(text: str) -> dict[str, Any]:
    """LangGraph default: wrap a plain string into MessagesState shape."""
    from langchain_core.messages import HumanMessage

    return {"messages": [HumanMessage(content=text)]}


_OUTPUT_KEYS = ("response", "output", "answer", "result", "text", "content")


def _default_output_mapper(result: Any) -> str:
    """Best-effort extraction of a flat string from a graph result.

    For ``MessagesState`` we take the last message's ``content``. For
    custom states we look for a known output key. As a last resort we
    stringify the whole thing.
    """
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        msgs = result.get("messages")
        if msgs:
            last = msgs[-1]
            if hasattr(last, "content"):
                content = last.content
                return content if isinstance(content, str) else str(content)
        for key in _OUTPUT_KEYS:
            if key in result and result[key] is not None:
                value = result[key]
                return value if isinstance(value, str) else str(value)
    raw_result: Any = result
    if hasattr(raw_result, "content"):
        return str(raw_result.content)
    return str(raw_result)


def as_evaluable(
    graph_or_chain: Any,
    *,
    input_mapper: Callable[[str], Any] | None = None,
    output_mapper: Callable[[Any], str] | None = None,
) -> Callable[[str], _EvaluableResult]:
    """Adapt a LangGraph compiled graph (or a LangChain Runnable) for
    use with ``fastaiagent.evaluate(...)``.

    Opens an outer ``eval.case`` span so we can capture ``trace_id``
    *while* it is still active — the LangChain handler's per-run spans
    have already closed by the time ``invoke`` returns.
    """
    _require()
    from fastaiagent.trace.otel import get_tracer

    handler = get_callback_handler()
    in_map = input_mapper or _default_input_mapper
    out_map = output_mapper or _default_output_mapper
    tracer = get_tracer("fastaiagent.integrations.langchain")

    def _evaluable(text: str) -> _EvaluableResult:
        graph_input = in_map(text)
        with tracer.start_as_current_span("eval.case"):
            result = graph_or_chain.invoke(
                graph_input, config={"callbacks": [handler]}
            )
            return _EvaluableResult(
                output=out_map(result), trace_id=_current_trace_id()
            )

    return _evaluable


def _register_global_handler(handler: Any) -> None:
    """Hook the handler into LangChain's global callback registry.

    LangChain ≥ 0.3 exposes ``register_configure_hook`` in
    ``langchain_core.tracers.context``. The hook is consulted for every
    ``RunnableConfig`` that flows through the system, so registering a
    ContextVar that defaults to our handler injects it into
    ``config["callbacks"]`` without the user passing it explicitly.

    LangChain does not expose a public ``unregister`` counterpart, so
    ``disable()`` flips a flag instead of removing the registration —
    but this is benign because span lifecycle still respects ``_enabled``
    via the singleton handler's idempotency.
    """
    import contextvars

    from langchain_core.tracers.context import register_configure_hook

    cv: contextvars.ContextVar[Any] = contextvars.ContextVar(
        "_fastaiagent_lc_handler", default=handler
    )
    register_configure_hook(cv, inheritable=True)


def enable() -> None:
    """Enable LangChain/LangGraph auto-tracing.

    Idempotent — calling twice is a no-op (no double-registered handlers,
    no duplicated spans).
    """
    global _enabled
    if _enabled:
        return
    _require()
    handler = get_callback_handler()
    try:
        _register_global_handler(handler)
    except Exception:
        # If the configure-hook surface ever changes, fall back silently —
        # the user can still pass ``get_callback_handler()`` explicitly
        # into their ``RunnableConfig``'s ``callbacks=`` list.
        pass
    _enabled = True


def disable() -> None:
    """Disable LangChain/LangGraph auto-tracing.

    LangChain ≥ 0.3 has no public way to unregister a configure-hook,
    so we flip the module flag — subsequent ``enable()`` calls remain
    idempotent and the singleton handler's spans only fire when callers
    re-pass it explicitly. Use process restart for a clean reset.
    """
    global _enabled
    _enabled = False


def _extract_input_text(graph_input: Any) -> str:
    """Pull the text content out of a LangGraph/LangChain invocation input.

    Recognises (a) plain strings, (b) ``MessagesState``-style dicts where
    we read the last ``HumanMessage.content``, and (c) custom-state
    dicts where we look for common keys (``input``, ``query``).
    """
    if isinstance(graph_input, str):
        return graph_input
    if isinstance(graph_input, dict):
        msgs: Any = graph_input.get("messages")
        if msgs:
            last: Any = msgs[-1]
            content = getattr(last, "content", None)
            if isinstance(content, str):
                return content
            if content is not None:
                return str(content)
        for key in ("input", "query", "question", "text"):
            if key in graph_input and graph_input[key] is not None:
                value = graph_input[key]
                return value if isinstance(value, str) else str(value)
    return str(graph_input)


def _extract_output_text(result: Any) -> str:
    """Inverse of ``_extract_input_text`` — see ``_default_output_mapper``."""
    return _default_output_mapper(result)


def _run_guardrails(
    text: str,
    guardrails: list[Any] | None,
    *,
    side: str,
    agent_name: str | None,
) -> None:
    """Block-only guardrail loop (decision A in harness.md).

    For each guardrail, calls ``g.execute(text)``. If it fails AND the
    guardrail is blocking, we log the event with ``framework`` tagged
    on it and raise ``GuardrailBlocked``. Filtering/redaction is not
    supported (the SDK's ``GuardrailResult`` has no ``filtered_text``);
    documented in ``docs/integrations/overview.md``.
    """
    if not guardrails:
        return

    from fastaiagent.integrations._registry import GuardrailBlocked

    for g in guardrails:
        result = g.execute(text)
        if not result.passed and getattr(g, "blocking", True):
            try:
                from fastaiagent.ui.events import log_guardrail_event

                # Stamp framework on the metadata so the UI can show a
                # framework badge on the guardrail event row.
                merged_metadata = dict(result.metadata or {})
                merged_metadata.setdefault("framework", "langchain")
                merged_metadata.setdefault("side", side)
                result.metadata = merged_metadata
                log_guardrail_event(g, result, agent_name=agent_name)
            except Exception:
                # Logging is best-effort; never fail a guardrail check
                # because the event store hiccupped.
                pass
            raise GuardrailBlocked(
                f"{side} blocked by {g.name}: {result.message or ''}"
            )


class _GuardedRunnable:
    """Proxy that wraps a LangChain ``Runnable`` with input / output
    guardrails. Forwards every other attribute access to the wrapped
    object so consumers see the original interface.
    """

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
        # ``__getattr__`` only fires for misses, so methods we override
        # below take precedence.
        return getattr(self._wrapped, item)

    def _check_input(self, graph_input: Any) -> None:
        _run_guardrails(
            _extract_input_text(graph_input),
            self._input_guardrails,
            side="input",
            agent_name=self._fastaiagent_name,
        )

    def _check_output(self, output: Any) -> None:
        _run_guardrails(
            _extract_output_text(output),
            self._output_guardrails,
            side="output",
            agent_name=self._fastaiagent_name,
        )

    def invoke(self, input: Any, *args: Any, **kwargs: Any) -> Any:
        self._check_input(input)
        out = self._wrapped.invoke(input, *args, **kwargs)
        self._check_output(out)
        return out

    async def ainvoke(self, input: Any, *args: Any, **kwargs: Any) -> Any:
        self._check_input(input)
        out = await self._wrapped.ainvoke(input, *args, **kwargs)
        self._check_output(out)
        return out

    def stream(self, input: Any, *args: Any, **kwargs: Any) -> Any:
        # Output guardrails buffer the stream until completion (decision
        # in harness.md — input-only stream is the no-latency path).
        self._check_input(input)
        chunks: list[Any] = []
        for chunk in self._wrapped.stream(input, *args, **kwargs):
            chunks.append(chunk)
            yield chunk
        if self._output_guardrails:
            self._check_output(chunks[-1] if chunks else "")

    async def astream(self, input: Any, *args: Any, **kwargs: Any) -> Any:
        self._check_input(input)
        chunks: list[Any] = []
        async for chunk in self._wrapped.astream(input, *args, **kwargs):
            chunks.append(chunk)
            yield chunk
        if self._output_guardrails:
            self._check_output(chunks[-1] if chunks else "")

    def batch(self, inputs: list[Any], *args: Any, **kwargs: Any) -> list[Any]:
        for i in inputs:
            self._check_input(i)
        outs: list[Any] = self._wrapped.batch(inputs, *args, **kwargs)
        for o in outs:
            self._check_output(o)
        return outs


def with_guardrails(
    agent: Any,
    *,
    name: str | None = None,
    input_guardrails: list[Any] | None = None,
    output_guardrails: list[Any] | None = None,
) -> _GuardedRunnable:
    """Wrap a LangChain ``Runnable`` (LangGraph compiled graph,
    LCEL chain, etc.) with FastAIAgent input/output guardrails.

    Block-only semantics: a failing blocking guardrail raises
    :class:`fastaiagent.integrations._registry.GuardrailBlocked` *and*
    writes a row to the ``guardrail_events`` store the Local UI reads.
    """
    _require()
    return _GuardedRunnable(
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
) -> Any:
    """Return a ``ChatPromptTemplate`` backed by the FastAIAgent prompt
    registry.

    The template uses ``mustache`` placeholder syntax so the registry's
    ``{{var}}`` markup round-trips into LangChain's
    ``ChatPromptTemplate.from_template`` without translation.

    Lineage: when the template is rendered (``format_messages``), we
    stamp ``fastaiagent.prompt.slug`` and ``fastaiagent.prompt.version``
    on the current OTel span so the Prompt detail page's "Traces using
    this prompt" panel can find them.
    """
    _require()
    from langchain_core.prompts import ChatPromptTemplate

    from fastaiagent.prompt import PromptRegistry

    resolved_version: int | None = None if version in (None, "latest") else int(version)

    reg = PromptRegistry()
    prompt = reg.get(slug, version=resolved_version)

    base = ChatPromptTemplate.from_template(prompt.template, template_format="mustache")

    # Subclass at runtime so we can override ``format_messages`` without
    # touching the public class.
    class _TrackedTemplate(type(base)):  # type: ignore[misc]
        def format_messages(self, **kwargs: Any) -> Any:
            # Push slug + version onto the per-thread lineage stack so
            # the next ``on_*_start`` callback can stamp them on the
            # LLM span. We can't use a ContextVar here because LCEL
            # isolates each step with ``copy_context().run(...)``, and
            # we can't tag the current OTel span because the handler
            # uses ``start_span`` (not ``start_as_current_span``).
            _push_prompt_lineage(slug, int(prompt.version))
            return super().format_messages(**kwargs)

    tracked = _TrackedTemplate(
        messages=base.messages, input_variables=list(base.input_variables)
    )

    if agent:
        # Auto-attach to the external_agent_attachments table so the
        # dependency graph picks it up (Phase 8 wires this — until then
        # the call no-ops because the helper module isn't installed).
        try:
            from fastaiagent.integrations._registry import attach as _attach

            _attach(agent, "prompt", slug, version=str(prompt.version))
        except Exception:
            pass

    return tracked


def _classify_node_data(data: Any) -> str:
    """Best-effort node classifier for ``register_agent`` topology
    extraction. Falls back to ``function`` for anything we don't
    recognise — the spec calls this out explicitly and the
    dependency-graph UI handles the generic case."""
    try:
        from langchain_core.language_models import BaseChatModel, BaseLanguageModel
        from langchain_core.retrievers import BaseRetriever
        from langchain_core.tools import BaseTool

        if isinstance(data, (BaseChatModel, BaseLanguageModel)):
            return "llm"
        if isinstance(data, BaseTool):
            return "tool"
        if isinstance(data, BaseRetriever):
            return "retriever"
    except Exception:
        pass
    return "function"


def _extract_topology(compiled: Any) -> dict[str, Any]:
    """Walk a LangGraph compiled ``Pregel`` and return ``{nodes, edges}``.

    Falls back to ``{}`` when the compiled object isn't a LangGraph
    (e.g. a plain LCEL ``Runnable``) — the dependency-graph endpoint
    treats an empty topology as "no native graph" and shows just the
    harness layers (guardrails / KBs / prompts).
    """
    if not hasattr(compiled, "get_graph"):
        return {}
    try:
        graph = compiled.get_graph()
    except Exception:
        return {}

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    raw_nodes = getattr(graph, "nodes", None) or {}
    for node_id, node in raw_nodes.items():
        data = getattr(node, "data", None)
        nodes.append(
            {
                "id": str(node_id),
                "type": _classify_node_data(data),
            }
        )
    for edge in getattr(graph, "edges", None) or []:
        edges.append(
            {
                "source": str(getattr(edge, "source", "")),
                "target": str(getattr(edge, "target", "")),
                "conditional": bool(getattr(edge, "conditional", False)),
            }
        )
    return {"nodes": nodes, "edges": edges}


def _extract_model(compiled: Any) -> tuple[str | None, str | None]:
    """Pull a ``(model, provider)`` pair out of the first LLM node."""
    topo = _extract_topology(compiled)
    if not topo:
        return None, None
    raw_nodes = getattr(compiled.get_graph(), "nodes", None) or {}
    for node in raw_nodes.values():
        data = getattr(node, "data", None)
        if data is None:
            continue
        if _classify_node_data(data) != "llm":
            continue
        model = (
            getattr(data, "model_name", None)
            or getattr(data, "model", None)
        )
        provider = type(data).__name__.lower().replace("chat", "")
        return (str(model) if model else None, provider or None)
    return None, None


def register_agent(compiled: Any, *, name: str) -> None:
    """Persist a LangGraph compiled graph (or LCEL ``Runnable``) to the
    external-agent registry so the Local UI's dependency graph and
    workflow visualisation can render it.

    Inspection is best-effort — we extract whatever the framework
    exposes (nodes, edges, first-LLM model) and hand the rest off to
    the dependency endpoint, which already knows how to render a
    partial tree.
    """
    _require()
    from fastaiagent.integrations._registry import upsert_agent

    topology = _extract_topology(compiled)
    model, provider = _extract_model(compiled)
    upsert_agent(
        name,
        framework="langchain",
        model=model,
        provider=provider,
        topology=topology,
    )


def kb_as_retriever(
    kb_name: str,
    *,
    top_k: int = 5,
    agent: str | None = None,
) -> Any:
    """Wrap a FastAIAgent ``LocalKB`` in a LangChain ``BaseRetriever``.

    The returned object satisfies LangChain's retriever interface so it
    drops into LCEL chains, `RunnableParallel`, etc. without further
    adaptation. Search runs against the named KB's hybrid (or
    configured) backend; the ``Document`` payload mirrors what
    LangChain's own retrievers return so downstream prompt templates
    can format ``page_content``/``metadata`` as usual.
    """
    _require()
    from langchain_core.callbacks import CallbackManagerForRetrieverRun
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever

    from fastaiagent.kb import LocalKB

    kb = LocalKB(name=kb_name)
    # Capture under different names so the class-body assignments
    # below don't shadow the closure variables (Python's class scope
    # treats ``kb_name: str = kb_name`` as a self-reference).
    _kb_name_default = kb_name
    _top_k_default = top_k

    class _FastAIAgentLocalKBRetriever(BaseRetriever):
        """LangChain retriever that delegates to a FastAIAgent ``LocalKB``."""

        # ``BaseRetriever`` inherits from pydantic.BaseModel in LangChain
        # ≥ 0.3, so attributes need to be declared as fields.
        kb_name: str = _kb_name_default
        top_k: int = _top_k_default

        def _get_relevant_documents(
            self,
            query: str,
            *,
            run_manager: CallbackManagerForRetrieverRun,
        ) -> list[Document]:
            results = kb.search(query, top_k=self.top_k)
            return [
                Document(
                    page_content=getattr(r.chunk, "content", "") or "",
                    metadata={
                        **(getattr(r.chunk, "metadata", None) or {}),
                        "score": float(getattr(r, "score", 0.0)),
                    },
                )
                for r in results
            ]

    if agent:
        try:
            from fastaiagent.integrations._registry import attach as _attach

            _attach(agent, "kb", kb_name)
        except Exception:
            pass

    return _FastAIAgentLocalKBRetriever()


__all__ = [
    "enable",
    "disable",
    "get_callback_handler",
    "as_evaluable",
    "with_guardrails",
    "prompt_from_registry",
    "register_agent",
    "kb_as_retriever",
]
