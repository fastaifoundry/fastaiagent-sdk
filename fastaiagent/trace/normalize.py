"""Normalize foreign instrumentor span attributes into canonical keys.

Spans emitted by third-party OpenTelemetry / OpenInference / OpenLLMetry
instrumentors use attribute conventions the Local UI does not read, so their
tokens / cost / model / IO panels render blank and they get no framework badge.

:func:`normalize_attributes` maps those foreign conventions onto the **exact**
canonical keys the rest of the stack already reads:

- ``gen_ai.request.model`` — model (read-time cost lookup, ``traces.py``)
- ``gen_ai.usage.input_tokens`` / ``gen_ai.usage.output_tokens`` — token counts
- ``gen_ai.prompt`` — prompt text (FTS ``input_text`` trigger + IO panel)
- ``gen_ai.completion`` — response text (FTS ``output_text`` trigger + IO panel)
- ``fastaiagent.runner.type`` — span classification (``agent``/``chain``/``tool``/…)
- ``fastaiagent.framework`` — the FA/LC/CA badge + framework filter (root span only)
- ``gen_ai.system`` / ``gen_ai.request.temperature`` / ``gen_ai.request.max_tokens``
  — best-effort extras when the source carries them

It is a **pure** function with no I/O. It only fills a canonical key when that
key is **absent** (or empty) — it never overwrites or removes anything, so
native fastaiagent spans (already canonical) pass through unchanged and the
original foreign keys are always preserved alongside the added canonical ones.
"""

from __future__ import annotations

import json
from typing import Any

# Foreign attribute key -> canonical key. Applied only when the source key is
# present and the destination is still absent. Ordered so the first source to
# populate a destination wins (e.g. ``llm.system`` before ``llm.provider``).
_DIRECT_MAP: dict[str, str] = {
    # --- OpenInference (Arize) ---
    "llm.model_name": "gen_ai.request.model",
    "llm.token_count.prompt": "gen_ai.usage.input_tokens",
    "llm.token_count.completion": "gen_ai.usage.output_tokens",
    # NOTE: input.value / output.value are handled separately (see
    # ``_PROMPT_TARGETS`` / ``_COMPLETION_TARGETS``) because the prompt/response
    # text has to fan out to BOTH the FTS keys and the UI IO-panel keys.
    "llm.system": "gen_ai.system",
    "llm.provider": "gen_ai.system",
    "tool.name": "fastaiagent.tool.name",
    # --- OpenLLMetry / Traceloop legacy spellings ---
    "llm.request.model": "gen_ai.request.model",
    "llm.response.model": "gen_ai.response.model",
    "gen_ai.usage.prompt_tokens": "gen_ai.usage.input_tokens",
    "gen_ai.usage.completion_tokens": "gen_ai.usage.output_tokens",
    "llm.usage.prompt_tokens": "gen_ai.usage.input_tokens",
    "llm.usage.completion_tokens": "gen_ai.usage.output_tokens",
}

# Prompt/completion text must land on *several* canonical keys because different
# readers look in different places (verified against the shipped code):
#   - the FTS search trigger reads ``gen_ai.prompt`` / ``gen_ai.completion``;
#   - the UI span IO panel (SpanInspector) reads ``gen_ai.request.messages`` for
#     input and ``gen_ai.response.content`` for output.
# Populating all of them makes a foreign span both *searchable* and *rendered*.
_PROMPT_TARGETS: tuple[str, ...] = ("gen_ai.prompt", "gen_ai.request.messages")
_COMPLETION_TARGETS: tuple[str, ...] = ("gen_ai.completion", "gen_ai.response.content")

# OpenInference span-kind -> fastaiagent runner.type. Display classification in
# the UI accepts any string; the list filter only narrows on
# agent/chain/swarm/supervisor, so the richer kinds (llm/tool/…) are still
# useful for the per-span badge without breaking the filter.
_SPAN_KIND_MAP: dict[str, str] = {
    "LLM": "llm",
    "CHAIN": "chain",
    "AGENT": "agent",
    "TOOL": "tool",
    "RETRIEVER": "retrieval",
    "EMBEDDING": "embedding",
    "RERANKER": "reranker",
    "GUARDRAIL": "guardrail",
}

# Substrings looked for in the instrumentation scope name to label the root
# span's framework badge, e.g. ``openinference.instrumentation.openai`` ->
# ``openai``. The value also feeds the (open-ended) framework filter.
_SCOPE_FRAMEWORK_HINTS: tuple[str, ...] = (
    "langchain",
    "langgraph",
    "llama_index",
    "llama-index",
    "llamaindex",
    "crewai",
    "haystack",
    "semantic_kernel",
    "autogen",
    "guardrails",
    "dspy",
    "litellm",
    "bedrock",
    "vertexai",
    "google_genai",
    "mistralai",
    "anthropic",
    "openai",
    "groq",
    "cohere",
)


def _framework_from_scope(scope_name: str | None) -> str | None:
    """Derive a framework slug from an instrumentation scope name."""
    if not scope_name:
        return None
    lowered = scope_name.lower()
    for hint in _SCOPE_FRAMEWORK_HINTS:
        if hint in lowered:
            return hint.replace("_", "-")
    # Fall back to the last dotted segment, e.g.
    # ``opentelemetry.instrumentation.foo`` -> ``foo``.
    segment = lowered.rsplit(".", 1)[-1].strip()
    return segment or None


def _consolidate_indexed(attrs: dict[str, Any], base: str) -> str | None:
    """Join ``base.{N}.content`` values (OpenLLMetry message arrays) in order.

    Traceloop spreads prompts/completions across ``gen_ai.prompt.0.content``,
    ``gen_ai.prompt.1.content`` … rather than a single string. Collapse them
    into one newline-joined string so the UI/FTS see real content.
    """
    items: list[tuple[int, Any]] = []
    prefix = base + "."
    suffix = ".content"
    for key, value in attrs.items():
        if key.startswith(prefix) and key.endswith(suffix):
            middle = key[len(prefix) : -len(suffix)]
            if middle.isdigit():
                items.append((int(middle), value))
    if not items:
        return None
    items.sort(key=lambda kv: kv[0])
    parts = [str(v) for _, v in items if v is not None and str(v) != ""]
    if not parts:
        return None
    return "\n".join(parts)


def normalize_attributes(
    attrs: dict[str, Any],
    *,
    scope_name: str | None = None,
    is_root: bool = False,
    framework_override: str | None = None,
) -> dict[str, Any]:
    """Return a copy of ``attrs`` enriched with canonical gen_ai/runner keys.

    Pure, no I/O. Only fills a canonical key when it is absent or empty; never
    overwrites or removes existing keys. ``scope_name`` is the span's
    instrumentation-scope name (used for the framework badge); ``is_root``
    gates the framework stamp to the root span (matching how the UI reads it).
    ``framework_override`` wins over scope derivation when provided.
    """
    out = dict(attrs)

    def fill(key: str, value: Any) -> None:
        if value is None:
            return
        existing = out.get(key)
        if existing is None or existing == "":
            out[key] = value

    # 1:1 spelling maps (OpenInference + OpenLLMetry legacy).
    for src, dst in _DIRECT_MAP.items():
        if src in attrs:
            fill(dst, attrs[src])

    # Prompt / completion text fan out to both the FTS keys and the UI keys.
    if "input.value" in attrs:
        for dst in _PROMPT_TARGETS:
            fill(dst, attrs["input.value"])
    if "output.value" in attrs:
        for dst in _COMPLETION_TARGETS:
            fill(dst, attrs["output.value"])

    # Span classification.
    kind = attrs.get("openinference.span.kind")
    if kind is not None:
        runner = _SPAN_KIND_MAP.get(str(kind).upper())
        if runner:
            fill("fastaiagent.runner.type", runner)
    if "tool.name" in attrs:
        fill("fastaiagent.runner.type", "tool")

    # OpenLLMetry indexed message arrays -> single prompt/completion strings,
    # fanned out to the same canonical targets as input.value / output.value.
    consolidated_prompt = _consolidate_indexed(attrs, "gen_ai.prompt")
    if consolidated_prompt is not None:
        for dst in _PROMPT_TARGETS:
            fill(dst, consolidated_prompt)
    consolidated_completion = _consolidate_indexed(attrs, "gen_ai.completion")
    if consolidated_completion is not None:
        for dst in _COMPLETION_TARGETS:
            fill(dst, consolidated_completion)

    # OpenInference packs request params in a JSON string.
    invocation = attrs.get("llm.invocation_parameters")
    if isinstance(invocation, str):
        try:
            params = json.loads(invocation)
        except (ValueError, TypeError):
            params = None
        if isinstance(params, dict):
            if params.get("temperature") is not None:
                fill("gen_ai.request.temperature", params["temperature"])
            max_tokens = params.get("max_tokens")
            if max_tokens is None:
                max_tokens = params.get("max_completion_tokens")
            if max_tokens is not None:
                fill("gen_ai.request.max_tokens", max_tokens)

    # Framework badge — root span only, to match the UI's read of the root.
    if is_root:
        framework = framework_override or _framework_from_scope(scope_name)
        if framework:
            fill("fastaiagent.framework", framework)

    return out
