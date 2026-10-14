"""One vocabulary for why a model stopped (1.89.0).

Providers say it differently: OpenAI ``length``, Anthropic ``max_tokens`` and
``model_context_window_exceeded``, Gemini ``MAX_TOKENS``, Ollama ``length``.
:attr:`LLMResponse.finish_reason` keeps the provider's raw value (decision D5);
callers that need to *act* on it (the Playground's truncation banner) normalise
it here.
"""

from __future__ import annotations

from typing import Literal

FinishKind = Literal["stop", "length", "tool_calls", "content_filter", "refusal", "other"]

_MAP: dict[str, FinishKind] = {
    "stop": "stop",
    "end_turn": "stop",
    "stop_sequence": "stop",
    "eos": "stop",
    "pause_turn": "stop",
    "length": "length",
    "max_tokens": "length",
    "model_context_window_exceeded": "length",
    "tool_calls": "tool_calls",
    "tool_use": "tool_calls",
    "function_call": "tool_calls",
    "content_filter": "content_filter",
    "safety": "content_filter",
    "recitation": "content_filter",
    "blocklist": "content_filter",
    "prohibited_content": "content_filter",
    "spii": "content_filter",
    "refusal": "refusal",
}


def normalize_finish_reason(raw: str | None) -> FinishKind | None:
    """``stop`` / ``length`` / ``tool_calls`` / ``content_filter`` / ``refusal`` /
    ``other``; ``None`` when the provider gave none."""
    if not raw:
        return None
    return _MAP.get(str(raw).strip().lower(), "other")


def is_truncated(raw: str | None) -> bool:
    """The output was cut off by a token limit."""
    return normalize_finish_reason(raw) == "length"


__all__ = ["FinishKind", "is_truncated", "normalize_finish_reason"]
