"""What a model can take and give — ``LLMClient.model_info`` (1.91.0, 2F).

One prefix-matched table beside the pricing table, filled only from the providers'
own sources (read 2026-10-18; ``claude_files/model_data_2026-10-18.md`` keeps the
citations):

* OpenAI — the per-model pages under ``developers.openai.com/api/docs/models``;
* Anthropic — the Models API (``GET /v1/models``: ``max_input_tokens``,
  ``max_tokens``) and the models overview;
* Google — the Gemini Models API (``inputTokenLimit`` / ``outputTokenLimit`` /
  ``thinking``).

A model the table does not know gets ``None`` for every field: an unknown is not a
guess. Image and PDF support are not copied here — they come from the multimodal
registry, the one place the request formatter reads them from.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["ModelInfo", "model_info"]


@dataclass(frozen=True)
class ModelInfo:
    """Limits and capabilities of one model. ``None`` means "not known"."""

    model: str
    #: Tokens the model reads in one request (input, history and tools).
    context_window: int | None = None
    #: Tokens one reply may be — what Anthropic and Bedrock send as ``max_tokens``
    #: when the client sets none.
    max_output_tokens: int | None = None
    #: The model thinks before answering (reasoning tokens / extended thinking).
    supports_reasoning: bool | None = None
    #: The provider caches repeated prompt prefixes and bills them at a lower rate.
    supports_cache: bool | None = None
    #: Takes image input (from the multimodal registry).
    supports_vision: bool | None = None
    #: Takes PDF documents natively (from the multimodal registry).
    supports_pdf: bool | None = None


# prefix -> (context_window, max_output_tokens, reasoning, cache). Longest prefix wins.
_TABLE: dict[str, tuple[int, int, bool, bool]] = {
    # OpenAI — model pages
    "gpt-6-astra": (1_050_000, 128_000, True, True),
    "gpt-6.1-sol": (1_050_000, 128_000, True, True),
    "gpt-6-luna": (1_050_000, 128_000, True, True),
    "gpt-5.1": (400_000, 128_000, True, True),
    "gpt-5-mini": (400_000, 128_000, True, True),
    "gpt-5-nano": (400_000, 128_000, True, True),
    "gpt-5": (400_000, 128_000, True, True),
    "gpt-4.1": (1_047_576, 32_768, False, True),
    "gpt-4o": (128_000, 16_384, False, True),
    "o3": (200_000, 100_000, True, True),
    "o4-mini": (200_000, 100_000, True, True),
    # Anthropic — Models API
    "claude-fable-5": (1_000_000, 128_000, True, True),
    "claude-mythos-5": (1_000_000, 128_000, True, True),
    "claude-opus-5": (1_000_000, 128_000, True, True),
    "claude-sonnet-5": (1_000_000, 128_000, True, True),
    "claude-haiku-5": (1_000_000, 128_000, True, True),
    "claude-opus-4-8": (1_000_000, 128_000, True, True),
    "claude-opus-4-7": (1_000_000, 128_000, True, True),
    "claude-opus-4-6": (1_000_000, 128_000, True, True),
    "claude-sonnet-4-6": (1_000_000, 128_000, True, True),
    "claude-opus-4-5": (200_000, 64_000, True, True),
    "claude-sonnet-4-5": (200_000, 64_000, True, True),
    "claude-haiku-4-5": (200_000, 64_000, True, True),
    # Google — Gemini Models API
    "gemini-4-argon": (1_048_576, 32_768, True, True),
    "gemini-3": (1_048_576, 65_536, True, True),
    "gemini-2.5": (1_048_576, 65_536, True, True),
    "gemini-flash-latest": (1_048_576, 65_536, True, True),
    "gemini-flash-lite-latest": (1_048_576, 65_536, True, True),
    "gemini-pro-latest": (1_048_576, 65_536, True, True),
}


def _lookup_id(model: str) -> str:
    """The id to look up: Bedrock ids carry a vendor (and region) prefix —
    ``us.anthropic.claude-…`` — and a ``-v1:0`` suffix the prefix match ignores."""
    normalised = model.lower()
    marker = "anthropic."
    if marker in normalised:
        normalised = normalised.split(marker, 1)[1]
    return normalised


def model_info(provider: str, model: str) -> ModelInfo:
    """What the table knows about ``model`` on ``provider``."""
    from fastaiagent.multimodal.registry import is_vision_capable, supports_native_pdf

    key = _lookup_id(model)
    best: tuple[int, tuple[int, int, bool, bool]] | None = None
    for prefix, row in _TABLE.items():
        if key.startswith(prefix) and (best is None or len(prefix) > best[0]):
            best = (len(prefix), row)
    vision = is_vision_capable(provider, model)
    pdf = supports_native_pdf(provider, model)
    if best is None:
        return ModelInfo(model=model, supports_vision=vision, supports_pdf=pdf)
    context, max_out, reasoning, cache = best[1]
    return ModelInfo(
        model=model,
        context_window=context,
        max_output_tokens=max_out,
        supports_reasoning=reasoning,
        supports_cache=cache,
        supports_vision=vision,
        supports_pdf=pdf,
    )
