"""Best-effort USD cost lookup for common LLM models.

The SDK will populate ``agent.cost_usd`` (or ``fastaiagent.cost.total_usd``)
when it can compute cost directly.
When that attribute is missing but we know the model + token counts, this
table lets the UI show a reasonable estimate instead of "—".

Prices are USD per 1M tokens, current as of early 2026. Keep this
conservative — we'd rather under-report than mislead users. Update by
dropping new entries here; unknown models return ``None`` and the UI
falls back to the dash.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class _Rate:
    input_per_m: float
    output_per_m: float


# Prefix-matched. Longer prefixes win.
_PRICING: dict[str, _Rate] = {
    # OpenAI
    "gpt-4o-mini": _Rate(0.15, 0.60),
    "gpt-4o": _Rate(2.50, 10.00),
    "gpt-4-turbo": _Rate(10.00, 30.00),
    "gpt-4.1-mini": _Rate(0.40, 1.60),
    "gpt-4.1": _Rate(2.00, 8.00),
    "gpt-3.5-turbo": _Rate(0.50, 1.50),
    "o3-mini": _Rate(1.10, 4.40),
    "o1-mini": _Rate(1.10, 4.40),
    "o1-preview": _Rate(15.00, 60.00),
    "o1": _Rate(15.00, 60.00),
    # Anthropic
    "claude-3-5-haiku": _Rate(0.80, 4.00),
    "claude-3-5-sonnet": _Rate(3.00, 15.00),
    "claude-3-haiku": _Rate(0.25, 1.25),
    "claude-3-sonnet": _Rate(3.00, 15.00),
    "claude-3-opus": _Rate(15.00, 75.00),
    "claude-sonnet-4": _Rate(3.00, 15.00),
    "claude-opus-4": _Rate(15.00, 75.00),
    "claude-haiku-4-5": _Rate(1.00, 5.00),
    "claude-haiku-4": _Rate(1.00, 5.00),
    # Google
    "gemini-1.5-flash": _Rate(0.075, 0.30),
    "gemini-1.5-pro": _Rate(1.25, 5.00),
    "gemini-2.0-flash": _Rate(0.10, 0.40),
    # Local / Mistral / Groq
    "mixtral-8x7b": _Rate(0.24, 0.24),
    "llama-3.1-70b": _Rate(0.59, 0.79),
    "llama-3.1-8b": _Rate(0.05, 0.08),
}


def compute_cost_usd(
    model: str | None,
    input_tokens: int | float | None,
    output_tokens: int | float | None,
) -> float | None:
    """Return USD cost from model + token counts, or ``None`` if unknown.

    Prefix-matches ``model`` against the pricing table. Longest matching
    prefix wins, so ``gpt-4o-mini-2024-07-18`` still resolves to the
    ``gpt-4o-mini`` rate.
    """
    if not model:
        return None
    rate = _match(model)
    if rate is None:
        return None
    inp = float(input_tokens or 0)
    out = float(output_tokens or 0)
    if inp == 0 and out == 0:
        return None
    return (inp * rate.input_per_m + out * rate.output_per_m) / 1_000_000.0


def _match(model: str) -> _Rate | None:
    normalised = model.lower()
    best: tuple[int, _Rate] | None = None
    for prefix, rate in _PRICING.items():
        if normalised.startswith(prefix):
            length = len(prefix)
            if best is None or length > best[0]:
                best = (length, rate)
    return best[1] if best else None
