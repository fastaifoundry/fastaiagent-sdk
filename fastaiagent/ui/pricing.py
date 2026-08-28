"""Best-effort USD cost estimate for common LLM models.

The SDK will populate ``agent.cost_usd`` (or ``fastaiagent.cost.total_usd``)
when it can compute cost directly.
When that attribute is missing but we know the model + token counts, this
table lets the UI show a reasonable estimate instead of "—".

**These are public list prices and will not match your invoice.** They know
nothing about negotiated or committed-use discounts, Amazon Bedrock / Google
Vertex partner rates (billed by those platforms, not Anthropic/OpenAI), the
Batch API's 50% reduction, or prompt-cache multipliers (cache reads bill at
roughly 0.1x and writes at 1.25-2x, and the token counts we get here don't
separate cached from uncached input). Treat every figure as an
order-of-magnitude sanity check, not accounting.

Organisations with their own rates should override them rather than patch this
table — see :func:`set_rate_overrides` and the ``pricing`` block in
:mod:`fastaiagent.ui.model_catalog`'s ``models.json``. Overrides are picked up
by *every* caller of :func:`compute_cost_usd` (traces, analytics, evals, trace
export, framework integrations), not just the Playground.

Prices are USD per 1M tokens, current as of 2026-08. Unknown models return
``None`` and the UI falls back to a dash.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Rate:
    input_per_m: float
    output_per_m: float


# Prefix-matched. Longer prefixes win.
_PRICING: dict[str, _Rate] = {
    # OpenAI
    "gpt-5-nano": _Rate(0.05, 0.40),
    "gpt-5-mini": _Rate(0.25, 2.00),
    "gpt-5": _Rate(1.25, 10.00),
    "gpt-4o-mini": _Rate(0.15, 0.60),
    "gpt-4o": _Rate(2.50, 10.00),
    "gpt-4-turbo": _Rate(10.00, 30.00),
    "gpt-4.1-nano": _Rate(0.10, 0.40),
    "gpt-4.1-mini": _Rate(0.40, 1.60),
    "gpt-4.1": _Rate(2.00, 8.00),
    "gpt-3.5-turbo": _Rate(0.50, 1.50),
    "o4-mini": _Rate(1.10, 4.40),
    "o3-mini": _Rate(1.10, 4.40),
    "o3": _Rate(2.00, 8.00),
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
    "claude-sonnet-4-6": _Rate(3.00, 15.00),
    # Opus 4.0/4.1 were $15/$75; from Opus 4.5 onward the tier is $5/$25.
    # These longer prefixes must stay ahead of the broad "claude-opus-4" row,
    # which otherwise reports 3x the real cost for every 4.5+ model.
    "claude-opus-4": _Rate(15.00, 75.00),
    "claude-opus-4-5": _Rate(5.00, 25.00),
    "claude-opus-4-6": _Rate(5.00, 25.00),
    "claude-opus-4-7": _Rate(5.00, 25.00),
    "claude-opus-4-8": _Rate(5.00, 25.00),
    "claude-haiku-4-5": _Rate(1.00, 5.00),
    "claude-haiku-4": _Rate(1.00, 5.00),
    # Claude 5 family
    "claude-opus-5": _Rate(5.00, 25.00),
    "claude-sonnet-5": _Rate(3.00, 15.00),
    "claude-fable-5": _Rate(10.00, 50.00),
    "claude-mythos-5": _Rate(10.00, 50.00),
    # Google
    "gemini-1.5-flash": _Rate(0.075, 0.30),
    "gemini-1.5-pro": _Rate(1.25, 5.00),
    "gemini-2.0-flash": _Rate(0.10, 0.40),
    "gemini-2.5-flash": _Rate(0.30, 2.50),
    "gemini-2.5-pro": _Rate(1.25, 10.00),
    # Local / Mistral / Groq
    "mixtral-8x7b": _Rate(0.24, 0.24),
    "llama-3.1-70b": _Rate(0.59, 0.79),
    "llama-3.1-8b": _Rate(0.05, 0.08),
    "llama-3.3-70b": _Rate(0.59, 0.79),
    # Groq (often free / cheap; rates from groq.com/pricing)
    "llama-3.1-70b-versatile": _Rate(0.59, 0.79),
    "llama-3.1-8b-instant": _Rate(0.05, 0.08),
    "mixtral-8x7b-32768": _Rate(0.24, 0.24),
    # DeepSeek
    "deepseek-chat": _Rate(0.27, 1.10),
    "deepseek-reasoner": _Rate(0.55, 2.19),
    # Mistral
    "mistral-large": _Rate(2.00, 6.00),
    "mistral-small": _Rate(0.20, 0.60),
    "mistral-medium": _Rate(0.40, 2.00),
    "open-mistral-nemo": _Rate(0.15, 0.15),
    # Together AI
    "meta-llama/llama-3.1-70b-instruct-turbo": _Rate(0.88, 0.88),
    "meta-llama/llama-3.1-8b-instruct-turbo": _Rate(0.18, 0.18),
    # Fireworks
    "accounts/fireworks/models/llama-v3p1-70b-instruct": _Rate(0.90, 0.90),
    "accounts/fireworks/models/llama-v3p1-8b-instruct": _Rate(0.20, 0.20),
    # Perplexity (sonar online)
    "llama-3.1-sonar-small-128k-online": _Rate(0.20, 0.20),
    "llama-3.1-sonar-large-128k-online": _Rate(1.00, 1.00),
    # OpenRouter — let downstream model name match the underlying provider's
    # entry; it carries the prefix ``openai/``, ``anthropic/``, etc.
    "openai/gpt-5": _Rate(1.25, 10.00),
    "openai/gpt-4o-mini": _Rate(0.15, 0.60),
    "openai/gpt-4o": _Rate(2.50, 10.00),
    "anthropic/claude-sonnet-4": _Rate(3.00, 15.00),
    "anthropic/claude-haiku-4": _Rate(1.00, 5.00),
    "anthropic/claude-opus-4": _Rate(15.00, 75.00),
    "anthropic/claude-opus-4-5": _Rate(5.00, 25.00),
    "anthropic/claude-opus-4-8": _Rate(5.00, 25.00),
    "anthropic/claude-opus-5": _Rate(5.00, 25.00),
    "anthropic/claude-sonnet-5": _Rate(3.00, 15.00),
    "anthropic/claude-3-5-sonnet": _Rate(3.00, 15.00),
    "anthropic/claude-3-5-haiku": _Rate(0.80, 4.00),
}


# ---------------------------------------------------------------------------
# Organisation rate overrides
# ---------------------------------------------------------------------------

_OVERRIDES: dict[str, _Rate] = {}
_OVERRIDES_LOADED = False
_OVERRIDES_LOCK = threading.Lock()


def set_rate_overrides(rates: dict[str, tuple[float, float]] | None) -> None:
    """Replace the org rate table. Keys are model-id prefixes, matched like
    the built-in table; values are ``(input_per_1m, output_per_1m)`` in USD.

    Passing ``None`` (or ``{}``) clears the overrides and falls back to list
    price. Overrides win over built-ins at equal prefix length, so
    ``{"claude-opus-5": (4.0, 20.0)}`` re-rates that model everywhere.
    """
    global _OVERRIDES, _OVERRIDES_LOADED
    with _OVERRIDES_LOCK:
        _OVERRIDES = {
            k.lower(): _Rate(float(i), float(o)) for k, (i, o) in (rates or {}).items()
        }
        _OVERRIDES_LOADED = True


def reload_rate_overrides(db_path: str | None = None) -> dict[str, _Rate]:
    """Re-read the ``pricing`` block of ``models.json`` into the override table."""
    from fastaiagent.ui.model_catalog import read_catalog_file

    raw = read_catalog_file(db_path).get("pricing")
    parsed: dict[str, _Rate] = {}
    if raw is not None:
        if not isinstance(raw, dict):
            logger.warning(
                "Model catalog: 'pricing' must be an object mapping model prefix -> "
                "rates — ignoring it and using list prices."
            )
        else:
            for prefix, spec in raw.items():
                rate = _parse_rate(prefix, spec)
                if rate is not None:
                    parsed[prefix.lower()] = rate

    global _OVERRIDES, _OVERRIDES_LOADED
    with _OVERRIDES_LOCK:
        _OVERRIDES = parsed
        _OVERRIDES_LOADED = True
    return parsed


def _parse_rate(prefix: str, spec: object) -> _Rate | None:
    """Validate one ``pricing`` entry. Logs and returns ``None`` if malformed."""
    if not isinstance(spec, dict):
        logger.warning("Model catalog: pricing[%r] must be an object — ignoring.", prefix)
        return None
    try:
        return _Rate(float(spec["input_per_1m"]), float(spec["output_per_1m"]))
    except (KeyError, TypeError, ValueError):
        logger.warning(
            "Model catalog: pricing[%r] needs numeric 'input_per_1m' and "
            "'output_per_1m' — ignoring.",
            prefix,
        )
        return None


def _active_overrides() -> dict[str, _Rate]:
    """Override table, loading it from disk on first use."""
    if not _OVERRIDES_LOADED:
        try:
            reload_rate_overrides()
        except Exception:  # noqa: BLE001 — cost display must never be fatal
            logger.debug("Model catalog: rate override load failed", exc_info=True)
            set_rate_overrides(None)
    return _OVERRIDES


def compute_cost_usd(
    model: str | None,
    input_tokens: int | float | None,
    output_tokens: int | float | None,
) -> float | None:
    """Return an estimated USD cost from model + token counts, or ``None``.

    Prefix-matches ``model`` against the org override table first, then the
    built-in list-price table. Longest matching prefix wins, so
    ``gpt-4o-mini-2024-07-18`` still resolves to the ``gpt-4o-mini`` rate.

    This is an estimate — see the module docstring for what it can't account
    for (negotiated discounts, partner pricing, batch, prompt caching).
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
    # Overrides are checked with ">=" so an org rate beats a built-in of the
    # same prefix length; built-ins are checked first with ">".
    for prefix, rate in _PRICING.items():
        if normalised.startswith(prefix):
            length = len(prefix)
            if best is None or length > best[0]:
                best = (length, rate)
    for prefix, rate in _active_overrides().items():
        if normalised.startswith(prefix):
            length = len(prefix)
            if best is None or length >= best[0]:
                best = (length, rate)
    return best[1] if best else None
