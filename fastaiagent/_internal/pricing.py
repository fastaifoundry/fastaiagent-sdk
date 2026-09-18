"""Best-effort USD cost estimate for common LLM models.

Since 1.67.0 the SDK populates ``fastaiagent.cost.total_usd`` on every
``llm.*`` span it emits, and ``AgentResult.cost`` from the same numbers — the
key the langchain/crewai/pydanticai integrations have always set. When that
attribute is missing (an older trace, a foreign span) but we know the model +
token counts, this table lets the UI show a reasonable estimate instead of "—".

This module lives under ``_internal`` rather than ``ui`` because it is on the
LLM hot path: ``LLMClient`` costs every completion. ``fastaiagent.ui.pricing``
re-exports the whole surface unchanged, so the UI routes, the trace exporter
and the three framework integrations keep their existing import.

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
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Any

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


# ---------------------------------------------------------------------------
# Run-scoped usage accumulation (cost AND tokens)
# ---------------------------------------------------------------------------
#
# ``AgentResult.cost`` was declared in the very first release and never
# assigned. Meanwhile the three foreign-framework integrations each computed it
# and set ``fastaiagent.cost.total_usd`` — so the SDK's own runs were the only
# framework where the plane received no cost at all, and the Local UI hid it
# behind a read-time estimate from token counts.
#
# The accumulator is written in ONE place — ``LLMClient._acomplete_with_retries``,
# where the token attributes already go — so a run that makes three LLM calls
# (a tool loop turn, a structured re-ask, a guardrail re-ask) reports the sum of
# all three rather than whichever response happened to come back last.
#
# ``tokens_used`` is in the same bucket for exactly that reason. Until 1.68.0 it
# was computed at the call site from ``response.usage["total_tokens"]``, and
# ``execute_tool_loop`` returns only the LAST response — so a three-turn run
# reported one turn's tokens next to a cost that covered all three. Two numbers
# derived from the same completions have to be produced by the same hook, or
# they drift, and one of them is always the one nobody re-checks.
#
# "Unknown" is tracked separately from "zero" — for cost. ``compute_cost_usd``
# returns ``None`` for a model it has no rate for: ollama and lmstudio are
# genuinely free, but a bedrock/azure deployment id is partner-billed and a
# private fine-tune is simply unknown. Reporting any of those as ``$0.00 spent``
# would make a budget gate certify a run it never priced — the §2.4 shape of a
# check that cannot check reporting a clean verdict. So the run carries a
# ``cost_known`` flag alongside the number, and a budget check treats unknown as
# unknown. Tokens need no such flag: a provider either reported a count or it
# reported nothing, and nothing is zero.

_TOKEN_KEYS_IN = ("prompt_tokens", "input_tokens")
_TOKEN_KEYS_OUT = ("completion_tokens", "output_tokens")


def _tokens(usage: dict[str, Any] | None, keys: tuple[str, ...]) -> int:
    if not usage:
        return 0
    for key in keys:
        value = usage.get(key)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                return 0
    return 0


#: Providers that run the model on hardware the operator already pays for, so a
#: completion costs nothing *to the provider*. These are known-zero, not unknown.
#:
#: The distinction is the whole point of ``known``: a bedrock or azure deployment
#: id is partner-billed and a private fine-tune has a rate we simply do not hold,
#: and a budget gate must not certify either of those as free. A local run, by
#: contrast, is free in the only sense a cost gate measures — and treating it as
#: unpriced would block ``cost_limit`` on every laptop running ollama, which is
#: a false positive, not a safety property.
LOCAL_FREE_PROVIDERS: frozenset[str] = frozenset({"ollama", "lmstudio", "vllm"})


def is_local_free(provider: str | None, model: str | None = None) -> bool:
    """Whether this completion ran on self-hosted infrastructure.

    Matched on the **provider**, because a local model name carries no marker —
    ollama serves ``llama3``, which is indistinguishable from a hosted model of
    the same name. A ``provider/model`` prefix is honoured as a fallback for
    callers that only have the model string.
    """
    if provider and provider.strip().lower() in LOCAL_FREE_PROVIDERS:
        return True
    if model and "/" in model:
        return model.split("/", 1)[0].strip().lower() in LOCAL_FREE_PROVIDERS
    return False


def usage_cost(
    model: str | None,
    usage: dict[str, Any] | None,
    *,
    provider: str | None = None,
) -> tuple[float, bool]:
    """Price one completion. Returns ``(usd, known)``.

    ``known`` is ``False`` when the model has no rate in the table **or** the
    provider reported no tokens — in both cases the ``0.0`` means "we could not
    price this", not "this was free".

    The exception is a self-hosted provider (see :data:`LOCAL_FREE_PROVIDERS`),
    which is ``(0.0, True)``: genuinely free rather than unpriced.
    """
    if is_local_free(provider, model):
        return 0.0, True
    cost = compute_cost_usd(
        model, _tokens(usage, _TOKEN_KEYS_IN), _tokens(usage, _TOKEN_KEYS_OUT)
    )
    if cost is None:
        return 0.0, False
    return cost, True


def usage_tokens(usage: dict[str, Any] | None) -> int:
    """Total tokens one completion reported, however the provider spelled it.

    ``total_tokens`` when the provider gave one (every OpenAI-compatible wire
    does), else prompt + completion — which is what the Anthropic, Gemini,
    Bedrock and Ollama adapters normalise to, and the only shape a streamed
    ``Usage`` event has.
    """
    if not usage:
        return 0
    total = usage.get("total_tokens")
    if total is not None:
        try:
            return int(total)
        except (TypeError, ValueError):
            return 0
    return _tokens(usage, _TOKEN_KEYS_IN) + _tokens(usage, _TOKEN_KEYS_OUT)


@dataclass
class _RunUsage:
    """What one run has spent, in dollars and in tokens.

    One bucket rather than two because the two numbers describe the same set of
    completions. Keeping them apart is how ``cost`` ended up correct across
    every turn of a tool loop while ``tokens_used`` reported only the last.
    """

    usd: float = 0.0
    calls: int = 0
    priced_calls: int = 0
    tokens: int = 0

    @property
    def known(self) -> bool:
        """True only when EVERY call in the run could be priced.

        A run that made three calls and priced two of them does not know what
        it spent; reporting the partial sum as the total would under-report.

        Zero calls is *known* and zero: a scope that has not called a model yet
        has provably spent nothing. That is what lets an ``input``-position
        ``cost_limit`` rule give a real verdict instead of erroring.

        Cost only — tokens carry no such flag, see the section comment above.
        """
        return self.priced_calls == self.calls


_run_cost: ContextVar[_RunUsage | None] = ContextVar("fastaiagent_run_cost", default=None)


def start_run_cost() -> Token[_RunUsage | None]:
    """Begin accumulating cost + tokens for the current task. Returns the reset token.

    Nested runs (a swarm's child agent inside the swarm's own scope) each open
    their own accumulator, so a child's usage is reported on the child's result
    and re-counted on the parent only if the parent also accumulates.
    """
    return _run_cost.set(_RunUsage())


def record_run_cost(
    model: str | None,
    usage: dict[str, Any] | None,
    *,
    provider: str | None = None,
) -> tuple[float, bool]:
    """Price one completion, count its tokens, and add both to the active run."""
    cost, known = usage_cost(model, usage, provider=provider)
    bucket = _run_cost.get()
    if bucket is not None:
        bucket.calls += 1
        bucket.tokens += usage_tokens(usage)
        if known:
            bucket.priced_calls += 1
            bucket.usd += cost
    return cost, known


def record_run_tokens(usage: dict[str, Any] | None) -> None:
    """Count one completion's tokens without pricing it.

    For completions the SDK makes through something that is *not*
    ``LLMClient._acomplete_with_retries`` — the offline ``TestModel`` /
    ``FunctionModel``, which implement the client surface rather than extend its
    internals. They have no rate and must not be priced (a ``test-model`` with a
    price would make every offline run's ``cost_known`` false), but their token
    counts are real and a multi-turn offline run should report all of them.
    """
    bucket = _run_cost.get()
    if bucket is not None:
        bucket.tokens += usage_tokens(usage)


def run_cost() -> tuple[float, bool]:
    """``(usd, known)`` for the active run. ``(0.0, False)`` when not tracking."""
    bucket = _run_cost.get()
    if bucket is None:
        return 0.0, False
    return (bucket.usd, True) if bucket.known else (0.0, False)


def run_tokens() -> int:
    """Tokens billed to the active run so far. ``0`` when not tracking."""
    bucket = _run_cost.get()
    return bucket.tokens if bucket is not None else 0


def stop_run_cost(token: Token[_RunUsage | None]) -> None:
    _run_cost.reset(token)
