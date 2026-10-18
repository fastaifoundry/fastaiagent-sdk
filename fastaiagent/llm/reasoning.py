"""Reasoning control and prompt caching — one vocabulary, each provider's wire (1.91.0).

``LLMClient(reasoning=...)`` takes :data:`REASONING_LEVELS`; this module puts a
level in each provider's own field. The tables are from the providers' own docs,
read 2026-10-18 (platform.claude.com effort + extended-thinking pages,
ai.google.dev generate-content/thinking, developers.openai.com reasoning guide) and
the requests were checked live the same day:

* **OpenAI-compatible** — ``reasoning_effort``, as given. Which levels a model takes
  varies (``gpt-6-luna``: ``none``..``xhigh``); the provider says which in its 400.
* **Anthropic** — ``output_config.effort`` on the models that have it, with
  adaptive thinking and ``display: "summarized"`` (the 5.x models hide the text
  otherwise) where adaptive thinking exists; extended thinking with a
  ``budget_tokens`` where it does not. ``"none"`` turns thinking off
  (``between_tools`` on Sonnet 5.5, whose lowest setting that is).
* **Gemini** — ``thinkingLevel`` on 3.x / 4 (``minimal``..``high``),
  ``thinkingBudget`` on 2.5, with ``includeThoughts`` so summaries come back.

Prompt caching: :func:`cache_setting` validates ``cache=``; Anthropic and Bedrock
are the providers that need asking (the request builders in ``client.py``).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

#: Every level ``reasoning=`` accepts, lowest first.
REASONING_LEVELS: tuple[str, ...] = ("none", "minimal", "low", "medium", "high", "xhigh", "max")

#: ``cache=`` values: off, the provider's default lifetime, or one hour.
_CACHE_SETTINGS: dict[Any, str | None] = {
    False: None,
    None: None,
    True: "5m",
    "5m": "5m",
    "1h": "1h",
}

_WARNED: set[str] = set()


def _warn_once(key: str, message: str, *args: Any) -> None:
    if key not in _WARNED:
        _WARNED.add(key)
        logger.warning(message, *args)


def reasoning_level(value: Any) -> str | None:
    """Validate a ``reasoning=`` value. ``None`` means "not set" — nothing is sent."""
    if value is None:
        return None
    if value not in REASONING_LEVELS:
        raise ValueError(f"reasoning must be one of {REASONING_LEVELS} or None, got {value!r}")
    return str(value)


def cache_setting(value: Any) -> str | None:
    """Validate a ``cache=`` value into ``None`` (off), ``"5m"`` or ``"1h"``."""
    try:
        return _CACHE_SETTINGS[value]
    except (KeyError, TypeError):
        raise ValueError(f"cache must be True, False, '5m' or '1h', got {value!r}") from None


# ── Anthropic ──────────────────────────────────────────────────────────────


def _claude_id(model: str) -> str:
    """The bare Claude id from an Anthropic or Bedrock model id
    (``us.anthropic.claude-haiku-5-5-v1:0`` → ``claude-haiku-5-5-v1:0``)."""
    name = model.lower()
    if "anthropic." in name:
        name = name.split("anthropic.", 1)[1]
    return name


#: Models that take ``output_config.effort`` (effort page, ``supportedModels``).
#: Haiku 4.5, Sonnet 4.5 and older refuse it with a 400.
_EFFORT_MODELS = (
    "claude-fable-5",
    "claude-mythos-5",
    "claude-mythos-preview",
    "claude-opus-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-opus-4-6",
    "claude-opus-4-5",
    "claude-sonnet-5",
    "claude-sonnet-4-6",
    "claude-haiku-5-5",
)

#: Models that take ``thinking: {"type": "adaptive"}`` (thinking page, per-model
#: table). Opus 4.5, Sonnet 4.5 and Haiku 4.5 400 on it: extended thinking only.
_ADAPTIVE_MODELS = (
    "claude-fable-5",
    "claude-mythos-5",
    "claude-mythos-preview",
    "claude-opus-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-opus-4-6",
    "claude-sonnet-5",
    "claude-sonnet-4-6",
    "claude-haiku-5-5",
)

#: ``budget_tokens`` per level for the extended-thinking-only models. The API's
#: minimum is 1024; the budget always stays below ``max_tokens``.
_ANTHROPIC_BUDGETS = {
    "minimal": 1024,
    "low": 2048,
    "medium": 8192,
    "high": 16384,
    "xhigh": 32768,
    "max": 63999,
}


def anthropic_fields(model: str, level: str, max_tokens: int) -> dict[str, Any]:
    """The ``thinking`` / ``output_config`` request fields for ``level`` on ``model``."""
    name = _claude_id(model)
    if level == "none":
        if name.startswith("claude-sonnet-5-5"):
            return {"thinking": {"type": "between_tools"}}
        return {"thinking": {"type": "disabled"}}
    effort = "low" if level == "minimal" else level
    fields: dict[str, Any] = {}
    if name.startswith(_ADAPTIVE_MODELS):
        fields["thinking"] = {"type": "adaptive", "display": "summarized"}
    else:
        budget = min(_ANTHROPIC_BUDGETS[level], max_tokens - 1)
        fields["thinking"] = {"type": "enabled", "budget_tokens": max(budget, 1024)}
    if name.startswith(_EFFORT_MODELS):
        fields["output_config"] = {"effort": effort}
    return fields


# ── Gemini ─────────────────────────────────────────────────────────────────

#: ``thinkingBudget`` per level on Gemini 2.5 (2.5 Flash takes 0..24576; 2.5 Pro
#: 128..32768 and cannot turn thinking off — the API says so).
_GEMINI_BUDGETS = {
    "none": 0,
    "minimal": 512,
    "low": 1024,
    "medium": 8192,
    "high": 24576,
    "xhigh": 24576,
    "max": 24576,
}


def gemini_thinking_config(model: str, level: str) -> dict[str, Any] | None:
    """``generationConfig.thinkingConfig`` for ``level`` on ``model``, or ``None``
    for a model that does not think (1.5 / 2.0)."""
    name = model.lower().rsplit("/", 1)[-1]
    if name.startswith(("gemini-1.", "gemini-2.0")):
        _warn_once(
            f"gemini-nothink:{name}",
            "reasoning=%r is ignored: %s does not think.",
            level,
            model,
        )
        return None
    if name.startswith("gemini-2.5"):
        config: dict[str, Any] = {"thinkingBudget": _GEMINI_BUDGETS[level]}
    else:
        # 3.x, 4 and the -latest aliases: levels, minimal..high.
        if level == "none":
            _warn_once(
                f"gemini-none:{name}",
                "%s cannot turn thinking off; reasoning='none' sends its lowest "
                "level, 'minimal' (a model without 'minimal' rejects it).",
                model,
            )
            return {"thinkingLevel": "minimal"}
        config = {"thinkingLevel": "high" if level in ("xhigh", "max") else level}
    if level != "none":
        config["includeThoughts"] = True
    return config


# ── Ollama ─────────────────────────────────────────────────────────────────


def ollama_think(model: str, level: str) -> bool | str:
    """Ollama's ``think``: a level for gpt-oss (low/medium/high), a switch otherwise."""
    if level == "none":
        return False
    if "gpt-oss" in model.lower():
        return {"minimal": "low", "xhigh": "high", "max": "high"}.get(level, level)
    return True
