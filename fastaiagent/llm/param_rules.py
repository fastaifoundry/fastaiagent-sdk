"""Which sampling parameters a model accepts (1.89.0): data only.

Providers reject sampling parameters per model, with a 400 and nothing else to
go on. This table is what the Local UI uses to disable a control *before* the
call, and what the Playground's failure hint uses *after* one. It never blocks
a call and never drops a parameter (decision D4): rules go stale as providers
change, so the provider stays the authority.

Measured against the live APIs on 2026-10-14 (temperature 0.7 / 1.0, top_p 0.9
/ 1.0, and both together):

* Anthropic ``claude-opus-4-7``/``-4-8`` and the Claude 5 family (sonnet,
  opus, fable, haiku): "`temperature` is deprecated for this model" for any
  value but the default 1; "`top_p` is deprecated" for every value.
* Older Claude (``claude-haiku-4-5``, ``claude-sonnet-4-6``, …): either one,
  not both ("cannot both be specified for this model").
* OpenAI ``gpt-5``, ``gpt-5-mini``, ``gpt-5-nano``, ``o4-mini``: "Only the
  default (1) value is supported" for temperature; top_p likewise.
* OpenAI ``o1``/``o3``: temperature default only; top_p not supported.
* OpenAI ``gpt-5.1``, ``gpt-5.2``, ``gpt-4o``, ``gpt-4.1``: free.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

ParamSupport = Literal["free", "default_only", "unsupported"]


@dataclass(frozen=True)
class SamplingRule:
    temperature: ParamSupport = "free"
    top_p: ParamSupport = "free"
    exclusive: bool = False
    """Temperature and top_p cannot both be sent."""
    default_temperature: float = 1.0
    default_top_p: float = 1.0
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_FREE = SamplingRule()
_CLAUDE_NEW = SamplingRule(
    temperature="default_only",
    top_p="unsupported",
    note="This Claude model accepts only its default temperature (1) and no top_p.",
)
_CLAUDE_OLD = SamplingRule(
    exclusive=True,
    note="Claude accepts temperature or top_p, not both.",
)
_OPENAI_REASONING = SamplingRule(
    temperature="default_only",
    top_p="default_only",
    note="This OpenAI reasoning model accepts only the default temperature (1) and top_p (1).",
)
_OPENAI_O_SERIES = SamplingRule(
    temperature="default_only",
    top_p="unsupported",
    note="This OpenAI o-series model accepts only the default temperature (1) and no top_p.",
)

#: ``(provider, model prefix, rule)``. The longest matching prefix wins.
RULES: list[tuple[str, str, SamplingRule]] = [
    ("anthropic", "claude-", _CLAUDE_OLD),
    ("anthropic", "claude-opus-4-7", _CLAUDE_NEW),
    ("anthropic", "claude-opus-4-8", _CLAUDE_NEW),
    ("anthropic", "claude-sonnet-5", _CLAUDE_NEW),
    ("anthropic", "claude-opus-5", _CLAUDE_NEW),
    ("anthropic", "claude-haiku-5", _CLAUDE_NEW),
    ("anthropic", "claude-fable-5", _CLAUDE_NEW),
    ("openai", "gpt-5", _OPENAI_REASONING),
    ("openai", "gpt-5.1", _FREE),
    ("openai", "gpt-5.2", _FREE),
    ("openai", "o1", _OPENAI_O_SERIES),
    ("openai", "o3", _OPENAI_O_SERIES),
    ("openai", "o4", _OPENAI_REASONING),
]

#: Model-id prefixes added by routers and clouds in front of the vendor's id.
_STRIP = (
    "us.anthropic.",
    "eu.anthropic.",
    "apac.anthropic.",
    "anthropic.",
    "anthropic/",
    "openai/",
)

#: Providers whose wire format carries another vendor's models.
_VENDOR_OF = {"azure": "openai", "openrouter": None, "bedrock": "anthropic"}


def _vendor_and_model(provider: str, model: str) -> tuple[str, str]:
    m = (model or "").strip().lower()
    vendor = provider
    for prefix in _STRIP:
        if m.startswith(prefix):
            vendor = "anthropic" if "anthropic" in prefix else "openai"
            m = m[len(prefix) :]
            break
    else:
        mapped = _VENDOR_OF.get(provider, provider)
        if mapped:
            vendor = mapped
    return vendor, m


def sampling_rule(provider: str, model: str) -> SamplingRule:
    """The rule for ``model`` on ``provider``; free when nothing matches."""
    vendor, m = _vendor_and_model(provider, model)
    best: SamplingRule | None = None
    best_len = -1
    for p, prefix, rule in RULES:
        if p == vendor and m.startswith(prefix) and len(prefix) > best_len:
            best, best_len = rule, len(prefix)
    return best or _FREE


def rules_table() -> list[dict[str, Any]]:
    """The table for the UI, which resolves the free-text model field locally."""
    return [{"provider": p, "model_prefix": prefix, **rule.to_dict()} for p, prefix, rule in RULES]


__all__ = ["RULES", "SamplingRule", "rules_table", "sampling_rule"]
