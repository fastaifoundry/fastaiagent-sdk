"""Provider → model catalog that feeds the Playground's // MODEL dropdown.

Model ids rot. Providers decommission them on their own schedule, and a
hardcoded list in a released wheel can't keep up: at 1.52.0 every Groq entry
and one Anthropic entry in the shipped catalog were dead, and no amount of
care at release time prevents that recurring.

So the catalog is layered:

1. **Shipped defaults** — :data:`BUILTIN_MODELS` for the built-in providers,
   :data:`PRESET_MODEL_HINTS` for registered presets. Verified live at
   release time; a best-effort starting point, not a source of truth.
2. **A user override file** — JSON, resolved from ``$FASTAIAGENT_MODEL_CATALOG``
   or ``{db_dir}/models.json``. Replaces a provider's list outright, so you can
   fix a stale dropdown without waiting for an SDK release.
3. **Free-text entry in the UI** — the dropdown is a combobox; ``LLMClient``
   accepts any model string the upstream API does, so an unlisted model is
   always one keystroke away.

Override file shape (both forms accepted)::

    {
      "anthropic": {"models": ["claude-opus-5", "claude-sonnet-5"]},
      "openai": ["gpt-5.2", "gpt-4o-mini"],
      "pricing": {
        "claude-opus-5": {"input_per_1m": 4.0, "output_per_1m": 20.0}
      }
    }

``pricing`` is a reserved top-level key, not a provider. It carries your
organisation's negotiated per-million-token rates, which is the only way the
cost figure can be right: the shipped table is public list price and knows
nothing about committed-use discounts, Bedrock/Vertex partner pricing, or the
Batch API's 50% reduction. Keys are matched as model-id prefixes, longest
first, exactly like the built-in table. See :mod:`fastaiagent.ui.pricing`.

Only providers the SDK already knows (built-ins + presets registered via
:func:`fastaiagent.llm.providers.register_provider`) can be overridden —
declaring a brand-new provider needs a ``base_url`` and a wire format, which
is what ``register_provider`` is for. Unknown keys are logged and skipped.

A malformed file never breaks the dropdown: it is logged and ignored.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Env var holding an explicit path to the override file.
CATALOG_ENV_VAR = "FASTAIAGENT_MODEL_CATALOG"

#: Filename looked up next to ``local.db`` when the env var is unset.
CATALOG_FILENAME = "models.json"

#: Top-level keys in the override file that are *not* provider names.
RESERVED_KEYS = frozenset({"pricing"})


# ---------------------------------------------------------------------------
# Shipped defaults
# ---------------------------------------------------------------------------

#: Curated model lists for the three built-in providers. Every non-local entry
#: was verified against the live provider API on 2026-08-28. The first entry is
#: the one the UI selects when you switch to that provider, so it should be a
#: cheap, non-reasoning model that behaves well at a small ``max_tokens``.
BUILTIN_MODELS: dict[str, list[str]] = {
    "openai": [
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-4.1-mini",
        "gpt-4.1",
        "gpt-5.2",
        "gpt-5.1",
        "gpt-5-mini",
        "o3-mini",
    ],
    "anthropic": [
        "claude-haiku-4-5",
        "claude-sonnet-5",
        "claude-opus-5",
        "claude-opus-4-8",
        "claude-sonnet-4-6",
        "claude-fable-5",
    ],
    # Local server — can't be verified from here; whatever the user has pulled.
    "ollama": [
        "llama3.2",
        "llama3.2-vision",
        "qwen2.5",
    ],
    # OpenAI-compatible endpoint you point at yourself: an internal AI gateway,
    # a LiteLLM/vLLM proxy, a staging deployment. No model list can be shipped
    # for these — the endpoint decides. Supply Endpoint (+ token) in the UI.
    "custom": [],
    "azure": [],
}

#: Env var each built-in reads for its key. ``""`` means "no key required".
BUILTIN_ENV_KEY: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "ollama": "",  # local — no key required
    "custom": "OPENAI_API_KEY",
    "azure": "OPENAI_API_KEY",
}

#: Providers that talk to a server you run yourself, which usually needs no
#: auth. They are never key-gated in the UI.
#:
#: ``lmstudio`` and ``vllm`` declare ``LMSTUDIO_API_KEY`` / ``VLLM_API_KEY`` on
#: their presets, so the "blank env var means no key" rule never fired for them
#: and both showed as disabled — you had to invent a dummy env var to use a
#: local server that wanted no key at all.
LOCAL_PROVIDERS: frozenset[str] = frozenset({"ollama", "lmstudio", "vllm"})

#: Extra suggestions per preset provider, appended after the preset's own
#: ``default_model``. Adding entries here only affects the UI affordance —
#: ``LLMClient(provider=..., model="anything")`` accepts any upstream model.
#: Non-local entries verified live on 2026-08-28.
PRESET_MODEL_HINTS: dict[str, list[str]] = {
    "groq": [
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "groq/compound-mini",
        "qwen/qwen3.8-27b",
    ],
    "gemini": [
        "gemini-flash-latest",
        "gemini-pro-latest",
        "gemini-flash-lite-latest",
        "gemini-2.5-flash",
    ],
    "openrouter": [
        "openai/gpt-4o-mini",
        "openai/gpt-4o",
        "anthropic/claude-sonnet-5",
        "anthropic/claude-haiku-4-5",
        "meta-llama/llama-3.1-70b-instruct",
    ],
    "deepseek": [
        "deepseek-chat",
        "deepseek-reasoner",
    ],
    "together": [
        "meta-llama/Llama-3.1-70B-Instruct-Turbo",
        "meta-llama/Llama-3.1-8B-Instruct-Turbo",
    ],
    "fireworks": [
        "accounts/fireworks/models/llama-v3p1-70b-instruct",
        "accounts/fireworks/models/llama-v3p1-8b-instruct",
    ],
    "perplexity": [
        "sonar",
        "sonar-pro",
    ],
    "mistral": [
        "mistral-large-latest",
        "mistral-small-latest",
        "open-mistral-nemo",
    ],
    "lmstudio": ["local-model"],
    "vllm": ["local-model"],
    "sambanova": ["Meta-Llama-3.1-70B-Instruct"],
    "cerebras": ["llama3.1-70b", "llama3.1-8b"],
}


# ---------------------------------------------------------------------------
# Override file
# ---------------------------------------------------------------------------


def catalog_path(db_path: str | None) -> Path:
    """Where the override file lives.

    ``$FASTAIAGENT_MODEL_CATALOG`` wins when set. Otherwise the file sits
    beside ``local.db`` in the ``.fastaiagent`` directory, matching where
    playground datasets are written.
    """
    explicit = os.environ.get(CATALOG_ENV_VAR)
    if explicit:
        return Path(explicit)
    if db_path:
        db = Path(db_path)
        if db.parent.name == ".fastaiagent":
            return db.parent / CATALOG_FILENAME
    return Path.cwd() / ".fastaiagent" / CATALOG_FILENAME


def _coerce_entry(provider: str, raw: Any) -> dict[str, Any] | None:
    """Normalise one override entry to ``{"models": [...], "env_var": ...}``.

    Accepts either a bare list of model ids or a mapping. Returns ``None``
    (and logs) for anything else, so one bad entry can't take out the file.
    """
    if isinstance(raw, list):
        raw = {"models": raw}
    if not isinstance(raw, dict):
        logger.warning(
            "Model catalog: entry for %r must be a list or an object, got %s "
            "— ignoring.",
            provider,
            type(raw).__name__,
        )
        return None

    entry: dict[str, Any] = {}
    models = raw.get("models")
    if models is not None:
        if not isinstance(models, list) or not all(isinstance(m, str) for m in models):
            logger.warning(
                "Model catalog: %r.models must be a list of strings — ignoring.",
                provider,
            )
            return None
        # Dedupe, preserve order, drop blanks.
        seen: set[str] = set()
        cleaned: list[str] = []
        for m in models:
            m = m.strip()
            if m and m not in seen:
                seen.add(m)
                cleaned.append(m)
        entry["models"] = cleaned

    env_var = raw.get("env_var")
    if env_var is not None:
        if not isinstance(env_var, str):
            logger.warning(
                "Model catalog: %r.env_var must be a string — ignoring it.", provider
            )
        else:
            entry["env_var"] = env_var

    return entry or None


def read_catalog_file(db_path: str | None = None) -> dict[str, Any]:
    """Parse the override file into a raw dict. Never raises.

    A missing file is the normal case and returns ``{}`` silently. A malformed
    one is logged and treated as absent — a typo in a JSON file must not take
    the Playground's model picker offline.
    """
    path = catalog_path(db_path)
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except OSError as exc:
        logger.warning("Model catalog: cannot read %s (%s) — using built-ins.", path, exc)
        return {}

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.warning(
            "Model catalog: %s is not valid JSON (%s) — using built-ins.", path, exc
        )
        return {}

    if not isinstance(data, dict):
        logger.warning(
            "Model catalog: %s must contain a JSON object mapping provider -> models "
            "— using built-ins.",
            path,
        )
        return {}
    return data


def load_overrides(db_path: str | None = None) -> dict[str, dict[str, Any]]:
    """Provider → ``{"models": [...], "env_var": ...}`` overrides from the file."""
    path = catalog_path(db_path)
    data = read_catalog_file(db_path)
    if not data:
        return {}

    known = _known_providers()
    out: dict[str, dict[str, Any]] = {}
    for provider, raw in data.items():
        if provider in RESERVED_KEYS:
            continue  # handled elsewhere (``pricing`` -> fastaiagent.ui.pricing)
        if provider not in known:
            logger.warning(
                "Model catalog: unknown provider %r in %s — ignoring. Known: %s. "
                "New providers are added with "
                "fastaiagent.llm.providers.register_provider().",
                provider,
                path,
                ", ".join(sorted(known)),
            )
            continue
        entry = _coerce_entry(provider, raw)
        if entry:
            out[provider] = entry
    return out


def _known_providers() -> set[str]:
    """Built-ins the Playground can drive, plus every registered preset."""
    from fastaiagent.llm.providers import list_presets

    return set(BUILTIN_MODELS) | {p.key for p in list_presets()}


# ---------------------------------------------------------------------------
# Catalog assembly
# ---------------------------------------------------------------------------


def env_var_for(provider: str, overrides: dict[str, dict[str, Any]] | None = None) -> str | None:
    """Env-var name for ``provider``, or ``None`` if the provider is unknown.

    ``""`` means "no key required" (local providers such as ollama, lmstudio
    and vllm). Presets whose ``env_var`` is blank are treated the same way.
    """
    overrides = overrides or {}
    if provider in overrides and "env_var" in overrides[provider]:
        return str(overrides[provider]["env_var"])
    if provider in BUILTIN_ENV_KEY:
        return BUILTIN_ENV_KEY[provider]

    from fastaiagent.llm.providers import get_preset

    preset = get_preset(provider)
    if preset is None:
        return None
    return preset.env_var or ""


def env_key_is_set(
    provider: str, overrides: dict[str, dict[str, Any]] | None = None
) -> bool:
    """Is a *real* credential present in the environment for ``provider``?

    Unlike :func:`has_api_key` this does not treat local providers as
    satisfied — it answers only "would ``LLMClient`` find a key to send if we
    didn't supply one". Callers use it to decide whether falling back to the
    environment would leak that key somewhere it doesn't belong.
    """
    env_var = env_var_for(provider, overrides)
    if not env_var:
        return False
    return bool(os.environ.get(env_var))


def has_api_key(provider: str, overrides: dict[str, dict[str, Any]] | None = None) -> bool:
    """Whether the server process has a usable key for ``provider``.

    Providers in :data:`LOCAL_PROVIDERS` always qualify — you run the server,
    so there is usually nothing to authenticate against, and gating them on an
    env var made LM Studio and vLLM unselectable in the UI.

    Note this only reflects the *environment*. A request may still carry a
    per-run token (an AI-gateway credential, say), which the route accepts in
    place of the env var.
    """
    if provider in LOCAL_PROVIDERS:
        return True
    env_var = env_var_for(provider, overrides)
    if env_var is None:
        return False
    if env_var == "":
        return True
    return bool(os.environ.get(env_var))


def build_catalog(db_path: str | None = None) -> list[dict[str, Any]]:
    """Rows for ``GET /api/playground/models``.

    Order: built-ins in their declared order, then presets alphabetically.
    Each row carries ``has_key`` so the UI can disable providers whose key is
    missing, and ``env_var`` so it can say which one to set.
    """
    from fastaiagent.llm.providers import list_presets

    overrides = load_overrides(db_path)
    rows: list[dict[str, Any]] = []

    for provider, default_models in BUILTIN_MODELS.items():
        models = overrides.get(provider, {}).get("models", default_models)
        rows.append(
            {
                "provider": provider,
                "models": list(models),
                "has_key": has_api_key(provider, overrides),
                "env_var": env_var_for(provider, overrides) or None,
            }
        )

    for preset in list_presets():
        override_models = overrides.get(preset.key, {}).get("models")
        if override_models is not None:
            models = list(override_models)
        else:
            # Preset's own default first, then curated hints, deduped.
            seen: set[str] = set()
            models = []
            for m in [preset.default_model, *PRESET_MODEL_HINTS.get(preset.key, [])]:
                if m and m not in seen:
                    seen.add(m)
                    models.append(m)
        rows.append(
            {
                "provider": preset.key,
                "models": models,
                "has_key": has_api_key(preset.key, overrides),
                "env_var": env_var_for(preset.key, overrides) or None,
            }
        )

    return rows


__all__ = [
    "BUILTIN_ENV_KEY",
    "BUILTIN_MODELS",
    "CATALOG_ENV_VAR",
    "CATALOG_FILENAME",
    "PRESET_MODEL_HINTS",
    "RESERVED_KEYS",
    "build_catalog",
    "catalog_path",
    "env_key_is_set",
    "env_var_for",
    "has_api_key",
    "load_overrides",
    "read_catalog_file",
]
