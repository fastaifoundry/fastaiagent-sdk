"""SDK configuration with environment variable support.

Reachable publicly as ``fastaiagent.config`` (see ``docs/configuration/sdk-config.md``).
The model is mutable and :func:`get_config` is ``lru_cache``d, so
``fa.config.trace_full_images = True`` sticks for the process.
"""

from __future__ import annotations

import os
import warnings
from functools import lru_cache
from typing import Any

from pydantic import BaseModel, Field

from fastaiagent._internal.env import env_flag, env_path


def _ui_enabled_env() -> bool:
    """Resolver for ``FASTAIAGENT_UI_ENABLED`` (registered in ``ENV_FLAGS``)."""
    return env_flag("FASTAIAGENT_UI_ENABLED", default=False)


def _trace_full_images_env() -> bool:
    """Resolver for ``FASTAIAGENT_TRACE_FULL_IMAGES`` (registered in ``ENV_FLAGS``)."""
    return env_flag("FASTAIAGENT_TRACE_FULL_IMAGES", default=False)


def _trace_enabled_env() -> bool:
    """Resolver for ``FASTAIAGENT_TRACE_ENABLED``.

    Fails closed on an unparseable value: the master switch is the strongest
    privacy control the SDK has, so ``FASTAIAGENT_TRACE_ENABLED=flase`` must not
    keep capturing.
    """
    return env_flag("FASTAIAGENT_TRACE_ENABLED", default=True, on_unparsed=False)


class SDKConfig(BaseModel):
    """Configuration for the FastAIAgent SDK.

    All settings can be overridden via environment variables prefixed with FASTAIAGENT_.
    """

    trace_enabled: bool = Field(default=True)

    local_db_path: str = Field(default=".fastaiagent/local.db")

    trace_db_path: str | None = Field(default=None)
    checkpoint_db_path: str | None = Field(default=None)
    prompt_dir: str | None = Field(default=None)

    #: Knowledge-base collections root. Read by the local UI's KB routes and by
    #: the agent-detail payload; ``FASTAIAGENT_KB_DIR`` overrides it.
    kb_dir: str = Field(default=".fastaiagent/kb")

    ui_enabled: bool = Field(default=False)
    ui_host: str = Field(default="127.0.0.1")
    ui_port: int = Field(default=7842)

    # ── Parsed but not consulted by the SDK runtime ───────────────────────
    # Kept because they are public fields with shipped environment variables
    # (removing them would break anyone reading ``get_config().cache_dir``), but
    # nothing in the library reads them: the SDK has no disk cache of its own,
    # configures no logging handler, and every HTTP timeout is passed explicitly
    # at its call site. ``docs/configuration/environment-variables.md`` says so
    # rather than implying they do something.
    cache_dir: str = Field(default=".fastaiagent/cache/")
    log_level: str = Field(default="WARNING")
    default_timeout: int = Field(default=120)

    # Multimodal config — see docs/multimodal/ and docs/configuration/sdk-config.md.
    pdf_mode: str = Field(default="auto")
    #: ``None`` means "use the provider's own per-image cap" (Anthropic 5 MB,
    #: OpenAI 20 MB, …), which is what ``multimodal.format`` has always meant by
    #: ``max_image_size_mb=None``. Set a number to override every provider.
    max_image_size_mb: float | None = Field(default=None)
    max_pdf_pages: int = Field(default=20)
    trace_full_images: bool = Field(default=False)

    @property
    def resolved_trace_db_path(self) -> str:
        return self.trace_db_path or self.local_db_path

    @property
    def resolved_checkpoint_db_path(self) -> str:
        return self.checkpoint_db_path or self.local_db_path

    @classmethod
    def from_env(cls) -> SDKConfig:
        """Load configuration from environment variables.

        Path fields go through :func:`fastaiagent._internal.env.env_path`, which
        expands ``~`` and ``$VARS``. Before 1.67.0 they were passed through a
        bare ``str``, so ``FASTAIAGENT_LOCAL_DB=~/x/local.db`` created a literal
        ``./~/`` directory under whatever the current working directory happened
        to be — and a resume from a different directory found nothing.
        """
        kwargs: dict[str, Any] = {}

        path_map = {
            "local_db_path": "FASTAIAGENT_LOCAL_DB",
            "trace_db_path": "FASTAIAGENT_TRACE_DB_PATH",
            "checkpoint_db_path": "FASTAIAGENT_CHECKPOINT_DB_PATH",
            "prompt_dir": "FASTAIAGENT_PROMPT_DIR",
            "kb_dir": "FASTAIAGENT_KB_DIR",
            "cache_dir": "FASTAIAGENT_CACHE_DIR",
        }
        for field_name, env_var in path_map.items():
            value = env_path(env_var)
            if value is not None:
                kwargs[field_name] = value

        str_map = {
            "ui_host": "FASTAIAGENT_UI_HOST",
            "log_level": "FASTAIAGENT_LOG_LEVEL",
            "pdf_mode": "FASTAIAGENT_PDF_MODE",
        }
        for field_name, env_var in str_map.items():
            raw = os.environ.get(env_var)
            if raw is not None:
                kwargs[field_name] = raw

        num_map: dict[str, tuple[str, Any]] = {
            "ui_port": ("FASTAIAGENT_UI_PORT", int),
            "default_timeout": ("FASTAIAGENT_DEFAULT_TIMEOUT", int),
            "max_pdf_pages": ("FASTAIAGENT_MAX_PDF_PAGES", int),
            "max_image_size_mb": ("FASTAIAGENT_MAX_IMAGE_SIZE_MB", float),
        }
        for field_name, (env_var, caster) in num_map.items():
            raw = os.environ.get(env_var)
            if raw is not None and raw.strip():
                try:
                    kwargs[field_name] = caster(raw.strip())
                except ValueError:
                    warnings.warn(
                        f"{env_var}={raw!r} is not a valid {caster.__name__}; ignoring it.",
                        UserWarning,
                        stacklevel=2,
                    )

        if "FASTAIAGENT_TRACE_ENABLED" in os.environ:
            kwargs["trace_enabled"] = _trace_enabled_env()
        if "FASTAIAGENT_UI_ENABLED" in os.environ:
            kwargs["ui_enabled"] = _ui_enabled_env()
        if "FASTAIAGENT_TRACE_FULL_IMAGES" in os.environ:
            kwargs["trace_full_images"] = _trace_full_images_env()

        legacy_vars = (
            "FASTAIAGENT_TRACE_DB_PATH",
            "FASTAIAGENT_CHECKPOINT_DB_PATH",
            "FASTAIAGENT_PROMPT_DIR",
        )
        for legacy_var in legacy_vars:
            if legacy_var in os.environ:
                warnings.warn(
                    f"{legacy_var} is deprecated; use FASTAIAGENT_LOCAL_DB to point all "
                    f"local storage at a single SQLite file.",
                    DeprecationWarning,
                    stacklevel=2,
                )
        return cls(**kwargs)


@lru_cache(maxsize=1)
def get_config() -> SDKConfig:
    """Get the SDK configuration singleton."""
    return SDKConfig.from_env()


def reset_config() -> None:
    """Reset the config cache (useful for testing)."""
    get_config.cache_clear()
