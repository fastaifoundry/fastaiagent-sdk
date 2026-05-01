"""SDK configuration with environment variable support."""

from __future__ import annotations

import os
import warnings
from functools import lru_cache
from typing import Any

from pydantic import BaseModel, Field


class SDKConfig(BaseModel):
    """Configuration for the FastAIAgent SDK.

    All settings can be overridden via environment variables prefixed with FASTAIAGENT_.
    """

    trace_enabled: bool = Field(default=True)

    local_db_path: str = Field(default=".fastaiagent/local.db")

    trace_db_path: str | None = Field(default=None)
    checkpoint_db_path: str | None = Field(default=None)
    prompt_dir: str | None = Field(default=None)

    ui_enabled: bool = Field(default=False)
    ui_host: str = Field(default="127.0.0.1")
    ui_port: int = Field(default=7842)

    cache_dir: str = Field(default=".fastaiagent/cache/")
    log_level: str = Field(default="WARNING")
    default_timeout: int = Field(default=120)

    # Multimodal config — see docs/multimodal/.
    pdf_mode: str = Field(default="auto")
    max_image_size_mb: float = Field(default=20.0)
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
        """Load configuration from environment variables."""
        env_map = {
            "trace_enabled": ("FASTAIAGENT_TRACE_ENABLED", lambda v: v.lower() in ("1", "true")),
            "local_db_path": ("FASTAIAGENT_LOCAL_DB", str),
            "trace_db_path": ("FASTAIAGENT_TRACE_DB_PATH", str),
            "checkpoint_db_path": ("FASTAIAGENT_CHECKPOINT_DB_PATH", str),
            "prompt_dir": ("FASTAIAGENT_PROMPT_DIR", str),
            "ui_enabled": ("FASTAIAGENT_UI_ENABLED", lambda v: v.lower() in ("1", "true")),
            "ui_host": ("FASTAIAGENT_UI_HOST", str),
            "ui_port": ("FASTAIAGENT_UI_PORT", int),
            "cache_dir": ("FASTAIAGENT_CACHE_DIR", str),
            "log_level": ("FASTAIAGENT_LOG_LEVEL", str),
            "default_timeout": ("FASTAIAGENT_DEFAULT_TIMEOUT", int),
        }
        kwargs: dict[str, Any] = {}
        for field_name, (env_var, converter) in env_map.items():
            value = os.environ.get(env_var)
            if value is not None:
                kwargs[field_name] = converter(value)  # type: ignore[operator]

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
