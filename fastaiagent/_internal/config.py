"""SDK configuration with environment variable support."""

from __future__ import annotations

import os
from functools import lru_cache

from pydantic import BaseModel, Field


class SDKConfig(BaseModel):
    """Configuration for the FastAIAgent SDK.

    All settings can be overridden via environment variables prefixed with FASTAIAGENT_.
    """

    trace_enabled: bool = Field(default=True)
    trace_db_path: str = Field(default=".fastaiagent/traces.db")
    checkpoint_db_path: str = Field(default=".fastaiagent/checkpoints.db")
    prompt_dir: str = Field(default=".prompts/")
    cache_dir: str = Field(default=".fastaiagent/cache/")
    log_level: str = Field(default="WARNING")
    default_timeout: int = Field(default=120)

    @classmethod
    def from_env(cls) -> SDKConfig:
        """Load configuration from environment variables."""
        env_map = {
            "trace_enabled": ("FASTAIAGENT_TRACE_ENABLED", lambda v: v.lower() in ("1", "true")),
            "trace_db_path": ("FASTAIAGENT_TRACE_DB_PATH", str),
            "checkpoint_db_path": ("FASTAIAGENT_CHECKPOINT_DB_PATH", str),
            "prompt_dir": ("FASTAIAGENT_PROMPT_DIR", str),
            "cache_dir": ("FASTAIAGENT_CACHE_DIR", str),
            "log_level": ("FASTAIAGENT_LOG_LEVEL", str),
            "default_timeout": ("FASTAIAGENT_DEFAULT_TIMEOUT", int),
        }
        kwargs: dict = {}
        for field_name, (env_var, converter) in env_map.items():
            value = os.environ.get(env_var)
            if value is not None:
                kwargs[field_name] = converter(value)
        return cls(**kwargs)


@lru_cache(maxsize=1)
def get_config() -> SDKConfig:
    """Get the SDK configuration singleton."""
    return SDKConfig.from_env()


def reset_config() -> None:
    """Reset the config cache (useful for testing)."""
    get_config.cache_clear()
