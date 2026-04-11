"""End-to-end quality gate — PromptRegistry (local + platform).

Exercises two paths:

- **Local registry** — register a prompt + fragments, load it back,
  format with variables. No platform dependency; runs on CI.
- **Platform registry** — publish a prompt to the platform, fetch it
  back via ``source="platform"``, format with variables. Platform-gated
  so CI with ``E2E_SKIP_PLATFORM=1`` skips the platform half cleanly.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest

from tests.e2e.conftest import require_env, require_platform

pytestmark = pytest.mark.e2e


class TestPromptRegistryLocalGate:
    """Local file-based prompt registry — no platform required."""

    def test_01_register_and_load_with_fragment(self, tmp_path: Path) -> None:
        require_env()
        from fastaiagent import PromptRegistry

        reg = PromptRegistry(path=str(tmp_path / ".prompts"))
        reg.register_fragment(
            name="tone", content="Be professional and concise."
        )
        reg.register(
            name="greeting",
            template="Hello {{name}}. {{@tone}}",
            fragments=["tone"],
        )
        prompt = reg.load("greeting")
        assert prompt is not None
        assert prompt.version >= 1

        formatted = prompt.format(name="World")
        assert "Hello World." in formatted, (
            f"variable substitution failed: {formatted!r}"
        )
        assert "professional" in formatted.lower(), (
            f"fragment was not resolved: {formatted!r}"
        )

    def test_02_register_bumps_version(self, tmp_path: Path) -> None:
        require_env()
        from fastaiagent import PromptRegistry

        reg = PromptRegistry(path=str(tmp_path / ".prompts"))
        reg.register(name="versioned", template="v1 template")
        reg.register(name="versioned", template="v2 template")
        reg.register(name="versioned", template="v3 template")

        latest = reg.load("versioned")
        assert latest.version == 3, (
            f"expected version 3 after three registers, got {latest.version}"
        )


class TestPromptRegistryPlatformGate:
    """Platform-side publish/fetch round-trip. Platform-gated."""

    def test_01_publish_and_fetch(self, gate_state: dict[str, Any]) -> None:
        require_env()
        require_platform()
        import fastaiagent as fa
        import os
        from fastaiagent import PromptRegistry

        # Connect to the configured platform so publish/fetch have a target.
        fa.connect(
            api_key=os.environ["FASTAIAGENT_API_KEY"],
            target=os.environ["FASTAIAGENT_TARGET"],
        )
        assert fa.is_connected, "fa.connect failed for prompt registry gate"

        reg = PromptRegistry()
        slug = f"gate-prompt-{int(time.time())}"

        reg.publish(
            slug=slug,
            content="You are a {{role}} for {{company}}. Be concise.",
            variables=["role", "company"],
        )

        fetched = reg.get(slug, source="platform")
        assert fetched is not None
        assert fetched.variables == ["role", "company"] or set(fetched.variables) == {
            "role",
            "company",
        }, f"variables did not round-trip: {fetched.variables}"

        formatted = fetched.format(role="support agent", company="Acme Corp")
        assert "support agent" in formatted
        assert "Acme Corp" in formatted
