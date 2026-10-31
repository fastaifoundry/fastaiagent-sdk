"""Tests for framework integration enable/disable functions."""

from __future__ import annotations

import importlib

import pytest


def _can_import(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _can_import("openai"), reason="openai not installed")
class TestOpenAIIntegration:
    def test_enable_disable_cycle(self) -> None:
        """enable() then disable() should restore original state."""
        from fastaiagent.integrations import openai as openai_int

        openai_int.disable()  # ensure clean state
        openai_int.enable()
        openai_int.disable()

    def test_double_enable_is_safe(self) -> None:
        """Calling enable() twice should not error."""
        from fastaiagent.integrations import openai as openai_int

        openai_int.enable()
        openai_int.enable()  # should not crash
        openai_int.disable()


@pytest.mark.skipif(not _can_import("anthropic"), reason="anthropic not installed")
class TestAnthropicIntegration:
    def test_enable_disable_cycle(self) -> None:
        from fastaiagent.integrations import anthropic as anthropic_int

        anthropic_int.disable()
        anthropic_int.enable()
        anthropic_int.disable()

    def test_double_enable_is_safe(self) -> None:
        from fastaiagent.integrations import anthropic as anthropic_int

        anthropic_int.enable()
        anthropic_int.enable()
        anthropic_int.disable()


def test_the_framework_integrations_are_gone() -> None:
    """1.96.0 removed the LangChain / CrewAI / Pydantic AI adapters."""
    import fastaiagent.integrations as integrations

    assert integrations.__all__ == ["anthropic", "openai"]
    for name in ("langchain", "crewai", "pydanticai"):
        with pytest.raises(AttributeError):
            getattr(integrations, name)
