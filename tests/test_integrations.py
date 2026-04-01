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


@pytest.mark.skipif(not _can_import("langchain_core"), reason="langchain-core not installed")
class TestLangChainIntegration:
    def test_enable_disable_cycle(self) -> None:
        from fastaiagent.integrations import langchain as lc_int

        lc_int.disable()
        lc_int.enable()
        lc_int.disable()
