"""Smoke tests — no live LLM calls.

Coverage:
  * shared/ assets import and are well-formed
  * each integration's public API surface is what we expect
  * support-prompt registers + reads back unchanged
  * support-kb (default path) is populated with the fixture content
  * each framework example imports without error if the framework is installed;
    skipped otherwise (so a missing optional dep doesn't fail the suite)

The framework sub-examples themselves can't be exercised end-to-end without
an LLM call — those live in eval_suite.py.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

import fastaiagent as fa

_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


# ─── Shared assets ──────────────────────────────────────────────────────────


def test_shared_imports() -> None:
    from shared.guardrails import input_guardrails, output_guardrails
    from shared.kb import support_kb
    from shared.prompts import register_support_prompt

    assert support_kb.name == "support-kb"
    assert support_kb.status()["chunk_count"] > 0, "shared KB should be seeded"
    assert len(input_guardrails) == 1
    assert len(output_guardrails) == 1
    # Idempotent across calls.
    text1 = register_support_prompt()
    text2 = register_support_prompt()
    assert text1 == text2 and len(text1) > 50


def test_kb_lives_at_default_path() -> None:
    """The integrations re-instantiate ``LocalKB(name=...)`` with the default
    path, so the shared KB MUST also be at the default path or the
    integration retriever / tool will see an empty store."""
    from shared.kb import support_kb

    # Default path is computed by the SDK; ours should match what
    # LocalKB(name="support-kb") with no path= would compute.
    default_kb = fa.LocalKB(name="support-kb")
    assert default_kb.status()["chunk_count"] == support_kb.status()["chunk_count"]


def test_prompt_registry_round_trip() -> None:
    from shared.prompts import PROMPT_SLUG, register_support_prompt

    register_support_prompt()
    registry = fa.PromptRegistry()
    prompt = registry.get(PROMPT_SLUG, source="local")
    assert prompt.name == PROMPT_SLUG
    assert "knowledge base" in prompt.template.lower()


# ─── Integration public API ─────────────────────────────────────────────────


def test_langchain_integration_surface() -> None:
    from fastaiagent.integrations import langchain as lc_int

    for name in (
        "enable", "disable", "as_evaluable", "with_guardrails",
        "prompt_from_registry", "register_agent", "kb_as_retriever",
        "get_callback_handler",
    ):
        assert hasattr(lc_int, name), f"langchain integration missing {name}"


def test_crewai_integration_surface() -> None:
    from fastaiagent.integrations import crewai as ca_int

    for name in (
        "enable", "disable", "as_evaluable", "with_guardrails",
        "prompt_from_registry", "register_agent", "kb_as_tool",
    ):
        assert hasattr(ca_int, name), f"crewai integration missing {name}"


def test_pydanticai_integration_surface() -> None:
    from fastaiagent.integrations import pydanticai as pa_int

    for name in (
        "enable", "disable", "as_evaluable", "with_guardrails",
        "prompt_from_registry", "register_agent", "kb_as_tool",
    ):
        assert hasattr(pa_int, name), f"pydanticai integration missing {name}"


# ─── Sub-examples import (gated on optional deps) ───────────────────────────


def _has(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


@pytest.mark.skipif(
    not (_has("langchain") and _has("langchain_openai") and _has("langgraph")),
    reason="langchain / langchain-openai / langgraph not installed",
)
def test_langchain_example_imports() -> None:
    import langchain_example  # noqa: F401


@pytest.mark.skipif(not _has("crewai"), reason="crewai not installed")
def test_crewai_example_imports() -> None:
    import crewai_example  # noqa: F401


@pytest.mark.skipif(not _has("pydantic_ai"), reason="pydantic-ai not installed")
def test_pydanticai_example_imports() -> None:
    import pydanticai_example  # noqa: F401


# ─── Eval-suite dataset shape ───────────────────────────────────────────────


def test_eval_dataset_shape() -> None:
    from eval_suite import EVAL_DATASET

    assert len(EVAL_DATASET) >= 3
    for row in EVAL_DATASET:
        assert "input" in row
        assert "expected" in row
        assert isinstance(row["input"], str)
        assert isinstance(row["expected"], str)
