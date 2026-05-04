"""Phase 7 — ``prompt_from_registry()`` helper tests (real LLM where needed).

Spec test IDs covered: #27 (template content match), #28 (lineage attrs
on the LLM span when the registry-backed template is rendered inside a
LangGraph run).

#27 is fully offline (no LLM) — it only verifies the template content
round-trips through the registry. #28 needs a real LLM call so the
LangChain handler emits an LLM span we can inspect.
"""

from __future__ import annotations

import os
import time
import uuid

import pytest

HAS_OPENAI = bool(os.environ.get("OPENAI_API_KEY"))

needs_openai = pytest.mark.skipif(not HAS_OPENAI, reason="OPENAI_API_KEY not set")

pytestmark = pytest.mark.e2e


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def seeded_prompt() -> tuple[str, str]:
    """Insert a one-off prompt into the registry, return its slug + body."""
    from fastaiagent.prompt import PromptRegistry

    slug = f"harness-prompt-{uuid.uuid4().hex[:8]}"
    body = (
        "You are a terse assistant. Reply with exactly one word: {{topic}}."
    )
    reg = PromptRegistry()
    reg.register(name=slug, template=body)
    return slug, body


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_27_langchain_template_content(seeded_prompt: tuple[str, str]) -> None:
    """Spec #27: prompt_from_registry returns a ChatPromptTemplate
    whose content matches the registry version verbatim."""
    from langchain_core.prompts import ChatPromptTemplate

    from fastaiagent.integrations import langchain as lc

    slug, body = seeded_prompt
    template = lc.prompt_from_registry(slug)
    assert isinstance(template, ChatPromptTemplate), type(template)

    # Mustache placeholders parse out as the same input variable name.
    assert "topic" in template.input_variables, template.input_variables

    # Render and confirm the literal body survived.
    rendered = template.format_messages(topic="ping")
    assert len(rendered) == 1
    text = rendered[0].content
    assert "terse assistant" in text and "ping" in text, text
    # And the original mustache template body matches what we registered.
    body_clean = body.replace("{{topic}}", "ping")
    assert text == body_clean


def test_27b_crewai_and_pydanticai_string_form(seeded_prompt: tuple[str, str]) -> None:
    """The CrewAI / PydanticAI helpers return raw template strings —
    the caller drops them straight into Agent backstory / system_prompt."""
    from fastaiagent.integrations import crewai as ca
    from fastaiagent.integrations import pydanticai as pa

    slug, body = seeded_prompt
    assert ca.prompt_from_registry(slug) == body
    assert pa.prompt_from_registry(slug) == body


def _trace_store():
    from fastaiagent.trace.storage import TraceStore

    return TraceStore.default()


def _wait_for_root_span(predicate, timeout: float = 10.0):
    store = _trace_store()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for summary in store.list_traces():
            try:
                trace = store.get_trace(summary.trace_id)
            except Exception:
                continue
            for span in trace.spans:
                if predicate(span):
                    return trace
        time.sleep(0.2)
    return None


@needs_openai
def test_28_lineage_attrs_on_llm_span(seeded_prompt: tuple[str, str]) -> None:
    """Spec #28: when a registry prompt is used in a traced LangGraph
    run, the LLM span carries ``fastaiagent.prompt.slug`` and
    ``fastaiagent.prompt.version`` so the Prompt detail page's
    "Traces using this prompt" panel can find it."""
    from langchain_openai import ChatOpenAI

    from fastaiagent.integrations import langchain as lc

    lc.enable()
    handler = lc.get_callback_handler()

    slug, _body = seeded_prompt
    template = lc.prompt_from_registry(slug)
    # The chain renders the template (which stamps the slug/version on
    # the current span) and feeds the resulting messages into the LLM,
    # whose call produces an llm.* span as a child.
    chain = template | ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain.invoke({"topic": "blue"}, config={"callbacks": [handler]})

    # Wait until the chain's trace lands and find a span whose attrs
    # carry our slug.
    def predicate(span) -> bool:
        attrs = span.attributes or {}
        return attrs.get("fastaiagent.prompt.slug") == slug

    trace = _wait_for_root_span(predicate, timeout=10.0)
    assert trace is not None, (
        f"no span tagged with fastaiagent.prompt.slug={slug!r} after invoke()"
    )
    matched = [s for s in trace.spans if (s.attributes or {}).get(
        "fastaiagent.prompt.slug"
    ) == slug]
    assert matched
    span = matched[0]
    version = (span.attributes or {}).get("fastaiagent.prompt.version")
    assert version is not None and int(version) >= 1, span.attributes
