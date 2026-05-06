"""
Smoke tests — fast deterministic checks, no live LLM calls.

These exercise the parts of the example that don't need an API key:

  * top-level imports (catches API drift in the SDK public surface)
  * Pydantic schemas round-trip
  * mock web_search returns expected shapes against ``DeepResearchDeps``
  * web_fetch HTML→text stripping (no network — feeds raw HTML through)
  * topology builders construct without error
  * spans.py helpers don't raise when called with a real OTel span

Run from the example directory:

    pytest tests/

Live-LLM coverage lives in ``tests/integration/test_deep_research_e2e.py``
at the repo root and is gated on ``OPENAI_API_KEY``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Make ``import topology / tools / spans / agent`` work whether pytest is run
# from the example dir or from the repo root.
_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


# ─── Module imports ──────────────────────────────────────────────────────────


def test_imports() -> None:
    import agent  # noqa: F401
    import eval_suite  # noqa: F401
    import memory_setup  # noqa: F401
    import replay_demo  # noqa: F401
    import spans  # noqa: F401
    import streaming_demo  # noqa: F401
    import tools  # noqa: F401
    import topology  # noqa: F401


# ─── Pydantic schemas ────────────────────────────────────────────────────────


def test_research_brief_round_trip() -> None:
    from topology import ResearchBrief, Subtopic

    brief = ResearchBrief(
        topic="t",
        summary="s",
        subtopics=[Subtopic(title="a", rationale="r")],
    )
    payload = brief.model_dump_json()
    rebuilt = ResearchBrief.model_validate_json(payload)
    assert rebuilt == brief


def test_research_findings_round_trip() -> None:
    from topology import Citation, ResearchFindings

    f = ResearchFindings(
        subtopic="s",
        summary="ok",
        citations=[Citation(title="t", url="https://x", relevance="r")],
    )
    payload = f.model_dump_json()
    rebuilt = ResearchFindings.model_validate_json(payload)
    assert rebuilt == f


# ─── tools — mock backend + html stripper ────────────────────────────────────


def test_mock_search_corpus_hit() -> None:
    from tools import _mock_search

    hits = _mock_search("retrieval-augmented generation", top_k=2)
    assert len(hits) >= 1
    assert all({"title", "url", "snippet"} <= set(h) for h in hits)


def test_mock_search_corpus_miss_falls_back() -> None:
    from tools import _mock_search

    hits = _mock_search("zzzzz unknown topic xxxxx", top_k=3)
    assert len(hits) == 1
    assert "no high-confidence" in hits[0]["title"]


def test_resolve_backend_auto_picks_mock_without_key(monkeypatch) -> None:
    from tools import _resolve_backend

    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    assert _resolve_backend("auto") == "mock"


def test_strip_html_drops_script_and_returns_visible_text() -> None:
    from tools import _strip_html

    html = """<html><head><script>alert(1)</script><style>x{}</style></head>
              <body><p>hello</p><p>world</p></body></html>"""
    txt = _strip_html(html)
    assert "alert" not in txt
    assert "x{}" not in txt
    assert "hello" in txt
    assert "world" in txt


# ─── topology builders ───────────────────────────────────────────────────────


def test_build_scope_agent() -> None:
    from topology import ResearchBrief, build_scope_agent

    agent = build_scope_agent()
    assert agent.name == "scope"
    assert agent.output_type is ResearchBrief


def test_build_researcher() -> None:
    from topology import ResearchFindings, build_researcher

    agent = build_researcher("subtopic-x", "because")
    assert "subtopic-x" in agent.name
    # web_search + web_fetch
    tool_names = {t.name for t in agent.tools}
    assert "web_search" in tool_names
    assert "web_fetch" in tool_names
    assert agent.output_type is ResearchFindings


def test_build_writer_agent() -> None:
    from topology import build_writer_agent

    agent = build_writer_agent()
    assert agent.name == "writer"
    # Writer takes structured findings as input but emits Markdown text,
    # so output_type stays None (free-form).
    assert agent.output_type is None


# ─── spans.py — real OTel round-trip ─────────────────────────────────────────


def test_spans_helpers_set_attributes_via_real_otel() -> None:
    """Use the real OTel tracer to verify our helpers emit the expected
    attribute keys. We reset the tracer to ensure a clean provider, then
    drive ``trace_context`` and inspect the span before exit."""
    import spans
    from topology import Citation, ResearchBrief, ResearchFindings, Subtopic

    from fastaiagent.trace.otel import reset
    from fastaiagent.trace.tracer import trace_context

    reset()
    brief = ResearchBrief(
        topic="t", summary="s", subtopics=[Subtopic(title="a", rationale="r")]
    )
    findings = ResearchFindings(
        subtopic="a",
        summary="ok",
        citations=[Citation(title="x", url="https://y", relevance="z")],
    )

    captured: dict[str, object] = {}

    def _capture(span):
        # Wrap span.set_attribute so we can capture without depending on
        # OTel internals. The OTel SDK accepts unknown attrs; we just want
        # to confirm our helper writes the right keys with the right
        # JSON-encoded shape.
        original = span.set_attribute

        def wrapper(key, value):
            captured[key] = value
            return original(key, value)

        span.set_attribute = wrapper
        return span

    with trace_context("test.research") as span:
        span = _capture(span)
        spans.set_topic(span, "the-topic")
        spans.set_brief(span, brief)
        spans.set_plan(span, brief)
        spans.set_subtopic(span, "a")
        spans.set_findings(span, findings)
        spans.set_report_metadata(span, "hello [1] world [2] more [3] text")

    assert captured[spans.ATTR_TOPIC] == "the-topic"
    assert captured[spans.ATTR_SUBTOPIC] == "a"
    # JSON payloads — verify they round-trip.
    brief_back = json.loads(captured[spans.ATTR_BRIEF])
    assert brief_back["topic"] == "t"
    plan_back = json.loads(captured[spans.ATTR_PLAN])
    assert len(plan_back["subtopics"]) == 1
    findings_back = json.loads(captured[spans.ATTR_FINDINGS])
    assert findings_back["subtopic"] == "a"
    assert findings_back["citations"][0]["url"] == "https://y"
    # Report metadata is structural, not JSON.
    assert captured[spans.ATTR_REPORT_LEN] == len("hello [1] world [2] more [3] text")
    assert captured[spans.ATTR_REPORT_CITATIONS] == 3
