"""
Smoke tests — no live LLM calls.

Exercise the deterministic parts of the example so a developer iterating on
prompts / tools / scorers gets fast feedback before they spend tokens on
``eval_suite.py`` or live runs.

Run from the example directory:

    pytest tests/

What's covered:
  * top-level imports of every module the example exposes
  * Supervisor + Worker construction (catches API drift)
  * Mock web_search keyword matching against ``ResearchDeps`` trail
  * RequiredSourcesScorer scoring logic — pure Python, no LLM
  * Eval dataset shape

What's NOT covered:
  * full supervisor.arun loop — that's eval_suite.py's job
  * real Tavily / Brave / Serper backends — they hit live HTTP
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Make ``import topology / tools / ...`` work whether pytest is run from
# the example dir or from the repo root.
_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


# ─── Module imports ──────────────────────────────────────────────────────────


def test_imports() -> None:
    import agent  # noqa: F401
    import eval_suite  # noqa: F401
    import streaming_demo  # noqa: F401
    import tools  # noqa: F401
    import topology  # noqa: F401


# ─── Topology shape ─────────────────────────────────────────────────────────


def test_supervisor_topology() -> None:
    from topology import build_supervisor

    sup = build_supervisor()
    assert sup.name == "research-team"
    roles = [w.role for w in sup.workers]
    assert roles == ["researcher", "writer", "verifier"], (
        "Order matters: the SUPERVISOR_PROMPT depends on this delegation order."
    )

    # The researcher must be tool-equipped (web_search). Writer/verifier
    # are LLM-only by design.
    researcher = next(w for w in sup.workers if w.role == "researcher")
    assert any(t.name == "web_search" for t in researcher.agent.tools)

    # max_delegation_rounds must leave room for the revision loop:
    # researcher + writer + verifier + (writer + verifier) × 2 = 7
    assert sup.max_delegation_rounds >= 7


# ─── Mock web_search backend (no LLM, no network) ────────────────────────────


def test_mock_search_hits_corpus_keywords() -> None:
    from tools import _mock_search

    # Each EVAL_CASES topic must be discoverable in the mock corpus.
    for query in [
        "Transformer architecture",
        "Retrieval-augmented generation",
        "Constitutional AI",
    ]:
        results = _mock_search(query, top_k=3)
        assert results, f"mock corpus has no entry for {query!r}"
        # Real search backends don't guarantee a URL, but our mock always provides one.
        assert all(r.get("url", "").startswith("http") for r in results)


def test_mock_search_fallback_for_unknown_query() -> None:
    from tools import _mock_search

    results = _mock_search("totally novel quantum-spaceship-language-model", top_k=3)
    assert results, "fallback should return at least one stub result"
    # The fallback flag — used in tests + by the writer to know retrieval was empty.
    assert "no high-confidence sources" in results[0]["title"].lower()


def test_web_search_tool_appends_to_trail() -> None:
    """The verifier-audit pattern depends on the search trail being mutated
    by every web_search call. If the tool ever stops appending, the verifier
    can't cross-check claims against actual retrievals."""
    import asyncio

    import fastaiagent as fa
    from tools import ResearchDeps, web_search

    deps = ResearchDeps(backend="mock", top_k=2)
    ctx = fa.RunContext(state=deps)

    async def _go():
        # web_search is decorated with @fa.tool — call its underlying fn via
        # the tool wrapper's aexecute path so RunContext injection happens.
        return await web_search.aexecute(
            {"query": "Transformer architecture"}, context=ctx
        )

    result = asyncio.run(_go())
    # FunctionTool.aexecute returns ToolResult(output=<str JSON>) for our tool.
    assert result.error is None, f"web_search errored: {result.error}"
    payload = json.loads(result.output) if isinstance(result.output, str) else result.output
    assert isinstance(payload, list) and len(payload) > 0
    assert len(deps.trail) > 0, "web_search must populate the trail for the verifier audit"


# ─── Custom scorer (no LLM) ──────────────────────────────────────────────────


def test_required_sources_scorer_pass() -> None:
    from eval_suite import RequiredSourcesScorer

    scorer = RequiredSourcesScorer(
        required_for_case={"Topic A": ["https://example.com/paper-1"]}
    )
    result = scorer.score(
        input="Topic A",
        output="See https://example.com/paper-1 for details.",
    )
    assert result.passed
    assert result.score == 1.0


def test_required_sources_scorer_fail() -> None:
    from eval_suite import RequiredSourcesScorer

    scorer = RequiredSourcesScorer(
        required_for_case={"Topic A": ["https://example.com/paper-1"]}
    )
    result = scorer.score(
        input="Topic A",
        output="Some report text without the canonical link.",
    )
    assert not result.passed
    assert result.score == 0.0


def test_required_sources_scorer_partial() -> None:
    from eval_suite import RequiredSourcesScorer

    scorer = RequiredSourcesScorer(
        required_for_case={
            "Topic A": [
                "https://example.com/paper-1",
                "https://example.com/paper-2",
            ]
        }
    )
    result = scorer.score(
        input="Topic A",
        output="Only [paper-1](https://example.com/paper-1) cited.",
    )
    assert not result.passed  # threshold defaults to 1.0 (full coverage)
    assert result.score == 0.5


def test_required_sources_scorer_no_requirement_passes() -> None:
    from eval_suite import RequiredSourcesScorer

    scorer = RequiredSourcesScorer(required_for_case={})
    result = scorer.score(input="Untracked topic", output="Anything goes here.")
    assert result.passed
    assert result.score == 1.0


# ─── Dataset shape ──────────────────────────────────────────────────────────


def test_eval_cases_well_formed() -> None:
    from eval_suite import EVAL_CASES

    assert len(EVAL_CASES) >= 3
    for case in EVAL_CASES:
        assert "input" in case
        assert "expected" in case
        assert "required_sources" in case
        assert isinstance(case["required_sources"], list)
        assert all(u.startswith("http") for u in case["required_sources"])
