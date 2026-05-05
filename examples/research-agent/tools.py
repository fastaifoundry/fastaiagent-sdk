"""
Tools — what the research workers can call.

Today this ships with a fully mocked ``web_search`` backend so the example
runs offline and deterministically. The mock covers the topics the
``eval_suite.py`` golden cases ask about; for any other query it returns a
plausible-but-empty result so the writer/verifier loop still exercises.

Swap in a real backend by:

  1. Setting ``SEARCH_BACKEND`` in your .env to ``tavily`` / ``brave`` / ``serper``.
  2. Filling in the corresponding ``_real_search_*`` function below
     (template stubs included — they call the provider's HTTP API).
  3. Uncommenting the matching env-var line in .env.example.

The Agent contract is unchanged either way: ``web_search(query: str)`` returns
a JSON-serializable list of result dicts. The writer & verifier never know or
care which backend produced them.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import fastaiagent as fa


# ─── Mock search corpus ──────────────────────────────────────────────────────
#
# Each entry maps a (lower-cased) keyword that may appear in a query to a
# small ranked list of "search results". This is deliberately shallow — the
# point is to give the researcher worker something to chew on, the writer
# something to cite, and the verifier something to verify.

_MOCK_CORPUS: dict[str, list[dict]] = {
    "transformer": [
        {
            "title": "Attention Is All You Need (2017)",
            "url": "https://arxiv.org/abs/1706.03762",
            "snippet": (
                "Vaswani et al. introduce the Transformer, a sequence-to-sequence "
                "architecture based entirely on attention, removing recurrence and "
                "convolution. Achieves SOTA on WMT 2014 English-to-German."
            ),
        },
        {
            "title": "The Illustrated Transformer — Jay Alammar",
            "url": "https://jalammar.github.io/illustrated-transformer/",
            "snippet": (
                "Visual walkthrough of self-attention, multi-head attention, "
                "positional encoding, and the encoder-decoder stack."
            ),
        },
        {
            "title": "A Survey of Transformers (2023)",
            "url": "https://arxiv.org/abs/2106.04554",
            "snippet": (
                "Taxonomy of Transformer variants — efficient attention, sparse "
                "attention, retrieval-augmented, vision/audio/multimodal extensions."
            ),
        },
    ],
    "retrieval-augmented generation": [
        {
            "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP (2020)",
            "url": "https://arxiv.org/abs/2005.11401",
            "snippet": (
                "Lewis et al. propose RAG, combining a parametric seq2seq model "
                "with a non-parametric memory (Wikipedia) accessed via DPR."
            ),
        },
        {
            "title": "Lost in the Middle: How LLMs Use Long Contexts (2023)",
            "url": "https://arxiv.org/abs/2307.03172",
            "snippet": (
                "Liu et al. show that models attend more to the start and end of "
                "long contexts than the middle, which has direct implications for "
                "RAG chunk ordering."
            ),
        },
        {
            "title": "Self-RAG: Self-Reflective Retrieval-Augmented Generation (2023)",
            "url": "https://arxiv.org/abs/2310.11511",
            "snippet": (
                "Asai et al. add reflection tokens that let the model decide when "
                "to retrieve, when to critique its draft, and when to stop."
            ),
        },
    ],
    "constitutional ai": [
        {
            "title": "Constitutional AI: Harmlessness from AI Feedback (2022)",
            "url": "https://arxiv.org/abs/2212.08073",
            "snippet": (
                "Bai et al. (Anthropic) train a harmless assistant using a written "
                "constitution — the model critiques and revises its own outputs "
                "against the principles, eliminating the need for human harm labels."
            ),
        },
        {
            "title": "Anthropic — Claude's Constitution",
            "url": "https://www.anthropic.com/news/claudes-constitution",
            "snippet": (
                "Public listing of the principles Claude is trained against, "
                "drawn from sources like the UN Declaration of Human Rights."
            ),
        },
    ],
    "agent eval": [
        {
            "title": "AgentBench: Evaluating LLMs as Agents (2023)",
            "url": "https://arxiv.org/abs/2308.03688",
            "snippet": (
                "Liu et al. propose 8 distinct environments for LLM-agent eval — "
                "OS, DB, knowledge graph, card game, lateral-thinking puzzles, etc."
            ),
        },
        {
            "title": "tau-bench: A Benchmark for Tool-Agent-User Interaction (2024)",
            "url": "https://arxiv.org/abs/2406.12045",
            "snippet": (
                "Yao et al. evaluate agents in real-world domain settings (airline, "
                "retail) with both tool-use and dialogue with simulated users."
            ),
        },
    ],
}


def _mock_search(query: str, top_k: int) -> list[dict]:
    q = query.lower()
    for keyword, hits in _MOCK_CORPUS.items():
        if keyword in q:
            return hits[:top_k]
    # Fallback — keep the writer/verifier loop on a defined surface area
    # rather than returning random text. Encourage citations even on misses.
    return [
        {
            "title": "(no high-confidence sources for this query)",
            "url": "https://example.com/no-results",
            "snippet": (
                "The mock search backend has no entries for this query. "
                "Plug in a real provider in tools.py to extend coverage."
            ),
        }
    ]


# ─── Real-backend stubs ──────────────────────────────────────────────────────
#
# Each of these is intentionally a thin wrapper around the provider's HTTP
# API. Fill in the body, set SEARCH_BACKEND in .env, and you're live.


def _real_search_tavily(query: str, top_k: int) -> list[dict]:
    raise NotImplementedError(
        "Implement Tavily search: POST https://api.tavily.com/search with "
        "{api_key, query, max_results=top_k}, then map response['results'] to "
        "[{title, url, snippet=content}]."
    )


def _real_search_brave(query: str, top_k: int) -> list[dict]:
    raise NotImplementedError(
        "Implement Brave search: GET https://api.search.brave.com/res/v1/web/search "
        "with header X-Subscription-Token, params {q, count=top_k}, map "
        "response['web']['results'] to [{title, url, snippet=description}]."
    )


def _real_search_serper(query: str, top_k: int) -> list[dict]:
    raise NotImplementedError(
        "Implement Serper search: POST https://google.serper.dev/search with "
        "header X-API-KEY, body {q, num=top_k}, map response['organic'] to "
        "[{title, url, snippet}]."
    )


_BACKENDS = {
    "mock": _mock_search,
    "tavily": _real_search_tavily,
    "brave": _real_search_brave,
    "serper": _real_search_serper,
}


# ─── RunContext for the research team ────────────────────────────────────────


@dataclass
class ResearchDeps:
    """Shared state across every worker run.

    ``trail`` is mutated by the ``web_search`` tool so the verifier can later
    inspect *which sources the researcher actually pulled* and cross-check that
    the writer cited only those — independently of whatever the writer's draft
    claims to have used.
    """

    backend: str = "mock"
    top_k: int = 4
    trail: list[dict] = field(default_factory=list)


# ─── Tools ───────────────────────────────────────────────────────────────────


@fa.tool()
def web_search(query: str, ctx: fa.RunContext[ResearchDeps]) -> str:
    """Search the web for sources on the given query.

    Returns a JSON-encoded list of results, each with ``title``, ``url``, and
    ``snippet``. Use this before drafting any factual claim — the writer's
    citations must point at URLs you saw here.
    """
    backend = ctx.state.backend
    fn = _BACKENDS.get(backend, _mock_search)
    results = fn(query, ctx.state.top_k)
    # Append to the trail so the verifier can audit retrievals later.
    for r in results:
        if r not in ctx.state.trail:
            ctx.state.trail.append(r)
    return json.dumps(results, indent=2)


def make_deps() -> ResearchDeps:
    return ResearchDeps(
        backend=os.getenv("SEARCH_BACKEND", "mock"),
        top_k=int(os.getenv("SEARCH_TOP_K", "4")),
    )
