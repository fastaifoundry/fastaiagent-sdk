"""
Tools — what the deep-research workers can call.

Two tools are exposed to researcher agents:

  * ``web_search(query)`` — returns a JSON-encoded list of result dicts
    (``title``, ``url``, ``snippet``). Backend selectable via the
    ``SEARCH_BACKEND`` env var: ``tavily`` (default if ``TAVILY_API_KEY`` is
    set), ``brave``, ``serper``, or ``mock`` (offline fallback that mirrors
    the corpus from ``examples/research-agent/``).

  * ``web_fetch(url)`` — fetches a URL with ``httpx`` and returns readable
    text. HTML is stripped to text via the stdlib ``html.parser`` so the
    template has no extra dependency surface.

The contract is identical regardless of backend. Researcher agents call
``web_search`` to find sources, then optionally call ``web_fetch`` to read
the full content of a promising URL before pruning their findings.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from html.parser import HTMLParser

import fastaiagent as fa

# ─── Mock corpus (fallback when no real backend is configured) ───────────────
#
# Matches the corpus in examples/research-agent/tools.py so the two templates
# behave consistently in offline / CI environments. Keys are lower-cased
# substrings; the first key found in the query wins.

_MOCK_CORPUS: dict[str, list[dict]] = {
    "transformer": [
        {
            "title": "Attention Is All You Need (2017)",
            "url": "https://arxiv.org/abs/1706.03762",
            "snippet": (
                "Vaswani et al. introduce the Transformer, a sequence-to-sequence "
                "architecture based entirely on attention."
            ),
        },
        {
            "title": "The Illustrated Transformer — Jay Alammar",
            "url": "https://jalammar.github.io/illustrated-transformer/",
            "snippet": "Visual walkthrough of self-attention and the encoder-decoder stack.",
        },
    ],
    "retrieval-augmented generation": [
        {
            "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP (2020)",
            "url": "https://arxiv.org/abs/2005.11401",
            "snippet": (
                "Lewis et al. propose RAG, combining a parametric seq2seq "
                "model with a non-parametric memory."
            ),
        },
        {
            "title": "Self-RAG: Self-Reflective Retrieval-Augmented Generation (2023)",
            "url": "https://arxiv.org/abs/2310.11511",
            "snippet": (
                "Asai et al. add reflection tokens that let the model decide "
                "when to retrieve."
            ),
        },
    ],
    "mcp": [
        {
            "title": "Model Context Protocol — Anthropic",
            "url": "https://www.anthropic.com/news/model-context-protocol",
            "snippet": (
                "MCP is an open protocol that standardizes how applications "
                "provide context to LLMs."
            ),
        },
    ],
    "agent eval": [
        {
            "title": "AgentBench: Evaluating LLMs as Agents (2023)",
            "url": "https://arxiv.org/abs/2308.03688",
            "snippet": "Liu et al. propose 8 distinct environments for LLM-agent evaluation.",
        },
    ],
}


_CORPUS_ALIASES = {
    # short-form / acronym shortcuts so realistic LLM queries hit
    "rag": "retrieval-augmented generation",
    "self-rag": "retrieval-augmented generation",
    "self rag": "retrieval-augmented generation",
    "attention": "transformer",
    "self-attention": "transformer",
    "model context protocol": "mcp",
    "agent benchmark": "agent eval",
    "agent benchmarks": "agent eval",
    "evaluating agents": "agent eval",
}


def _mock_search(query: str, top_k: int) -> list[dict]:
    q = query.lower()
    # 1. exact substring match against canonical corpus keys
    for keyword, hits in _MOCK_CORPUS.items():
        if keyword in q:
            return hits[:top_k]
    # 2. alias map (acronyms / common short forms)
    for alias, target in _CORPUS_ALIASES.items():
        if alias in q and target in _MOCK_CORPUS:
            return _MOCK_CORPUS[target][:top_k]
    # 3. token-overlap fallback — pick the corpus entry sharing the most words
    q_tokens = {t for t in q.replace("-", " ").split() if len(t) > 3}
    best = (0, None)
    for keyword, hits in _MOCK_CORPUS.items():
        k_tokens = {t for t in keyword.replace("-", " ").split() if len(t) > 3}
        overlap = len(q_tokens & k_tokens)
        if overlap > best[0]:
            best = (overlap, hits)
    if best[0] > 0 and best[1] is not None:
        return best[1][:top_k]
    return [
        {
            "title": "(no high-confidence sources for this query)",
            "url": "https://example.com/no-results",
            "snippet": (
                "The mock backend has no entries for this query. Set "
                "TAVILY_API_KEY or SEARCH_BACKEND to use a real provider."
            ),
        }
    ]


# ─── Real backends ───────────────────────────────────────────────────────────


def _real_search_tavily(query: str, top_k: int) -> list[dict]:
    """Live Tavily search. Requires ``TAVILY_API_KEY``."""
    import httpx

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError(
            "SEARCH_BACKEND=tavily but TAVILY_API_KEY is not set. "
            "Get a free key at https://tavily.com or fall back to "
            "SEARCH_BACKEND=mock."
        )
    response = httpx.post(
        "https://api.tavily.com/search",
        json={
            "api_key": api_key,
            "query": query,
            "max_results": top_k,
            "search_depth": "basic",
        },
        timeout=20.0,
    )
    response.raise_for_status()
    payload = response.json()
    return [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "snippet": r.get("content", ""),
        }
        for r in payload.get("results", [])
    ]


def _real_search_brave(query: str, top_k: int) -> list[dict]:
    import httpx

    api_key = os.getenv("BRAVE_SEARCH_API_KEY")
    if not api_key:
        raise RuntimeError("SEARCH_BACKEND=brave but BRAVE_SEARCH_API_KEY is not set.")
    response = httpx.get(
        "https://api.search.brave.com/res/v1/web/search",
        params={"q": query, "count": top_k},
        headers={"X-Subscription-Token": api_key, "Accept": "application/json"},
        timeout=20.0,
    )
    response.raise_for_status()
    web = response.json().get("web", {})
    return [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "snippet": r.get("description", ""),
        }
        for r in web.get("results", [])
    ]


def _real_search_serper(query: str, top_k: int) -> list[dict]:
    import httpx

    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        raise RuntimeError("SEARCH_BACKEND=serper but SERPER_API_KEY is not set.")
    response = httpx.post(
        "https://google.serper.dev/search",
        json={"q": query, "num": top_k},
        headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
        timeout=20.0,
    )
    response.raise_for_status()
    return [
        {
            "title": r.get("title", ""),
            "url": r.get("link", ""),
            "snippet": r.get("snippet", ""),
        }
        for r in response.json().get("organic", [])[:top_k]
    ]


_BACKENDS = {
    "mock": _mock_search,
    "tavily": _real_search_tavily,
    "brave": _real_search_brave,
    "serper": _real_search_serper,
}


def _resolve_backend(name: str) -> str:
    """Pick a backend. ``auto`` (default) → tavily if key present, else mock."""
    if name != "auto":
        return name
    if os.getenv("TAVILY_API_KEY"):
        return "tavily"
    return "mock"


# ─── HTML → text via stdlib (no extra deps) ──────────────────────────────────


class _TextExtractor(HTMLParser):
    """Strip HTML to readable text. Drops <script>, <style>, and inline noise."""

    SKIP_TAGS = {"script", "style", "noscript", "svg", "head"}

    def __init__(self) -> None:
        super().__init__()
        self._chunks: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in self.SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in self.SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            text = data.strip()
            if text:
                self._chunks.append(text)

    def text(self) -> str:
        return " ".join(self._chunks)


def _strip_html(html: str) -> str:
    parser = _TextExtractor()
    try:
        parser.feed(html)
    except Exception:
        # Pathological HTML — fall back to crude regex strip.
        import re

        return re.sub(r"<[^>]+>", " ", html).strip()
    return parser.text()


# ─── RunContext for the deep research team ───────────────────────────────────


@dataclass
class DeepResearchDeps:
    """Shared state across every worker run.

    ``trail`` records every URL ever returned by a ``web_search`` call —
    the writer's citations must come from this list, and the eval suite
    cross-checks the final report against it.
    """

    backend: str = "auto"
    top_k: int = 5
    fetch_max_chars: int = 8000
    trail: list[dict] = field(default_factory=list)


# ─── Tools ───────────────────────────────────────────────────────────────────


@fa.tool()
def web_search(query: str, ctx: fa.RunContext[DeepResearchDeps]) -> str:
    """Search the web for sources on the given query.

    Returns a JSON-encoded list of results, each with ``title``, ``url``, and
    ``snippet``. Use this before drafting any factual claim — citations must
    point at URLs you saw here.
    """
    backend = _resolve_backend(ctx.state.backend)
    fn = _BACKENDS.get(backend, _mock_search)
    results = fn(query, ctx.state.top_k)
    for r in results:
        if r not in ctx.state.trail:
            ctx.state.trail.append(r)
    return json.dumps(results, indent=2)


@fa.tool()
def web_fetch(url: str, ctx: fa.RunContext[DeepResearchDeps]) -> str:
    """Fetch a URL and return readable text content.

    HTML is stripped to text. Output is truncated to ``fetch_max_chars`` (8K
    by default) so a single fetch can't blow the agent's context window.
    Use this when a search snippet is too short and you need the full page
    to draw a citation-worthy claim.
    """
    import httpx

    try:
        response = httpx.get(url, timeout=20.0, follow_redirects=True)
        response.raise_for_status()
    except Exception as e:
        return f"Error fetching {url}: {e}"

    content_type = response.headers.get("content-type", "").lower()
    if "html" in content_type:
        text = _strip_html(response.text)
    else:
        text = response.text

    max_chars = ctx.state.fetch_max_chars
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n\n[truncated at {max_chars} chars]"
    return text


def make_deps() -> DeepResearchDeps:
    return DeepResearchDeps(
        backend=os.getenv("SEARCH_BACKEND", "auto"),
        top_k=int(os.getenv("SEARCH_TOP_K", "5")),
        fetch_max_chars=int(os.getenv("FETCH_MAX_CHARS", "8000")),
    )
