"""Provider-run tools on the wire (1.91.0, 2D) — the one place that knows each
provider's shape for :mod:`fastaiagent.tool.provider`.

Two directions:

* **Out:** a :class:`~fastaiagent.tool.provider.ProviderTool` reaches a request
  builder as the marker ``{"type": "provider", "provider_tool": {...}}`` among
  the function tools. :func:`split_tools` separates the two; ``*_tool`` turns a
  marker into the provider's own definition, refusing an option the provider
  cannot honour. :func:`refuse_provider_tools` is what every request builder
  whose provider runs none of these calls.
* **In:** ``parse_*`` read what the provider reports back into
  :class:`~fastaiagent.llm.message.ProviderToolCall` records (with the text a
  ``tool_result`` guardrail judges), the turn verbatim
  (:class:`~fastaiagent.llm.message.ProviderContent`) when it must go back, and
  the billable counts.

Shapes are from the providers' own documentation and were checked against the
live APIs on 2026-10-19: OpenAI Responses (``web_search``, ``code_interpreter``,
``file_search``), Anthropic server tools (``web_search_20250305``,
``code_execution_20250825``), Gemini ``google_search`` / ``code_execution`` /
``file_search``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from fastaiagent._internal.errors import LLMError
from fastaiagent.llm.message import Message, MessageRole, ProviderContent, ProviderToolCall

#: ``ProviderContent.provider`` for each wire. OpenAI's Responses items are
#: never valid on Chat Completions, so that wire has its own name.
ANTHROPIC = "anthropic"
GEMINI = "gemini"
OPENAI_RESPONSES = "openai-responses"

#: Anthropic's tool versions. ``web_search_20260209`` and later run the search
#: from inside code execution ("dynamic filtering"), so its results do not come
#: back as search results; the basic version does.
ANTHROPIC_WEB_SEARCH = "web_search_20250305"
ANTHROPIC_CODE_EXECUTION = "code_execution_20250825"


@dataclass
class ProviderToolSpec:
    """One provider-run tool requested for a call."""

    kind: str
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProviderToolReport:
    """What provider-run tools did in one response."""

    calls: list[ProviderToolCall] = field(default_factory=list)
    content: ProviderContent | None = None
    tool_requests: dict[str, int] = field(default_factory=dict)
    containers: list[str] = field(default_factory=list)

    def apply_to_usage(self, usage: dict[str, Any]) -> None:
        """Put the billable counts on a usage dict, for pricing."""
        requests = {k: v for k, v in self.tool_requests.items() if v}
        if requests:
            usage["tool_requests"] = requests
        if self.containers:
            usage["containers"] = list(self.containers)


def split_tools(
    tools: list[dict[str, Any]] | None,
) -> tuple[list[dict[str, Any]], list[ProviderToolSpec]]:
    """``(function tool definitions, provider-run tool specs)``."""
    functions: list[dict[str, Any]] = []
    specs: list[ProviderToolSpec] = []
    for t in tools or []:
        if t.get("type") == "provider":
            raw = t.get("provider_tool") or {}
            specs.append(
                ProviderToolSpec(kind=raw.get("kind", ""), options=raw.get("options") or {})
            )
        else:
            functions.append(t)
    return functions, specs


def refuse_provider_tools(tools: list[dict[str, Any]] | None, where: str) -> None:
    """Raise when a provider-run tool reaches a wire that cannot run it — the
    tool is never silently dropped."""
    _, specs = split_tools(tools)
    if not specs:
        return
    names = ", ".join(sorted({s.kind for s in specs}))
    raise LLMError(
        f"{names} {'is' if len(specs) == 1 else 'are'} run by the model's provider, and {where} "
        f"runs none. Use LLMClient(provider='openai', api='responses'), "
        f"provider='anthropic' or provider='gemini' (see docs/tools/provider-tools.md)."
    )


def _refuse_option(spec: ProviderToolSpec, option: str, where: str) -> None:
    raise LLMError(f"{spec.kind}({option}=...) is not supported by {where}; remove it.")


# ---------------------------------------------------------------------------
# Out: tool definitions
# ---------------------------------------------------------------------------


def anthropic_tool(spec: ProviderToolSpec) -> dict[str, Any]:
    """An Anthropic server tool definition."""
    opts = spec.options
    if spec.kind == "web_search":
        if opts.get("allowed_domains") and opts.get("blocked_domains"):
            raise LLMError(
                "Anthropic's web search takes allowed_domains or blocked_domains, not both."
            )
        tool: dict[str, Any] = {"type": ANTHROPIC_WEB_SEARCH, "name": "web_search"}
        if opts.get("max_uses") is not None:
            tool["max_uses"] = int(opts["max_uses"])
        for key in ("allowed_domains", "blocked_domains"):
            if opts.get(key):
                tool[key] = list(opts[key])
        if opts.get("user_location"):
            tool["user_location"] = {"type": "approximate", **opts["user_location"]}
        return tool
    if spec.kind == "code_execution":
        return {"type": ANTHROPIC_CODE_EXECUTION, "name": "code_execution"}
    raise LLMError(
        f"Anthropic has no hosted {spec.kind}. Use LLMClient(provider='openai', "
        f"api='responses') or provider='gemini' for file_search."
    )


def responses_tool(spec: ProviderToolSpec) -> dict[str, Any]:
    """An OpenAI Responses built-in tool definition."""
    opts = spec.options
    if spec.kind == "web_search":
        if opts.get("max_uses") is not None:
            _refuse_option(spec, "max_uses", "OpenAI's web search")
        tool: dict[str, Any] = {"type": "web_search"}
        filters = {k: list(opts[k]) for k in ("allowed_domains", "blocked_domains") if opts.get(k)}
        if filters:
            tool["filters"] = filters
        if opts.get("user_location"):
            tool["user_location"] = {"type": "approximate", **opts["user_location"]}
        return tool
    if spec.kind == "code_execution":
        return {"type": "code_interpreter", "container": {"type": "auto"}}
    tool = {"type": "file_search", "vector_store_ids": list(opts.get("stores") or [])}
    if opts.get("max_results") is not None:
        tool["max_num_results"] = int(opts["max_results"])
    return tool


def responses_includes(specs: list[ProviderToolSpec]) -> list[str]:
    """The ``include`` values that return each tool's results."""
    wanted = {
        "web_search": "web_search_call.action.sources",
        "code_execution": "code_interpreter_call.outputs",
        "file_search": "file_search_call.results",
    }
    return [wanted[s.kind] for s in specs if s.kind in wanted]


def gemini_tool(spec: ProviderToolSpec) -> dict[str, Any]:
    """A Gemini built-in tool definition."""
    opts = spec.options
    if spec.kind == "web_search":
        for option in ("max_uses", "allowed_domains", "blocked_domains", "user_location"):
            if opts.get(option):
                _refuse_option(spec, option, "Gemini's google_search")
        return {"google_search": {}}
    if spec.kind == "code_execution":
        return {"code_execution": {}}
    if opts.get("max_results") is not None:
        _refuse_option(spec, "max_results", "Gemini's file_search")
    return {"file_search": {"file_search_store_names": list(opts.get("stores") or [])}}


# ---------------------------------------------------------------------------
# In: what the provider reports
# ---------------------------------------------------------------------------


def _source_lines(sources: list[tuple[str, str]]) -> list[str]:
    return [f"- {title} — {url}" if title else f"- {url}" for title, url in sources]


def _web_output(queries: list[str], sources: list[tuple[str, str]], cited: list[str]) -> str:
    lines = [f"query: {q}" for q in queries if q]
    if sources:
        lines.append("sources:")
        lines.extend(_source_lines(sources))
    if cited:
        lines.append("cited:")
        lines.extend(f"- {c}" for c in cited)
    return "\n".join(lines)


def _code_output(code: str, stdout: str = "", stderr: str = "", extra: str = "") -> str:
    parts = [f"code:\n{code}"] if code else []
    if stdout:
        parts.append(f"stdout:\n{stdout}")
    if stderr:
        parts.append(f"stderr:\n{stderr}")
    if extra:
        parts.append(extra)
    return "\n".join(parts)


_ANTHROPIC_NAMES = {
    "web_search": "web_search",
    "code_execution": "code_execution",
    "bash_code_execution": "code_execution",
    "text_editor_code_execution": "code_execution",
}


def _anthropic_server_uses(history: list[Message]) -> dict[str, dict[str, Any]]:
    """``server_tool_use`` blocks in earlier turns, by id — a search requested
    beside a client tool runs, and reports, in the next response."""
    uses: dict[str, dict[str, Any]] = {}
    for m in history:
        if m.role == MessageRole.assistant:
            for block in m.content_for(ANTHROPIC) or []:
                if block.get("type") == "server_tool_use":
                    uses[str(block.get("id"))] = block
    return uses


def parse_anthropic(
    content: list[dict[str, Any]],
    usage: dict[str, Any] | None,
    history: list[Message],
) -> ProviderToolReport:
    """Server-tool blocks of one Anthropic response."""
    report = ProviderToolReport()
    blocks = [b for b in content if isinstance(b, dict)]
    is_server = [
        b.get("type") == "server_tool_use"
        or (str(b.get("type", "")).endswith("_tool_result") and b.get("type") != "tool_result")
        for b in blocks
    ]
    if not any(is_server):
        return report
    report.content = ProviderContent(provider=ANTHROPIC, data=[dict(b) for b in blocks])

    uses = _anthropic_server_uses(history)
    uses.update({str(b.get("id")): b for b in blocks if b.get("type") == "server_tool_use"})
    cited: dict[str, list[str]] = {}
    for b in blocks:
        if b.get("type") != "text":
            continue
        for c in b.get("citations") or []:
            if c.get("url") and c.get("cited_text"):
                cited.setdefault(c["url"], []).append(str(c["cited_text"]))

    for b in blocks:
        kind = str(b.get("type", ""))
        if not kind.endswith("_tool_result") or kind == "tool_result":
            continue
        use = uses.get(str(b.get("tool_use_id")), {})
        name = _ANTHROPIC_NAMES.get(str(use.get("name", "")), "")
        if not name:
            name = "web_search" if kind.startswith("web_search") else "code_execution"
        if kind == "web_fetch_tool_result":
            continue  # not a tool the SDK offers
        call = ProviderToolCall(
            id=str(b.get("tool_use_id", "")), name=name, arguments=dict(use.get("input") or {})
        )
        result = b.get("content")
        if isinstance(result, dict) and str(result.get("type", "")).endswith("_error"):
            call.error = str(result.get("error_code") or result.get("type"))
        elif name == "web_search":
            sources = [
                (str(r.get("title") or ""), str(r.get("url") or ""))
                for r in result or []
                if isinstance(r, dict)
            ]
            quotes = [q for _, url in sources for q in cited.get(url, [])]
            call.output = _web_output([str(use.get("input", {}).get("query", ""))], sources, quotes)
        elif isinstance(result, dict):
            code = str(
                (use.get("input") or {}).get("command")
                or (use.get("input") or {}).get("code")
                or json.dumps(use.get("input") or {})
            )
            extra = ""
            if result.get("return_code") not in (None, 0):
                extra = f"return code: {result['return_code']}"
            if result.get("content") and isinstance(result["content"], str):
                extra = (extra + "\n" + result["content"]).strip()
            call.output = _code_output(
                code, str(result.get("stdout") or ""), str(result.get("stderr") or ""), extra
            )
        report.calls.append(call)

    server = (usage or {}).get("server_tool_use") or {}
    if server.get("web_search_requests"):
        report.tool_requests["web_search"] = int(server["web_search_requests"])
    return report


def parse_responses(data: dict[str, Any]) -> ProviderToolReport:
    """Built-in tool items of one OpenAI Responses response."""
    report = ProviderToolReport()
    output = [o for o in data.get("output") or [] if isinstance(o, dict)]
    kinds = {"web_search_call", "code_interpreter_call", "file_search_call"}
    if not any(o.get("type") in kinds for o in output):
        return report
    report.content = ProviderContent(provider=OPENAI_RESPONSES, data=[dict(o) for o in output])

    citations: list[tuple[str, str]] = []
    for o in output:
        if o.get("type") == "message":
            for part in o.get("content") or []:
                for a in part.get("annotations") or []:
                    if a.get("type") == "url_citation":
                        citations.append((str(a.get("title") or ""), str(a.get("url") or "")))

    searches = 0
    containers: list[str] = []
    for o in output:
        kind = o.get("type")
        failed = o.get("status") == "failed"
        if kind == "web_search_call":
            action = dict(o.get("action") or {})
            queries = [str(q) for q in action.get("queries") or [] if q] or (
                [str(action["query"])] if action.get("query") else []
            )
            sources = [
                ("", str(s.get("url"))) for s in action.pop("sources", None) or [] if s.get("url")
            ]
            if action.get("type") == "search":
                searches += 1
            if action.get("url"):
                sources.append(("", str(action["url"])))
            cited_urls = {url for _, url in sources}
            titled = [c for c in citations if c[1] in cited_urls or not cited_urls]
            call = ProviderToolCall(
                id=str(o.get("id", "")),
                name="web_search",
                arguments=action,
                output=_web_output(queries, titled or sources, []),
            )
        elif kind == "code_interpreter_call":
            if o.get("container_id") and o["container_id"] not in containers:
                containers.append(str(o["container_id"]))
            logs = "\n".join(
                str(x.get("logs") or "") for x in o.get("outputs") or [] if x.get("type") == "logs"
            )
            images = [str(x.get("url")) for x in o.get("outputs") or [] if x.get("type") == "image"]
            call = ProviderToolCall(
                id=str(o.get("id", "")),
                name="code_execution",
                arguments={"code": o.get("code") or ""},
                output=_code_output(
                    str(o.get("code") or ""),
                    logs,
                    extra="\n".join(f"image: {u}" for u in images),
                ),
            )
        elif kind == "file_search_call":
            results = o.get("results") or o.get("search_results") or []
            lines = [f"query: {q}" for q in o.get("queries") or []]
            for r in results:
                lines.append(f"- {r.get('filename') or r.get('file_id')}: {r.get('text', '')}")
            call = ProviderToolCall(
                id=str(o.get("id", "")),
                name="file_search",
                arguments={"queries": list(o.get("queries") or [])},
                output="\n".join(lines),
            )
            report.tool_requests["file_search"] = report.tool_requests.get("file_search", 0) + 1
        else:
            continue
        if failed:
            call.error = "the provider reported the call failed"
        report.calls.append(call)

    tool_usage = (data.get("tool_usage") or {}).get("web_search") or {}
    report.tool_requests["web_search"] = int(tool_usage.get("num_requests") or searches)
    report.containers = containers
    return report


def parse_gemini(
    parts: list[dict[str, Any]], grounding: dict[str, Any] | None
) -> ProviderToolReport:
    """Built-in tool parts and grounding of one Gemini candidate.

    A search is recorded from ``groundingMetadata`` — the response that carries
    its sources. With function tools (Gemini 3+), the ``toolCall`` /
    ``toolResponse`` parts of the turn that ran it carry only the queries and
    Google's rendered suggestions; the sources arrive with the answer.
    """
    report = ProviderToolReport()
    server_parts = [
        p
        for p in parts
        if any(
            k in p for k in ("toolCall", "toolResponse", "executableCode", "codeExecutionResult")
        )
    ]
    if server_parts:
        report.content = ProviderContent(provider=GEMINI, data=[dict(p) for p in parts])

    pending: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for i, p in enumerate(parts):
        if "executableCode" in p:
            code = p["executableCode"]
            key = str(code.get("id") or f"code_{i}")
            pending[key] = code
            order.append(key)
        elif "codeExecutionResult" in p:
            res = p["codeExecutionResult"]
            key = str(res.get("id") or (order[-1] if order else f"code_{i}"))
            code = pending.pop(key, {})
            call = ProviderToolCall(
                id=key,
                name="code_execution",
                arguments={"code": code.get("code", ""), "language": code.get("language", "")},
                output=_code_output(str(code.get("code") or ""), str(res.get("output") or "")),
            )
            if res.get("outcome") not in (None, "OUTCOME_OK"):
                call.error = str(res.get("outcome"))
            report.calls.append(call)

    if grounding:
        chunks = grounding.get("groundingChunks") or []
        web = [c["web"] for c in chunks if isinstance(c.get("web"), dict)]
        files = [
            c["retrievedContext"] for c in chunks if isinstance(c.get("retrievedContext"), dict)
        ]
        supported = [
            str((s.get("segment") or {}).get("text") or "")
            for s in grounding.get("groundingSupports") or []
        ]
        queries = [str(q) for q in grounding.get("webSearchQueries") or [] if q]
        if web or queries:
            report.calls.append(
                ProviderToolCall(
                    id=f"search_{len(report.calls)}",
                    name="web_search",
                    arguments={"queries": queries},
                    output=_web_output(
                        queries,
                        [(str(w.get("title") or ""), str(w.get("uri") or "")) for w in web],
                        [s for s in supported if s],
                    ),
                )
            )
            report.tool_requests["web_search"] = len(queries)
        if files:
            lines = [
                f"- {f.get('title') or f.get('uri') or ''}: {f.get('text') or ''}" for f in files
            ]
            report.calls.append(
                ProviderToolCall(
                    id=f"file_search_{len(report.calls)}",
                    name="file_search",
                    arguments={},
                    output="\n".join(lines),
                )
            )
    return report
