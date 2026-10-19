"""OpenAI's Responses API (``POST /v1/responses``) for ``LLMClient`` (1.91.0, 2C).

Opt in with ``LLMClient(provider="openai", api="responses")``; Chat Completions
stays the default. Why it exists: on OpenAI's current models, tool calling with
reasoning is served only here. Measured 2026-10-19 — ``gpt-6-luna`` on Chat
Completions answers a request with tools and ``reasoning_effort`` other than
``"none"`` with HTTP 400 ("Function tools with reasoning_effort are not supported
for gpt-6-luna in /v1/chat/completions. To use function tools, use
/v1/responses"), and OpenAI's migration guide says GPT-6 Astra and GPT-6.1 Sol
need this API for tool calling at all. It is also where OpenAI's provider-run
tools (``web_search()``, ``code_execution()``, ``file_search()``) live.

Like the Decisions API (``/v1/decisions``) it has its own body builder and
parser, and returns the same :class:`~fastaiagent.llm.client.LLMResponse` — so
retries, fallbacks, the ``llm.openai.<model>`` span, recorded replay, pricing and
the run's limits are the client's, unchanged.

Stateless by design: every request is ``store: false`` and carries the whole
history, as Chat Completions does — the SDK's messages and checkpoints are the
state, and nothing is kept at OpenAI. Reasoning items come back (summary +
encrypted content) as ``ReasoningBlock(provider="openai-responses")`` and go
back with the turn's function calls, which OpenAI requires. A turn in which a
provider-run tool ran goes back verbatim (``Message.provider_content``).
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator, AsyncIterator
from typing import TYPE_CHECKING, Any

from fastaiagent._internal.errors import LLMProviderError
from fastaiagent.llm.message import Message, MessageRole, ReasoningBlock, ToolCall
from fastaiagent.llm.provider_tools import (
    OPENAI_RESPONSES,
    ProviderToolReport,
    parse_responses,
    responses_includes,
    responses_tool,
    split_tools,
)
from fastaiagent.llm.stream import (
    ProviderTurn,
    ReasoningDelta,
    StreamDone,
    StreamEvent,
    TextDelta,
    ToolCallEnd,
    ToolCallStart,
)

if TYPE_CHECKING:
    from fastaiagent.llm.client import LLMClient, LLMResponse

logger = logging.getLogger(__name__)

#: Chat Completions settings the Responses API has no field for. Sent nowhere;
#: warned about once each, as ``tool_choice`` is on a provider without one.
_NOT_ON_RESPONSES = ("stop", "seed", "frequency_penalty", "presence_penalty")
_WARNED: set[str] = set()


def _text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(p for p in content if isinstance(p, str))
    return str(content)


def _input_part(part: dict[str, Any]) -> dict[str, Any]:
    """One Chat Completions content part as a Responses input part."""
    kind = part.get("type")
    if kind == "text":
        return {"type": "input_text", "text": part.get("text", "")}
    if kind == "image_url":
        image = part.get("image_url") or {}
        out: dict[str, Any] = {"type": "input_image", "image_url": image.get("url", "")}
        if image.get("detail"):
            out["detail"] = image["detail"]
        return out
    if kind == "file":
        return {"type": "input_file", **(part.get("file") or {})}
    return part


def _input_parts(client: LLMClient, content: list[Any]) -> list[dict[str, Any]]:
    """Multimodal content (text, ``Image``, ``PDF``, ``File``) as Responses parts —
    through the same formatter Chat Completions uses, so ``pdf_mode`` and the
    image caps apply."""
    formatted = Message(role=MessageRole.user, content=content).to_provider_dict(
        "openai", **client._provider_dict_kwargs()
    )
    parts = formatted.get("content")
    if isinstance(parts, str):
        return [{"type": "input_text", "text": parts}]
    return [_input_part(p) for p in parts or []]


def build_input(client: LLMClient, messages: list[Message]) -> list[dict[str, Any]]:
    """The SDK's history as Responses ``input`` items."""
    items: list[dict[str, Any]] = []
    for m in messages:
        if m.role == MessageRole.system:
            items.append({"role": "system", "content": _text(m.content)})
        elif m.role == MessageRole.user:
            if m.has_multimodal_content():
                items.append(
                    {"role": "user", "content": _input_parts(client, list(m.content or []))}
                )
            else:
                items.append({"role": "user", "content": _text(m.content)})
        elif m.role == MessageRole.assistant:
            verbatim = m.content_for(OPENAI_RESPONSES)
            if verbatim is not None:
                items.extend(verbatim)
                continue
            items.extend(m.reasoning_for(OPENAI_RESPONSES))
            if m.content:
                items.append({"role": "assistant", "content": _text(m.content)})
            for tc in m.tool_calls or []:
                items.append(
                    {
                        "type": "function_call",
                        "call_id": tc.id,
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    }
                )
        elif m.role == MessageRole.tool:
            output: Any = (
                _input_parts(client, list(m.content or []))
                if m.has_multimodal_content()
                else _text(m.content)
            )
            items.append(
                {"type": "function_call_output", "call_id": m.tool_call_id or "", "output": output}
            )
    return items


def _text_format(response_format: dict[str, Any]) -> dict[str, Any] | None:
    """Chat's ``response_format`` as Responses ``text.format``."""
    kind = response_format.get("type")
    if kind == "json_schema":
        schema = dict(response_format.get("json_schema") or {})
        return {"type": "json_schema", **schema}
    if kind == "json_object":
        return {"type": "json_object"}
    return None


def build_body(
    client: LLMClient,
    messages: list[Message],
    tools: list[dict[str, Any]] | None,
    *,
    stream: bool,
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    """The request body. Credentials and pass-through options are added by the
    client, as for Chat Completions."""
    body: dict[str, Any] = {
        "model": client.model,
        "input": build_input(client, messages),
        "store": False,
    }
    if stream:
        body["stream"] = True
    functions, specs = split_tools(tools)
    defs: list[dict[str, Any]] = []
    for t in functions:
        func = t.get("function", t)
        defs.append(
            {
                "type": "function",
                "name": func["name"],
                "description": func.get("description", ""),
                "parameters": func.get("parameters", {"type": "object", "properties": {}}),
            }
        )
    defs.extend(responses_tool(s) for s in specs)
    include = ["reasoning.encrypted_content", *responses_includes(specs)]
    if defs:
        body["tools"] = defs
        tool_choice = client._requested_tool_choice(tools, kwargs)
        if tool_choice is not None:
            body["tool_choice"] = tool_choice
        ptc = kwargs.get("parallel_tool_calls", client.parallel_tool_calls)
        if ptc is not None:
            body["parallel_tool_calls"] = ptc
    body["include"] = include

    temperature = kwargs.get("temperature", client.temperature)
    if temperature is not None:
        body["temperature"] = temperature
    top_p = kwargs.get("top_p", client.top_p)
    if top_p is not None:
        body["top_p"] = top_p
    max_tok = kwargs.get("max_tokens", client.max_tokens)
    if max_tok is not None:
        body["max_output_tokens"] = max_tok
    for name in _NOT_ON_RESPONSES:
        if kwargs.get(name, getattr(client, name, None)) is not None and name not in _WARNED:
            _WARNED.add(name)
            logger.warning("%s is not supported by the Responses API; it is not sent.", name)

    response_format = kwargs.get("response_format")
    if isinstance(response_format, dict):
        fmt = _text_format(response_format)
        if fmt is not None:
            body["text"] = {"format": fmt}

    level = client._reasoning(kwargs)
    if level is not None:
        body["reasoning"] = {"effort": level}
        if level != "none":
            # The summary is the only reasoning text OpenAI shows.
            body["reasoning"]["summary"] = "auto"
    return body


def _finish(data: dict[str, Any], has_calls: bool) -> str:
    status = data.get("status") or ""
    if status == "incomplete":
        reason = (data.get("incomplete_details") or {}).get("reason") or "incomplete"
        return {"max_output_tokens": "length", "content_filter": "content_filter"}.get(
            reason, str(reason)
        )
    if has_calls:
        return "tool_calls"
    return "stop" if status in ("completed", "") else str(status)


def _usage(raw: dict[str, Any] | None) -> dict[str, Any]:
    """Responses usage in the SDK's shape. ``input_tokens`` includes the cache."""
    from fastaiagent.llm.client import _usage as usage_shape

    raw = raw or {}
    in_details = raw.get("input_tokens_details") or {}
    out_details = raw.get("output_tokens_details") or {}
    return usage_shape(
        int(raw.get("input_tokens", 0)),
        int(raw.get("output_tokens", 0)),
        total=raw.get("total_tokens"),
        reasoning=out_details.get("reasoning_tokens"),
        cached=in_details.get("cached_tokens"),
        cache_write=in_details.get("cache_write_tokens") or None,
    )


def _raise_failed(client: LLMClient, data: dict[str, Any]) -> None:
    if data.get("status") == "failed":
        error = data.get("error") or {}
        raise client._provider_error(
            "OpenAI Responses API failed",
            json.dumps({"error": error}),
            None,
        )


def _reasoning_text(item: dict[str, Any]) -> str:
    return "\n".join(
        str(s.get("text") or "")
        for s in item.get("summary") or []
        if s.get("type") == "summary_text"
    )


def parse(client: LLMClient, data: dict[str, Any]) -> LLMResponse:
    """A Responses response as an :class:`LLMResponse`."""
    from fastaiagent.llm.client import LLMResponse

    _raise_failed(client, data)
    text: list[str] = []
    reasoning: list[str] = []
    blocks: list[ReasoningBlock] = []
    calls: list[ToolCall] = []
    for item in data.get("output") or []:
        kind = item.get("type")
        if kind == "message":
            for part in item.get("content") or []:
                if part.get("type") == "output_text":
                    text.append(str(part.get("text") or ""))
                elif part.get("type") == "refusal":
                    text.append(str(part.get("refusal") or ""))
        elif kind == "function_call":
            raw_args = item.get("arguments") or "{}"
            calls.append(
                ToolCall(
                    id=str(item.get("call_id") or item.get("id") or ""),
                    name=str(item.get("name") or ""),
                    arguments=json.loads(raw_args) if isinstance(raw_args, str) else raw_args,
                )
            )
        elif kind == "reasoning":
            if summary := _reasoning_text(item):
                reasoning.append(summary)
            blocks.append(ReasoningBlock(provider=OPENAI_RESPONSES, data=dict(item)))

    report = parse_responses(data)
    usage = _usage(data.get("usage"))
    report.apply_to_usage(usage)
    return LLMResponse(
        content="".join(text) or None,
        tool_calls=calls,
        usage=usage,
        model=str(data.get("model") or client.model),
        finish_reason=_finish(data, bool(calls)),
        reasoning="\n".join(reasoning) or None,
        reasoning_blocks=blocks,
        provider_tool_calls=report.calls,
        provider_content=report.content,
    )


def _turn_event(report: ProviderToolReport) -> ProviderTurn | None:
    if not report.calls and report.content is None and not report.tool_requests:
        return None
    return ProviderTurn(
        calls=report.calls,
        content=report.content,
        tool_requests={k: v for k, v in report.tool_requests.items() if v},
        containers=list(report.containers),
    )


async def stream_events(
    client: LLMClient, events: AsyncIterator[dict[str, Any]]
) -> AsyncGenerator[StreamEvent, None]:
    """Turn Responses SSE events into the SDK's stream events."""
    from fastaiagent.llm.client import _usage_event

    finished = False
    async for event in events:
        kind = event.get("type")
        if kind == "response.output_text.delta":
            if delta := event.get("delta"):
                yield TextDelta(text=str(delta))
        elif kind == "response.reasoning_summary_text.delta":
            if delta := event.get("delta"):
                yield ReasoningDelta(text=str(delta))
        elif kind == "response.output_item.added":
            item = event.get("item") or {}
            if item.get("type") == "function_call":
                yield ToolCallStart(
                    call_id=str(item.get("call_id") or ""), tool_name=str(item.get("name") or "")
                )
        elif kind == "response.output_item.done":
            item = event.get("item") or {}
            if item.get("type") == "function_call":
                raw_args = item.get("arguments") or "{}"
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
                except json.JSONDecodeError:
                    args = {}
                yield ToolCallEnd(
                    call_id=str(item.get("call_id") or ""),
                    tool_name=str(item.get("name") or ""),
                    arguments=args,
                )
            elif item.get("type") == "reasoning":
                # The encrypted content is final only here, not on ``.added``.
                yield ReasoningDelta(
                    block=ReasoningBlock(provider=OPENAI_RESPONSES, data=dict(item))
                )
        elif kind in ("response.completed", "response.incomplete", "response.failed"):
            data = event.get("response") or {}
            _raise_failed(client, data)
            report = parse_responses(data)
            if (turn := _turn_event(report)) is not None:
                yield turn
            usage = _usage(data.get("usage"))
            if usage["prompt_tokens"] or usage["completion_tokens"]:
                yield _usage_event(usage)
            has_calls = any(o.get("type") == "function_call" for o in data.get("output") or [])
            yield StreamDone(finish_reason=_finish(data, has_calls))
            finished = True
        elif kind == "error":
            raise client._provider_error(
                "OpenAI Responses API stream error", json.dumps({"error": event}), None
            )
    if not finished:
        raise LLMProviderError(
            "OpenAI Responses API stream ended without a final response event",
            provider=client.provider,
        )
