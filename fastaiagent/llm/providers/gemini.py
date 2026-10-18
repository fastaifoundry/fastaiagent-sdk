"""Native Google Gemini wire for ``LLMClient``.

Gemini's REST API is not OpenAI-compatible at the deepest level — it has a
different message envelope (``contents`` instead of ``messages``), uses
``role="model"`` in place of ``"assistant"``, and routes tool calls through
``functionCall`` / ``functionResponse`` parts. Rather than maintaining a
brittle compatibility shim, we implement a thin native client here.

Only ``httpx`` is used — no ``google-generativeai`` runtime dependency.

Public entry points:
    acomplete_gemini(client, messages, tools, **kwargs) -> LLMResponse
    astream_gemini(client, messages, tools, **kwargs)   -> AsyncGenerator[StreamEvent]

Both expect a ready-configured ``LLMClient`` (so ``client.api_key``,
``client.base_url``, ``client.model``, ``client.temperature`` etc. are
respected). They normalise into the same ``LLMResponse`` /
``StreamEvent`` types as the other providers, so the rest of the SDK
(tracing, replay, agent loop, UI) sees no difference.
"""

from __future__ import annotations

import json as _json
import os
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

from fastaiagent._internal.errors import LLMProviderError
from fastaiagent.llm.client import _augment_system_for_response_format
from fastaiagent.llm.message import Message, MessageRole, ToolCall
from fastaiagent.llm.stream import (
    ReasoningDelta,
    StreamDone,
    StreamEvent,
    TextDelta,
    ToolCallEnd,
    ToolCallStart,
)

if TYPE_CHECKING:
    from fastaiagent.llm.client import LLMClient, LLMResponse


_DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

#: What Gemini accepts in place of a thought signature on a function call the
#: model did not sign. Measured 2026-10-18: accepted by 2.5-flash, 3.8-flash and
#: 4-argon; a missing signature is a 400 on every model after 2.5.
_SKIP_SIGNATURE = "skip_thought_signature_validator"


def _coerce_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # System/assistant/tool roles are text-only; keep just the strings.
        # User-role media is handled by _user_parts (Gemini inlineData).
        return "\n".join(p for p in content if isinstance(p, str))
    return str(content)


def _user_parts(content: Any) -> list[dict[str, Any]]:
    """Convert a user message's content into Gemini ``parts``.

    Text stays text; ``Image`` and ``PDF`` become native ``inlineData`` blobs
    (base64) — Gemini reads both directly (it even extracts a PDF's embedded
    text for free), so no local rendering is involved and ``pdf_mode``
    doesn't apply. Inline data is best for smaller payloads; very large PDFs
    would need the Gemini File API (not yet wired here).
    """
    from fastaiagent.multimodal.file import File
    from fastaiagent.multimodal.image import Image
    from fastaiagent.multimodal.pdf import PDF

    if content is None:
        return [{"text": ""}]
    if isinstance(content, str):
        return [{"text": content}]
    if not isinstance(content, list):
        return [{"text": str(content)}]

    parts: list[dict[str, Any]] = []
    for p in content:
        if isinstance(p, str):
            parts.append({"text": p})
        elif isinstance(p, Image):
            parts.append({"inlineData": {"mimeType": p.media_type, "data": p.to_base64()}})
        elif isinstance(p, PDF):
            parts.append(
                {"inlineData": {"mimeType": "application/pdf", "data": p.to_base64()}}
            )
        elif isinstance(p, File):
            # Gemini reads any of these natively (images, audio, video, pdf,
            # text/csv/html…) — no local processing, just the raw bytes + mime.
            parts.append({"inlineData": {"mimeType": p.mime_type, "data": p.to_base64()}})
    return parts or [{"text": ""}]


def _convert_messages(messages: list[Message]) -> tuple[str, list[dict[str, Any]]]:
    """Return (system_text, gemini_contents).

    Gemini conventions:
      - ``role`` for user is ``"user"``, for assistant is ``"model"``.
      - System messages collapse into ``systemInstruction``.
      - Tool calls become ``functionCall`` parts inside a ``"model"`` turn.
      - Tool results (our ``role=tool``) become ``functionResponse`` parts
        inside a ``"user"`` turn (Gemini convention — the tool reply is
        spoken on the user's behalf).
    """
    system_parts: list[str] = []
    contents: list[dict[str, Any]] = []
    # tool_call_id -> tool name, from the turn that declared the call. Gemini
    # restarts its ids every turn, so this is refreshed per assistant turn.
    call_names: dict[str, str] = {}

    for m in messages:
        if m.role == MessageRole.system:
            text = _coerce_text(m.content)
            if text:
                system_parts.append(text)
            continue

        if m.role == MessageRole.assistant:
            parts: list[dict[str, Any]] = []
            text = _coerce_text(m.content)
            if text:
                parts.append({"text": text})
            calls = m.tool_calls or []
            call_names = {tc.id: tc.name for tc in calls}
            # Gemini 3+ signs the first call of a turn and 400s when the
            # signature is not sent back (1.91.0, X11). A turn nobody signed —
            # history from another provider, ``messages=``, a pre-1.91
            # checkpoint — carries Google's validator-skip value on its first
            # call instead, which every Gemini model accepts.
            signed = any(tc.signature for tc in calls)
            for i, tc in enumerate(calls):
                part: dict[str, Any] = {"functionCall": {"name": tc.name, "args": tc.arguments}}
                if tc.signature:
                    part["thoughtSignature"] = tc.signature
                elif i == 0 and not signed:
                    part["thoughtSignature"] = _SKIP_SIGNATURE
                parts.append(part)
            if not parts:
                parts.append({"text": ""})
            contents.append({"role": "model", "parts": parts})
            continue

        if m.role == MessageRole.tool:
            # Function response — named after the call it answers (1.91.0, X4).
            # ``ToolMessage`` carries only the id, so the name comes from the
            # declaring turn; an explicit ``Message.name`` still wins, and
            # ``"tool"`` stays the last resort for a result with no parent.
            tool_name = m.name or call_names.get(m.tool_call_id or "") or "tool"
            try:
                response_payload: Any = (
                    _json.loads(m.content) if isinstance(m.content, str) else m.content
                )
            except (ValueError, TypeError):
                response_payload = {"output": _coerce_text(m.content)}
            if not isinstance(response_payload, dict):
                response_payload = {"output": response_payload}
            contents.append(
                {
                    "role": "user",
                    "parts": [
                        {
                            "functionResponse": {
                                "name": tool_name,
                                "response": response_payload,
                            }
                        }
                    ],
                }
            )
            continue

        # user role — text plus any native inline media (images, PDFs)
        contents.append({"role": "user", "parts": _user_parts(m.content)})

    system_text = "\n\n".join(system_parts) if system_parts else ""
    return system_text, contents


def _convert_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    if not tools:
        return None
    declarations: list[dict[str, Any]] = []
    for t in tools:
        func = t.get("function", t)
        declarations.append(
            {
                "name": func["name"],
                "description": func.get("description", ""),
                "parameters": func.get("parameters", {"type": "object", "properties": {}}),
            }
        )
    return [{"functionDeclarations": declarations}]


def _completion_tokens(usage_meta: dict[str, Any]) -> int:
    """Output tokens as billed: the reply plus the model's thinking (1.90.0, F7).

    A thinking model reports ``thoughtsTokenCount`` beside
    ``candidatesTokenCount``; Google bills it at the output rate and counts it in
    ``totalTokenCount``. Mapping completion to the candidates alone left the
    thinking unpriced, and made ``astream``'s total disagree with ``arun``'s.
    """
    return int(usage_meta.get("candidatesTokenCount", 0)) + int(
        usage_meta.get("thoughtsTokenCount", 0)
    )


def _usage(usage_meta: dict[str, Any]) -> dict[str, Any]:
    """Gemini usage in the SDK's shape. ``promptTokenCount`` already includes
    ``cachedContentTokenCount`` (implicit caching, 2.5 and later)."""
    from fastaiagent.llm.client import _usage as usage_shape

    return usage_shape(
        int(usage_meta.get("promptTokenCount", 0)),
        _completion_tokens(usage_meta),
        total=int(usage_meta.get("totalTokenCount", 0)),
        reasoning=usage_meta.get("thoughtsTokenCount"),
        cached=usage_meta.get("cachedContentTokenCount"),
    )


def _build_generation_config(client: LLMClient, **kwargs: Any) -> dict[str, Any]:
    from fastaiagent.llm.reasoning import gemini_thinking_config

    cfg: dict[str, Any] = {}
    level = client._reasoning(kwargs)
    if level is not None:
        thinking = gemini_thinking_config(client.model, level)
        if thinking is not None:
            cfg["thinkingConfig"] = thinking
    temperature = kwargs.get("temperature", client.temperature)
    if temperature is not None:
        cfg["temperature"] = temperature
    max_tok = kwargs.get("max_tokens", client.max_tokens)
    if max_tok is not None:
        cfg["maxOutputTokens"] = max_tok
    top_p = kwargs.get("top_p", client.top_p)
    if top_p is not None:
        cfg["topP"] = top_p
    stop = kwargs.get("stop", client.stop)
    if stop is not None:
        cfg["stopSequences"] = [stop] if isinstance(stop, str) else list(stop)

    response_format = kwargs.get("response_format")
    if response_format is not None:
        rf_type = (
            response_format.get("type", "text") if isinstance(response_format, dict) else "text"
        )
        if rf_type in ("json_object", "json_schema"):
            cfg["responseMimeType"] = "application/json"
            if rf_type == "json_schema":
                schema = (response_format.get("json_schema") or {}).get("schema")
                if schema:
                    cfg["responseSchema"] = schema
    return cfg


def _api_key(client: LLMClient) -> str:
    api_key = client.api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get(
        "GOOGLE_API_KEY", ""
    )
    if not api_key:
        raise LLMProviderError(
            "No API key for provider 'gemini'. Set the api_key parameter or "
            "the GEMINI_API_KEY environment variable.\n"
            "Example: LLMClient(provider='gemini', "
            "model='gemini-2.0-flash', api_key='AIza...')"
        )
    return api_key


def _base_url(client: LLMClient) -> str:
    return (client.base_url or _DEFAULT_BASE_URL).rstrip("/")


def _parse_response(client: LLMClient, data: dict[str, Any]) -> LLMResponse:
    from fastaiagent.llm.client import LLMResponse  # late import to avoid cycle

    candidates = data.get("candidates") or []
    if not candidates:
        # Some Gemini error / safety responses have no candidates.
        prompt_feedback = data.get("promptFeedback") or {}
        block_reason = prompt_feedback.get("blockReason")
        return LLMResponse(
            content=None,
            tool_calls=[],
            usage={},
            model=client.model,
            finish_reason=block_reason or "stop",
        )

    candidate = candidates[0]
    parts = (candidate.get("content") or {}).get("parts") or []
    content_text = ""
    reasoning_text = ""
    tool_calls: list[ToolCall] = []
    for i, part in enumerate(parts):
        if part.get("thought"):
            # A thought summary (``includeThoughts``) — reasoning, never the answer.
            reasoning_text += part.get("text", "")
            continue
        if "text" in part:
            content_text += part.get("text", "")
        if "functionCall" in part:
            fc = part["functionCall"]
            tool_calls.append(
                ToolCall(
                    id=f"call_{i}",
                    name=fc.get("name", ""),
                    arguments=fc.get("args") or {},
                    signature=part.get("thoughtSignature") or None,
                )
            )

    usage = _usage(data.get("usageMetadata") or {})
    finish_reason_raw = candidate.get("finishReason", "")
    finish_reason = {
        "STOP": "stop",
        "MAX_TOKENS": "length",
        "SAFETY": "content_filter",
        "RECITATION": "content_filter",
    }.get(finish_reason_raw, finish_reason_raw.lower() if finish_reason_raw else "")
    if tool_calls:
        finish_reason = "tool_calls"

    return LLMResponse(
        content=content_text or None,
        tool_calls=tool_calls,
        usage=usage,
        model=data.get("modelVersion", client.model),
        finish_reason=finish_reason,
        reasoning=reasoning_text or None,
    )


def _build_request(
    client: LLMClient,
    messages: list[Message],
    tools: list[dict[str, Any]] | None,
    *,
    stream: bool,
    **kwargs: Any,
) -> tuple[str, dict[str, Any], dict[str, str]]:
    """Return (url, body, headers) for a Gemini generateContent request.

    The key travels in the ``x-goog-api-key`` header, not a ``?key=`` query
    parameter (1.89.0): a URL is written to logs — httpx's own INFO line
    ``HTTP Request: POST …`` included — and into connection-error messages.
    """
    api_key = _api_key(client)
    system_text, contents = _convert_messages(messages)

    response_format = kwargs.get("response_format")
    # Gemini's responseMimeType / responseSchema covers most cases, but if
    # neither tool calling nor responseSchema are honoured by a given model,
    # we still inject the JSON instruction into the system prompt.
    if response_format is not None and not system_text:
        system_text = ""
    if response_format is not None:
        system_text = _augment_system_for_response_format(system_text, response_format)

    body: dict[str, Any] = {"contents": contents}
    if system_text:
        body["systemInstruction"] = {"parts": [{"text": system_text}]}

    converted_tools = _convert_tools(tools)
    if converted_tools:
        body["tools"] = converted_tools
        tool_choice = client._requested_tool_choice(tools, kwargs)
        if tool_choice is not None:
            mode = {"required": "ANY", "none": "NONE"}[tool_choice]
            body["toolConfig"] = {"functionCallingConfig": {"mode": mode}}

    gen_cfg = _build_generation_config(client, **kwargs)
    if gen_cfg:
        body["generationConfig"] = gen_cfg

    method = "streamGenerateContent" if stream else "generateContent"
    suffix = "?alt=sse" if stream else ""
    url = f"{_base_url(client)}/models/{client.model}:{method}{suffix}"
    headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}
    client._apply_extras(body, headers, kwargs)  # extra_body / extra_headers (1.91.0)
    return url, body, headers


async def acomplete_gemini(
    client: LLMClient,
    messages: list[Message],
    tools: list[dict[str, Any]] | None = None,
    **kwargs: Any,
) -> LLMResponse:
    """Non-streaming Gemini call. Returns a normalised :class:`LLMResponse`."""
    url, body, headers = _build_request(client, messages, tools, stream=False, **kwargs)
    async with client._new_async_client() as h:
        resp = await h.post(url, json=body, headers=headers)
        if resp.status_code != 200:
            raise client._provider_error(
                f"Gemini API error {resp.status_code}", resp.text, resp.status_code, resp.headers
            )
        data = resp.json()

    return _parse_response(client, data)


async def astream_gemini(
    client: LLMClient,
    messages: list[Message],
    tools: list[dict[str, Any]] | None = None,
    **kwargs: Any,
) -> AsyncGenerator[StreamEvent, None]:
    """Streaming Gemini call.

    Gemini SSE deltas come as full ``GenerateContentResponse`` objects with
    incremental ``parts``. We emit one :class:`TextDelta` per text chunk and
    fire ``ToolCallStart`` / ``ToolCallEnd`` whole-shot for ``functionCall``
    blocks (Gemini doesn't stream function-call args incrementally).
    """
    url, body, headers = _build_request(client, messages, tools, stream=True, **kwargs)

    usage: dict[str, Any] = {}
    seen_tool_index = 0
    finish_reason = ""

    async with client._new_async_client() as h:
        async with h.stream("POST", url, json=body, headers=headers) as resp:
            if resp.status_code != 200:
                # Drain so error text is informative.
                err_bytes = b""
                async for chunk in resp.aiter_bytes():
                    err_bytes += chunk
                    if len(err_bytes) > 4096:
                        break
                raise client._provider_error(
                    f"Gemini stream error {resp.status_code}",
                    err_bytes.decode(errors="replace"),
                    resp.status_code,
                    resp.headers,
                )

            async for line in resp.aiter_lines():
                if not line:
                    continue
                if line.startswith("data:"):
                    payload = line[len("data:") :].strip()
                else:
                    payload = line.strip()
                if not payload or payload == "[DONE]":
                    continue
                try:
                    chunk = _json.loads(payload)
                except _json.JSONDecodeError:
                    continue

                candidates = chunk.get("candidates") or []
                if candidates:
                    cand0 = candidates[0]
                    if raw_finish := cand0.get("finishReason"):
                        # Same mapping as a non-streamed call.
                        finish_reason = {
                            "STOP": "stop",
                            "MAX_TOKENS": "length",
                            "SAFETY": "content_filter",
                            "RECITATION": "content_filter",
                        }.get(raw_finish, str(raw_finish).lower())
                    parts = (cand0.get("content") or {}).get("parts") or []
                    for part in parts:
                        if part.get("thought"):
                            if thought := part.get("text", ""):
                                yield ReasoningDelta(text=thought)
                            continue
                        if "text" in part:
                            text = part.get("text", "")
                            if text:
                                yield TextDelta(text=text)
                        if "functionCall" in part:
                            fc = part["functionCall"]
                            call_id = f"call_{seen_tool_index}"
                            seen_tool_index += 1
                            yield ToolCallStart(call_id=call_id, tool_name=fc.get("name", ""))
                            yield ToolCallEnd(
                                call_id=call_id,
                                tool_name=fc.get("name", ""),
                                arguments=fc.get("args") or {},
                                signature=part.get("thoughtSignature") or None,
                            )

                usage_meta = chunk.get("usageMetadata") or {}
                if usage_meta:
                    # Each chunk reports the running totals; keep the largest.
                    for key, value in _usage(usage_meta).items():
                        usage[key] = max(int(usage.get(key, 0)), int(value))

    if usage.get("prompt_tokens") or usage.get("completion_tokens"):
        from fastaiagent.llm.client import _usage_event

        yield _usage_event(usage)
    yield StreamDone(finish_reason=finish_reason)


# Re-exports for convenience. Callers should import these from this module
# directly; ``LLMClient._get_provider_fn`` does the routing internally.
__all__ = ["acomplete_gemini", "astream_gemini"]
