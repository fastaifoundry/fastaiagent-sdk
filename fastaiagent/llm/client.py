"""LLMClient — unified multi-provider LLM abstraction."""

from __future__ import annotations

import asyncio
import json as _json
import os
import re
import time
from collections.abc import AsyncGenerator
from typing import Any

from pydantic import BaseModel, Field

from fastaiagent._internal.async_utils import run_sync
from fastaiagent._internal.errors import LLMError, LLMProviderError
from fastaiagent.llm.message import Message, MessageRole, ToolCall
from fastaiagent.llm.stream import (
    StreamDone,
    StreamEvent,
    TextDelta,
    ToolCallEnd,
    ToolCallStart,
    Usage,
)

# --- Structured output helpers (aligned with platform) ---

_CODE_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*\n?(.*?)\n?\s*```\s*$", re.DOTALL)


def _strip_code_fences(text: str) -> str:
    """Strip markdown code fences from LLM response (Anthropic sometimes wraps JSON)."""
    m = _CODE_FENCE_RE.match(text.strip())
    return m.group(1).strip() if m else text


def _serialize_for_span(value: Any) -> str | None:
    """JSON-encode an arbitrary structure for span attributes, swallowing errors."""
    if value is None:
        return None
    try:
        return _json.dumps(value, default=str)
    except Exception:
        return None


def _augment_system_for_response_format(
    system_text: str | None, response_format: dict[str, Any]
) -> str:
    """Inject JSON instructions into system prompt for providers without native response_format."""
    rf_type = response_format.get("type", "text") if isinstance(response_format, dict) else "text"
    if rf_type == "json_object":
        return (system_text or "") + (
            "\n\nYou must respond with valid JSON only. Do not include any text, "
            "markdown formatting, or code fences outside the JSON object. "
            "Your entire response must be parseable by JSON.parse()."
        )
    elif rf_type == "json_schema":
        js = response_format.get("json_schema", {})
        schema_name = js.get("name", "response")
        schema_body = _json.dumps(js.get("schema", {}), indent=2)
        return (system_text or "") + (
            f"\n\nYou must respond with valid JSON matching this schema ('{schema_name}'):\n"
            f"```json\n{schema_body}\n```\n"
            "Respond with the raw JSON object only. Do not wrap it in markdown code fences "
            "or add any text outside the JSON. "
            "Your entire response must be parseable by JSON.parse()."
        )
    return system_text or ""


def _coerce_system_content_to_text(content: Any) -> str:
    """System prompts must be strings. Reject multimodal system content explicitly."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    raise LLMError(
        "system messages must be strings; multimodal content is not allowed in system prompts"
    )


def _anthropic_tool_result_content(content: Any) -> Any:
    """Build the ``content`` value for an Anthropic ``tool_result`` block.

    Anthropic accepts either a string or a list of typed blocks (text/image).
    String content passes through unchanged for backward compatibility.
    Multimodal tool returns (a ``list[ContentPart]``) get formatted via
    :func:`format_multimodal_message` so images embedded in tool results
    reach the model.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        from fastaiagent.multimodal.format import format_multimodal_message

        formatted = format_multimodal_message(content, "anthropic")
        return formatted.get("content", "")
    return str(content)


def _ollama_format_from_response_format(
    response_format: dict[str, Any],
) -> str | dict[str, Any] | None:
    """Convert OpenAI response_format to Ollama 'format' parameter."""
    rf_type = response_format.get("type", "text") if isinstance(response_format, dict) else "text"
    if rf_type == "json_object":
        return "json"
    elif rf_type == "json_schema":
        schema = response_format.get("json_schema", {}).get("schema")
        return schema if schema else "json"
    return None


class LLMResponse(BaseModel):
    """Normalized response from any LLM provider."""

    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: dict[str, Any] = Field(default_factory=dict)
    model: str = ""
    finish_reason: str = ""
    latency_ms: int = 0


class LLMClient:
    """Unified LLM client supporting multiple providers.

    Providers: openai, anthropic, ollama, azure, bedrock, custom.

    Example:
        llm = LLMClient(provider="openai", model="gpt-4o", api_key="sk-...")
        response = llm.complete([UserMessage("Hello")])
    """

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        max_retries: int = 0,
        top_p: float | None = None,
        stop: str | list[str] | None = None,
        seed: int | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        parallel_tool_calls: bool | None = None,
        pdf_mode: str = "auto",
        max_pdf_pages: int = 20,
        max_image_size_mb: float | None = None,
        **kwargs: Any,
    ):
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.base_url = base_url or self._default_base_url(provider)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.top_p = top_p
        self.stop = stop
        self.seed = seed
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.parallel_tool_calls = parallel_tool_calls
        self.pdf_mode = pdf_mode
        self.max_pdf_pages = max_pdf_pages
        self.max_image_size_mb = max_image_size_mb
        self._extra = kwargs

    def _provider_dict_kwargs(self) -> dict[str, Any]:
        """Build the kwargs passed to ``Message.to_provider_dict`` from this client's config."""
        from fastaiagent.multimodal.registry import is_vision_capable

        return {
            "model": self.model,
            "pdf_mode": self.pdf_mode,
            "is_vision_capable": is_vision_capable(self.provider, self.model),
            "max_pdf_pages": self.max_pdf_pages,
            "max_image_size_mb": self.max_image_size_mb,
        }

    @staticmethod
    def _should_retry(status_code: int | None) -> bool:
        """Retry on rate limit (429) and server errors (5xx)."""
        if status_code is None:
            return False
        return status_code == 429 or status_code >= 500

    @staticmethod
    def _retry_delay(attempt: int) -> float:
        """Exponential backoff: 1s, 2s, 4s, 8s, ... capped at 30s."""
        return min(2**attempt, 30)

    @staticmethod
    def _default_base_url(provider: str) -> str:
        defaults = {
            "openai": "https://api.openai.com/v1",
            "anthropic": "https://api.anthropic.com/v1",
            "ollama": "http://localhost:11434",
            "azure": "",
            "bedrock": "",
            "custom": "",
        }
        return defaults.get(provider, "")

    def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Synchronous completion."""
        return run_sync(self.acomplete(messages, tools=tools, **kwargs))

    async def acomplete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Async completion — routes to the appropriate provider.

        Wraps the call in an OTel span so every LLM call shows up on the
        replay timeline regardless of provider. The integration-level monkey
        patches in ``fastaiagent/integrations/`` only fire for users calling
        the bare provider SDKs directly; LLMClient hits provider HTTP APIs
        with httpx, so this wrapper is what produces spans for the agent
        flow.
        """
        from fastaiagent.trace.otel import get_tracer
        from fastaiagent.trace.span import set_genai_attributes

        tracer = get_tracer("fastaiagent.llm.client")
        with tracer.start_as_current_span(f"llm.{self.provider}.{self.model}") as span:
            set_genai_attributes(
                span,
                system=self.provider,
                model=self.model,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                request_messages=_serialize_for_span([m.to_openai_format() for m in messages]),
                request_tools=_serialize_for_span(tools),
            )
            return await self._acomplete_with_retries(span, messages, tools, **kwargs)

    async def _acomplete_with_retries(
        self,
        span: Any,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        from fastaiagent.trace.span import set_genai_attributes

        start = time.monotonic()
        provider_fn = self._get_provider_fn()

        for attempt in range(self.max_retries + 1):
            try:
                response: LLMResponse = await provider_fn(messages, tools, **kwargs)
                response.latency_ms = int((time.monotonic() - start) * 1000)
                set_genai_attributes(
                    span,
                    input_tokens=response.usage.get("prompt_tokens")
                    or response.usage.get("input_tokens"),
                    output_tokens=response.usage.get("completion_tokens")
                    or response.usage.get("output_tokens"),
                    response_content=response.content,
                    response_tool_calls=_serialize_for_span(
                        [
                            {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                            for tc in response.tool_calls
                        ]
                    )
                    if response.tool_calls
                    else None,
                    finish_reason=response.finish_reason or None,
                )
                return response
            except LLMProviderError as e:
                if attempt < self.max_retries and self._should_retry(e.status_code):
                    await asyncio.sleep(self._retry_delay(attempt))
                    continue
                raise

        raise LLMProviderError("Retries exhausted")  # unreachable — satisfies type checker

    async def astream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Async streaming — yields StreamEvent objects as tokens arrive.

        Example:
            async for event in llm.astream([UserMessage("Hello")]):
                if isinstance(event, TextDelta):
                    print(event.text, end="", flush=True)
        """
        stream_providers = {
            "openai": self._stream_openai,
            "anthropic": self._stream_anthropic,
            "ollama": self._stream_ollama,
            "azure": self._stream_openai,
            "bedrock": None,
            "custom": self._stream_openai,
        }
        fn = stream_providers.get(self.provider)
        if fn is None:
            raise LLMError(
                f"Streaming not supported for provider '{self.provider}'. "
                f"Supported streaming providers: openai, anthropic, ollama, azure, custom."
            )

        for attempt in range(self.max_retries + 1):
            try:
                async for event in fn(messages, tools, **kwargs):
                    yield event
                return
            except LLMProviderError as e:
                if attempt < self.max_retries and self._should_retry(e.status_code):
                    await asyncio.sleep(self._retry_delay(attempt))
                    continue
                raise

    def stream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Synchronous streaming — collects stream into a single LLMResponse.

        For true streaming, use ``astream()`` in an async context.
        """

        async def _collect() -> LLMResponse:
            content_parts: list[str] = []
            tool_calls: list[ToolCall] = []
            usage_data: dict[str, int] = {}
            # Accumulate tool call args per call_id
            pending_tools: dict[str, dict[str, Any]] = {}

            async for event in self.astream(messages, tools=tools, **kwargs):
                if isinstance(event, TextDelta):
                    content_parts.append(event.text)
                elif isinstance(event, ToolCallStart):
                    pending_tools[event.call_id] = {"name": event.tool_name, "args": ""}
                elif isinstance(event, ToolCallEnd):
                    tool_calls.append(
                        ToolCall(
                            id=event.call_id,
                            name=event.tool_name,
                            arguments=event.arguments,
                        )
                    )
                    pending_tools.pop(event.call_id, None)
                elif isinstance(event, Usage):
                    usage_data = {
                        "prompt_tokens": event.prompt_tokens,
                        "completion_tokens": event.completion_tokens,
                        "total_tokens": event.prompt_tokens + event.completion_tokens,
                    }

            content = "".join(content_parts) or None
            finish = "tool_calls" if tool_calls else "stop"
            return LLMResponse(
                content=content,
                tool_calls=tool_calls,
                usage=usage_data,
                model=self.model,
                finish_reason=finish,
            )

        return run_sync(_collect())

    def _get_provider_fn(self) -> Any:
        providers = {
            "openai": self._call_openai,
            "anthropic": self._call_anthropic,
            "ollama": self._call_ollama,
            "azure": self._call_openai,  # Azure uses OpenAI-compatible API
            "bedrock": self._call_bedrock,
            "custom": self._call_openai,  # Custom endpoints are OpenAI-compatible
        }
        fn = providers.get(self.provider)
        if fn is None:
            supported = ", ".join(sorted(providers.keys()))
            raise LLMError(
                f"Unsupported provider '{self.provider}'. "
                f"Supported providers: {supported}.\n"
                f"Example: LLMClient(provider='openai', model='gpt-4o')"
            )
        return fn

    def _split_multimodal_tool_messages_for_openai(
        self, messages: list[Message]
    ) -> list[Message]:
        """Split tool messages with multimodal content into (tool, user) pairs.

        OpenAI's chat-completions API rejects ``image_url`` blocks inside
        tool-result messages — only user messages may carry images. When a
        tool returns an :class:`Image` or :class:`PDF` (in vision mode),
        the executor records a ``ToolMessage`` whose content is a
        ``list[ContentPart]``. For OpenAI we split that into:

        * a tool message with the textual summary only (satisfies the API)
        * a synthetic user message with the multimodal content prefixed by
          a short label (``"Here is the result of <tool>:"``)

        Anthropic and Bedrock accept images directly inside tool-result
        blocks, so this rewrite is OpenAI-only.
        """
        out: list[Message] = []
        for m in messages:
            if (
                m.role != MessageRole.tool
                or not m.has_multimodal_content()
                or not isinstance(m.content, list)
            ):
                out.append(m)
                continue
            text_parts = [p for p in m.content if isinstance(p, str)]
            media_parts = [p for p in m.content if not isinstance(p, str)]
            summary = "\n".join(text_parts) if text_parts else "[tool returned multimodal content]"
            out.append(
                Message(
                    role=MessageRole.tool,
                    content=summary,
                    tool_call_id=m.tool_call_id,
                )
            )
            user_parts: list[Any] = ["Here is the multimodal result from the previous tool call:"]
            user_parts.extend(media_parts)
            out.append(Message(role=MessageRole.user, content=user_parts))
        return out

    async def _call_openai(
        self, messages: list[Message], tools: list[dict[str, Any]] | None = None, **kwargs: Any
    ) -> LLMResponse:
        """Call OpenAI-compatible endpoint."""
        import httpx

        prepared = self._split_multimodal_tool_messages_for_openai(messages)
        msg_dicts = [
            m.to_provider_dict("openai", **self._provider_dict_kwargs()) for m in prepared
        ]
        body: dict[str, Any] = {
            "model": self.model,
            "messages": msg_dicts,
        }
        if tools:
            body["tools"] = tools
        if self.temperature is not None:
            body["temperature"] = self.temperature
        # OpenAI uses max_completion_tokens for newer models
        max_tok = kwargs.get("max_tokens", self.max_tokens)
        if max_tok is not None:
            if self.provider == "custom":
                body["max_tokens"] = max_tok
            else:
                body["max_completion_tokens"] = max_tok
        # Structured output
        response_format = kwargs.get("response_format")
        if response_format is not None:
            body["response_format"] = response_format
        # Additional parameters
        top_p = kwargs.get("top_p", self.top_p)
        if top_p is not None:
            body["top_p"] = top_p
        stop = kwargs.get("stop", self.stop)
        if stop is not None:
            body["stop"] = stop
        seed = kwargs.get("seed", self.seed)
        if seed is not None:
            body["seed"] = seed
        freq_pen = kwargs.get("frequency_penalty", self.frequency_penalty)
        if freq_pen is not None:
            body["frequency_penalty"] = freq_pen
        pres_pen = kwargs.get("presence_penalty", self.presence_penalty)
        if pres_pen is not None:
            body["presence_penalty"] = pres_pen
        ptc = kwargs.get("parallel_tool_calls", self.parallel_tool_calls)
        if ptc is not None and tools:
            body["parallel_tool_calls"] = ptc

        api_key = self.api_key or os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise LLMProviderError(
                f"No API key for provider '{self.provider}'. "
                f"Set the api_key parameter or the OPENAI_API_KEY environment variable.\n"
                f"Example: LLMClient(provider='{self.provider}', "
                f"model='{self.model}', api_key='sk-...')"
            )
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        url = f"{self.base_url.rstrip('/')}/chat/completions"
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(url, json=body, headers=headers)
            if resp.status_code != 200:
                raise LLMProviderError(
                    f"OpenAI API error {resp.status_code}: {resp.text}",
                    status_code=resp.status_code,
                )
            data = resp.json()

        return self._parse_openai_response(data)

    def _parse_openai_response(self, data: dict[str, Any]) -> LLMResponse:
        """Parse OpenAI-compatible response into LLMResponse."""
        import json

        choice = data.get("choices", [{}])[0]
        msg = choice.get("message", {})

        tool_calls = []
        if raw_tc := msg.get("tool_calls"):
            for tc in raw_tc:
                func = tc.get("function", {})
                args = func.get("arguments", "{}")
                if isinstance(args, str):
                    args = json.loads(args) if args else {}
                tool_calls.append(ToolCall(id=tc["id"], name=func["name"], arguments=args))

        return LLMResponse(
            content=msg.get("content"),
            tool_calls=tool_calls,
            usage=data.get("usage", {}),
            model=data.get("model", self.model),
            finish_reason=choice.get("finish_reason", ""),
        )

    async def _call_anthropic(
        self, messages: list[Message], tools: list[dict[str, Any]] | None = None, **kwargs: Any
    ) -> LLMResponse:
        """Call Anthropic Messages API."""
        import httpx

        # Extract system messages — Anthropic uses a separate 'system' field
        # Convert OpenAI message format to Anthropic format:
        #   - system messages → separate 'system' field
        #   - assistant messages with tool_calls → content blocks with type: tool_use
        #   - tool messages → user messages with content blocks type: tool_result
        system_parts: list[str] = []
        filtered_msgs: list[dict[str, Any]] = []
        for m in messages:
            if m.role == MessageRole.system:
                system_parts.append(_coerce_system_content_to_text(m.content))
            elif m.role == MessageRole.assistant and m.tool_calls:
                # Convert to Anthropic tool_use content blocks
                content: list[dict[str, Any]] = []
                if m.content:
                    content.append({"type": "text", "text": m.content})
                for tc in m.tool_calls:
                    content.append(
                        {
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.name,
                            "input": tc.arguments,
                        }
                    )
                filtered_msgs.append({"role": "assistant", "content": content})
            elif m.role == MessageRole.tool:
                # Convert to Anthropic tool_result content block
                filtered_msgs.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": m.tool_call_id,
                                "content": _anthropic_tool_result_content(m.content),
                            }
                        ],
                    }
                )
            else:
                filtered_msgs.append(
                    m.to_provider_dict("anthropic", **self._provider_dict_kwargs())
                )

        system_text = "\n\n".join(system_parts) if system_parts else None

        # Augment system prompt for response_format (Anthropic has no native support)
        response_format = kwargs.get("response_format")
        if response_format is not None:
            system_text = _augment_system_for_response_format(system_text, response_format)

        body: dict[str, Any] = {
            "model": self.model,
            "messages": filtered_msgs,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens) or 4096,
        }
        if system_text:
            body["system"] = system_text
        if self.temperature is not None:
            body["temperature"] = self.temperature
        # Additional parameters (Anthropic supports top_p and stop_sequences)
        top_p = kwargs.get("top_p", self.top_p)
        if top_p is not None:
            body["top_p"] = top_p
        stop = kwargs.get("stop", self.stop)
        if stop is not None:
            body["stop_sequences"] = [stop] if isinstance(stop, str) else stop

        # Convert tools from OpenAI format to Anthropic format
        if tools:
            anthropic_tools = []
            for t in tools:
                func = t.get("function", t)
                anthropic_tools.append(
                    {
                        "name": func["name"],
                        "description": func.get("description", ""),
                        "input_schema": func.get(
                            "parameters", {"type": "object", "properties": {}}
                        ),
                    }
                )
            body["tools"] = anthropic_tools

        api_key = self.api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise LLMProviderError(
                f"No API key for provider 'anthropic'. "
                f"Set the api_key parameter or the ANTHROPIC_API_KEY environment variable.\n"
                f"Example: LLMClient(provider='anthropic', "
                f"model='{self.model}', api_key='sk-ant-...')"
            )
        headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        }

        url = f"{self.base_url.rstrip('/')}/messages"
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(url, json=body, headers=headers)
            if resp.status_code != 200:
                raise LLMProviderError(
                    f"Anthropic API error {resp.status_code}: {resp.text}",
                    status_code=resp.status_code,
                )
            data = resp.json()

        # Parse Anthropic response
        content_text = ""
        tool_calls = []
        for block in data.get("content", []):
            if block.get("type") == "text":
                content_text += block.get("text", "")
            elif block.get("type") == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block["id"],
                        name=block["name"],
                        arguments=block.get("input", {}),
                    )
                )

        # Strip markdown code fences if response_format was requested
        if content_text and response_format is not None:
            rf_type = (
                response_format.get("type", "text") if isinstance(response_format, dict) else "text"
            )
            if rf_type in ("json_object", "json_schema"):
                content_text = _strip_code_fences(content_text)

        # Normalize usage
        usage_raw = data.get("usage", {})
        usage = {
            "prompt_tokens": usage_raw.get("input_tokens", 0),
            "completion_tokens": usage_raw.get("output_tokens", 0),
            "total_tokens": (usage_raw.get("input_tokens", 0) + usage_raw.get("output_tokens", 0)),
        }

        # Normalize finish reason
        stop_reason = data.get("stop_reason", "")
        finish_reason = {
            "end_turn": "stop",
            "tool_use": "tool_calls",
        }.get(stop_reason, stop_reason)

        return LLMResponse(
            content=content_text or None,
            tool_calls=tool_calls,
            usage=usage,
            model=data.get("model", self.model),
            finish_reason=finish_reason,
        )

    async def _call_ollama(
        self, messages: list[Message], tools: list[dict[str, Any]] | None = None, **kwargs: Any
    ) -> LLMResponse:
        """Call Ollama API."""
        import httpx

        msg_dicts = [m.to_provider_dict("ollama", **self._provider_dict_kwargs()) for m in messages]
        body: dict[str, Any] = {
            "model": self.model,
            "messages": msg_dicts,
            "stream": False,
        }
        if tools:
            body["tools"] = tools

        options: dict[str, Any] = {}
        if self.temperature is not None:
            options["temperature"] = self.temperature
        max_tok = kwargs.get("max_tokens", self.max_tokens)
        if max_tok is not None:
            options["num_predict"] = max_tok
        # Additional parameters
        top_p = kwargs.get("top_p", self.top_p)
        if top_p is not None:
            options["top_p"] = top_p
        stop = kwargs.get("stop", self.stop)
        if stop is not None:
            options["stop"] = [stop] if isinstance(stop, str) else stop
        seed = kwargs.get("seed", self.seed)
        if seed is not None:
            options["seed"] = seed
        freq_pen = kwargs.get("frequency_penalty", self.frequency_penalty)
        if freq_pen is not None:
            options["frequency_penalty"] = freq_pen
        pres_pen = kwargs.get("presence_penalty", self.presence_penalty)
        if pres_pen is not None:
            options["presence_penalty"] = pres_pen
        if options:
            body["options"] = options
        # Structured output
        response_format = kwargs.get("response_format")
        if response_format is not None:
            fmt = _ollama_format_from_response_format(response_format)
            if fmt:
                body["format"] = fmt

        url = f"{self.base_url.rstrip('/')}/api/chat"
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(url, json=body)
            if resp.status_code != 200:
                raise LLMProviderError(
                    f"Ollama API error {resp.status_code}: {resp.text}",
                    status_code=resp.status_code,
                )
            data = resp.json()

        msg = data.get("message", {})
        tool_calls = []
        if raw_tc := msg.get("tool_calls"):
            for i, tc in enumerate(raw_tc):
                func = tc.get("function", {})
                tool_calls.append(
                    ToolCall(
                        id=f"call_{i}",
                        name=func.get("name", ""),
                        arguments=func.get("arguments", {}),
                    )
                )

        usage = {
            "prompt_tokens": data.get("prompt_eval_count", 0),
            "completion_tokens": data.get("eval_count", 0),
            "total_tokens": (data.get("prompt_eval_count", 0) + data.get("eval_count", 0)),
        }

        finish_reason = "tool_calls" if tool_calls else "stop"

        return LLMResponse(
            content=msg.get("content"),
            tool_calls=tool_calls,
            usage=usage,
            model=self.model,
            finish_reason=finish_reason,
        )

    async def _call_bedrock(
        self, messages: list[Message], tools: list[dict[str, Any]] | None = None, **kwargs: Any
    ) -> LLMResponse:
        """Call AWS Bedrock (via boto3)."""
        try:
            import boto3
        except ImportError:
            raise LLMError(
                "boto3 is required for Bedrock provider. Install it with: pip install boto3"
            )

        client = boto3.client("bedrock-runtime", region_name=self._extra.get("region", "us-east-1"))
        msg_dicts = [
            m.to_provider_dict("bedrock", **self._provider_dict_kwargs()) for m in messages
        ]

        # Extract system for Bedrock/Anthropic models
        system_parts = []
        filtered = []
        for m in msg_dicts:
            if m["role"] == "system":
                system_parts.append({"text": m.get("content", "")})
            else:
                filtered.append(m)

        body: dict[str, Any] = {
            "messages": filtered,
        }
        if system_parts:
            body["system"] = system_parts

        inference_config: dict[str, Any] = {}
        if self.temperature is not None:
            inference_config["temperature"] = self.temperature
        max_tok = kwargs.get("max_tokens", self.max_tokens) or 4096
        inference_config["maxTokens"] = max_tok
        # Additional parameters (Bedrock supports topP and stopSequences)
        top_p = kwargs.get("top_p", self.top_p)
        if top_p is not None:
            inference_config["topP"] = top_p
        stop = kwargs.get("stop", self.stop)
        if stop is not None:
            inference_config["stopSequences"] = [stop] if isinstance(stop, str) else stop
        body["inferenceConfig"] = inference_config

        response = client.converse(modelId=self.model, **body)

        output = response.get("output", {})
        msg_out = output.get("message", {})
        content_text = ""
        tool_calls = []
        for block in msg_out.get("content", []):
            if "text" in block:
                content_text += block["text"]
            elif "toolUse" in block:
                tu = block["toolUse"]
                tool_calls.append(
                    ToolCall(
                        id=tu.get("toolUseId", ""),
                        name=tu.get("name", ""),
                        arguments=tu.get("input", {}),
                    )
                )

        usage_raw = response.get("usage", {})
        usage = {
            "prompt_tokens": usage_raw.get("inputTokens", 0),
            "completion_tokens": usage_raw.get("outputTokens", 0),
            "total_tokens": (usage_raw.get("inputTokens", 0) + usage_raw.get("outputTokens", 0)),
        }

        stop_reason = response.get("stopReason", "")
        finish_reason = {
            "end_turn": "stop",
            "tool_use": "tool_calls",
        }.get(stop_reason, stop_reason)

        return LLMResponse(
            content=content_text or None,
            tool_calls=tool_calls,
            usage=usage,
            model=self.model,
            finish_reason=finish_reason,
        )

    # --- Streaming provider methods ---

    async def _stream_openai(
        self, messages: list[Message], tools: list[dict[str, Any]] | None = None, **kwargs: Any
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream from OpenAI-compatible endpoint via SSE."""
        import httpx

        prepared = self._split_multimodal_tool_messages_for_openai(messages)
        msg_dicts = [
            m.to_provider_dict("openai", **self._provider_dict_kwargs()) for m in prepared
        ]
        body: dict[str, Any] = {
            "model": self.model,
            "messages": msg_dicts,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            body["tools"] = tools
        if self.temperature is not None:
            body["temperature"] = self.temperature
        max_tok = kwargs.get("max_tokens", self.max_tokens)
        if max_tok is not None:
            if self.provider == "custom":
                body["max_tokens"] = max_tok
            else:
                body["max_completion_tokens"] = max_tok
        # Structured output
        response_format = kwargs.get("response_format")
        if response_format is not None:
            body["response_format"] = response_format
        # Additional parameters
        top_p = kwargs.get("top_p", self.top_p)
        if top_p is not None:
            body["top_p"] = top_p
        stop = kwargs.get("stop", self.stop)
        if stop is not None:
            body["stop"] = stop
        seed = kwargs.get("seed", self.seed)
        if seed is not None:
            body["seed"] = seed
        freq_pen = kwargs.get("frequency_penalty", self.frequency_penalty)
        if freq_pen is not None:
            body["frequency_penalty"] = freq_pen
        pres_pen = kwargs.get("presence_penalty", self.presence_penalty)
        if pres_pen is not None:
            body["presence_penalty"] = pres_pen
        ptc = kwargs.get("parallel_tool_calls", self.parallel_tool_calls)
        if ptc is not None and tools:
            body["parallel_tool_calls"] = ptc

        api_key = self.api_key or os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise LLMProviderError(
                f"No API key for provider '{self.provider}'. "
                f"Set the api_key parameter or the OPENAI_API_KEY environment variable."
            )
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        url = f"{self.base_url.rstrip('/')}/chat/completions"
        # Accumulate tool call arguments across chunks
        tool_calls_acc: dict[int, dict[str, Any]] = {}

        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream("POST", url, json=body, headers=headers) as resp:
                if resp.status_code != 200:
                    body_text = await resp.aread()
                    raise LLMProviderError(
                        f"OpenAI API error {resp.status_code}: {body_text.decode()}",
                        status_code=resp.status_code,
                    )

                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:]
                    if payload.strip() == "[DONE]":
                        break

                    chunk = _json.loads(payload)
                    choices = chunk.get("choices", [])
                    delta = choices[0].get("delta", {}) if choices else {}

                    # Text content
                    if text := delta.get("content"):
                        yield TextDelta(text=text)

                    # Tool calls (streamed incrementally)
                    if raw_tcs := delta.get("tool_calls"):
                        for tc_delta in raw_tcs:
                            idx = tc_delta.get("index", 0)
                            if idx not in tool_calls_acc:
                                tool_calls_acc[idx] = {
                                    "id": tc_delta.get("id", ""),
                                    "name": tc_delta.get("function", {}).get("name", ""),
                                    "arguments_str": "",
                                }
                                if tool_calls_acc[idx]["name"]:
                                    yield ToolCallStart(
                                        call_id=tool_calls_acc[idx]["id"],
                                        tool_name=tool_calls_acc[idx]["name"],
                                    )
                            # Accumulate argument chunks
                            arg_chunk = tc_delta.get("function", {}).get("arguments", "")
                            if arg_chunk:
                                tool_calls_acc[idx]["arguments_str"] += arg_chunk

                    # Usage (typically in the final chunk)
                    if usage := chunk.get("usage"):
                        yield Usage(
                            prompt_tokens=usage.get("prompt_tokens", 0),
                            completion_tokens=usage.get("completion_tokens", 0),
                        )

        # Emit ToolCallEnd for each accumulated tool call
        for _idx, tc_acc in sorted(tool_calls_acc.items()):
            args_str = tc_acc["arguments_str"]
            try:
                args = _json.loads(args_str) if args_str else {}
            except _json.JSONDecodeError:
                args = {}
            yield ToolCallEnd(
                call_id=tc_acc["id"],
                tool_name=tc_acc["name"],
                arguments=args,
            )

        yield StreamDone()

    async def _stream_anthropic(
        self, messages: list[Message], tools: list[dict[str, Any]] | None = None, **kwargs: Any
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream from Anthropic Messages API via SSE."""
        import httpx

        # Build Anthropic request body (same conversion as _call_anthropic)
        system_parts: list[str] = []
        filtered_msgs: list[dict[str, Any]] = []
        for m in messages:
            if m.role == MessageRole.system:
                system_parts.append(_coerce_system_content_to_text(m.content))
            elif m.role == MessageRole.assistant and m.tool_calls:
                content: list[dict[str, Any]] = []
                if m.content:
                    content.append({"type": "text", "text": m.content})
                for tc in m.tool_calls:
                    content.append(
                        {"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments}
                    )
                filtered_msgs.append({"role": "assistant", "content": content})
            elif m.role == MessageRole.tool:
                filtered_msgs.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": m.tool_call_id,
                                "content": _anthropic_tool_result_content(m.content),
                            }
                        ],
                    }
                )
            else:
                filtered_msgs.append(
                    m.to_provider_dict("anthropic", **self._provider_dict_kwargs())
                )

        stream_system_text = "\n\n".join(system_parts) if system_parts else None

        # Augment system prompt for response_format (Anthropic has no native support)
        response_format = kwargs.get("response_format")
        if response_format is not None:
            stream_system_text = _augment_system_for_response_format(
                stream_system_text, response_format
            )

        body: dict[str, Any] = {
            "model": self.model,
            "messages": filtered_msgs,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens) or 4096,
            "stream": True,
        }
        if stream_system_text:
            body["system"] = stream_system_text
        if self.temperature is not None:
            body["temperature"] = self.temperature
        # Additional parameters (Anthropic supports top_p and stop_sequences)
        top_p = kwargs.get("top_p", self.top_p)
        if top_p is not None:
            body["top_p"] = top_p
        stop = kwargs.get("stop", self.stop)
        if stop is not None:
            body["stop_sequences"] = [stop] if isinstance(stop, str) else stop

        if tools:
            anthropic_tools = []
            for t in tools:
                func = t.get("function", t)
                anthropic_tools.append(
                    {
                        "name": func["name"],
                        "description": func.get("description", ""),
                        "input_schema": func.get(
                            "parameters", {"type": "object", "properties": {}}
                        ),
                    }
                )
            body["tools"] = anthropic_tools

        api_key = self.api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise LLMProviderError(
                "No API key for provider 'anthropic'. "
                "Set the api_key parameter or the ANTHROPIC_API_KEY environment variable."
            )
        headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        }

        url = f"{self.base_url.rstrip('/')}/messages"

        # Track current tool call state
        current_tool_id = ""
        current_tool_name = ""
        current_tool_input_str = ""
        prompt_tokens = 0
        completion_tokens = 0

        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream("POST", url, json=body, headers=headers) as resp:
                if resp.status_code != 200:
                    body_text = await resp.aread()
                    raise LLMProviderError(
                        f"Anthropic API error {resp.status_code}: {body_text.decode()}",
                        status_code=resp.status_code,
                    )

                event_type = ""
                async for line in resp.aiter_lines():
                    if line.startswith("event: "):
                        event_type = line[7:].strip()
                        continue
                    if not line.startswith("data: "):
                        continue
                    data = _json.loads(line[6:])

                    if event_type == "content_block_start":
                        block = data.get("content_block", {})
                        if block.get("type") == "tool_use":
                            current_tool_id = block.get("id", "")
                            current_tool_name = block.get("name", "")
                            current_tool_input_str = ""
                            yield ToolCallStart(
                                call_id=current_tool_id,
                                tool_name=current_tool_name,
                            )

                    elif event_type == "content_block_delta":
                        delta = data.get("delta", {})
                        if delta.get("type") == "text_delta":
                            yield TextDelta(text=delta.get("text", ""))
                        elif delta.get("type") == "input_json_delta":
                            current_tool_input_str += delta.get("partial_json", "")

                    elif event_type == "content_block_stop":
                        if current_tool_id:
                            try:
                                args = (
                                    _json.loads(current_tool_input_str)
                                    if current_tool_input_str
                                    else {}
                                )
                            except _json.JSONDecodeError:
                                args = {}
                            yield ToolCallEnd(
                                call_id=current_tool_id,
                                tool_name=current_tool_name,
                                arguments=args,
                            )
                            current_tool_id = ""
                            current_tool_name = ""
                            current_tool_input_str = ""

                    elif event_type == "message_start":
                        msg_usage = data.get("message", {}).get("usage", {})
                        prompt_tokens = msg_usage.get("input_tokens", 0)

                    elif event_type == "message_delta":
                        delta_usage = data.get("usage", {})
                        completion_tokens = delta_usage.get("output_tokens", 0)

        if prompt_tokens or completion_tokens:
            yield Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
        yield StreamDone()

    async def _stream_ollama(
        self, messages: list[Message], tools: list[dict[str, Any]] | None = None, **kwargs: Any
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream from Ollama API via newline-delimited JSON."""
        import httpx

        msg_dicts = [m.to_provider_dict("ollama", **self._provider_dict_kwargs()) for m in messages]
        body: dict[str, Any] = {
            "model": self.model,
            "messages": msg_dicts,
            "stream": True,
        }
        if tools:
            body["tools"] = tools

        options: dict[str, Any] = {}
        if self.temperature is not None:
            options["temperature"] = self.temperature
        max_tok = kwargs.get("max_tokens", self.max_tokens)
        if max_tok is not None:
            options["num_predict"] = max_tok
        # Additional parameters
        top_p = kwargs.get("top_p", self.top_p)
        if top_p is not None:
            options["top_p"] = top_p
        stop = kwargs.get("stop", self.stop)
        if stop is not None:
            options["stop"] = [stop] if isinstance(stop, str) else stop
        seed = kwargs.get("seed", self.seed)
        if seed is not None:
            options["seed"] = seed
        freq_pen = kwargs.get("frequency_penalty", self.frequency_penalty)
        if freq_pen is not None:
            options["frequency_penalty"] = freq_pen
        pres_pen = kwargs.get("presence_penalty", self.presence_penalty)
        if pres_pen is not None:
            options["presence_penalty"] = pres_pen
        if options:
            body["options"] = options
        # Structured output
        response_format = kwargs.get("response_format")
        if response_format is not None:
            fmt = _ollama_format_from_response_format(response_format)
            if fmt:
                body["format"] = fmt

        url = f"{self.base_url.rstrip('/')}/api/chat"
        prompt_tokens = 0
        completion_tokens = 0

        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream("POST", url, json=body) as resp:
                if resp.status_code != 200:
                    body_text = await resp.aread()
                    raise LLMProviderError(
                        f"Ollama API error {resp.status_code}: {body_text.decode()}",
                        status_code=resp.status_code,
                    )

                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    chunk = _json.loads(line)

                    # Text content
                    msg = chunk.get("message", {})
                    if text := msg.get("content"):
                        yield TextDelta(text=text)

                    # Ollama sends tool calls only in the final message (done=true)
                    if chunk.get("done"):
                        prompt_tokens = chunk.get("prompt_eval_count", 0)
                        completion_tokens = chunk.get("eval_count", 0)

                        # Tool calls in final message
                        if raw_tcs := msg.get("tool_calls"):
                            for i, tc in enumerate(raw_tcs):
                                func = tc.get("function", {})
                                call_id = f"call_{i}"
                                name = func.get("name", "")
                                args = func.get("arguments", {})
                                yield ToolCallStart(call_id=call_id, tool_name=name)
                                yield ToolCallEnd(call_id=call_id, tool_name=name, arguments=args)

        if prompt_tokens or completion_tokens:
            yield Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
        yield StreamDone()

    def to_dict(self) -> dict[str, Any]:
        """Serialize to canonical format."""
        data: dict[str, Any] = {
            "provider": self.provider,
            "model": self.model,
        }
        if self.base_url and self.base_url != self._default_base_url(self.provider):
            data["base_url"] = self.base_url
        if self.temperature is not None:
            data["temperature"] = self.temperature
        if self.max_tokens is not None:
            data["max_tokens"] = self.max_tokens
        if self.max_retries:
            data["max_retries"] = self.max_retries
        if self.top_p is not None:
            data["top_p"] = self.top_p
        if self.stop is not None:
            data["stop"] = self.stop
        if self.seed is not None:
            data["seed"] = self.seed
        if self.frequency_penalty is not None:
            data["frequency_penalty"] = self.frequency_penalty
        if self.presence_penalty is not None:
            data["presence_penalty"] = self.presence_penalty
        if self.parallel_tool_calls is not None:
            data["parallel_tool_calls"] = self.parallel_tool_calls
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LLMClient:
        """Deserialize from canonical format."""
        return cls(
            provider=data.get("provider", "openai"),
            model=data.get("model", "gpt-4o-mini"),
            api_key=data.get("api_key"),
            base_url=data.get("base_url"),
            temperature=data.get("temperature"),
            max_tokens=data.get("max_tokens"),
            max_retries=data.get("max_retries", 0),
            top_p=data.get("top_p"),
            stop=data.get("stop"),
            seed=data.get("seed"),
            frequency_penalty=data.get("frequency_penalty"),
            presence_penalty=data.get("presence_penalty"),
            parallel_tool_calls=data.get("parallel_tool_calls"),
        )
