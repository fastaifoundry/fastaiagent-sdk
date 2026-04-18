"""Agent class — the central component of the SDK."""

from __future__ import annotations

import json
import time
from collections.abc import AsyncGenerator, Callable, Sequence
from typing import Any

from pydantic import BaseModel, Field

from fastaiagent._internal.async_utils import run_sync
from fastaiagent.agent.context import RunContext
from fastaiagent.agent.executor import execute_tool_loop, stream_tool_loop
from fastaiagent.agent.memory import AgentMemory
from fastaiagent.agent.middleware import (
    AgentMiddleware,
    MiddlewareContext,
    _MiddlewarePipeline,
)
from fastaiagent.guardrail.executor import execute_guardrails
from fastaiagent.guardrail.guardrail import Guardrail, GuardrailPosition
from fastaiagent.llm.client import LLMClient, _strip_code_fences
from fastaiagent.llm.message import Message, SystemMessage, UserMessage
from fastaiagent.llm.stream import StreamEvent, TextDelta
from fastaiagent.tool.base import Tool


class AgentConfig(BaseModel):
    """Agent execution configuration."""

    max_iterations: int = Field(default=10, ge=1, le=100)
    tool_choice: str = "auto"  # "auto", "required", "none"
    temperature: float | None = None
    max_tokens: int | None = None


class AgentResult(BaseModel):
    """Result of an agent execution."""

    output: str = ""
    parsed: Any | None = None
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    tokens_used: int = 0
    cost: float = 0.0
    latency_ms: int = 0
    trace_id: str | None = None

    model_config = {"arbitrary_types_allowed": True}


class Agent:
    """An AI agent with tools, guardrails, and full tracing.

    Example:
        agent = Agent(
            name="support-bot",
            system_prompt="You are a helpful support agent.",
            llm=LLMClient(provider="openai", model="gpt-4o"),
            tools=[search_tool, refund_tool],
            guardrails=[no_pii()],
        )
        result = agent.run("How do I get a refund?")
    """

    def __init__(
        self,
        name: str,
        system_prompt: str | Callable[..., str] = "",
        llm: LLMClient | None = None,
        tools: Sequence[Tool] | None = None,
        guardrails: Sequence[Guardrail] | None = None,
        memory: AgentMemory | None = None,
        config: AgentConfig | None = None,
        output_type: type | None = None,
        middleware: Sequence[AgentMiddleware] | None = None,
    ):
        self.name = name
        self.system_prompt = system_prompt
        self.llm = llm or LLMClient()
        self.tools: list[Tool] = list(tools) if tools else []
        self.guardrails: list[Guardrail] = list(guardrails) if guardrails else []
        self.memory = memory
        self.config = config or AgentConfig()
        self.output_type = output_type
        self.middleware: list[AgentMiddleware] = list(middleware) if middleware else []
        self._mw_pipeline = _MiddlewarePipeline(self.middleware)

    def _build_response_format(self) -> dict[str, Any] | None:
        """Build response_format dict from output_type for structured output."""
        if self.output_type is None:
            return None
        return {
            "type": "json_schema",
            "json_schema": {
                "name": self.output_type.__name__,
                "schema": self.output_type.model_json_schema(),
            },
        }

    def _parse_output(self, text: str) -> Any | None:
        """Parse LLM text output into output_type Pydantic model."""
        if self.output_type is None or not text:
            return None
        try:
            clean = _strip_code_fences(text)
            data = json.loads(clean)
            return self.output_type.model_validate(data)
        except Exception:
            return None

    def run(
        self, input: str, *, context: RunContext | None = None, trace: bool = True, **kwargs: Any
    ) -> AgentResult:
        """Synchronous execution."""
        return run_sync(self.arun(input, context=context, trace=trace, **kwargs))

    async def arun(
        self, input: str, *, context: RunContext | None = None, trace: bool = True, **kwargs: Any
    ) -> AgentResult:
        """Async execution with tool-calling loop."""
        if trace:
            return await self._arun_traced(input, context=context, **kwargs)
        return await self._arun_core(input, context=context, **kwargs)

    async def _arun_traced(
        self, input: str, *, context: RunContext | None = None, **kwargs: Any
    ) -> AgentResult:
        """Execute with OTel tracing."""
        from fastaiagent.trace.otel import get_tracer
        from fastaiagent.trace.span import trace_payloads_enabled

        tracer = get_tracer()
        with tracer.start_as_current_span(f"agent.{self.name}") as span:
            span.set_attribute("agent.name", self.name)
            span.set_attribute("agent.input", input)

            # Reconstruction metadata for ForkedReplay.arerun (always captured —
            # structural, not payload).
            span.set_attribute("agent.config", json.dumps(self.config.model_dump()))
            span.set_attribute(
                "agent.tools", json.dumps([t.to_dict() for t in self.tools])
            )
            span.set_attribute(
                "agent.guardrails", json.dumps([g.to_dict() for g in self.guardrails])
            )
            span.set_attribute("agent.llm.provider", self.llm.provider)
            span.set_attribute("agent.llm.model", self.llm.model)
            span.set_attribute("agent.llm.config", json.dumps(self.llm.to_dict()))

            # Resolved system prompt (payload-gated).
            if trace_payloads_enabled():
                try:
                    resolved_prompt = self._resolve_system_prompt(context)
                    if resolved_prompt:
                        span.set_attribute("agent.system_prompt", resolved_prompt)
                except Exception:
                    # Callable system_prompt that needs context — best-effort only.
                    pass

            result = await self._arun_core(input, context=context, **kwargs)

            span.set_attribute("agent.output", result.output)
            span.set_attribute("agent.tokens_used", result.tokens_used)
            span.set_attribute("agent.latency_ms", result.latency_ms)

            # Set trace_id on result
            ctx = span.get_span_context()
            result.trace_id = format(ctx.trace_id, "032x")
            return result

    async def _arun_core(
        self, input: str, *, context: RunContext | None = None, **kwargs: Any
    ) -> AgentResult:
        """Core execution without tracing."""
        start = time.monotonic()

        # Execute input guardrails (blocking)
        if self.guardrails:
            await execute_guardrails(self.guardrails, input, GuardrailPosition.input)

        # Build messages
        messages = self._build_messages(input, context=context)

        # Inject response_format for structured output
        response_format = self._build_response_format()
        if response_format is not None:
            kwargs["response_format"] = response_format

        # Middleware context shared across the whole run.
        mw_ctx: MiddlewareContext | None = None
        if self._mw_pipeline:
            mw_ctx = MiddlewareContext(run_context=context, agent_name=self.name)

        # Execute tool-calling loop. StopAgent raised by middleware is caught
        # inside the loop so that partial tool-call records are preserved.
        response, tool_calls = await execute_tool_loop(
            llm=self.llm,
            messages=messages,
            tools=self.tools,
            max_iterations=self.config.max_iterations,
            tool_choice=self.config.tool_choice,
            context=context,
            guardrails=self.guardrails or None,
            mw_pipeline=self._mw_pipeline if self._mw_pipeline else None,
            mw_ctx=mw_ctx,
            **kwargs,
        )

        output = response.content or ""
        parsed = self._parse_output(output)

        # Execute output guardrails
        if self.guardrails:
            await execute_guardrails(self.guardrails, output, GuardrailPosition.output)

        # Store in memory
        if self.memory:
            self.memory.add(UserMessage(input))
            from fastaiagent.llm.message import AssistantMessage

            self.memory.add(AssistantMessage(output))

        latency = int((time.monotonic() - start) * 1000)
        tokens = response.usage.get("total_tokens", 0)

        return AgentResult(
            output=output,
            parsed=parsed,
            tool_calls=tool_calls,
            tokens_used=tokens,
            latency_ms=latency,
        )

    async def astream(
        self, input: str, *, context: RunContext | None = None, trace: bool = True, **kwargs: Any
    ) -> AsyncGenerator[StreamEvent, None]:
        """Async streaming execution — yields StreamEvent objects as tokens arrive.

        Runs input guardrails before streaming begins. Output guardrails
        run after streaming completes. Memory is updated at the end.

        Example:
            async for event in agent.astream("Hello"):
                if isinstance(event, TextDelta):
                    print(event.text, end="", flush=True)
        """
        # Execute input guardrails (blocking)
        if self.guardrails:
            await execute_guardrails(self.guardrails, input, GuardrailPosition.input)

        messages = self._build_messages(input, context=context)

        # Inject response_format for structured output
        response_format = self._build_response_format()
        if response_format is not None:
            kwargs["response_format"] = response_format

        # Stream tool loop — yields events to caller
        accumulated_text = ""
        async for event in stream_tool_loop(
            llm=self.llm,
            messages=messages,
            tools=self.tools,
            max_iterations=self.config.max_iterations,
            tool_choice=self.config.tool_choice,
            context=context,
            guardrails=self.guardrails or None,
            **kwargs,
        ):
            if isinstance(event, TextDelta):
                accumulated_text += event.text
            yield event

        output = accumulated_text

        # Execute output guardrails
        if self.guardrails:
            await execute_guardrails(self.guardrails, output, GuardrailPosition.output)

        # Store in memory
        if self.memory:
            self.memory.add(UserMessage(input))
            from fastaiagent.llm.message import AssistantMessage

            self.memory.add(AssistantMessage(output))

    def stream(
        self, input: str, *, context: RunContext | None = None, trace: bool = True, **kwargs: Any
    ) -> AgentResult:
        """Synchronous streaming — collects stream into AgentResult.

        For true streaming, use ``astream()`` in an async context.
        """

        async def _collect() -> AgentResult:
            start = time.monotonic()
            text_parts: list[str] = []
            async for event in self.astream(input, context=context, trace=trace, **kwargs):
                if isinstance(event, TextDelta):
                    text_parts.append(event.text)
            latency = int((time.monotonic() - start) * 1000)
            output = "".join(text_parts)
            parsed = self._parse_output(output)
            return AgentResult(
                output=output,
                parsed=parsed,
                latency_ms=latency,
            )

        return run_sync(_collect())

    def _resolve_system_prompt(self, context: RunContext | None = None) -> str:
        """Resolve system_prompt to a string. Calls it if callable."""
        if callable(self.system_prompt):
            return self.system_prompt(context)
        return self.system_prompt

    def _build_messages(self, input: str, context: RunContext | None = None) -> list[Message]:
        """Build the message array for the LLM."""
        messages: list[Message] = []

        system_text = self._resolve_system_prompt(context)
        if system_text:
            messages.append(SystemMessage(system_text))

        # Add memory context
        if self.memory:
            messages.extend(self.memory.get_context())

        messages.append(UserMessage(input))
        return messages

    def to_dict(self) -> dict[str, Any]:
        """Serialize to canonical format for platform push."""
        if callable(self.system_prompt):
            raise ValueError(
                f"Agent '{self.name}' has a callable system_prompt which cannot be "
                f"serialized. Use a static string for agents pushed to the platform."
            )
        d: dict[str, Any] = {
            "name": self.name,
            "agent_type": "single",
            "system_prompt": self.system_prompt,
            "llm_endpoint": self.llm.to_dict(),
            "tools": [t.to_dict() for t in self.tools],
            "guardrails": [g.to_dict() for g in self.guardrails],
            "config": self.config.model_dump(),
        }
        # Include response_format schema if output_type is set.
        # Note: output_type (a Python class) cannot be restored from JSON.
        response_format = self._build_response_format()
        if response_format is not None:
            d["config"]["response_format"] = response_format
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Agent:
        """Deserialize from canonical format (platform pull).

        Note: output_type cannot be restored from serialized data because
        it is a Python class. The response_format schema in config is
        informational and will be passed through to the LLM if present.
        """
        return cls(
            name=data["name"],
            system_prompt=data.get("system_prompt", ""),
            llm=LLMClient.from_dict(data.get("llm_endpoint", {})),
            tools=[Tool.from_dict(t) for t in data.get("tools", [])],
            guardrails=[Guardrail.from_dict(g) for g in data.get("guardrails", [])],
            config=AgentConfig(**data.get("config", {})),
        )
