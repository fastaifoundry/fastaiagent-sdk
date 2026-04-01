"""Agent class — the central component of the SDK."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, Field

from fastaiagent._internal.async_utils import run_sync
from fastaiagent.agent.executor import execute_tool_loop
from fastaiagent.agent.memory import AgentMemory
from fastaiagent.guardrail.executor import execute_guardrails
from fastaiagent.guardrail.guardrail import Guardrail, GuardrailPosition
from fastaiagent.llm.client import LLMClient
from fastaiagent.llm.message import Message, SystemMessage, UserMessage
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
        system_prompt: str = "",
        llm: LLMClient | None = None,
        tools: Sequence[Tool] | None = None,
        guardrails: Sequence[Guardrail] | None = None,
        memory: AgentMemory | None = None,
        config: AgentConfig | None = None,
    ):
        self.name = name
        self.system_prompt = system_prompt
        self.llm = llm or LLMClient()
        self.tools: list[Tool] = list(tools) if tools else []
        self.guardrails: list[Guardrail] = list(guardrails) if guardrails else []
        self.memory = memory
        self.config = config or AgentConfig()

    def run(self, input: str, *, trace: bool = True, **kwargs: Any) -> AgentResult:
        """Synchronous execution."""
        return run_sync(self.arun(input, trace=trace, **kwargs))

    async def arun(self, input: str, *, trace: bool = True, **kwargs: Any) -> AgentResult:
        """Async execution with tool-calling loop."""
        start = time.monotonic()

        # Execute input guardrails (blocking)
        if self.guardrails:
            await execute_guardrails(self.guardrails, input, GuardrailPosition.input)

        # Build messages
        messages = self._build_messages(input)

        # Execute tool-calling loop
        response, tool_calls = await execute_tool_loop(
            llm=self.llm,
            messages=messages,
            tools=self.tools,
            max_iterations=self.config.max_iterations,
            tool_choice=self.config.tool_choice,
        )

        output = response.content or ""

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
            tool_calls=tool_calls,
            tokens_used=tokens,
            latency_ms=latency,
        )

    def _build_messages(self, input: str) -> list[Message]:
        """Build the message array for the LLM."""
        messages: list[Message] = []

        if self.system_prompt:
            messages.append(SystemMessage(self.system_prompt))

        # Add memory context
        if self.memory:
            messages.extend(self.memory.get_context())

        messages.append(UserMessage(input))
        return messages

    def to_dict(self) -> dict[str, Any]:
        """Serialize to canonical format for platform push."""
        return {
            "name": self.name,
            "agent_type": "single",
            "system_prompt": self.system_prompt,
            "llm_endpoint": self.llm.to_dict(),
            "tools": [t.to_dict() for t in self.tools],
            "guardrails": [g.to_dict() for g in self.guardrails],
            "config": self.config.model_dump(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Agent:
        """Deserialize from canonical format (platform pull)."""
        return cls(
            name=data["name"],
            system_prompt=data.get("system_prompt", ""),
            llm=LLMClient.from_dict(data.get("llm_endpoint", {})),
            tools=[Tool.from_dict(t) for t in data.get("tools", [])],
            guardrails=[Guardrail.from_dict(g) for g in data.get("guardrails", [])],
            config=AgentConfig(**data.get("config", {})),
        )
