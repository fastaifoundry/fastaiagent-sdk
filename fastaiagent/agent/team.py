"""Supervisor/Worker team delegation pattern."""

from __future__ import annotations

import time
from collections.abc import AsyncGenerator, Callable, Sequence
from typing import Any

from fastaiagent._internal.async_utils import run_sync
from fastaiagent.agent.agent import Agent, AgentConfig, AgentResult
from fastaiagent.agent.context import RunContext
from fastaiagent.llm.client import LLMClient
from fastaiagent.llm.stream import StreamEvent, TextDelta
from fastaiagent.tool.base import Tool
from fastaiagent.tool.function import FunctionTool


class Worker:
    """An agent assigned a specific role in a team.

    Example:
        researcher = Agent(name="researcher", system_prompt="You research topics.", ...)
        worker = Worker(agent=researcher, role="researcher", description="Searches for info")
    """

    def __init__(
        self,
        agent: Agent,
        role: str = "",
        description: str = "",
    ):
        self.agent = agent
        self.role = role or agent.name
        if description:
            self.description = description
        else:
            prompt = agent.system_prompt
            self.description = prompt[:200] if isinstance(prompt, str) else ""


class Supervisor:
    """Manages a team of Worker agents, delegating tasks.

    The supervisor LLM decides which worker(s) to invoke based on
    the task. Workers run as full agents with their own tools and
    system prompts.

    Example:
        supervisor = Supervisor(
            name="team-lead",
            llm=LLMClient(provider="openai", model="gpt-4o"),
            workers=[
                Worker(agent=researcher, role="researcher", description="Searches for info"),
                Worker(agent=writer, role="writer", description="Writes content"),
            ],
        )
        result = supervisor.run("Research and write a report on AI trends")
    """

    def __init__(
        self,
        name: str,
        llm: LLMClient | None = None,
        workers: list[Worker] | None = None,
        system_prompt: str | Callable[..., str] = "",
        max_delegation_rounds: int = 3,
    ):
        self.name = name
        self.llm = llm or LLMClient()
        self.workers = workers or []
        self.max_delegation_rounds = max_delegation_rounds
        self.system_prompt = system_prompt if system_prompt else self._build_supervisor_prompt()

    def _build_supervisor_prompt(self) -> str:
        worker_desc = "\n".join(f"- {w.role}: {w.description}" for w in self.workers)
        return (
            f"You are a supervisor managing a team of workers.\n"
            f"Available workers:\n{worker_desc}\n\n"
            f"Delegate tasks by calling the appropriate worker tool. "
            f"Synthesize results into a final answer."
        )

    def _build_worker_tools(self, context: RunContext | None = None) -> Sequence[Tool]:
        """Create tool definitions for each worker.

        Rebuilds tools per call to capture the current context
        for forwarding to workers. This is stateless and concurrent-safe.
        """
        tools: list[Tool] = []
        for worker in self.workers:

            async def delegate(
                task: str,
                _worker: Worker = worker,
                _ctx: RunContext | None = context,
            ) -> str:
                result = await _worker.agent.arun(task, context=_ctx)
                return result.output

            tools.append(
                FunctionTool(
                    name=f"delegate_to_{worker.role}",
                    fn=delegate,
                    description=f"Delegate a task to {worker.role}: {worker.description}",
                    parameters={
                        "type": "object",
                        "properties": {
                            "task": {
                                "type": "string",
                                "description": "The task to delegate",
                            }
                        },
                        "required": ["task"],
                    },
                )
            )
        return tools

    def run(self, input: str, *, context: RunContext | None = None, **kwargs: Any) -> AgentResult:
        """Run the supervisor synchronously."""
        return run_sync(self.arun(input, context=context, **kwargs))

    async def arun(self, input: str, *, context: RunContext | None = None, **kwargs: Any) -> AgentResult:
        """Run the supervisor — delegates to workers via tool calls."""
        from fastaiagent.trace.otel import get_tracer

        tracer = get_tracer()
        # Root span wraps the supervisor run so the delegated worker spans
        # nest as children in the UI span tree.
        with tracer.start_as_current_span(f"supervisor.{self.name}") as span:
            span.set_attribute("supervisor.name", self.name)
            span.set_attribute(
                "supervisor.worker_count",
                len(getattr(self, "workers", []) or []),
            )
            span.set_attribute("fastaiagent.runner.type", "supervisor")
            span.set_attribute("supervisor.input", input)

            agent = Agent(
                name=self.name,
                system_prompt=self.system_prompt,
                llm=self.llm,
                tools=self._build_worker_tools(context=context),
                config=AgentConfig(max_iterations=self.max_delegation_rounds * 2),
            )
            result = await agent.arun(input, context=context, **kwargs)
            span.set_attribute("supervisor.output", result.output)
            return result

    async def astream(
        self, input: str, *, context: RunContext | None = None, **kwargs: Any
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream the supervisor — yields events as tokens arrive.

        Yields TextDelta for the supervisor's synthesis, and
        ToolCallStart/ToolCallEnd for worker delegations.
        Worker execution itself is not streamed.
        """
        agent = Agent(
            name=self.name,
            system_prompt=self.system_prompt,
            llm=self.llm,
            tools=self._build_worker_tools(context=context),
            config=AgentConfig(max_iterations=self.max_delegation_rounds * 2),
        )
        async for event in agent.astream(input, context=context, **kwargs):
            yield event

    def stream(self, input: str, *, context: RunContext | None = None, **kwargs: Any) -> AgentResult:
        """Synchronous streaming — collects stream into AgentResult."""

        async def _collect() -> AgentResult:
            start = time.monotonic()
            text_parts: list[str] = []
            async for event in self.astream(input, context=context, **kwargs):
                if isinstance(event, TextDelta):
                    text_parts.append(event.text)
            latency = int((time.monotonic() - start) * 1000)
            return AgentResult(output="".join(text_parts), latency_ms=latency)

        return run_sync(_collect())
