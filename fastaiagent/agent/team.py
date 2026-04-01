"""Supervisor/Worker team delegation pattern."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from fastaiagent._internal.async_utils import run_sync
from fastaiagent.agent.agent import Agent, AgentConfig, AgentResult
from fastaiagent.llm.client import LLMClient
from fastaiagent.tool.base import Tool


class Worker:
    """An agent assigned a specific role in a team."""

    def __init__(
        self,
        agent: Agent,
        role: str = "",
        description: str = "",
    ):
        self.agent = agent
        self.role = role or agent.name
        self.description = description or agent.system_prompt[:200]


class Supervisor:
    """Manages a team of Worker agents, delegating tasks.

    The supervisor LLM decides which worker(s) to invoke.

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
        system_prompt: str = "",
        max_delegation_rounds: int = 3,
    ):
        self.name = name
        self.llm = llm or LLMClient()
        self.workers = workers or []
        self.max_delegation_rounds = max_delegation_rounds
        self.system_prompt = system_prompt or self._build_supervisor_prompt()

    def _build_supervisor_prompt(self) -> str:
        worker_desc = "\n".join(f"- {w.role}: {w.description}" for w in self.workers)
        return (
            f"You are a supervisor managing a team of workers.\n"
            f"Available workers:\n{worker_desc}\n\n"
            f"Delegate tasks by calling the appropriate worker tool. "
            f"Synthesize results into a final answer."
        )

    def _build_worker_tools(self) -> Sequence[Tool]:
        """Create tool definitions for each worker."""
        from fastaiagent.tool.function import FunctionTool

        tools: list[Tool] = []
        for worker in self.workers:

            async def delegate(task: str, _worker: Worker = worker) -> str:
                result = await _worker.agent.arun(task)
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

    def run(self, input: str, **kwargs: Any) -> AgentResult:
        """Run the supervisor synchronously."""
        return run_sync(self.arun(input, **kwargs))

    async def arun(self, input: str, **kwargs: Any) -> AgentResult:
        """Run the supervisor — delegates to workers via tool calls."""
        agent = Agent(
            name=self.name,
            system_prompt=self.system_prompt,
            llm=self.llm,
            tools=self._build_worker_tools(),
            config=AgentConfig(max_iterations=self.max_delegation_rounds * 2),
        )
        return await agent.arun(input, **kwargs)
