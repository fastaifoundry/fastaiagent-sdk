"""Supervisor/Worker team delegation pattern."""

from __future__ import annotations

import time
from collections.abc import AsyncGenerator, Callable, Sequence
from typing import Any

from fastaiagent._internal.async_utils import run_sync
from fastaiagent.agent.agent import Agent, AgentConfig, AgentResult
from fastaiagent.agent.context import RunContext
from fastaiagent.agent.executor import _AgentInterrupted
from fastaiagent.chain.interrupt import (
    Resume,
    _execution_id,
    _resume_value,
)
from fastaiagent.checkpointers import Checkpointer, SQLiteCheckpointer
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
        checkpointer: Checkpointer | None = None,
    ):
        self.name = name
        self.llm = llm or LLMClient()
        self.workers = workers or []
        self.max_delegation_rounds = max_delegation_rounds
        self.system_prompt = system_prompt if system_prompt else self._build_supervisor_prompt()
        self._checkpointer: Checkpointer | None = checkpointer

    def _build_supervisor_prompt(self) -> str:
        worker_desc = "\n".join(f"- {w.role}: {w.description}" for w in self.workers)
        return (
            f"You are a supervisor managing a team of workers.\n"
            f"Available workers:\n{worker_desc}\n\n"
            f"Delegate tasks by calling the appropriate worker tool. "
            f"Synthesize results into a final answer."
        )

    def _build_worker_clone(self, worker: Worker) -> Agent:
        """Build the agent that actually runs when ``delegate_to_<role>`` fires.

        The clone shares everything with the user's worker.agent but:
            - uses the supervisor's checkpointer (so its checkpoints land
              under the supervisor's execution_id);
            - uses ``"worker:<role>"`` as its agent_path label so paths nest
              as ``supervisor:<s>/worker:<role>/...``.
        """
        base = worker.agent
        return Agent(
            name=base.name,
            system_prompt=base.system_prompt,
            llm=base.llm,
            tools=list(base.tools),
            guardrails=list(base.guardrails),
            memory=base.memory,
            config=base.config,
            output_type=base.output_type,
            middleware=list(base.middleware),
            checkpointer=self._checkpointer,
            agent_path_label=f"worker:{worker.role}",
        )

    def _has_worker_state(self, execution_id: str, worker_role: str) -> bool:
        """True if any checkpoint exists under this worker's path prefix."""
        if self._checkpointer is None:
            return False
        prefix = f"supervisor:{self.name}/worker:{worker_role}"
        for cp in self._checkpointer.list(execution_id, limit=500):
            if (cp.agent_path or "").startswith(prefix):
                return True
        return False

    def _build_worker_tools(self, context: RunContext | None = None) -> Sequence[Tool]:
        """Create durability-aware ``delegate_to_<role>`` tools.

        Each tool, when fired:
          1. Reads the active execution_id from the ``_execution_id`` ContextVar.
          2. If a worker checkpoint exists for this (execution, role) — i.e.
             we're inside :meth:`Supervisor.aresume` and the worker is mid-flight
             — calls ``worker_clone.aresume(...)`` (with the supervisor's
             ``_resume_value`` if any). Otherwise calls ``worker_clone.arun(task)``.
          3. If the worker returns ``status="paused"``, raises
             :class:`_AgentInterrupted` so the supervisor's tool loop bubbles
             paused state up — the worker has already persisted its
             interrupted checkpoint and ``pending_interrupts`` row with the
             full nested ``agent_path``.
        """
        tools: list[Tool] = []
        for worker in self.workers:

            async def delegate(
                task: str,
                _worker: Worker = worker,
                _ctx: RunContext | None = context,
                _supervisor: Supervisor = self,
            ) -> str:
                clone = _supervisor._build_worker_clone(_worker)
                exec_id = _execution_id.get()
                rv = _resume_value.get()
                # Branch: resume vs. fresh run.
                if exec_id is not None and _supervisor._has_worker_state(exec_id, _worker.role):
                    # Scope the resume to the worker's subtree so it doesn't
                    # accidentally pick up the supervisor's own pre-tool
                    # checkpoint as "latest".
                    worker_prefix = f"supervisor:{_supervisor.name}/worker:{_worker.role}"
                    result = await clone.aresume(
                        exec_id,
                        resume_value=rv,
                        context=_ctx,
                        agent_path_prefix=worker_prefix,
                    )
                else:
                    result = await clone.arun(task, context=_ctx, execution_id=exec_id)

                if result.status == "paused":
                    pi = result.pending_interrupt or {}
                    raise _AgentInterrupted(
                        reason=str(pi.get("reason", "")),
                        context=dict(pi.get("context", {})),
                        node_id=str(pi.get("node_id", "")),
                        agent_path=pi.get("agent_path"),
                    )
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

    def _build_inner_agent(self, context: RunContext[Any] | None = None) -> Agent:
        """Build the supervisor-as-agent with durability + custom path label."""
        return Agent(
            name=self.name,
            system_prompt=self.system_prompt,
            llm=self.llm,
            tools=self._build_worker_tools(context=context),
            config=AgentConfig(max_iterations=self.max_delegation_rounds * 2),
            checkpointer=self._checkpointer,
            agent_path_label=f"supervisor:{self.name}",
        )

    def run(
        self,
        input: str,
        *,
        context: RunContext | None = None,
        execution_id: str | None = None,
        **kwargs: Any,
    ) -> AgentResult:
        """Run the supervisor synchronously."""
        return run_sync(self.arun(input, context=context, execution_id=execution_id, **kwargs))

    async def arun(
        self,
        input: str,
        *,
        context: RunContext | None = None,
        execution_id: str | None = None,
        **kwargs: Any,
    ) -> AgentResult:
        """Run the supervisor — delegates to workers via tool calls.

        ``execution_id`` (optional) names this run for resume; pair with a
        ``checkpointer`` on the Supervisor to get crash- and interrupt-
        recovery via :meth:`resume`.
        """
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

            if self._checkpointer is not None:
                self._checkpointer.setup()

            agent = self._build_inner_agent(context=context)
            result = await agent.arun(input, context=context, execution_id=execution_id, **kwargs)
            span.set_attribute("supervisor.output", result.output)
            return result

    def resume(
        self,
        execution_id: str,
        *,
        resume_value: Resume | None = None,
        context: RunContext[Any] | None = None,
        **kwargs: Any,
    ) -> AgentResult:
        """Synchronous resume wrapper around :meth:`aresume`."""
        return run_sync(
            self.aresume(execution_id, resume_value=resume_value, context=context, **kwargs)
        )

    async def aresume(
        self,
        execution_id: str,
        *,
        resume_value: Resume | None = None,
        context: RunContext[Any] | None = None,
        **kwargs: Any,
    ) -> AgentResult:
        """Resume a paused or crashed supervisor run.

        Strategy: rebuild the supervisor's inner Agent, recover the original
        input from the supervisor's earliest checkpoint, and call
        ``inner_agent.arun(input, execution_id=...)`` again. The supervisor's
        LLM is re-issued; it will deterministically call the same
        ``delegate_to_<role>`` tools in order. Each delegate tool detects
        that worker state exists for this execution and calls
        ``worker_clone.aresume(...)`` (with the supervisor's ``resume_value``
        in scope) instead of running the worker fresh. So the resumed
        worker picks up exactly where it left off; subsequent workers run
        normally.
        """
        from fastaiagent._internal.errors import ChainCheckpointError

        store: Checkpointer = self._checkpointer or SQLiteCheckpointer()
        store.setup()

        latest = store.get_last(execution_id)
        if latest is None:
            raise ChainCheckpointError(
                f"No checkpoint found for supervisor execution '{execution_id}'"
            )
        if resume_value is None and latest.status == "interrupted":
            raise ChainCheckpointError(
                f"Supervisor execution '{execution_id}' is suspended on interrupt(); "
                "pass resume_value=Resume(...) to supervisor.resume()."
            )

        # Recover the original supervisor input by walking the supervisor's
        # own earliest checkpoint and pulling the first user message.
        input_str = self._hydrate_input(store, execution_id)
        if input_str is None:
            raise ChainCheckpointError(
                f"Cannot recover original input for supervisor execution '{execution_id}'."
            )

        # Bind the resume value so every worker the supervisor delegates to
        # during this resumed run can pick it up via _resume_value.
        rv_token = _resume_value.set(resume_value) if resume_value is not None else None
        try:
            return await self.arun(
                input_str,
                context=context,
                execution_id=execution_id,
                **kwargs,
            )
        finally:
            if rv_token is not None:
                _resume_value.reset(rv_token)

    def _hydrate_input(self, store: Checkpointer, execution_id: str) -> str | None:
        """Pull the original supervisor input from its earliest turn checkpoint."""
        prefix_exact = f"supervisor:{self.name}"
        # ``list`` returns checkpoints in chronological order — first match
        # wins. Filter to checkpoints written by the supervisor itself
        # (not by a worker nested under the supervisor).
        for cp in store.list(execution_id, limit=500):
            if (cp.agent_path or "") != prefix_exact:
                continue
            for raw in cp.state_snapshot.get("messages", []) or []:
                if (raw or {}).get("role") == "user" and raw.get("content"):
                    return str(raw["content"])
        return None

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

    def stream(
        self, input: str, *, context: RunContext | None = None, **kwargs: Any
    ) -> AgentResult:
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

    def to_dict(self) -> dict[str, Any]:
        """Serialize the supervisor structure for the Local UI topology view.

        Worker agents are referenced by name + role; rebuilding requires the
        caller to pass the live :class:`Worker` instances back in.
        """
        return {
            "name": self.name,
            "supervisor_llm": {
                "provider": getattr(self.llm, "provider", ""),
                "model": getattr(self.llm, "model", ""),
            },
            "workers": [
                {
                    "role": w.role,
                    "agent_name": w.agent.name,
                    "description": w.description,
                    "model": getattr(w.agent.llm, "model", ""),
                    "tools": [t.name for t in (w.agent.tools or [])],
                }
                for w in self.workers
            ],
            "max_delegation_rounds": self.max_delegation_rounds,
        }
