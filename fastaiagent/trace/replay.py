"""Agent Replay — fork-and-rerun debugging for AI agents."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from fastaiagent._internal.errors import ReplayError
from fastaiagent.trace.storage import SpanData, TraceData, TraceStore


class ReplayStep(BaseModel):
    """A single step in a replay."""

    step: int
    span_name: str = ""
    span_id: str = ""
    input: dict[str, Any] = Field(default_factory=dict)
    output: dict[str, Any] = Field(default_factory=dict)
    attributes: dict[str, Any] = Field(default_factory=dict)
    timestamp: str = ""


class ReplayResult(BaseModel):
    """Result of a forked replay re-execution."""

    original_output: Any = None
    new_output: Any = None
    steps_executed: int = 0
    trace_id: str | None = None


class ComparisonResult(BaseModel):
    """Side-by-side comparison between original and forked execution."""

    original_steps: list[ReplayStep] = Field(default_factory=list)
    new_steps: list[ReplayStep] = Field(default_factory=list)
    diverged_at: int | None = None


class ForkedReplay:
    """A forked execution that can be modified and rerun."""

    def __init__(
        self,
        original_trace: TraceData,
        fork_point: int,
        steps: list[ReplayStep],
    ):
        self._trace = original_trace
        self._fork_point = fork_point
        self._steps = steps
        self._modifications: dict[str, Any] = {}

    def modify_input(self, new_input: dict) -> ForkedReplay:
        self._modifications["input"] = new_input
        return self

    def modify_prompt(self, new_prompt: str) -> ForkedReplay:
        self._modifications["prompt"] = new_prompt
        return self

    def modify_config(self, **kwargs: Any) -> ForkedReplay:
        self._modifications["config"] = kwargs
        return self

    def modify_state(self, new_state: dict) -> ForkedReplay:
        self._modifications["state"] = new_state
        return self

    async def arerun(self) -> ReplayResult:
        """Rerun from fork point with modifications.

        In a full implementation this reconstructs the agent/chain from
        trace metadata and re-executes. For now returns a placeholder.
        """
        return ReplayResult(
            original_output=self._steps[-1].output if self._steps else None,
            new_output=None,
            steps_executed=len(self._steps) - self._fork_point,
            trace_id=self._trace.trace_id,
        )

    def rerun(self) -> ReplayResult:
        import asyncio

        return asyncio.run(self.arerun())

    def compare(self, new_result: ReplayResult) -> ComparisonResult:
        return ComparisonResult(
            original_steps=self._steps,
            diverged_at=self._fork_point,
        )


class Replay:
    """Replay and debug agent/chain executions from traces.

    Example:
        replay = Replay.load("trace_abc123")
        replay.summary()
        replay.step_through()
        forked = replay.fork_at(step=3)
        forked.modify_prompt("Be more specific...")
        result = forked.rerun()
    """

    def __init__(self, trace_data: TraceData):
        self._trace = trace_data
        self._steps = self._build_steps()

    @classmethod
    def load(cls, trace_id: str, store: TraceStore | None = None) -> Replay:
        """Load a replay from a stored trace."""
        store = store or TraceStore.default()
        trace_data = store.get_trace(trace_id)
        return cls(trace_data)

    def _build_steps(self) -> list[ReplayStep]:
        steps = []
        for i, span in enumerate(self._trace.spans):
            steps.append(
                ReplayStep(
                    step=i,
                    span_name=span.name,
                    span_id=span.span_id,
                    attributes=span.attributes,
                    timestamp=span.start_time,
                )
            )
        return steps

    def summary(self) -> str:
        """Rich-formatted summary of the execution."""
        lines = [
            f"Trace: {self._trace.trace_id}",
            f"Name: {self._trace.name}",
            f"Status: {self._trace.status}",
            f"Spans: {len(self._trace.spans)}",
            f"Duration: {self._trace.start_time} → {self._trace.end_time}",
            "",
            "Steps:",
        ]
        for step in self._steps:
            lines.append(f"  [{step.step}] {step.span_name}")
        return "\n".join(lines)

    def steps(self) -> list[ReplayStep]:
        """Get all steps in the replay."""
        return list(self._steps)

    def inspect(self, step: int) -> ReplayStep:
        """Inspect a specific step."""
        if step < 0 or step >= len(self._steps):
            raise ReplayError(f"Step {step} out of range (0-{len(self._steps) - 1})")
        return self._steps[step]

    def step_through(self) -> list[ReplayStep]:
        """Step through all steps (returns all steps for programmatic use)."""
        return list(self._steps)

    def fork_at(self, step: int) -> ForkedReplay:
        """Fork the execution at a specific step for re-execution."""
        if step < 0 or step >= len(self._steps):
            raise ReplayError(f"Step {step} out of range (0-{len(self._steps) - 1})")
        return ForkedReplay(
            original_trace=self._trace,
            fork_point=step,
            steps=self._steps,
        )
