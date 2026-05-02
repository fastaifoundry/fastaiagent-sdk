"""Agent Replay — fork-and-rerun debugging for AI agents."""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import BaseModel, Field

from fastaiagent._internal.errors import ReplayError
from fastaiagent.trace.storage import SpanData, TraceData, TraceStore

_log = logging.getLogger(__name__)


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

    def modify_input(self, new_input: Any) -> ForkedReplay:
        """Override the input the rerun will use.

        Accepts the same shapes as :py:meth:`Agent.run`:

        * a string (legacy)
        * a single ``Image`` / ``PDF`` instance
        * a ``list[ContentPart]`` (mix of strings, ``Image``, ``PDF``)
        * a ``dict`` with a ``"input"`` key (legacy contract — flattened to
          a string before rerun)
        """
        self._modifications["input"] = new_input
        return self

    def modify_prompt(self, new_prompt: str) -> ForkedReplay:
        self._modifications["prompt"] = new_prompt
        return self

    def modify_config(self, **kwargs: Any) -> ForkedReplay:
        self._modifications["config"] = kwargs
        return self

    def modify_state(self, new_state: dict[str, Any]) -> ForkedReplay:
        self._modifications["state"] = new_state
        return self

    def with_tools(self, tools: list[Any]) -> ForkedReplay:
        """Override the tools used during rerun.

        When the original trace used tools with dynamic callables (e.g.,
        ``kb.as_tool()``), those callables cannot be reconstructed from
        span attributes. Pass the live tool instances here so the rerun
        agent can actually execute them.

        Example::

            forked = replay.fork_at(step=0)
            forked.with_tools([kb.as_tool(), my_other_tool])
            forked.modify_prompt("Be more concise.")
            result = forked.rerun()
        """
        self._modifications["tools"] = tools
        return self

    async def arerun(self) -> ReplayResult:
        """Rerun the agent with modifications applied.

        Reconstructs an ``Agent`` from span attributes captured on the original
        trace (see ``fastaiagent/agent/agent.py::_arun_traced``), applies any
        user modifications (modify_input / modify_prompt / modify_config), and
        re-executes via ``agent.arun``.

        v1 note: modifications are applied to the agent as a whole, then the
        agent re-runs from the top with the (possibly modified) input. True
        mid-trace resume — replaying messages up to ``fork_point`` then
        continuing — is a v2 concern and requires a stable on-span message
        history representation across providers. ``compare()`` still marks
        ``diverged_at = fork_point`` so downstream tooling sees where the
        user asked for divergence.
        """
        from fastaiagent.agent.agent import Agent

        root = self._find_root_span()
        if root is None:
            raise ReplayError(
                f"Trace {self._trace.trace_id} has no spans — cannot reconstruct agent."
            )

        agent_dict = self._build_agent_dict(root)
        self._apply_agent_modifications(agent_dict)

        from fastaiagent.multimodal.image import Image as _MMImage
        from fastaiagent.multimodal.pdf import PDF as _MMPDF

        original_input = self._extract_original_input(root)
        new_input = self._modifications.get("input", original_input)

        # Accept the same input shapes as ``Agent.run``. Plain strings,
        # ``Image``, ``PDF``, and lists of ContentPart are passed through
        # unchanged. Legacy dict form is flattened.
        if isinstance(new_input, dict):
            new_input = new_input.get("input") or json.dumps(new_input, default=str)
        elif isinstance(new_input, (_MMImage, _MMPDF, list)):
            pass  # Multimodal-ready — Agent.arun accepts these directly.
        elif not isinstance(new_input, str):
            new_input = str(new_input)

        try:
            agent = Agent.from_dict(agent_dict)
        except Exception as e:
            raise ReplayError(
                f"Failed to reconstruct agent from trace {self._trace.trace_id}: {e}"
            ) from e

        # Apply tool overrides — replaces the serialization-only tools with
        # live callable instances so the rerun can actually execute them.
        if "tools" in self._modifications:
            agent.tools = list(self._modifications["tools"])

        new_result = await agent.arun(new_input)

        original_output = root.attributes.get("agent.output")
        return ReplayResult(
            original_output=original_output,
            new_output=new_result.output,
            steps_executed=len(self._steps) - self._fork_point,
            trace_id=new_result.trace_id,
        )

    def rerun(self) -> ReplayResult:
        from fastaiagent._internal.async_utils import run_sync

        return run_sync(self.arerun())

    def compare(self, new_result: ReplayResult) -> ComparisonResult:
        """Build a side-by-side comparison between the original and rerun traces."""
        new_steps: list[ReplayStep] = []
        if new_result.trace_id:
            try:
                new_replay = Replay.load(new_result.trace_id)
                new_steps = new_replay.steps()
            except Exception as e:
                _log.warning(
                    "compare(): could not load rerun trace %s: %s",
                    new_result.trace_id,
                    e,
                )
        return ComparisonResult(
            original_steps=self._steps,
            new_steps=new_steps,
            diverged_at=self._fork_point,
        )

    # ── Internals ──────────────────────────────────────────────────────────

    def _find_root_span(self) -> SpanData | None:
        """Locate the root agent span (parent_span_id is falsy, or name starts
        with 'agent.'). Falls back to the first span in document order."""
        for span in self._trace.spans:
            if not span.parent_span_id and span.name.startswith("agent."):
                return span
        for span in self._trace.spans:
            if not span.parent_span_id:
                return span
        return self._trace.spans[0] if self._trace.spans else None

    def _build_agent_dict(self, root: SpanData) -> dict[str, Any]:
        """Reconstruct the canonical Agent.from_dict payload from span attrs."""
        attrs = root.attributes or {}

        def _load_json(key: str, default: Any) -> Any:
            raw = attrs.get(key)
            if raw is None:
                return default
            if isinstance(raw, str):
                try:
                    return json.loads(raw)
                except json.JSONDecodeError:
                    return default
            return raw

        return {
            "name": attrs.get("agent.name", "replayed-agent"),
            "agent_type": "single",
            "system_prompt": attrs.get("agent.system_prompt", ""),
            "llm_endpoint": _load_json("agent.llm.config", {}),
            "tools": _load_json("agent.tools", []),
            "guardrails": _load_json("agent.guardrails", []),
            "config": _load_json("agent.config", {}),
        }

    def _apply_agent_modifications(self, agent_dict: dict[str, Any]) -> None:
        if "prompt" in self._modifications:
            agent_dict["system_prompt"] = self._modifications["prompt"]
        if "config" in self._modifications:
            cfg = agent_dict.setdefault("config", {})
            if isinstance(cfg, dict):
                cfg.update(self._modifications["config"])

    def _extract_original_input(self, root: SpanData) -> str:
        raw = (root.attributes or {}).get("agent.input", "")
        return raw if isinstance(raw, str) else json.dumps(raw, default=str)


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

    @classmethod
    def from_platform(cls, trace_id: str) -> Replay:
        """Pull a trace from the platform and create a Replay.

        The platform API returns a different schema than local SQLite
        storage. This method maps the platform response into the SDK's
        internal ``TraceData`` / ``SpanData`` models:

        Platform span shape::

            {
                "id": "44aea511e72b02ee",       ← maps to span_id
                "span_type": "sdk",
                "name": "agent.support-bot",
                "status": "unset",
                "input": { ...all span attributes... },
                "output": { ...may contain additional attrs... },
                "start_time": "...",
                "end_time": "...",
                "metadata": {}
            }

        SDK ``SpanData`` shape::

            SpanData(span_id, trace_id, parent_span_id, name, attributes, ...)

        Key differences:
        - ``id`` (platform) → ``span_id`` (SDK)
        - ``trace_id`` is on the trace envelope, not on each span
        - ``parent_span_id`` is not provided by the platform — set to None
        - ``input`` + ``output`` dicts are merged into ``attributes``
        - ``span_type`` and ``metadata`` are platform-specific fields
        """
        from fastaiagent._internal.errors import PlatformNotConnectedError
        from fastaiagent._platform.api import get_platform_api
        from fastaiagent.client import _connection

        if not _connection.is_connected:
            raise PlatformNotConnectedError(
                "Not connected to platform. Call fa.connect() first."
            )
        api = get_platform_api()
        data = api.get(f"/public/v1/traces/{trace_id}")

        resolved_trace_id = data.get("trace_id") or data.get("id") or trace_id

        spans = []
        for s in data.get("spans", []):
            # Merge input + output dicts into a single attributes dict.
            # The platform stores pre-execution attrs in "input" and
            # post-execution attrs in "output". The SDK expects one flat
            # "attributes" dict.
            attrs: dict[str, Any] = {}
            if isinstance(s.get("input"), dict):
                attrs.update(s["input"])
            if isinstance(s.get("output"), dict):
                attrs.update(s["output"])

            spans.append(
                SpanData(
                    span_id=s.get("id") or s.get("span_id", ""),
                    trace_id=resolved_trace_id,
                    parent_span_id=s.get("parent_span_id"),
                    name=s.get("name", ""),
                    start_time=s.get("start_time", ""),
                    end_time=s.get("end_time", ""),
                    status=s.get("status", "OK"),
                    attributes=attrs,
                    events=s.get("events", []),
                )
            )

        # Derive trace-level fields from the first span if the platform
        # response doesn't carry them at the top level.
        first_span = spans[0] if spans else None
        trace_data = TraceData(
            trace_id=resolved_trace_id,
            name=data.get("name") or (first_span.name if first_span else ""),
            start_time=data.get("start_time") or (first_span.start_time if first_span else ""),
            end_time=data.get("end_time") or (spans[-1].end_time if spans else ""),
            status=data.get("status", "OK"),
            metadata=data.get("metadata", {}),
            spans=spans,
        )
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
