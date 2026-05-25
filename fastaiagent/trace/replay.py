"""Agent Replay — fork-and-rerun debugging for AI agents."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from fastaiagent._internal.errors import ReplayError
from fastaiagent.trace.storage import SpanData, TraceData, TraceStore

_log = logging.getLogger(__name__)

DeterminismMode = Literal["live", "recorded", "deterministic"]


def _recorded_response_from_span(span: SpanData) -> Any | None:
    """Reconstruct an :class:`LLMResponse` from a captured ``llm.*`` span.

    Looks at the GenAI semantic-convention attributes
    (``gen_ai.response.content``, ``gen_ai.response.finish_reason``,
    ``gen_ai.response.tool_calls``) and rebuilds an ``LLMResponse``
    instance that ``LLMClient.acomplete`` can return verbatim under
    ``determinism="recorded"``.

    Returns ``None`` when the span lacks a response content — meaning
    payload capture was disabled on the original run or this span isn't
    an LLM call. Callers must check.
    """
    from fastaiagent.llm.client import LLMResponse
    from fastaiagent.llm.message import ToolCall

    attrs = span.attributes or {}
    content = attrs.get("gen_ai.response.content")
    if content is None:
        return None

    tool_calls: list[ToolCall] = []
    raw_tool_calls = attrs.get("gen_ai.response.tool_calls")
    if raw_tool_calls:
        try:
            parsed = (
                json.loads(raw_tool_calls) if isinstance(raw_tool_calls, str) else raw_tool_calls
            )
            if isinstance(parsed, list):
                for tc in parsed:
                    tool_calls.append(
                        ToolCall(
                            id=tc.get("id", ""),
                            name=tc.get("name", ""),
                            arguments=tc.get("arguments", {}),
                        )
                    )
        except (json.JSONDecodeError, TypeError, KeyError):
            # Recorded responses degrade gracefully — missing tool calls
            # are treated as "no tools called this turn" rather than
            # failing the entire replay.
            pass

    return LLMResponse(
        content=content if isinstance(content, str) else str(content),
        tool_calls=tool_calls,
        finish_reason=str(attrs.get("gen_ai.response.finish_reason", "stop")),
        usage={},
        latency_ms=0,
    )


def _first_divergence(original: list[ReplayStep], new: list[ReplayStep]) -> int | None:
    """Return the first step index where ``original`` and ``new`` differ.

    "Differ" means: different span name OR different ``agent.output`` /
    ``gen_ai.response.content`` attribute. Returns ``None`` when the two
    lists agree step-for-step up to the shorter length. When the lists
    are different lengths but the overlap matches, returns the index of
    the first missing-from-shorter step (i.e. ``min(len(original), len(new))``).
    """
    if not original or not new:
        return None
    n = min(len(original), len(new))
    for i in range(n):
        a = original[i]
        b = new[i]
        if a.span_name != b.span_name:
            return i
        a_out = (a.attributes or {}).get("agent.output") or (a.attributes or {}).get(
            "gen_ai.response.content"
        )
        b_out = (b.attributes or {}).get("agent.output") or (b.attributes or {}).get(
            "gen_ai.response.content"
        )
        if a_out != b_out:
            return i
    if len(original) != len(new):
        return n
    return None


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

    def save_as_test(
        self,
        dataset_path: str | Path,
        input: str,
        expected_output: str,
        source_trace_id: str | None = None,
    ) -> Path:
        """Append this rerun as a regression-test case to a JSONL dataset.

        The written record uses the same field names ``evaluate()`` reads
        (``input``, ``expected_output``), so the dataset is immediately
        consumable as a regression suite::

            replay = Replay.load(failed.trace_id)
            forked = replay.fork_at(step=3).modify_prompt("Be specific.")
            rerun = forked.rerun()
            rerun.save_as_test(
                "regression_tests.jsonl",
                input="What is our refund policy?",
                expected_output=rerun.new_output,
            )

        Args:
            dataset_path: JSONL file to append to. Parent dirs are created.
            input: The agent input that should be replayed in future eval runs.
            expected_output: The known-good output to compare against.
            source_trace_id: Origin trace for provenance. Defaults to the
                rerun's own ``trace_id``; pass the *original failure's*
                trace_id to keep the link back to the bug.

        Returns:
            The path written to.
        """
        path = Path(dataset_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "input": input,
            "expected_output": expected_output,
            "trace_id": source_trace_id or self.trace_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        with path.open("a") as f:
            f.write(json.dumps(record) + "\n")
        return path


class ComparisonResult(BaseModel):
    """Side-by-side comparison between original and forked execution.

    ``diverged_at`` is the first step index where the original and new
    traces produce different outputs — computed by walking both step
    lists in parallel and finding the first mismatch in span name or
    captured output. ``None`` means the two traces match step-for-step
    up to the shorter one's length.

    ``compare_status`` distinguishes successful comparisons from cases
    where the rerun trace couldn't be loaded — useful for UIs that want
    to show "rerun failed" instead of pretending the comparison is
    complete.
    """

    original_steps: list[ReplayStep] = Field(default_factory=list)
    new_steps: list[ReplayStep] = Field(default_factory=list)
    diverged_at: int | None = None
    compare_status: Literal["ok", "rerun_failed"] = "ok"


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
        self._tool_overrides: dict[str, Any] = {}
        self._determinism: DeterminismMode = "live"

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
        """Override the tools used during rerun (full replacement).

        When the original trace used tools with dynamic callables (e.g.,
        ``kb.as_tool()``), those callables cannot be reconstructed from
        span attributes. Pass the live tool instances here so the rerun
        agent can actually execute them.

        Example::

            forked = replay.fork_at(step=0)
            forked.with_tools([kb.as_tool(), my_other_tool])
            forked.modify_prompt("Be more concise.")
            result = forked.rerun()

        See also :py:meth:`with_tool_override` for replacing a single tool
        while keeping the others intact.
        """
        self._modifications["tools"] = tools
        return self

    def with_tool_override(self, name: str, tool: Any) -> ForkedReplay:
        """Replace a single tool by name; other tools keep their original
        (re-constructed-from-trace) implementations.

        Useful for regression-from-trace flows where you've fixed one
        buggy tool and want to verify the agent now produces the right
        answer — without rebuilding the entire toolset.

        Multiple calls compose: the last override for a given name wins::

            forked.with_tool_override("search_kb", new_search_tool)
            forked.with_tool_override("create_ticket", patched_ticket_tool)

        ``with_tools(...)`` (full replacement) takes precedence over
        ``with_tool_override(...)`` if both are used on the same fork.
        """
        if not name:
            raise ReplayError("with_tool_override requires a non-empty tool name")
        self._tool_overrides[name] = tool
        return self

    def with_determinism(self, mode: DeterminismMode) -> ForkedReplay:
        """Control how the LLM is invoked during rerun.

        * ``"live"`` *(default)*: call the LLM provider with whatever
          temperature/seed the captured config carried. Output is
          nondeterministic for non-zero temperature.
        * ``"recorded"``: skip the LLM HTTP call entirely; return the
          captured response from the original trace's GenAI attributes.
          Use for byte-identical regression tests.
        * ``"deterministic"``: call the LLM live, but force
          ``temperature=0`` (and ``seed=42`` where the provider supports
          it). Reduces nondeterminism without skipping the call.

        See ``docs/replay/guarantees.md`` for per-provider support and
        known limitations (e.g. streaming chunks aren't recorded
        granularly — ``recorded`` mode reconstructs from final text).
        """
        if mode not in ("live", "recorded", "deterministic"):
            raise ReplayError(
                f"Unknown determinism mode {mode!r}; "
                "expected 'live', 'recorded', or 'deterministic'"
            )
        self._determinism = mode
        return self

    async def arerun(self) -> ReplayResult:
        """Rerun the agent with modifications applied.

        Reconstructs an ``Agent`` from span attributes captured on the original
        trace (see ``fastaiagent/agent/agent.py::_arun_traced``), applies any
        user modifications (modify_input / modify_prompt / modify_config /
        with_tool_override / with_determinism), and re-executes via
        ``agent.arun``.

        v1 note: modifications are applied to the agent as a whole, then the
        agent re-runs from the top with the (possibly modified) input. True
        mid-trace resume — replaying messages up to ``fork_point`` then
        continuing — is a v2 concern and requires a stable on-span message
        history representation across providers.
        """
        from fastaiagent.agent.agent import Agent
        from fastaiagent.llm.client import _replay_recorded_response

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

        # Tool plumbing precedence: with_tools(...) full replacement wins,
        # then per-name overrides patch on top of the trace-reconstructed
        # list.
        if "tools" in self._modifications:
            agent.tools = list(self._modifications["tools"])
        if self._tool_overrides:
            agent.tools = self._apply_tool_overrides(agent.tools)

        # Determinism mode plumbing — see ForkedReplay.with_determinism.
        if self._determinism == "deterministic":
            # Force temperature=0 and seed=42 on the LLM client where
            # the provider supports them. Bedrock has no seed concept,
            # which we accept as a documented gap.
            if hasattr(agent, "llm") and agent.llm is not None:
                try:
                    agent.llm.temperature = 0
                    agent.llm.seed = 42
                except AttributeError:
                    pass

        if self._determinism == "recorded":
            recorded = self._first_llm_response()
            if recorded is None:
                raise ReplayError(
                    f"Trace {self._trace.trace_id} has no captured LLM response "
                    f"(gen_ai.response.content). determinism='recorded' requires "
                    f"FASTAIAGENT_TRACE_PAYLOADS=1 (the default) on the original run."
                )
            token = _replay_recorded_response.set(recorded)
            try:
                new_result = await agent.arun(new_input)
            finally:
                _replay_recorded_response.reset(token)
        else:
            new_result = await agent.arun(new_input)

        original_output = root.attributes.get("agent.output")
        return ReplayResult(
            original_output=original_output,
            new_output=new_result.output,
            steps_executed=len(self._steps) - self._fork_point,
            trace_id=new_result.trace_id,
        )

    def _apply_tool_overrides(self, tools: list[Any]) -> list[Any]:
        """Substitute tools whose ``.name`` matches a per-name override.

        Tools without an override keep their trace-reconstructed
        implementation. Override names that don't match any
        reconstructed tool are appended to the end so the caller can
        introduce a tool that didn't exist on the original run.
        """
        out: list[Any] = []
        matched: set[str] = set()
        for t in tools:
            name = getattr(t, "name", None)
            if name and name in self._tool_overrides:
                out.append(self._tool_overrides[name])
                matched.add(name)
            else:
                out.append(t)
        # Append any overrides that didn't match a reconstructed tool.
        for name, tool in self._tool_overrides.items():
            if name not in matched:
                out.append(tool)
        return out

    def _first_llm_response(self) -> Any | None:
        """Return the first :class:`LLMResponse` recoverable from any
        span carrying GenAI response attributes. Walks every span (not
        just those named ``llm.*``) so traces captured under older span
        naming conventions still work for ``determinism="recorded"``."""
        for span in self._trace.spans:
            recovered = _recorded_response_from_span(span)
            if recovered is not None:
                return recovered
        return None

    def rerun(self) -> ReplayResult:
        from fastaiagent._internal.async_utils import run_sync

        return run_sync(self.arerun())

    def compare(self, new_result: ReplayResult) -> ComparisonResult:
        """Build a side-by-side comparison between the original and rerun traces.

        ``diverged_at`` is computed by walking both step lists and finding
        the first index where span name or captured output differ. This
        replaces the v1.13 behavior of hardcoding ``diverged_at`` to the
        fork point — which was misleading because the actual divergence
        often happens later (the fork point is just where the user *asked*
        to diverge).

        When the rerun trace can't be loaded, ``compare_status`` is set to
        ``"rerun_failed"`` and ``diverged_at`` stays ``None`` so callers
        can distinguish "no divergence" from "couldn't tell".
        """
        new_steps: list[ReplayStep] = []
        status: Literal["ok", "rerun_failed"] = "ok"
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
                status = "rerun_failed"
        else:
            status = "rerun_failed"

        diverged = None if status == "rerun_failed" else _first_divergence(self._steps, new_steps)
        return ComparisonResult(
            original_steps=self._steps,
            new_steps=new_steps,
            diverged_at=diverged,
            compare_status=status,
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
            raise PlatformNotConnectedError("Not connected to platform. Call fa.connect() first.")
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
