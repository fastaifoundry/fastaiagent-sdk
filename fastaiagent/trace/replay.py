"""Agent Replay — fork-and-rerun debugging for AI agents."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from fastaiagent._internal.errors import ReplayError
from fastaiagent.trace import _replay_roles as _roles
from fastaiagent.trace.storage import SpanData, TraceData, TraceStore

_log = logging.getLogger(__name__)

DeterminismMode = Literal["live", "recorded", "deterministic"]

#: What a rerun does with tools it was not given. See ``with_tool_policy``.
_TOOL_POLICIES = ("block", "error", "live")


def _default_tool_policy() -> str:
    """``block``, unless ``FASTAIAGENT_REPLAY_TOOL_POLICY`` names another policy.

    The env var is a migration aid (1.89.0): ``live`` restores the old default,
    under which a rerun ran every tool it could rebuild.
    """
    import os

    raw = os.environ.get("FASTAIAGENT_REPLAY_TOOL_POLICY") or ""
    value = raw.strip().lower()
    if value in _TOOL_POLICIES:
        return value
    if value:
        _log.warning(
            "FASTAIAGENT_REPLAY_TOOL_POLICY=%r is not one of %s; using 'block'.",
            raw,
            list(_TOOL_POLICIES),
        )
    return "block"


def _has_redacted_header(spec: dict[str, Any]) -> bool:
    from fastaiagent.tool.rest import RESTTool

    headers = (spec.get("config") or {}).get("headers") or {}
    return isinstance(headers, dict) and any(
        v == RESTTool._REDACTED_HEADER_VALUE for v in headers.values()
    )


def _coerce_replay_class(value: Any) -> str:
    from fastaiagent.tool._replay_blocked import _coerce_replay_class as coerce

    return coerce(value)


class _ReplayBlockedTool:
    """Typing alias, resolved lazily: see ``fastaiagent.tool._replay_blocked``."""

    def __new__(cls, spec: dict[str, Any], *, reason: str) -> Any:
        from fastaiagent.tool._replay_blocked import ReplayBlockedTool

        return ReplayBlockedTool(spec, reason=reason)

    name: str
    calls: int

# Attribute keys whose values represent the *input* side of a span (the
# question that was asked / the prompt / the tool args). Used by
# ``ReplayStep`` to partition a span's attributes into the input panel
# the UI's diff view reads from. Mirrors INPUT_KEYS in
# ``ui-frontend/src/components/traces/SpanInspector.tsx`` — keep in sync
# when adding new attribute conventions.
_INPUT_ATTR_KEYS: frozenset[str] = frozenset(
    {
        "gen_ai.request.messages",
        "gen_ai.request.prompt",
        "agent.input",
        "chain.input",
        "swarm.input",
        "supervisor.input",
        "tool.input",
        "tool.args",
        "retrieval.query",
        "input",
    }
)

# Attribute keys whose values represent the *output* side of a span (the
# response / the tool result / the agent's reply). Mirrors OUTPUT_KEYS in
# SpanInspector.tsx.
_OUTPUT_ATTR_KEYS: frozenset[str] = frozenset(
    {
        "gen_ai.response.content",
        "gen_ai.response.tool_calls",
        "agent.output",
        "chain.output",
        "swarm.output",
        "supervisor.output",
        "tool.output",
        "tool.result",
        "retrieval.doc_ids",
        "retrieval.result_count",
        "output",
    }
)


def _partition_span_io(attrs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split a span's flat ``attributes`` dict into ``(input, output)``.

    The router's UI diff view (``ReplayDiffView.tsx``) compares each step's
    ``input`` and ``output`` fields side by side; before v1.14.1 those
    fields were always empty because ``ReplayStep`` never populated them.
    This helper centralizes the partition so the Python ``ReplayStep`` and
    the TypeScript ``SpanInspector`` agree on what belongs where.

    JSON-string values are pretty-decoded so the UI's diff renders the
    structured payload, not an escaped string.
    """
    input_part: dict[str, Any] = {}
    output_part: dict[str, Any] = {}
    for key, raw_value in attrs.items():
        value = _maybe_parse_json(raw_value)
        if key in _INPUT_ATTR_KEYS:
            input_part[key] = value
        elif key in _OUTPUT_ATTR_KEYS:
            output_part[key] = value
    return input_part, output_part


def _maybe_parse_json(value: Any) -> Any:
    """Decode JSON-string payloads so the UI sees structured data.

    Several capture sites JSON-stringify list/dict attributes before
    setting them on a span (OTel attributes can't hold nested
    structures). Parse those back into native types so the diff view
    can render keys, not escaped quotes.
    """
    if not isinstance(value, str):
        return value
    trimmed = value.strip()
    if not (trimmed.startswith("{") or trimmed.startswith("[")):
        return value
    try:
        return json.loads(trimmed)
    except json.JSONDecodeError:
        return value


def _recorded_decision_from_span(span: SpanData) -> Any | None:
    """Rebuild a ``DecisionResult`` from a captured Decisions API span.

    Those spans are named ``llm.<provider>.decisions.<model>`` and carry
    ``gen_ai.operation.name="decide"``; the attribute is what identifies them.
    ``None`` for any other span, or one whose answers are missing/unreadable (a
    trace from before 1.84.0 has none) — the rerun then treats that decision as
    a miss rather than inventing answers.
    """
    from fastaiagent.llm.client import is_decision_span
    from fastaiagent.llm.decisions import _ANSWERS_ADAPTER, DecisionResult

    attrs = span.attributes or {}
    if not is_decision_span(span.name, attrs):
        return None
    raw = attrs.get("fastaiagent.decision.answers")
    if raw is None:
        return None
    try:
        answers = _ANSWERS_ADAPTER.validate_python(json.loads(raw) if isinstance(raw, str) else raw)
    except Exception:
        _log.warning("replay: unreadable decision answers on span %s", span.span_id)
        return None
    usage: dict[str, Any] = {}
    for key, wire in (("gen_ai.usage.input_tokens", "input_tokens"), ("gen_ai.usage.output_tokens", "output_tokens")):
        if attrs.get(key) is not None:
            usage[wire] = attrs[key]
    return DecisionResult(
        answers=answers,
        model=str(attrs.get("gen_ai.request.model") or ""),
        usage=usage,
    )


def _recorded_response_from_span(span: SpanData) -> Any | None:
    """Reconstruct an :class:`LLMResponse` from a captured ``llm.*`` span.

    Looks at the GenAI semantic-convention attributes
    (``gen_ai.response.content``, ``gen_ai.response.finish_reason``,
    ``gen_ai.response.tool_calls``) and rebuilds an ``LLMResponse``
    instance that ``LLMClient.acomplete`` can return verbatim under
    ``determinism="recorded"``.

    Returns ``None`` when the span carries neither response content nor tool
    calls — this span isn't an LLM call. Callers must check.

    A turn that only called tools has no content: OpenAI sends ``content:
    null`` there, so ``gen_ai.response.content`` is never written. Until 1.84.0
    such a span returned ``None`` and was skipped, so a recorded rerun of any
    tool-using agent jumped straight to the captured final answer, never ran a
    tool, and still matched the original output.
    """
    from fastaiagent.llm.client import LLMResponse
    from fastaiagent.llm.message import ToolCall

    attrs = span.attributes or {}
    content = attrs.get("gen_ai.response.content")
    if content is None and not attrs.get("gen_ai.response.tool_calls"):
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

    if content is None and not tool_calls:
        # Tool calls were recorded but are unreadable, and there is no text:
        # nothing to replay for this turn.
        return None

    return LLMResponse(
        content=None if content is None else content if isinstance(content, str) else str(content),
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
    parent_span_id: str | None = None
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
    # 1.89.0. All defaulted, so existing constructors keep working.
    tool_dispositions: dict[str, str] = Field(default_factory=dict)
    """``{tool_name: override | stub | live | blocked | no_function}``."""
    blocked_tool_calls: dict[str, int] = Field(default_factory=dict)
    """Calls the rerun's model made to a blocked tool, by name. Counted on the
    blocked tool itself, so it is right with tracing off too."""
    source_trace_id: str | None = None
    determinism: str = "live"
    exported: bool = False
    """Whether the rerun's spans may leave the machine (off by default)."""
    trace_db_path: str | None = None
    """Where the rerun's trace was written, when not the default store."""

    def save_as_test(
        self,
        dataset_path: str | Path,
        input: str,
        expected_output: str,
        source_trace_id: str | None = None,
        *,
        fork_step: int | None = None,
        modifications: dict[str, Any] | None = None,
    ) -> Path:
        """Append this rerun as a regression-test case to a JSONL dataset.

        The written record uses the same field names ``evaluate()`` reads
        (``input``, ``expected_output``), so the dataset is immediately
        consumable as a regression suite. v1.14.1 added provenance fields
        (``source_trace_id``, ``fixed_trace_id``, ``fork_step``,
        ``modifications``) so the audit trail back to the production
        failure and the fix that was applied is recoverable::

            replay = Replay.load(failed.trace_id)
            forked = replay.fork_at(step=3).modify_prompt("Be specific.")
            rerun = forked.rerun()
            rerun.save_as_test(
                "regression_tests.jsonl",
                input="What is our refund policy?",
                expected_output=rerun.new_output,
                source_trace_id=failed.trace_id,
                fork_step=3,
                modifications={"prompt": "Be specific."},
            )

        Args:
            dataset_path: JSONL file to append to. Parent dirs are created.
            input: The agent input that should be replayed in future eval runs.
            expected_output: The known-good output to compare against.
            source_trace_id: The *original failure's* trace id. Kept in
                its own field (no longer overwriting ``trace_id``) so
                the link back to the bug doesn't bury the rerun's id.
            fork_step: The step in the source trace where the fork
                happened. Useful when a future debugger wants to
                re-fork to the same point.
            modifications: Provenance for the fix that was applied —
                e.g. ``{"prompt": "Be specific.", "tool_overrides": ["search"]}``.
                Free-form dict; intended for human inspection.

        Returns:
            The path written to.

        Schema (v1.14.1, JSONL line):
            ``input``, ``expected_output``, ``trace_id`` (the rerun's id),
            ``source_trace_id``, ``fixed_trace_id`` (alias of trace_id for
            clarity), ``fork_step``, ``modifications``, ``created_at``.

        Backwards-compat: existing v1.14.0 readers that only consume
        ``input`` + ``expected_output`` keep working. ``trace_id`` retains
        its v1.13/v1.14.0 meaning of "the rerun's id" (which is also what
        ``fixed_trace_id`` carries) so any code that grew to read it
        keeps seeing the same value.
        """
        path = Path(dataset_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fixed_trace_id = self.trace_id
        record: dict[str, Any] = {
            "input": input,
            "expected_output": expected_output,
            "trace_id": fixed_trace_id,
            "source_trace_id": source_trace_id,
            "fixed_trace_id": fixed_trace_id,
            "fork_step": fork_step,
            "modifications": modifications or {},
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        with path.open("a") as f:
            f.write(json.dumps(record) + "\n")
        return path


class ToolDiff(BaseModel):
    """One aligned tool call whose outcome changed between the runs (1.89.0)."""

    original_step: int | None = None
    new_step: int | None = None
    name: str = ""
    original_status: str | None = None
    new_status: str | None = None
    original_result: Any = None
    new_result: Any = None
    disposition: str | None = None
    """What the rerun did with this tool: ``blocked``, ``stub``, ``live``…"""


def _tool_result(step: ReplayStep) -> Any:
    attrs = step.attributes or {}
    for key in ("tool.result", "tool.output", "tool.error"):
        if key in attrs:
            return attrs[key]
    return None


def _tool_status(step: ReplayStep) -> str | None:
    attrs = step.attributes or {}
    status = attrs.get("tool.status")
    if status is None and "tool.error" in attrs:
        status = "error"
    return str(status) if status is not None else None


def _turn_indices(steps: list[ReplayStep]) -> list[int]:
    """The steps a comparison aligns: the root and its direct children that
    are not ``memory.*``. Memory subtrees (and the LLM calls under them), and
    anything nested under a tool, are left out."""
    root = next((s for s in steps if not s.parent_span_id), None)
    if root is None:
        return list(range(len(steps)))
    out: list[int] = []
    for i, s in enumerate(steps):
        if s.span_id == root.span_id:
            out.append(i)
        elif s.parent_span_id == root.span_id and not s.span_name.startswith("memory."):
            out.append(i)
    return out


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
    """An index into ``original_steps``. Since 1.89.0 only the run's own turns
    are compared (the root and its direct children, minus ``memory.*``): a
    memory agent's rerun has no memory, and its missing ``memory.*`` spans used
    to shift every later index, so identical recorded reruns "diverged"."""
    compare_status: Literal["ok", "rerun_failed"] = "ok"
    # 1.89.0, all additive.
    new_diverged_at: int | None = None
    """The same divergence, as an index into ``new_steps``."""
    skipped_steps: list[int] = Field(default_factory=list)
    """``original_steps`` left out of the alignment (memory, nested spans)."""
    new_skipped_steps: list[int] = Field(default_factory=list)
    aligned: list[tuple[int | None, int | None]] = Field(default_factory=list)
    """``(original_index, new_index)`` pairs, in order; ``None`` where one side
    ran fewer turns."""
    tool_diffs: list[ToolDiff] = Field(default_factory=list)
    """Aligned tool calls whose status or result differ. Kept apart from
    ``diverged_at`` so a recorded rerun with a blocked tool still reports the
    model's answers as identical."""
    tools_diverged_at: int | None = None
    """Index into ``original_steps`` of the first tool call that differs."""


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
        self._stub_names: set[str] = set()
        self._determinism: DeterminismMode = "live"
        self._on_miss: str = "live"
        # 1.89.0: a rerun runs only the tools you supply or allow.
        self._tool_policy: str = _default_tool_policy()
        self._allow: set[str] = set()
        self._export: bool | None = None
        self._trace_db: str | None = None
        self._surface: str = "python"
        self._source_store: TraceStore | None = None

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
        # Honest surface: this was never wired into the agent rerun path, and
        # wiring it would grow trace/replay.py into a state-counterfactual
        # engine — which is deliberately the Enterprise plane's job, not the
        # SDK's. For checkpoint-state forking, use the SDK primitive
        # ``Chain.afork(..., modified_state=...)`` / ``Agent.afork(...)``.
        raise NotImplementedError(
            "ForkedReplay.modify_state() is not supported. trace/replay.py is a "
            "read-only inspect/diff surface; trace-based state counterfactuals "
            "live in the Enterprise Replay engine. To fork a run from a saved "
            "step with modified state, use Chain.afork(execution_id, "
            "modified_state=...) or Agent.afork(execution_id, input=...)."
        )

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

        ``tool`` is a :class:`~fastaiagent.tool.Tool` (used as-is) or, since
        1.89.0, a plain callable, wrapped in a ``FunctionTool`` that is **not**
        registered globally. A ``FunctionTool`` you built with the default
        ``register=True`` *is* in the global registry, where it replaces the
        real tool for every later ``Agent.from_dict``; that case logs a warning.
        """
        if not name:
            raise ReplayError("with_tool_override requires a non-empty tool name")
        from fastaiagent.tool.base import Tool
        from fastaiagent.tool.function import FunctionTool
        from fastaiagent.tool.registry import ToolRegistry

        if not isinstance(tool, Tool) and callable(tool):
            tool = FunctionTool(name=name, fn=tool, register=False)
        elif isinstance(tool, FunctionTool) and ToolRegistry.get(name) is tool:
            _log.warning(
                "with_tool_override(%r): this FunctionTool is registered globally, so it "
                "replaces the real %r tool for every later Agent.from_dict and replay in "
                "this process. Build it with FunctionTool(..., register=False), pass a "
                "plain callable, or use with_tool_stub(%r, returns=...).",
                name,
                name,
                name,
            )
        self._tool_overrides[name] = tool
        self._stub_names.discard(name)
        return self

    def with_tool_stub(self, name: str, returns: Any) -> ForkedReplay:
        """Replace a tool with a stub that returns ``returns`` (1.89.0).

        The stub is never registered globally, so it affects this rerun only.
        """
        from fastaiagent.tool.function import FunctionTool

        if not name:
            raise ReplayError("with_tool_stub requires a non-empty tool name")

        def _stub(**_kwargs: Any) -> Any:
            return returns

        _stub.__name__ = f"stubbed_{name}"
        _stub.__doc__ = f"Stub for tool {name!r}; returns a canned response."
        self._tool_overrides[name] = FunctionTool(name=name, fn=_stub, register=False)
        self._stub_names.add(name)
        return self

    def with_tool_policy(
        self, policy: Literal["block", "error", "live"] = "block", *, allow: Any = ()
    ) -> ForkedReplay:
        """Choose what the rerun does with tools you did not supply (1.89.0).

        * ``"block"`` *(default)*: they are not run. The model gets an error
          result naming the tool and how to allow it, and the call is counted
          in ``ReplayResult.blocked_tool_calls``.
        * ``"error"``: the same, and the rerun raises :class:`ReplayError`
          (before anything runs, in ``recorded`` mode, when the recorded calls
          name a tool you did not supply).
        * ``"live"``: a function tool runs the implementation registered in
          this process, as reruns did before 1.89.0.

        ``allow`` names tools that run live under any policy. REST and MCP
        tools rebuilt from the trace run **only** when named here, and never
        when their credentials were redacted from the trace.
        """
        if policy not in _TOOL_POLICIES:
            raise ReplayError(f"Unknown tool policy {policy!r}; expected one of {_TOOL_POLICIES}")
        self._tool_policy = policy
        self._allow = {str(a) for a in allow}
        return self

    def with_export(self, export: bool = True) -> ForkedReplay:
        """Export the rerun's trace like a real run (1.89.0; off by default).

        The spans keep ``fastaiagent.source=replay``. Auto-registration stays
        suppressed either way: a rerun never pushes its agent definition.
        """
        self._export = bool(export)
        return self

    def with_trace_db(self, path: str | Path) -> ForkedReplay:
        """Write the rerun's trace to another local store (1.89.0)."""
        self._trace_db = str(path)
        return self

    def with_determinism(self, mode: DeterminismMode, *, on_miss: str = "live") -> ForkedReplay:
        """Control how the LLM is invoked during rerun.

        * ``"live"`` *(default)*: call the LLM provider with whatever
          temperature/seed the captured config carried. Output is
          nondeterministic for non-zero temperature.
        * ``"recorded"``: skip the LLM HTTP call entirely; return the
          captured response from the original trace's GenAI attributes.
          Use for byte-identical regression tests.
        * ``"deterministic"``: call the LLM live, but force
          ``temperature=0`` (and ``seed=42`` where the provider supports
          it). Reduces nondeterminism without skipping the call. A model that
          takes only its default temperature — GPT-5, GPT-5.5+, GPT-6, Claude
          5.x, or a model with reasoning on that locks it — keeps its own,
          with a warning (1.91.0): forcing 0 there was a 400 on every call.

        ``on_miss`` (``"recorded"`` mode only) controls what happens when the
        rerun makes MORE LLM calls than the original trace captured:

        * ``"live"`` *(default)*: fall through to a real provider call —
          billed and nondeterministic — with a prominent warning logged.
        * ``"error"``: raise :class:`ReplayError` instead. Recommended for
          regression tests, where a silent live call would mask drift.

        See ``docs/replay/guarantees.md`` for per-provider support and
        known limitations (e.g. streaming chunks aren't recorded
        granularly — ``recorded`` mode reconstructs from final text).
        """
        if mode not in ("live", "recorded", "deterministic"):
            raise ReplayError(
                f"Unknown determinism mode {mode!r}; "
                "expected 'live', 'recorded', or 'deterministic'"
            )
        if on_miss not in ("live", "error"):
            raise ReplayError(f"Unknown on_miss {on_miss!r}; expected 'live' or 'error'")
        self._determinism = mode
        self._on_miss = on_miss
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

        Since 1.89.0 the rerun runs inside a ``replay`` provenance scope: its
        spans carry ``fastaiagent.source=replay`` and are not exported by
        default, the rebuilt agent is never registered with the plane, and
        tools follow the tool plan (``with_tool_policy``; default ``block``).
        """
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

        from fastaiagent._internal.scope import scoped_tools
        from fastaiagent.trace.provenance import experiments_exported, provenance_scope

        # The tool plan (1.89.0): a rerun runs only tools you supplied or
        # allowed. It used to rebuild every function tool from the global
        # registry (the live implementation, side effects and all) and every
        # REST/MCP tool from the trace, so a rerun charged the card again.
        tools, dispositions, blocked = self._build_tool_plan(agent_dict.get("tools") or [])
        if self._tool_policy == "error" and self._determinism == "recorded":
            self._preflight_recorded_tools(dispositions)

        scope_attrs = {
            "fastaiagent.replay.source_trace_id": self._trace.trace_id,
            "fastaiagent.replay.fork_step": self._fork_point,
            "fastaiagent.replay.determinism": self._determinism,
            "fastaiagent.replay.on_miss": self._on_miss,
            "fastaiagent.replay.tool_policy": self._tool_policy,
            "fastaiagent.replay.modifications": self._modification_kinds(),
            "fastaiagent.replay.tool_overrides": sorted(self._tool_overrides),
            "fastaiagent.replay.surface": self._surface,
        }
        # Tools built during the rerun register into a private overlay, never
        # into the process-global registry.
        tools_token = scoped_tools.set(dict(scoped_tools.get() or {}))
        try:
            with provenance_scope(
                "replay",
                attributes=scope_attrs,
                export=self._export,
                trace_db=self._trace_db,
                tool_dispositions=dispositions,
                suppress_registration=True,
            ) as prov:
                new_result = await self._arun_in_scope(agent_dict, tools, new_input)
                exported = experiments_exported(prov)
        finally:
            scoped_tools.reset(tools_token)

        blocked_calls = {name: t.calls for name, t in blocked.items() if t.calls}
        if blocked_calls:
            _log.warning(
                "Replay of trace %s blocked %s; the rerun's answer does not show their "
                "effect. Supply them with with_tool_override / with_tool_stub, or allow "
                "them with with_tool_policy('live', allow=[...]).",
                self._trace.trace_id,
                ", ".join(f"{n} (x{c})" for n, c in sorted(blocked_calls.items())),
            )
            if self._tool_policy == "error":
                raise ReplayError(
                    f"Replay of trace {self._trace.trace_id} called tools it was not given: "
                    f"{sorted(blocked_calls)} (tool policy 'error')."
                )

        root_output = root.attributes.get("agent.output")
        return ReplayResult(
            original_output=root_output,
            new_output=new_result.output,
            steps_executed=len(self._steps) - self._fork_point,
            trace_id=new_result.trace_id,
            tool_dispositions=dispositions,
            blocked_tool_calls=blocked_calls,
            source_trace_id=self._trace.trace_id,
            determinism=self._determinism,
            exported=exported,
            trace_db_path=self._trace_db,
        )

    async def _arun_in_scope(
        self, agent_dict: dict[str, Any], tools: list[Any], new_input: Any
    ) -> Any:
        """Rebuild the agent (without its tools) and run it, honouring determinism."""
        from fastaiagent.agent.agent import Agent
        from fastaiagent.llm.client import (
            _replay_on_miss,
            _replay_recorded_decisions,
            _replay_recorded_response,
        )

        try:
            agent = Agent.from_dict({**agent_dict, "tools": []})
        except Exception as e:
            raise ReplayError(
                f"Failed to reconstruct agent from trace {self._trace.trace_id}: {e}"
            ) from e
        agent.tools = tools

        # Determinism mode plumbing — see ForkedReplay.with_determinism.
        if self._determinism == "deterministic":
            # Force temperature=0 and seed=42 on the LLM client where
            # the provider supports them. Bedrock has no seed concept,
            # which we accept as a documented gap.
            if hasattr(agent, "llm") and agent.llm is not None:
                from fastaiagent.llm.param_rules import pins_temperature

                llm = agent.llm
                try:
                    llm.seed = 42
                    if pins_temperature(
                        getattr(llm, "provider", ""),
                        getattr(llm, "model", ""),
                        getattr(llm, "reasoning", None),
                    ):
                        llm.temperature = 0
                    else:
                        # The model refuses temperature=0 (1.91.0, X24): forcing
                        # it made every deterministic rerun a 400.
                        llm.temperature = None
                        _log.warning(
                            "determinism='deterministic': %s/%s takes only its default "
                            "temperature, so the rerun pins seed=42 but not temperature.",
                            getattr(llm, "provider", ""),
                            getattr(llm, "model", ""),
                        )
                except AttributeError:
                    pass

        if self._determinism == "recorded":
            # v1.14.1: install the *full sequence* of captured LLM
            # responses (ordered by start_time) so multi-turn tool-loop
            # reruns replay each turn in order. The LLM client pops the
            # front entry per acomplete() call.
            recorded_queue = self._all_llm_responses()
            if not recorded_queue:
                raise ReplayError(
                    f"Trace {self._trace.trace_id} has no captured LLM response "
                    f"(gen_ai.response.content). determinism='recorded' requires "
                    f"FASTAIAGENT_TRACE_PAYLOADS=1 (the default) on the original run."
                )
            # 1.89.0: a guardrail's own model call (an llm_judge, say) is served
            # from the guardrail recordings, never from the agent's turns. See
            # fastaiagent.trace._replay_roles.
            responses: Any
            decisions: Any
            if _roles.role_tracking_available():
                responses = _roles.RoleQueue(
                    _roles.split_by_role(self._trace.spans, _recorded_response_from_span)
                )
                decisions = _roles.RoleQueue(
                    _roles.split_by_role(self._trace.spans, _recorded_decision_from_span)
                )
            else:
                _log.warning(
                    "Replay of trace %s: tracing is off, so a guardrail's model call cannot "
                    "be told from the agent's. Recorded responses are served from one shared "
                    "queue, and a guardrail that is not in the original trace takes an "
                    "agent turn.",
                    self._trace.trace_id,
                )
                responses = list(recorded_queue)
                decisions = self._all_decisions()
            token = _replay_recorded_response.set(responses)
            # Decisions API answers replay from their own queue (installed even
            # when empty, so a rerun that asks a decision the original never did
            # is a miss governed by ``on_miss`` — not a silent live call).
            decisions_token = _replay_recorded_decisions.set(decisions)
            miss_token = _replay_on_miss.set(self._on_miss)
            roles_token = _roles.start_tracking()
            try:
                return await self._arun_unpaused(agent, new_input)
            finally:
                _roles.stop_tracking(roles_token)
                _replay_on_miss.reset(miss_token)
                _replay_recorded_decisions.reset(decisions_token)
                _replay_recorded_response.reset(token)
        return await self._arun_unpaused(agent, new_input)

    # ── The tool plan (1.89.0) ─────────────────────────────────────────────

    def _build_tool_plan(
        self, specs: list[dict[str, Any]]
    ) -> tuple[list[Any], dict[str, str], dict[str, _ReplayBlockedTool]]:
        """``(tools, {name: disposition}, {name: blocked_tool})`` for the rerun.

        Priority: ``with_tools`` (full replacement), then a per-name override or
        stub, then ``live`` (the policy, or the name in ``allow``), else blocked.
        """
        from fastaiagent.tool.base import Tool
        from fastaiagent.tool.function import FunctionTool
        from fastaiagent.tool.registry import ToolRegistry

        if "tools" in self._modifications:
            replacement = list(self._modifications["tools"])
            return replacement, {getattr(t, "name", "?"): "override" for t in replacement}, {}

        tools: list[Any] = []
        dispositions: dict[str, str] = {}
        blocked: dict[str, _ReplayBlockedTool] = {}

        def block(spec: dict[str, Any], reason: str) -> None:
            stand_in = _ReplayBlockedTool(spec, reason=reason)
            tools.append(stand_in)
            blocked[stand_in.name] = stand_in
            dispositions[stand_in.name] = "blocked"

        for spec in specs:
            if not isinstance(spec, dict) or not spec.get("name"):
                continue
            name = str(spec["name"])
            if name in self._tool_overrides:
                tools.append(self._tool_overrides[name])
                dispositions[name] = "stub" if name in self._stub_names else "override"
                continue
            tool_type = spec.get("tool_type", "function")
            allowed = name in self._allow
            if tool_type in ("rest_api", "mcp"):
                # An endpoint rebuilt from the trace is not something this
                # process has: never called unless named in ``allow``.
                if not allowed:
                    block(spec, "its endpoint was rebuilt from the trace")
                elif tool_type == "rest_api" and _has_redacted_header(spec):
                    block(
                        spec,
                        "its credentials were redacted from the trace; pass the live "
                        "tool with with_tool_override",
                    )
                else:
                    tools.append(Tool.from_dict(spec))
                    dispositions[name] = "live"
                continue
            if not (allowed or self._tool_policy == "live"):
                block(spec, "a rerun runs only tools you supply")
                continue
            registered = ToolRegistry.get(name)
            if isinstance(registered, FunctionTool) and registered.fn is not None:
                tools.append(registered)
                dispositions[name] = "live"
            else:
                tools.append(
                    FunctionTool(
                        name=name,
                        description=spec.get("description", ""),
                        parameters=spec.get("parameters"),
                        replay_class=_coerce_replay_class(spec.get("replay_class")),
                        register=False,
                    )
                )
                dispositions[name] = "no_function"

        # Overrides for tools the original run did not have are appended.
        for name, tool in self._tool_overrides.items():
            if name not in dispositions:
                tools.append(tool)
                dispositions[name] = "stub" if name in self._stub_names else "override"
        return tools, dispositions, blocked

    def _preflight_recorded_tools(self, dispositions: dict[str, str]) -> None:
        """Policy ``error`` in recorded mode: refuse before anything runs when the
        recorded model turns call a tool the rerun was not given."""
        called: set[str] = set()
        for response in self._all_llm_responses():
            for call in getattr(response, "tool_calls", None) or []:
                name = getattr(call, "name", None) or (
                    call.get("name") if isinstance(call, dict) else None
                )
                if name:
                    called.add(str(name))
        missing = sorted(
            n for n in called if dispositions.get(n) not in ("override", "stub", "live")
        )
        if missing:
            raise ReplayError(
                f"Replay of trace {self._trace.trace_id}: the recorded run calls {missing}, "
                "which this rerun was not given (tool policy 'error'). Supply them with "
                "with_tool_override / with_tool_stub, or allow them with "
                "with_tool_policy('live', allow=[...])."
            )

    def _modification_kinds(self) -> list[str]:
        """What was modified, as kinds only: never the modified text itself."""
        kinds = [k for k in ("prompt", "input", "config", "tools") if k in self._modifications]
        overrides = set(self._tool_overrides) - self._stub_names
        if overrides:
            kinds.append("tool_overrides")
        if self._stub_names:
            kinds.append("tool_stubs")
        return kinds

    async def _arun_unpaused(self, agent: Any, new_input: Any) -> Any:
        """``agent.arun`` for a rerun, refusing a pause with a ``ReplayError``.

        The rebuilt agent has no checkpointer (and no ``agent_id``, so managed
        governance never gates a replay), so a tool that calls ``interrupt()``
        raises the bare ``InterruptSignal``. Before 1.78.0 that escaped
        ``arerun()``. A replay cannot hold a pause, so it says which tool
        stopped it and how to replay past it.
        """
        from fastaiagent._internal.pause import what_paused
        from fastaiagent.chain.interrupt import InterruptSignal

        try:
            result = await agent.arun(new_input)
        except InterruptSignal as sig:
            paused = what_paused(sig)
        else:
            paused = what_paused(result)
            if not paused:
                return result
        raise ReplayError(
            f"Replay of trace {self._trace.trace_id} paused: {paused}. A replay rebuilds "
            f"the agent without a checkpointer, so it cannot hold a pause. Override the "
            f"tool that paused with with_tool_override(name, tool) to replay past it."
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

    def _all_llm_responses(self) -> list[Any]:
        """Return every :class:`LLMResponse` recoverable from the trace,
        in capture order (by ``start_time``). Walks every span — not just
        those named ``llm.*`` — so traces captured under older span
        naming conventions still work for ``determinism="recorded"``.

        v1.14.1: replaced the v1.14.0 ``_first_llm_response`` which
        returned only the *first* response. Multi-turn tool-loop traces
        replay each turn in the order they were captured; ``acomplete``
        pops the front entry per call.
        """
        ordered = sorted(self._trace.spans, key=lambda s: s.start_time or "")
        out: list[Any] = []
        for span in ordered:
            recovered = _recorded_response_from_span(span)
            if recovered is not None:
                out.append(recovered)
        return out

    def _all_decisions(self) -> list[Any]:
        """Every Decisions API result recoverable from the trace, in capture order."""
        ordered = sorted(self._trace.spans, key=lambda s: s.start_time or "")
        out: list[Any] = []
        for span in ordered:
            recovered = _recorded_decision_from_span(span)
            if recovered is not None:
                out.append(recovered)
        return out

    def _first_llm_response(self) -> Any | None:
        """Deprecated v1.14.0 alias. Kept so anyone monkey-patching this
        for tests keeps working; new code should use
        :py:meth:`_all_llm_responses`.
        """
        responses = self._all_llm_responses()
        return responses[0] if responses else None

    def rerun(self) -> ReplayResult:
        from fastaiagent._internal.async_utils import run_sync

        return run_sync(self.arerun())

    def compare(
        self, new_result: ReplayResult, *, store: TraceStore | None = None
    ) -> ComparisonResult:
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

        The rerun is read from ``store``, else the store it was written to
        (``with_trace_db``), else the default store.
        """
        new_steps: list[ReplayStep] = []
        status: Literal["ok", "rerun_failed"] = "ok"
        if store is None:
            sink = new_result.trace_db_path or self._trace_db
            store = TraceStore(db_path=sink) if sink else TraceStore.default()
        if new_result.trace_id:
            try:
                new_replay = Replay.load(new_result.trace_id, store=store)
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

        if status == "rerun_failed":
            return ComparisonResult(
                original_steps=self._steps, new_steps=new_steps, compare_status=status
            )

        orig_idx = _turn_indices(self._steps)
        new_idx = _turn_indices(new_steps)
        orig_turns = [self._steps[i] for i in orig_idx]
        new_turns = [new_steps[i] for i in new_idx]
        aligned: list[tuple[int | None, int | None]] = [
            (
                orig_idx[k] if k < len(orig_idx) else None,
                new_idx[k] if k < len(new_idx) else None,
            )
            for k in range(max(len(orig_idx), len(new_idx)))
        ]
        at = _first_divergence(orig_turns, new_turns)

        def _map(k: int | None, idx: list[int], total: int) -> int | None:
            if k is None:
                return None
            return idx[k] if k < len(idx) else total

        tool_diffs: list[ToolDiff] = []
        for o, n in aligned:
            a = self._steps[o] if o is not None else None
            b = new_steps[n] if n is not None else None
            either = a or b
            name = either.span_name if either is not None else ""
            if not name.startswith("tool."):
                continue
            a_status, b_status = (_tool_status(a) if a else None), (_tool_status(b) if b else None)
            a_res, b_res = (_tool_result(a) if a else None), (_tool_result(b) if b else None)
            if a_status != b_status or a_res != b_res or a is None or b is None:
                tool_diffs.append(
                    ToolDiff(
                        original_step=o,
                        new_step=n,
                        name=name[len("tool.") :],
                        original_status=a_status,
                        new_status=b_status,
                        original_result=a_res,
                        new_result=b_res,
                        disposition=(b.attributes or {}).get("fastaiagent.replay.tool_disposition")
                        if b
                        else None,
                    )
                )
        aligned_orig = {o for o, _ in aligned if o is not None}
        aligned_new = {n for _, n in aligned if n is not None}
        return ComparisonResult(
            original_steps=self._steps,
            new_steps=new_steps,
            diverged_at=_map(at, orig_idx, len(self._steps)),
            new_diverged_at=_map(at, new_idx, len(new_steps)),
            compare_status=status,
            skipped_steps=[i for i in range(len(self._steps)) if i not in aligned_orig],
            new_skipped_steps=[i for i in range(len(new_steps)) if i not in aligned_new],
            aligned=aligned,
            tool_diffs=tool_diffs,
            tools_diverged_at=tool_diffs[0].original_step if tool_diffs else None,
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

    def __init__(self, trace_data: TraceData, store: TraceStore | None = None):
        self._trace = trace_data
        self._store = store
        self._steps = self._build_steps()

    @classmethod
    def load(cls, trace_id: str, store: TraceStore | None = None) -> Replay:
        """Load a replay from a stored trace."""
        store = store or TraceStore.default()
        trace_data = store.get_trace(trace_id)
        return cls(trace_data, store=store)

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
        # Populates ``input`` / ``output`` from the span's attributes via
        # :func:`_partition_span_io` so the UI's diff view (which reads
        # ``step.input`` / ``step.output``) has real content to render.
        # Before v1.14.1 these fields were always empty dicts and the UI
        # showed two blank cards per step.
        steps = []
        for i, span in enumerate(self._trace.spans):
            attrs = span.attributes or {}
            input_part, output_part = _partition_span_io(attrs)
            steps.append(
                ReplayStep(
                    step=i,
                    span_name=span.name,
                    span_id=span.span_id,
                    parent_span_id=span.parent_span_id,
                    input=input_part,
                    output=output_part,
                    attributes=attrs,
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
        forked = ForkedReplay(
            original_trace=self._trace,
            fork_point=step,
            steps=self._steps,
        )
        forked._source_store = self._store
        return forked
