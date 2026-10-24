"""Agent class — the central component of the SDK."""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import AsyncGenerator, Callable, Coroutine, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Generic, TypeVar, overload

from pydantic import BaseModel, Field, field_validator
from typing_extensions import TypeVar as TypeVarWithDefault

from fastaiagent._internal.async_utils import run_sync
from fastaiagent._internal.errors import StopAgent
from fastaiagent._internal.pricing import (
    _RunUsage,
    bind_run_cost,
    current_run_usage,
    run_cost,
    run_tokens,
    start_run_cost,
    stop_run_cost,
    usage_tokens,
)
from fastaiagent.agent import run_control
from fastaiagent.agent.context import (
    RunContext,
    reset_active_run_context,
    set_active_run_context,
)
from fastaiagent.agent.executor import (
    _AgentInterrupted,
    _governed_tool_call,
    _stopped_response,
    _StoppedResponse,
    _stream_stop,
    execute_tool_loop,
    stream_tool_loop,
)
from fastaiagent.agent.memory import MemoryLike
from fastaiagent.agent.middleware import (
    AgentMiddleware,
    MiddlewareContext,
    _DeferredTools,
    _MiddlewarePipeline,
)
from fastaiagent.chain.checkpoint import (
    Checkpoint,
    is_run_end,
    latest_forkable,
    latest_resumable,
    write_run_end,
    write_run_end_once,
)
from fastaiagent.chain.idempotent import bind_cache, unbind_cache
from fastaiagent.chain.interrupt import (
    AlreadyResumed,
    InterruptSignal,
    Resume,
    _agent_path,
    _execution_id,
    _resume_value,
)
from fastaiagent.checkpointers import Checkpointer, SQLiteCheckpointer
from fastaiagent.guardrail.executor import GuardrailOutcome, execute_guardrails
from fastaiagent.guardrail.guardrail import (
    Guardrail,
    GuardrailFiring,
    GuardrailPosition,
    collected_firings,
    start_firing_collection,
    stop_firing_collection,
)
from fastaiagent.llm.client import LLMClient, LLMResponse
from fastaiagent.llm.message import Message, SystemMessage, UserMessage
from fastaiagent.llm.stream import (
    PartialOutput,
    Paused,
    StreamDone,
    StreamEvent,
    TextDelta,
    ToolCallEnd,
    ToolCallStart,
    Usage,
)
from fastaiagent.llm.structured import OutputSpec, partial_json
from fastaiagent.multimodal.image import Image as MultimodalImage
from fastaiagent.multimodal.pdf import PDF as MultimodalPDF  # noqa: N811
from fastaiagent.multimodal.types import ContentPart, normalize_input
from fastaiagent.tool.base import Tool

logger = logging.getLogger(__name__)

AgentInput = str | MultimodalImage | MultimodalPDF | list[ContentPart]

#: What the model is told about a sibling tool call a resume stepped over. The
#: resume re-enters at a checkpoint, never in the middle of a turn, so the call
#: is answered rather than dispatched — its side effect never happens, and the
#: note has to say that plainly or the model will assume it did.
SIBLING_SKIPPED_NOTE = "the agent was interrupted before this tool ran; it was not run on resume"

#: The same repair on the fork path, said differently on purpose. A resume is a
#: recovery — the run was going to make this call and was interrupted. A fork is
#: a deliberate branch someone asked for, and the branch simply starts after the
#: call. Telling the model it was "interrupted" on a branch nobody interrupted
#: would be a small lie in the one place the model reasons from.
FORK_SKIPPED_NOTE = "this tool was not run: the conversation was branched before its result"

#: Bridge from :meth:`Agent.astream` back to the synchronous :meth:`Agent.stream`.
#:
#: ``astream`` is a generator: it yields events, never an :class:`AgentResult`,
#: so the run's identity (trace id, execution id, token count) had no way back
#: to the caller that collects those events — which is why a streamed run
#: returned ``trace_id=None``, ``tokens_used=0`` and ``execution_id=""`` right
#: up to 1.67.0. An async generator's body runs in the context of whoever drives
#: it, so a dict published here by ``stream()`` is filled in by ``astream``
#: while it iterates. Same property the run-scoped guardrail firing collection
#: relies on, and the reason that one is opened in ``stream()`` too.
_stream_outcome: ContextVar[dict[str, Any] | None] = ContextVar(
    "fastaiagent_stream_outcome", default=None
)


def _run_tokens_or(response: Any, retry_tokens: int) -> int:
    """This run's token total.

    The run-scoped accumulator is authoritative: ``LLMClient`` adds to it at the
    one point every completion passes through, so a three-turn tool loop, a
    structured re-ask and a guardrail re-ask are all counted once and the number
    agrees with ``cost``, which is built from the same hook. Before 1.68.0 this
    was ``response.usage["total_tokens"] + retry_tokens`` — and
    ``execute_tool_loop`` hands back only the LAST response, so every turn but
    the last went unbilled.

    The fallback covers a client that implements ``acomplete`` instead of
    extending ``LLMClient``'s internals — a user's own subclass, or a test
    double. Those never reach the accumulator, and the legacy sum is a better
    answer for them than zero.
    """
    accumulated = run_tokens()
    if accumulated:
        return accumulated
    return usage_tokens(getattr(response, "usage", None)) + retry_tokens


async def _one_event(event: StreamEvent) -> AsyncGenerator[StreamEvent, None]:
    yield event


def _end_state(mw_ctx: MiddlewareContext | None, *, strict: bool) -> dict[str, Any] | None:
    """Middleware state for a run-end marker. ``strict`` on success: state that
    cannot be saved fails the run, as at any checkpoint. Not on a failure path,
    where nothing may replace the exception being re-raised."""
    if mw_ctx is None:
        return None
    if strict:
        return mw_ctx.saved_state()
    try:
        return mw_ctx.saved_state()
    except TypeError:
        return None


class _FinalResultTool(Tool):
    """``final_result`` — how an ``output_mode="tool"`` agent answers (1.91.0,
    2G). The tool loop ends on the call and takes its arguments as the reply; it
    is never executed."""

    origin = "builtin"

    def __init__(self, schema: dict[str, Any]) -> None:
        super().__init__(
            name=FINAL_RESULT,
            description="Give your final answer. Call this once, when you have it.",
            parameters=schema,
        )

    async def aexecute(self, arguments: dict[str, Any], context: Any | None = None) -> Any:
        from fastaiagent.tool.base import ToolResult

        return ToolResult(error=f"{FINAL_RESULT} ends the run; it is not executed")


def _prefix_ids(llm_messages: list[Message], history: list[Message] | None) -> set[int]:
    """The messages ``_build_messages`` put ahead of the conversation — the system
    prompt, middleware instructions, memory context — by identity, so a
    middleware that trims or rewrites the list in place does not shift them."""
    prefix_len = max(len(llm_messages) - len(history or []) - 1, 0)
    return {id(m) for m in llm_messages[:prefix_len]}


def _conversation(
    llm_messages: list[Message], prefix_ids: set[int], final: str | None = None
) -> list[Message]:
    """``AgentResult.messages`` (1.91.0, 2J): the run's conversation, ready for
    ``messages=``. System messages never are (a resumed run's prefix is known
    only by role); the final answer is added when there is one."""
    from fastaiagent.llm.message import AssistantMessage, MessageRole

    turns = [m for m in llm_messages if id(m) not in prefix_ids and m.role != MessageRole.system]
    if final:
        turns.append(AssistantMessage(content=final))
    return turns


def _loop_end_turn(tool_calls: list[dict[str, Any]], start_iteration: int) -> int:
    """The turn index one past the tool loop's last turn.

    Used as the re-entry point for a run that died *after* the loop. Derived from
    the loop's own records rather than counted here, so it stays right whether
    the loop ended normally, on ``StopAgent``, or at ``max_iterations``.
    """
    iterations = [int(rec.get("iteration", 0)) for rec in tool_calls or []]
    return (max(iterations) + 1) if iterations else start_iteration


def _input_summary_text(parts: list[ContentPart]) -> str:
    """Concatenate the text portions of a multimodal input.

    Used wherever a string is required (memory, guardrails, span attributes)
    but the user passed a multimodal list. Image/PDF parts are replaced
    with a short marker so the trace stays readable.
    """
    pieces: list[str] = []
    for part in parts:
        if isinstance(part, str):
            pieces.append(part)
        elif isinstance(part, MultimodalImage):
            pieces.append(f"[image:{part.media_type}:{part.size_bytes()}b]")
        elif isinstance(part, MultimodalPDF):
            pieces.append(f"[pdf:{part.size_bytes()}b]")
    return " ".join(pieces)


def _resumed_input_text(messages: list[Message]) -> str:
    """The input of the run a checkpointed prompt belongs to, as memory text.

    That is the **last** user message: the prompt starts with the memory window
    and any caller history, so earlier user messages are old turns, and neither
    the tool loop nor a re-ask adds one to the saved prompt.
    """
    for m in reversed(messages):
        if m.role.value == "user" and m.content:
            if isinstance(m.content, str):
                return m.content
            return _input_summary_text(list(m.content))
    return ""


def _apply_input_rewrite(original: AgentInput, outcome: GuardrailOutcome) -> tuple[AgentInput, str]:
    """Substitute a masked/overridden input, or fail closed when we can't.

    Input guardrails judge ``_input_summary_text``, a *summary* of the
    normalized parts. For plain-text input the summary is the input, so the
    rewrite substitutes cleanly. For multimodal input it is not: swapping the
    summary in for the parts would drop the images the user sent, and passing
    the original parts through would hand the model the very text the rule
    redacted. Neither is acceptable, so the rewrite degrades to a block —
    the same rule masking applies everywhere: never pass the payload through
    untouched.
    """
    from fastaiagent._internal.errors import GuardrailBlockedError

    rewritten = str(outcome.data)
    if isinstance(original, str):
        return rewritten, rewritten

    outcome.refuse("multimodal input cannot be rewritten faithfully")
    raise GuardrailBlockedError(
        guardrail_name="input",
        message=(
            "A guardrail rewrote this input, but multimodal input cannot be "
            "rewritten faithfully (the text is a summary of the parts); blocked instead."
        ),
        results=outcome.results,
    )


def _refuse_stream_rewrite(outcome: GuardrailOutcome) -> None:
    """Block a streamed output whose guardrails wanted to rewrite or re-ask it.

    ``astream`` yields each ``TextDelta`` as it arrives, so by the time the
    output guardrails run the caller has already seen the raw text. Applying a
    mask now would record a redaction that never happened, and a re-ask would
    produce a second answer for a question the caller already has an answer to.
    """
    from fastaiagent._internal.errors import GuardrailBlockedError

    if not outcome.modified and outcome.reask is None:
        return
    what = "rewrite" if outcome.modified else "re-ask"
    outcome.refuse(f"a streamed reply cannot be {'rewritten' if outcome.modified else 're-asked'}")
    raise GuardrailBlockedError(
        guardrail_name="output",
        message=(
            f"A guardrail asked to {what} this reply, but it has already been "
            "streamed to the caller and cannot be taken back; blocked instead. "
            "Use run()/arun() if this rule must rewrite the output."
        ),
        results=outcome.results,
    )


#: ``AgentConfig`` fields that bound a run (1.91.0).
_RUN_LIMITS = ("timeout_seconds", "max_total_tokens", "max_cost_usd")
#: ``AgentConfig.output_mode`` values (1.91.0, 2G).
OUTPUT_MODES = ("auto", "native", "tool", "prompt")
#: The tool ``output_mode="tool"`` offers; calling it is the run's answer.
FINAL_RESULT = "final_result"


class AgentConfig(BaseModel):
    """Agent execution configuration."""

    max_iterations: int = Field(default=10, ge=1, le=100)
    # Sent to the model since 1.90.0; before, all three were read by nothing.
    # ``temperature`` / ``max_tokens`` go on every model call of the run, over
    # the LLMClient's own values (an explicit ``arun(..., temperature=)`` wins).
    # ``tool_choice``: "auto" sends nothing (the provider default); "required"
    # forces the FIRST turn only, so the loop can still answer; "none" applies
    # to every turn.
    tool_choice: str = "auto"  # "auto", "required", "none"
    temperature: float | None = None
    max_tokens: int | None = None
    # Opt-in concurrent execution of the tool calls the LLM emits within a
    # single turn. Default off preserves the sequential behavior exactly. The
    # parallel path is used only when no checkpointer, middleware, or managed
    # governance is engaged (those stay sequential for correct ordering / HITL);
    # otherwise it transparently falls back to sequential.
    parallel_tools: bool = False
    max_parallel_tools: int = Field(default=4, ge=1, le=64)
    # Structured output (``output_type``): on a parse/validation failure, re-ask
    # the model with the error up to this many times. 0 disables (previous
    # behavior — a failure yields ``parsed=None``). Only fires on failure, so it
    # adds no calls on the happy path.
    output_retries: int = Field(default=2, ge=0, le=5)
    # Output guardrails carrying ``action="reask"``: re-prompt the model with the
    # failure as feedback up to this many times before blocking. Deliberately
    # lower than ``output_retries`` — a guardrail re-ask is a full extra turn on
    # a reply that already cost one, and the failure it is correcting is a policy
    # judgement rather than a parse error. 0 makes a reask rule block outright.
    guardrail_retries: int = Field(default=1, ge=0, le=5)
    # Use OpenAI/Azure native *strict* Structured Outputs (hard schema guarantee)
    # for ``output_type``. Off by default; ignored by non-OpenAI providers.
    strict_output: bool = False
    # Run limits (1.91.0, 2I). ``timeout_seconds`` is the run's wall clock: past
    # it the run raises ``AgentTimeoutError``. ``max_total_tokens`` /
    # ``max_cost_usd`` are checked before every model call on the run's running
    # total and end it cooperatively, with ``stop_reason`` naming the cap. In
    # ``to_dict()`` only when set.
    timeout_seconds: float | None = Field(default=None, gt=0)
    max_total_tokens: int | None = Field(default=None, gt=0)
    max_cost_usd: float | None = Field(default=None, gt=0)
    # How structured output (``output_type``) is asked for (1.91.0, 2G):
    # "auto" — each provider's usual way; "native" — the provider's own
    # schema-constrained output (Anthropic ``output_config.format``, Bedrock
    # ``outputConfig.textFormat``); "tool" — a ``final_result`` tool the model
    # calls to answer; "prompt" — the schema in the prompt only. In
    # ``to_dict()`` only when not "auto".
    output_mode: str = "auto"

    def _to_dict(self) -> dict[str, Any]:
        """The config as ``Agent.to_dict()`` carries it: the run limits only when
        set, so a default agent's payload is what it was before 1.91.0."""
        data = self.model_dump()
        for key in _RUN_LIMITS:
            if data.get(key) is None:
                data.pop(key, None)
        if data.get("output_mode") == "auto":
            data.pop("output_mode")
        return data

    @field_validator("output_mode")
    @classmethod
    def _known_output_mode(cls, value: str) -> str:
        if value not in OUTPUT_MODES:
            raise ValueError(f"output_mode must be one of {OUTPUT_MODES}, got {value!r}")
        return value

    @field_validator("tool_choice")
    @classmethod
    def _known_tool_choice(cls, value: str) -> str:
        # Silently ignoring a value was how all three sampling fields went
        # unnoticed for so long; a value no provider understands is refused.
        from fastaiagent.llm.client import TOOL_CHOICES

        if value not in TOOL_CHOICES:
            raise ValueError(f"tool_choice must be one of {TOOL_CHOICES}, got {value!r}")
        return value

    def _llm_kwargs(self) -> dict[str, Any]:
        """The per-call LLM options this config sets (only the ones set)."""
        out: dict[str, Any] = {}
        if self.temperature is not None:
            out["temperature"] = self.temperature
        if self.max_tokens is not None:
            out["max_tokens"] = self.max_tokens
        return out


#: ``AgentResult``'s ``parsed`` type. Its default (PEP 696) makes a bare
#: ``AgentResult`` mean ``AgentResult[Any]`` to type checkers too, so annotations
#: written before 1.91.0 stay valid. ``typing_extensions`` comes with pydantic.
OutputT = TypeVarWithDefault("OutputT", default=Any)
#: The ``output_type`` of one ``run`` / ``arun`` call, for its typed overload.
ResultT = TypeVar("ResultT")


class AgentResult(BaseModel, Generic[OutputT]):
    """Result of an agent execution.

    ``status`` is ``"completed"`` for a normal run or ``"paused"`` when a
    tool called :func:`interrupt`. In the paused case ``pending_interrupt``
    holds ``{reason, context, node_id, agent_path}`` — the same payload the
    ``/approvals`` UI reads from the ``pending_interrupts`` table.
    ``execution_id`` is always populated when a checkpointer is configured.

    ``guardrails`` lists every guardrail that executed during the run, in order,
    across all four positions. It is the only way to observe a non-halting
    outcome — ``warn``, ``mask`` and ``override`` all let the run finish, so
    before 1.64.0 a run with the Local UI off and no plane attached reported a
    clean string whether or not a rule had fired::

        result = agent.run("my ssn is 123-45-6789")
        for g in result.guardrails:
            if g.fired():
                print(g.name, g.position, g.action_taken)

    A blocking failure still raises ``GuardrailBlockedError`` rather than
    returning; the list is for the outcomes that do not stop the run.
    """

    output: str = ""
    #: ``output`` parsed into the agent's ``output_type`` (1.91.0: typed —
    #: ``AgentResult[Invoice]`` for ``output_type=Invoice``; plain
    #: ``AgentResult`` is ``AgentResult[Any]``).
    parsed: OutputT | None = None
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    #: The run's conversation (1.91.0, 2J): prior ``messages=``, the input, the
    #: tool turns and the answer — not the system prompt, middleware
    #: instructions or memory context. Pass it back as ``messages=`` to continue.
    messages: list[Message] = Field(default_factory=list)
    tokens_used: int = 0
    #: Estimated USD spent on this run, summed over every LLM call it made —
    #: the tool loop's turns, a structured re-ask, a guardrail re-ask. Declared
    #: since the first release and never assigned until 1.67.0.
    #:
    #: ``0.0`` has two meanings, and ``cost_known`` is how you tell them apart:
    #: genuinely free, versus a model we could not price. A self-hosted provider
    #: (ollama, lmstudio, vllm) runs on hardware the operator already pays for,
    #: so it is a known zero; a bedrock/azure deployment id is partner-billed
    #: and a private fine-tune is simply unknown, so both are ``cost_known
    #: False``. Never read ``cost == 0`` alone as "this run was free".
    cost: float = 0.0
    #: ``True`` only when every LLM call in the run could be priced.
    cost_known: bool = False
    latency_ms: int = 0
    trace_id: str | None = None
    execution_id: str = ""
    status: str = "completed"
    pending_interrupt: dict[str, Any] | None = None
    guardrails: list[GuardrailFiring] = Field(default_factory=list)
    #: Set when middleware ended the run with ``StopAgent`` (1.90.0): the
    #: stop's ``reason``, or ``"stopped"`` when it gave none. ``output`` is the
    #: stop message. ``None`` for a run that was not stopped.
    stop_reason: str | None = None
    #: How a ``Supervisor(routing="decisions")`` routed this run — a
    #: :class:`~fastaiagent.agent.team.SupervisorRoute` (which worker, the
    #: confidence, whether it fell back, every answer). ``None`` everywhere else.
    route: Any | None = None

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
        memory: MemoryLike | None = None,
        config: AgentConfig | None = None,
        output_type: type | None = None,
        middleware: Sequence[AgentMiddleware] | None = None,
        checkpointer: Checkpointer | None = None,
        agent_path_label: str | None = None,
        agent_id: str | None = None,
        prompt_slug: str | None = None,
    ):
        self.name = name
        # The platform agent UUID this agent represents. Set it to enroll a
        # connected agent in managed governance — it's sent to POST /policy/decide
        # so the plane can match approval policies (Task C). Without it the
        # governance gate is a no-op (the agent isn't enrolled).
        self.agent_id = agent_id
        self.system_prompt = system_prompt
        # A control-plane prompt-registry reference. When set, to_dict() emits
        # this slug and sends system_prompt="" so a pushed agent links to the
        # governed prompt (console shows the slug, not "Inline") instead of
        # inlining the resolved text and dropping the linkage.
        self.prompt_slug = prompt_slug
        # Gap 4 provenance: if system_prompt is a registry Prompt, use its text at
        # runtime and capture its name/version so every run stamps the llm_call
        # span — the Local UI's prompt lineage finds a prompt's traces by name.
        # A control-plane prompt also carries slug/environment for Prompt
        # Analytics and auto-links prompt_slug (Gap 3) so the pushed definition
        # references the slug. A local prompt never links: the plane has no
        # such slug, and a linked agent pushes system_prompt="".
        self._prompt_provenance: dict[str, Any] | None = None
        try:
            from fastaiagent.prompt.prompt import Prompt as _Prompt

            if isinstance(system_prompt, _Prompt):
                self._prompt_provenance = {
                    "name": system_prompt.name,
                    "version": system_prompt.version,
                }
                if system_prompt.slug:
                    self._prompt_provenance["slug"] = system_prompt.slug
                    self._prompt_provenance["environment"] = system_prompt.environment
                    if not self.prompt_slug:
                        self.prompt_slug = system_prompt.slug
                self.system_prompt = system_prompt.template
        except Exception:
            pass
        if llm is not None and not isinstance(llm, LLMClient):
            # A raw openai/AzureOpenAI SDK client can't be used as ``llm``
            # directly — it has no ``model``/``provider`` and no ``acomplete``.
            # Point users at the supported wrapper, which delegates the HTTP
            # call (auth, gateway, api-version, Entra token refresh) to their
            # client while supplying the deployment id and building the body.
            looks_like_openai = callable(
                getattr(getattr(getattr(llm, "chat", None), "completions", None), "create", None)
            )
            hint = (
                " To use an Azure OpenAI client behind a gateway, wrap it: "
                "Agent(llm=LLMClient(provider='azure', model='<deployment>', "
                "openai_client=<your AzureOpenAI client>))."
                if looks_like_openai
                else ""
            )
            raise TypeError(
                f"Agent(llm=...) expects a fastaiagent LLMClient, got {type(llm).__name__}.{hint}"
            )
        self.llm = llm or LLMClient()
        self.tools: list[Tool] = list(tools) if tools else []
        self.guardrails: list[Guardrail] = list(guardrails) if guardrails else []
        self.memory = memory
        self.config = config or AgentConfig()
        self.output_type = output_type
        # Resolve structured-output plan once (schema + parser). Supports any
        # Pydantic-compatible type (BaseModel, list[Model], primitives, unions).
        self._output_spec: OutputSpec | None = (
            OutputSpec(output_type) if output_type is not None else None
        )
        self.middleware: list[AgentMiddleware] = list(middleware) if middleware else []
        self._mw_pipeline = self._make_pipeline()
        self._check_tool_names()
        self._checkpointer: Checkpointer | None = checkpointer
        # Override the segment this Agent contributes to ``_agent_path``.
        # Default ``"agent:<name>"``; ``Supervisor`` uses ``"supervisor:<name>"``,
        # delegated workers use ``"worker:<role>"``.
        self._agent_path_label: str = agent_path_label or f"agent:{self.name}"
        # Gap 2 registration: track this agent so a later connect() can flush it
        # to the plane (best-effort; never breaks construction).
        try:
            from fastaiagent._platform.push import track_agent

            track_agent(self)
        except Exception:
            pass

    def _make_pipeline(self) -> _MiddlewarePipeline:
        """The middleware pipeline for this agent's ``middleware`` and ``tools``.

        Deferred tools (1.91.0, 1G) are reached through a built-in middleware,
        innermost so the user's middleware sees every tool. It is only added
        when a tool is deferred: an empty pipeline keeps parallel tool calls
        allowed. Anything that copies an agent rebuilds through here.
        """
        stack: list[AgentMiddleware] = list(self.middleware)
        deferred = [t for t in self.tools if getattr(t, "deferred", False)]
        if deferred:
            stack.append(_DeferredTools(deferred))
        return _MiddlewarePipeline(stack)

    def _with_outer_middleware(self, outer: Sequence[AgentMiddleware]) -> Agent:
        """This agent with ``outer`` middleware around its own — how a Chain runs
        an agent node under the chain's middleware (1.91.0, 1C). A shallow copy:
        same tools, memory, model and checkpointer; ``self`` is not changed."""
        import copy

        if not outer:
            return self
        clone = copy.copy(self)
        clone.middleware = [*outer, *self.middleware]
        clone._mw_pipeline = clone._make_pipeline()
        clone._check_tool_names()
        return clone

    def _with_output_type(self, output_type: type | None) -> Agent:
        """This agent with a per-run ``output_type`` (1.91.0, 2H). A shallow copy,
        like :meth:`_with_outer_middleware`; ``self`` is not changed."""
        import copy

        if output_type is None or output_type is self.output_type:
            return self
        clone = copy.copy(self)
        clone.output_type = output_type
        clone._output_spec = OutputSpec(output_type)
        return clone

    def cancel(self, execution_id: str) -> bool:
        """Stop the run ``execution_id`` if it is in progress in this process (1.91.0).

        ``arun`` / ``run`` / ``aresume`` stop at once — inside a model or tool call
        — and return ``status="cancelled"``; ``astream`` stops at its next event
        and ends with ``StreamDone(finish_reason="cancelled")``. Safe from any
        thread. The run's checkpoint ends ``failed`` with ``run_reason``
        ``"cancelled"``. Returns ``False`` when no such run is in progress.
        """
        from fastaiagent.agent import run_control

        return run_control.cancel(execution_id)

    def _middleware_names(self) -> list[str]:
        """The run's middleware, in order — the ``fastaiagent.middleware`` span
        attribute (decision B1-6: one structural attribute, no new spans)."""
        return list(self._mw_pipeline._keys)

    def _run_tools(self) -> list[Tool]:
        """The tools a run offers: the agent's, then each middleware's (1.91.0,
        1B). ``self.tools`` stays the agent's own list — what ``to_dict`` and
        the plane see. With ``output_mode="tool"`` the ``final_result`` tool
        comes last (1.91.0, 2G)."""
        tools: list[Tool] = [*self.tools, *self._mw_pipeline.tools]
        if self._output_spec is not None and self.config.output_mode == "tool":
            tools.append(_FinalResultTool(self._output_spec.schema))
        return tools

    def _apply_structured(self, kwargs: dict[str, Any]) -> None:
        """Put this run's structured-output parameters on the model call
        (1.91.0, 2G): the provider's ``response_format`` for ``"auto"`` and
        ``"native"`` (which also says so, for the providers whose native form
        differs); nothing for ``"prompt"`` / ``"tool"``, whose instruction is in
        the prompt (see :meth:`_build_messages`)."""
        response_format = self._build_response_format()
        if response_format is None or self.config.output_mode in ("prompt", "tool"):
            return
        kwargs["response_format"] = response_format
        if self.config.output_mode == "native":
            kwargs["output_mode"] = "native"

    def _final_tool(self) -> str | None:
        """``final_result`` when this agent answers through it (``"tool"`` mode)."""
        if self._output_spec is not None and self.config.output_mode == "tool":
            return FINAL_RESULT
        return None

    def _structured_instruction(self) -> str | None:
        """The system instruction for ``output_mode="prompt"`` / ``"tool"``."""
        if self._output_spec is None or self.config.output_mode not in ("prompt", "tool"):
            return None
        if self.config.output_mode == "tool":
            return (
                f"When you have the final answer, call the {FINAL_RESULT} tool with it. "
                "Do not answer in plain text."
            )
        from fastaiagent.llm.client import _augment_system_for_response_format

        return _augment_system_for_response_format(
            "", self._output_spec.response_format(strict=False)
        ).strip()

    def _check_tool_names(self) -> None:
        """A middleware tool may not share a name with another tool: the model
        could only ever reach one of them, and which one would be an accident
        of ordering."""
        owners: dict[str, str] = {t.name: "the agent" for t in self.tools}
        if self._output_spec is not None and self.config.output_mode == "tool":
            if FINAL_RESULT in owners:
                raise ValueError(
                    f"Tool name {FINAL_RESULT!r} is reserved by output_mode='tool', which "
                    "answers through a tool of that name; rename the agent's tool."
                )
        for mw in self._mw_pipeline.middleware:
            label = (
                "deferred tools (load_tools)"
                if isinstance(mw, _DeferredTools)
                else f"middleware {mw.name or type(mw).__name__!r}"
            )
            for t in mw.tools or ():
                if t.name in owners:
                    raise ValueError(
                        f"Tool name {t.name!r} is used by {owners[t.name]} and by {label}; "
                        "tool names must be unique in an agent."
                    )
                owners[t.name] = label

    def push(self, *, force: bool = True) -> Any:
        """Register this agent's definition with the plane (Gap 2).

        Explicit, first-class replacement for hand-written ``httpx.post(...)``.
        Requires an active ``fa.connect()`` and an ``agent:write`` scope on the
        key. Returns a :class:`~fastaiagent._platform.push.PushResult`
        (``agent_id``, ``name``, ``version``, console ``url``). Idempotent by
        name on the plane; ``force=True`` (default) always re-pushes so a
        same-process definition change resyncs. Raises on platform errors so CI
        surfaces them.
        """
        from fastaiagent._platform.push import push_agent

        return push_agent(self, force=force, best_effort=False)

    def _pause_owner_run_id(self, exec_id: str) -> str:
        """The run a managed-policy pause belongs to — what its pending run is registered under.

        Call it BEFORE this run binds ``_execution_id`` to ``exec_id``: it reads the
        enclosing run's id from that ContextVar.

        * With its own checkpointer the agent stores its pause itself, under ``exec_id``.
        * With none, it cannot: the ``InterruptSignal`` rises to whatever encloses it —
          a Chain — which stores the pause under **its** run id, reports the pause and
          the resolution under that id, and on resume re-runs this agent under a fresh
          ``exec_id``. So the enclosing run id is the only one that lives from pause to
          resolution, and registering the pending run under ``exec_id`` left the plane
          a record it could never close (1.76.0). With nothing enclosing, ``exec_id``.
        """
        if self._checkpointer is not None:
            return exec_id
        return _execution_id.get() or exec_id

    def _build_response_format(self) -> dict[str, Any] | None:
        """Build response_format dict from output_type for structured output."""
        if self._output_spec is None:
            return None
        # Native strict Structured Outputs is OpenAI/Azure-only; other providers
        # ignore the flag and receive the normal (non-strict) json_schema.
        strict = self.config.strict_output and self.llm.provider in ("openai", "azure")
        return self._output_spec.response_format(strict=strict)

    def _try_parse(self, text: str) -> tuple[Any | None, str | None]:
        """Parse text into ``output_type``. Returns ``(value, error_reason)``."""
        if self._output_spec is None:
            return None, None
        if not text:
            return None, "the model returned an empty response"
        return self._output_spec.parse(text)

    def _parse_output(self, text: str) -> Any | None:
        """Parse LLM text into ``output_type`` (value only; ``None`` on failure)."""
        value, _ = self._try_parse(text)
        return value

    async def _ask_model(
        self,
        mw_ctx: MiddlewareContext | None,
        messages: list[Message],
        kwargs: dict[str, Any],
    ) -> LLMResponse:
        """A model call outside the tool loop — the structured-output and
        guardrail re-asks. Through ``wrap_model`` like every loop call (1.91.0,
        1A): before, a re-ask bypassed middleware, so a cache, a retry or a
        budget never saw it. Offers no tools.

        Held to the run's limits like a loop call (1.91.0, 2I): over a cap it
        raises ``StopAgent``, which each re-ask handles."""
        from fastaiagent.agent import run_control

        run_control.check_before_model_call()
        if mw_ctx is None or not self._mw_pipeline.overrides("wrap_model"):
            return await self.llm.acomplete(messages, **kwargs)
        mw_ctx.llm = self.llm
        mw_ctx.tools = []

        async def _terminal(msgs: list[Message]) -> LLMResponse:
            return await (mw_ctx.llm or self.llm).acomplete(msgs, **kwargs)

        return await self._mw_pipeline.call_model(mw_ctx, messages, _terminal)

    async def _reask_structured(
        self,
        messages: list[Message],
        bad_output: str,
        error: str,
        kwargs: dict[str, Any],
        mw_ctx: MiddlewareContext | None = None,
    ) -> tuple[str, Any | None, int]:
        """Re-ask the model to fix a structured-output parse/validation failure.

        Appends the rejected reply plus a correction message and calls the LLM
        again with the same ``response_format`` (no tools) up to
        ``config.output_retries`` times. Returns
        ``(final_output, parsed_or_None, extra_tokens)``.

        Works on a copy: the caller's list is the prompt the loop-end checkpoint
        saves, and a resume reads the run's input from its last user message —
        which must not be this correction prompt.
        """
        from fastaiagent.llm.message import AssistantMessage

        messages = list(messages)
        output = bad_output
        reason: str | None = error
        parsed: Any | None = None
        extra_tokens = 0
        for _ in range(self.config.output_retries):
            messages.append(AssistantMessage(content=output))
            messages.append(
                UserMessage(
                    "Your previous response could not be used: "
                    f"{reason}. Reply again with ONLY the JSON value that "
                    "satisfies the required schema — no explanation, no markdown."
                )
            )
            try:
                resp = await self._ask_model(mw_ctx, messages, kwargs)
            except StopAgent as stop:
                # A run limit (1.91.0): no more re-asks; the result says why.
                from fastaiagent.agent import run_control

                control = run_control.active_control()
                if control is not None:
                    control.limit_reason = stop.reason
                break
            output = resp.content or ""
            extra_tokens += resp.usage.get("total_tokens", 0)
            parsed, reason = self._try_parse(output)
            if parsed is not None:
                break
        return output, parsed, extra_tokens

    async def _guard_output(
        self,
        output: str,
        parsed: Any | None,
        guardrails: list[Guardrail],
        llm_messages: list[Message],
        kwargs: dict[str, Any],
        *,
        stopped: bool = False,
        mw_ctx: MiddlewareContext | None = None,
    ) -> tuple[str, Any | None, int]:
        """Run the output guardrails, honouring whatever action they carry.

        ``stopped``: middleware ended the run with ``StopAgent``. The rules still
        judge the stop message, but a ``reask`` cannot re-drive a run its
        middleware ended, so a failing one blocks at once (1.90.0, F8).

        Returns ``(output, parsed, extra_tokens)``. Three of the five actions
        resolve inside :func:`execute_guardrails`; the two that need the agent
        loop resolve here:

        * ``mask`` / ``override`` rewrote the payload — take the rewritten text
          and re-derive ``parsed``, since the value the caller receives is no
          longer the one we parsed.
        * ``reask`` re-prompts the model with the failure as feedback, bounded by
          ``config.guardrail_retries``. Exhausting the cap blocks: this is the
          one action the plane cannot perform, and a re-ask that never converges
          must not become a silent pass.
        """
        from fastaiagent._internal.errors import GuardrailBlockedError
        from fastaiagent.llm.message import AssistantMessage

        extra_tokens = 0
        attempts = 0 if stopped else max(0, self.config.guardrail_retries)

        for attempt in range(attempts + 1):
            outcome = await execute_guardrails(
                guardrails,
                output,
                GuardrailPosition.output,
                allow_reask=True,
                agent_name=self.name,
                attempt=attempt + 1,
            )

            if outcome.modified:
                output = str(outcome.data)
                parsed, _ = self._try_parse(output)

            if outcome.reask is None:
                return output, parsed, extra_tokens

            if attempt >= attempts:
                # Out of retries. The rule asked us to try again and we did;
                # the failure stands, so it costs what an unsatisfied guardrail
                # always costs.
                why = (
                    "the run was stopped by middleware, so it cannot be re-asked"
                    if stopped
                    else f"re-asked {attempts} time(s) without satisfying the rule"
                )
                outcome.refuse(why)
                raise GuardrailBlockedError(
                    guardrail_name="reask",
                    message=f"{outcome.reask.message or 'Guardrail failed'} ({why}; blocked)",
                    results=outcome.results,
                )

            reason = outcome.reask.message or "the reply did not satisfy a guardrail"
            llm_messages = [
                *llm_messages,
                AssistantMessage(content=output),
                UserMessage(
                    f"Your previous response was rejected: {reason}. "
                    "Reply again, addressing that problem. Do not mention this instruction."
                ),
            ]
            try:
                resp = await self._ask_model(mw_ctx, llm_messages, kwargs)
            except StopAgent as stop:
                # A run limit (1.91.0) leaves the rule unsatisfied: blocked, as
                # when the retries run out — never let through.
                why = f"the run reached {stop.reason}, so it cannot be re-asked"
                outcome.refuse(why)
                raise GuardrailBlockedError(
                    guardrail_name="reask",
                    message=f"{outcome.reask.message or 'Guardrail failed'} ({why}; blocked)",
                    results=outcome.results,
                ) from None
            output = resp.content or ""
            extra_tokens += resp.usage.get("total_tokens", 0)
            parsed, _ = self._try_parse(output)

        return output, parsed, extra_tokens

    @overload
    def run(
        self,
        input: AgentInput,
        *,
        output_type: type[ResultT],
        context: RunContext[Any] | None = None,
        trace: bool = True,
        execution_id: str | None = None,
        messages: list[Message] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AgentResult[ResultT]: ...

    @overload
    def run(
        self,
        input: AgentInput,
        *,
        output_type: None = None,
        context: RunContext[Any] | None = None,
        trace: bool = True,
        execution_id: str | None = None,
        messages: list[Message] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AgentResult[Any]: ...

    def run(
        self,
        input: AgentInput,
        *,
        output_type: type | None = None,
        context: RunContext[Any] | None = None,
        trace: bool = True,
        execution_id: str | None = None,
        messages: list[Message] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AgentResult[Any]:
        """Synchronous execution.

        ``input`` is a string, an :class:`Image`, a :class:`PDF`, or a list
        of those parts. The list form sends multimodal content to the LLM.

        ``messages`` (optional) are prior conversation turns prepended before
        ``input`` — used for multi-turn flows such as :func:`fastaiagent.simulate`.

        ``output_type`` (1.91.0) overrides the agent's for this run, and types the
        result: ``agent.run(..., output_type=Invoice)`` is an
        ``AgentResult[Invoice]``.
        """
        return run_sync(
            self.arun(
                input,
                output_type=output_type,
                context=context,
                trace=trace,
                execution_id=execution_id,
                messages=messages,
                metadata=metadata,
                **kwargs,
            )
        )

    def _effective_guardrails(self) -> list[Guardrail]:
        """The agent's local guardrails plus any plane-authored guardrails
        distributed to this connected agent via the cached governance policy
        (scoped by ``agent_ids``). Returns just the local list when not connected
        or nothing applies — so unconnected/local runs are unchanged and pay
        nothing. This is how a guardrail authored on the Enterprise plane is
        enforced at the edge by the runtime.
        """
        try:
            from fastaiagent.guardrail.from_policy import plane_guardrails_for_agent

            plane = plane_guardrails_for_agent(self.agent_id)
        except Exception:
            return self.guardrails
        if not plane:
            return self.guardrails
        # Don't enforce the same plane rule twice. A caller who pulled the rules
        # themselves and passed them in guardrails=[...] would otherwise get two
        # runs of every rule — double the llm_judge spend and double the
        # guardrail rows on the plane. Matched on plane-origin name only, so a
        # *locally* authored guardrail that merely shares a name still runs
        # alongside the plane's (deduping those could weaken enforcement).
        already = {g.name for g in self.guardrails if getattr(g, "origin", "local") == "plane"}
        extra = [g for g in plane if g.name not in already]
        if not extra:
            return self.guardrails
        return [*self.guardrails, *extra]

    @overload
    async def arun(
        self,
        input: AgentInput,
        *,
        output_type: type[ResultT],
        context: RunContext[Any] | None = None,
        trace: bool = True,
        execution_id: str | None = None,
        messages: list[Message] | None = None,
        wait_for_approval: bool = False,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AgentResult[ResultT]: ...

    @overload
    async def arun(
        self,
        input: AgentInput,
        *,
        output_type: None = None,
        context: RunContext[Any] | None = None,
        trace: bool = True,
        execution_id: str | None = None,
        messages: list[Message] | None = None,
        wait_for_approval: bool = False,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AgentResult[Any]: ...

    async def arun(
        self,
        input: AgentInput,
        *,
        output_type: type | None = None,
        context: RunContext[Any] | None = None,
        trace: bool = True,
        execution_id: str | None = None,
        messages: list[Message] | None = None,
        wait_for_approval: bool = False,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AgentResult[Any]:
        """Async execution with tool-calling loop.

        ``output_type`` (1.91.0): see :meth:`run`.

        Bounded by ``AgentConfig.timeout_seconds`` / ``max_total_tokens`` /
        ``max_cost_usd`` and stoppable with :meth:`cancel` (1.91.0, 2I).

        ``input`` may be a string or a multimodal list (text + Image + PDF).
        ``execution_id`` (optional) names this run for resume. If omitted,
        a UUID is generated. Pair with a ``checkpointer`` on the Agent to
        get crash- and interrupt-recovery via :meth:`resume`.

        ``messages`` (optional) are prior conversation turns prepended before
        ``input`` — used for multi-turn flows such as :func:`fastaiagent.simulate`.
        Default ``None`` reproduces the single-input behavior exactly.

        When a managed governance policy pauses a tool call for approval, the run
        returns ``status="paused"``. **Your application is the approver**: show
        ``result.pending_interrupt["context"]`` (``tool`` and ``tool_input``) to
        its user, then call :meth:`aresume` with ``Resume(approved=...,
        metadata={"resolver": <that user>})``. A rejection refuses the call — the
        model is told, and the tool never runs.

        ``wait_for_approval=True`` was **removed in 1.76.0** and raises
        ``ValueError`` before anything runs: it blocked polling the plane for a
        console decision, and the plane retired its console approve/deny
        (enterprise PR #199). ``False`` is still accepted and does nothing. The
        parameter stays so an explicit ``wait_for_approval=False`` keeps working
        instead of slipping through ``**kwargs`` into the model call.
        """
        if wait_for_approval:
            raise ValueError(
                "arun(wait_for_approval=True) was removed in fastaiagent 1.76.0: the "
                "control plane no longer approves tool calls, so there is nothing to "
                "wait for. Take the paused result (status='paused') and resume it from "
                "your application with aresume(run_id, resume_value=Resume(approved=..., "
                "metadata={'resolver': ...}))."
            )
        if output_type is not None and output_type is not self.output_type:
            return await self._with_output_type(output_type).arun(
                input,
                context=context,
                trace=trace,
                execution_id=execution_id,
                messages=messages,
                metadata=metadata,
                **kwargs,
            )
        exec_id = execution_id or str(uuid.uuid4())

        # Prompt provenance is bound in ``_arun_core``, which every non-streamed
        # run passes through (arun, aresume, a fork); astream binds its own.
        def work() -> Coroutine[Any, Any, AgentResult[Any]]:
            if trace:
                return self._arun_traced(
                    input,
                    context=context,
                    execution_id=exec_id,
                    messages=messages,
                    metadata=metadata,
                    **kwargs,
                )
            return self._arun_core(
                input, context=context, execution_id=exec_id, messages=messages, **kwargs
            )

        return await self._controlled(exec_id, work)

    async def _controlled(
        self, exec_id: str, work: Callable[[], Coroutine[Any, Any, AgentResult[Any]]]
    ) -> AgentResult[Any]:
        """Run ``work`` as a run :meth:`cancel` and ``timeout_seconds`` can stop
        (1.91.0, 2I).

        The work runs as a child task, so a stop takes effect at once, inside a
        model or tool call. A cancel returns ``status="cancelled"``; a timeout
        raises ``AgentTimeoutError``; the caller's own task being cancelled
        cancels the run and propagates. A run nested in another — a Swarm hop, a
        Supervisor worker, an agent called from a tool — runs inline, holding its
        own token and cost caps on its own spend; cancel and the timeout are the
        top-level run's (the same ``_agent_path`` test that decides who writes the
        run-end marker).
        """
        import asyncio

        from fastaiagent.agent import run_control

        control = self._run_control(exec_id)
        if _agent_path.get() is not None:
            token = run_control.bind(control)
            try:
                return await work()
            finally:
                run_control.unbind(token)
        run_control.add(control)
        token = run_control.bind(control)  # before the task, which copies it
        timer = None
        try:
            control.loop = asyncio.get_running_loop()
            task = asyncio.ensure_future(work())
            control.task = task
            timer = run_control.arm_timeout(control)
            try:
                return await task
            except (asyncio.CancelledError, run_control._RunCancelled):
                if control.reason is None:
                    # The caller's own task was cancelled: stop the run with it.
                    task.cancel()
                    try:
                        await task
                    except BaseException:  # noqa: BLE001 - we are unwinding already
                        pass
                    raise
            if control.reason == run_control.TIMEOUT:
                raise control.timeout_error() from None
            return self._cancelled_result(control)
        finally:
            if timer is not None:
                timer.cancel()
            run_control.unbind(token)
            run_control.remove(control)

    def _run_control(self, exec_id: str, **fields: Any) -> Any:
        """This run's :class:`~fastaiagent.agent.run_control.RunControl`, nested
        under the active one when there is one."""
        from fastaiagent.agent import run_control

        return run_control.RunControl(
            execution_id=exec_id,
            agent_name=self.name,
            model=getattr(self.llm, "model", "") or "",
            timeout_seconds=self.config.timeout_seconds,
            max_total_tokens=self.config.max_total_tokens,
            max_cost_usd=self.config.max_cost_usd,
            parent=run_control.active_control() if _agent_path.get() is not None else None,
            **fields,
        )

    def _cancelled_result(self, control: Any) -> AgentResult[Any]:
        """What a cancelled run returns: what it spent, and why it stopped."""
        from fastaiagent.agent import run_control

        bucket = control.usage_bucket
        return AgentResult(
            output="",
            tokens_used=bucket.tokens if bucket is not None else 0,
            cost=bucket.usd if bucket is not None and bucket.known else 0.0,
            cost_known=bucket.known if bucket is not None else False,
            latency_ms=int((time.monotonic() - control.started) * 1000),
            execution_id=control.execution_id,
            status=run_control.CANCELLED,
            stop_reason=run_control.CANCELLED,
            messages=_conversation(control.llm_messages or [], control.prefix_ids),
        )

    async def _arun_traced(
        self,
        input: AgentInput,
        *,
        context: RunContext[Any] | None = None,
        execution_id: str | None = None,
        messages: list[Message] | None = None,
        metadata: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AgentResult:
        """Execute with OTel tracing."""
        # The id is chosen HERE, not in ``_arun_core``, so the root span can name
        # it: ``agent.execution_id`` is what links a run to its checkpoints and to
        # every resume of it (each resume is its own trace — decision D7).
        exec_id = execution_id or str(uuid.uuid4())
        with self._open_run_span(
            input, context, execution_id=exec_id, metadata=metadata, extra=extra
        ) as span:
            result = await self._arun_core(
                input,
                context=context,
                execution_id=exec_id,
                messages=messages,
                **kwargs,
            )
            self._close_run_span(span, result)
            return result

    @contextmanager
    def _open_run_span(
        self,
        input: AgentInput,
        context: RunContext[Any] | None,
        *,
        execution_id: str,
        metadata: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Iterator[Any]:
        """Open the ``agent.<name>`` root span every non-streamed run gets.

        Shared by :meth:`arun`, :meth:`aresume` and a fork, so a resumed run is
        stamped exactly like a fresh one plus its ``extra`` link attributes.
        Until 1.89.0 only ``arun`` opened it: a resume's LLM calls were emitted
        as orphan ``llm.*`` roots and the result carried no ``trace_id``.
        """
        from fastaiagent.trace.otel import get_tracer
        from fastaiagent.trace.span import set_metadata_attributes, trace_id_of

        tracer = get_tracer()
        with tracer.start_as_current_span(f"agent.{self.name}") as span:
            span.set_attribute("agent.name", self.name)
            span.set_attribute("fastaiagent.framework", "fastaiagent")
            span.set_attribute("agent.execution_id", execution_id)
            if self._mw_pipeline:
                span.set_attribute("fastaiagent.middleware", self._middleware_names())
            for key, value in (extra or {}).items():
                if value is not None:
                    span.set_attribute(key, value)
            # Developer-supplied run tags (MLflow-style) → fastaiagent.meta.*
            # on the root span, tagging the whole trace. Open OTel map, no wire
            # change. The developer owns keys/values/PII.
            set_metadata_attributes(span, metadata)
            # Gap 2 first-run auto-register: an agent created after connect()
            # registers itself the first time it runs. Non-blocking, best-effort,
            # once per process — no-op when disconnected/opted-out/already pushed.
            try:
                from fastaiagent._platform.push import auto_register_async

                auto_register_async(self)
            except Exception:
                pass
            # Span attributes must be primitives — coerce multimodal input
            # to a readable text summary.
            normalized_input_parts: list[ContentPart] = (
                [input] if isinstance(input, str) else normalize_input(input)
            )
            # ``agent.input`` is captured to local.db unconditionally so the
            # local UI and Replay always have full fidelity. Payload privacy is
            # enforced at the *export* boundary (security_audit_2 N3, egress
            # model): see ``export_payloads_enabled()`` and the plane/OTel
            # exporters, which strip payload attributes when the operator opts
            # out via ``FASTAIAGENT_TRACE_PAYLOADS=0``.
            input_text = (
                input if isinstance(input, str) else _input_summary_text(normalized_input_parts)
            )
            span.set_attribute("agent.input", input_text)

            # Persist multimodal attachments (Image/PDF) to the
            # ``trace_attachments`` table so the UI / Replay can fetch
            # thumbnails and (optionally) the original bytes.
            try:
                from fastaiagent.multimodal.image import Image as _MMImage
                from fastaiagent.multimodal.pdf import PDF as _MMPDF

                if any(isinstance(p, (_MMImage, _MMPDF)) for p in normalized_input_parts):
                    from fastaiagent.trace.attachments import save_parts_for_span
                    from fastaiagent.trace.storage import TraceStore

                    span_ctx = span.get_span_context()
                    saved = save_parts_for_span(
                        db=TraceStore.default()._db,
                        trace_id=trace_id_of(span_ctx),
                        span_id=format(span_ctx.span_id, "016x"),
                        parts=normalized_input_parts,
                        role="input",
                    )
                    if saved:
                        span.set_attribute(
                            "fastaiagent.input.attachment_ids",
                            json.dumps([r.attachment_id for r in saved]),
                        )
                        span.set_attribute("fastaiagent.input.media_count", len(saved))
            except Exception:
                # Trace persistence must never fail the agent run.
                logger.debug("Failed to persist input attachments", exc_info=True)

            # Reconstruction metadata for ForkedReplay.arerun (always captured —
            # structural, not payload).
            span.set_attribute("agent.config", json.dumps(self.config.model_dump()))
            span.set_attribute("agent.tools", json.dumps([t.to_dict() for t in self.tools]))
            span.set_attribute(
                "agent.guardrails", json.dumps([g.to_dict() for g in self.guardrails])
            )
            span.set_attribute("agent.llm.provider", self.llm.provider)
            span.set_attribute("agent.llm.model", self.llm.model)
            span.set_attribute("agent.llm.config", json.dumps(self.llm.to_dict()))

            # Resolved system prompt — captured locally unconditionally so Replay
            # can reconstruct the run; stripped at export when payloads are off.
            try:
                resolved_prompt = self._resolve_system_prompt(context)
                if resolved_prompt:
                    span.set_attribute("agent.system_prompt", resolved_prompt)
            except Exception:
                # Callable system_prompt that needs context — best-effort only.
                logger.debug("Failed to resolve system prompt for trace", exc_info=True)

            yield span

    @staticmethod
    def _close_run_span(span: Any, result: AgentResult) -> None:
        """Stamp a finished (or paused) run onto its root span and its result."""
        from fastaiagent.trace.span import trace_id_of

        # Captured locally unconditionally (N3 egress model — privacy is
        # enforced at export, not capture).
        span.set_attribute("agent.output", result.output)
        span.set_attribute("agent.tokens_used", result.tokens_used)
        span.set_attribute("agent.latency_ms", result.latency_ms)
        # ``completed`` or ``paused`` — a paused run is still a successful span,
        # and without this a reader could not tell the two apart.
        span.set_attribute("agent.status", result.status)
        result.trace_id = trace_id_of(span.get_span_context())

    async def _arun_core(
        self,
        input: AgentInput,
        *,
        context: RunContext[Any] | None = None,
        execution_id: str | None = None,
        messages: list[Message] | None = None,
        _resumed_messages: list[Message] | None = None,
        _start_iteration: int = 0,
        _stopped: StopAgent | None = None,
        _seed_firings: list[Any] | None = None,
        _seed_usage: Any | None = None,
        _mw_ctx: MiddlewareContext | None = None,
        _mw_state: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AgentResult:
        """Core execution without tracing.

        ``_mw_state``: middleware state saved in the checkpoint a fork starts
        from (1.91.0, 1B) — the fork is a new run, from where that one was.

        ``_stopped``: middleware raised ``StopAgent`` around the call a resume
        re-ran, so the run ends here exactly as the loop ends a stopped run —
        without another model call.

        ``_mw_ctx``: a resume's middleware context. The resume already fired
        ``before_run`` and ran the saved tool with it; reusing it keeps one
        ``scratch`` for the whole run and ``before_run`` from firing twice.

        ``_seed_firings`` / ``_seed_usage``: what that re-run call recorded
        (guardrail firings, and any model spend — an ``llm_judge`` on it). It ran
        before this scope opened, and belongs to the same run's result.
        """
        start = time.monotonic()
        exec_id = execution_id or str(uuid.uuid4())
        # Read BEFORE this run rebinds ``_execution_id`` below.
        pause_run_id = self._pause_owner_run_id(exec_id)

        # Set up the execution-scoped ContextVars so interrupt() / @idempotent
        # can find the active execution + checkpointer. ``_agent_path`` is
        # **extended** rather than overwritten so a parent topology
        # (Swarm / Supervisor) can prefix it with its own segment.
        exec_token = _execution_id.set(exec_id)
        parent_path = _agent_path.get()
        new_path = (
            f"{parent_path}/{self._agent_path_label}" if parent_path else self._agent_path_label
        )
        ap_token = _agent_path.set(new_path)
        # ⚠ Whether this Agent is the OUTERMOST runner of ``exec_id``, and so the
        # one entitled to write its run-end marker (audit D5). A Swarm's child
        # agents and a Supervisor's worker clones run under the SAME
        # execution_id as their parent; without this, each child finishing its
        # own tool loop would stamp "the run ended" on a run that is still
        # handing off, and a swarm would accumulate one false terminal row per
        # hop. An inherited ``_agent_path`` is exactly the signal that something
        # above us owns this execution — a Supervisor's own inner agent sees
        # None here (the Supervisor sets no path of its own) and correctly
        # counts as outermost.
        owns_run = parent_path is None
        # An enclosing run (an Agent node of a Chain) keeps its cache (1.92.0).
        cache_tokens = bind_cache(exec_id, self._checkpointer)
        # Expose the RunContext so memory blocks can resolve a dynamic scope_id
        # (e.g. per-user) for this turn. Absent context → callable ids resolve
        # to "" → safe (no personal facts).
        rc_token = set_active_run_context(context)
        # Which registry prompt this run uses, for the LLM client to stamp
        # fastaiagent.prompt.* on each llm span. Bound even when None so an agent
        # run inside another (a tool, a worker) never inherits its caller's prompt.
        from fastaiagent.prompt.provenance import reset_prompt_provenance, set_prompt_provenance

        pv_token = set_prompt_provenance(self._prompt_provenance)
        # Collect every guardrail firing for this run, whatever position it ran
        # at — the tool_call/tool_result ones happen inside the tool loop, which
        # is why this is run-scoped rather than a local list here.
        gf_token = start_firing_collection(_seed_firings)
        # Accumulate this run's LLM spend. Run-scoped for the same reason the
        # firing collection is: the calls happen inside the tool loop, the
        # structured re-ask and the guardrail re-ask, none of which return a
        # cost up to here. ``LLMClient`` records into it at the one point every
        # completion passes through.
        rcost_token = start_run_cost(_seed_usage)
        # The run's stop control (1.91.0, 2I), when this run is the top-level one:
        # it reports a cancelled run's spend and messages from outside the run.
        from fastaiagent.agent import run_control

        control = run_control.active_control() if owns_run else None
        if control is not None and control.execution_id != exec_id:
            control = None
        if control is not None:
            control.usage_bucket = current_run_usage()
        # The agent's sampling options go on every model call of this run — the
        # tool loop and both re-asks all receive ``kwargs`` (1.90.0, F6).
        for key, value in self.config._llm_kwargs().items():
            kwargs.setdefault(key, value)

        # Run the agent's checkpointer setup once — cheap on subsequent calls.
        if self._checkpointer is not None:
            self._checkpointer.setup()

        # Middleware context shared across the whole run — a resume hands in
        # its own so the re-run tool call and the continued loop share it.
        mw_ctx: MiddlewareContext | None = _mw_ctx
        if mw_ctx is None and self._mw_pipeline:
            mw_ctx = MiddlewareContext(
                run_context=context, agent_name=self.name, execution_id=exec_id
            )
            mw_ctx.restore_state(_mw_state)

        try:
            # ``before_run`` (1.91.0, 1A): once per run, before the input
            # guardrails, so they judge the input as rewritten. A resume fired it
            # already (it hands in ``_mw_ctx``); a fork is a run of its own.
            # ``StopAgent`` ends the run here, before any model call.
            if mw_ctx is not None and _mw_ctx is None:
                try:
                    input = await self._mw_pipeline.apply_before_run(mw_ctx, input)
                except StopAgent as stop:
                    _stopped = stop

            # Normalize once — every downstream step uses the same shape.
            normalized_parts: list[ContentPart] = (
                [input] if isinstance(input, str) else normalize_input(input)
            )
            input_text = input if isinstance(input, str) else _input_summary_text(normalized_parts)

            # Effective guardrails = the agent's local list + any authored on the
            # plane and distributed to this connected agent. Computed once so
            # input/tool/output enforcement all see the same set.
            eff_guardrails = self._effective_guardrails()

            # Execute input guardrails (blocking). Guardrails are text-only
            # today; we pass the text summary so policies still trigger on
            # the textual portion of multimodal input. A run its middleware
            # stopped before it began sends nothing, so there is nothing to judge.
            guarded_input: AgentInput = input
            if eff_guardrails and _stopped is None:
                outcome = await execute_guardrails(
                    eff_guardrails, input_text, GuardrailPosition.input, agent_name=self.name
                )
                if outcome.modified:
                    # A mask/override rewrote the input. We can only substitute
                    # it when the input was plain text: for multimodal input
                    # ``input_text`` is a *summary* of the parts, so swapping it
                    # in would silently drop the images. Fail closed instead.
                    guarded_input, input_text = _apply_input_rewrite(input, outcome)

            # Build messages — when resuming, restore the saved history.
            llm_messages = (
                _resumed_messages
                if _resumed_messages is not None
                else self._build_messages(
                    guarded_input, context=context, history=messages, mw_ctx=mw_ctx
                )
            )
            # What ``AgentResult.messages`` leaves out (1.91.0, 2J). A resumed
            # run's prefix is not recorded, so there only system messages go.
            prefix = set() if _resumed_messages is not None else _prefix_ids(llm_messages, messages)
            if control is not None:
                control.llm_messages, control.prefix_ids = llm_messages, prefix

            # Structured output, as ``output_mode`` asks (1.91.0, 2G).
            self._apply_structured(kwargs)

            # The run-end marker has to cover EVERYTHING that can still kill the
            # run, not just the tool loop: 1.65.0's D5 fix wrapped only
            # ``execute_tool_loop``, so a run that died in the structured-output
            # re-ask, an output guardrail or a memory write left no tombstone at
            # all — and a run that crashed there was byte-identical to one that
            # finished. ``loop_done`` and ``run_ended`` are what let the handler
            # below tell the three cases apart.
            loop_done = False
            run_ended = False
            try:
                response: LLMResponse
                if _stopped is not None:
                    response = _stopped_response(_stopped)
                    tool_calls: list[dict[str, Any]] = []
                else:
                    response, tool_calls = await execute_tool_loop(
                        llm=self.llm,
                        messages=llm_messages,
                        tools=self._run_tools(),
                        max_iterations=self.config.max_iterations,
                        tool_choice=self.config.tool_choice,
                        context=context,
                        guardrails=eff_guardrails or None,
                        mw_pipeline=self._mw_pipeline if self._mw_pipeline else None,
                        mw_ctx=mw_ctx,
                        checkpointer=self._checkpointer,
                        execution_id=exec_id,
                        agent_name=self.name,
                        agent_id=self.agent_id,
                        start_iteration=_start_iteration,
                        parallel_tools=self.config.parallel_tools,
                        max_parallel_tools=self.config.max_parallel_tools,
                        governance_run_id=pause_run_id,
                        final_tool=self._final_tool(),
                        **kwargs,
                    )
                loop_done = True

                output = response.content or ""
                # Middleware ended the run (1.90.0, F8): keep the reason, and never
                # re-drive the model. The structured re-ask used to answer a stop
                # message that failed to parse — calling the model again outside
                # middleware, and replacing the stop with whatever it said.
                stop_reason = (
                    response.stop_reason if isinstance(response, _StoppedResponse) else None
                )
                parsed, perr = self._try_parse(output)
                retry_tokens = 0
                # Structured-output self-correction: on a parse/validation failure,
                # re-ask the model with the error (opt-out via output_retries=0).
                if (
                    parsed is None
                    and perr is not None
                    and self._output_spec is not None
                    and self.config.output_retries > 0
                    and stop_reason is None
                ):
                    output, parsed, retry_tokens = await self._reask_structured(
                        llm_messages, output, perr, kwargs, mw_ctx=mw_ctx
                    )

                # Execute output guardrails. Unlike every other position this one
                # can re-drive the model, so a `reask` rule is honoured here rather
                # than degraded to a block.
                if eff_guardrails:
                    output, parsed, guardrail_tokens = await self._guard_output(
                        output,
                        parsed,
                        eff_guardrails,
                        llm_messages,
                        kwargs,
                        stopped=stop_reason is not None,
                        mw_ctx=mw_ctx,
                    )
                    retry_tokens += guardrail_tokens

                # Store in memory. Memory backends are text-only; record the
                # text summary so multimodal calls don't break the memory store.
                # Wrapped in a ``memory.write`` span (+ per-block children).
                if self.memory:
                    from fastaiagent.agent._memory_tracing import traced_add
                    from fastaiagent.llm.message import AssistantMessage

                    traced_add(self.memory, UserMessage(input_text))
                    traced_add(self.memory, AssistantMessage(output))

                latency = int((time.monotonic() - start) * 1000)
                tokens = _run_tokens_or(response, retry_tokens)

                cost, cost_known = run_cost()
                result = AgentResult(
                    output=output,
                    parsed=parsed,
                    tool_calls=tool_calls,
                    tokens_used=tokens,
                    cost=cost,
                    cost_known=cost_known,
                    latency_ms=latency,
                    execution_id=exec_id,
                    status="completed",
                    guardrails=collected_firings(),
                    # A cap that ended a structured re-ask (1.91.0, 2I).
                    stop_reason=stop_reason or (control.limit_reason if control else None),
                    messages=_conversation(llm_messages, prefix, output),
                )
                # ``after_run`` (1.91.0, 1A) before the run-end marker: a hook
                # that raises fails the run, and the marker must say so.
                if mw_ctx is not None:
                    result = await self._mw_pipeline.apply_after_run(mw_ctx, result)

                # The run finished. Without this row a completed run and one that
                # crashed right after its last turn are byte-identical — both leave a
                # ``completed`` turn checkpoint as the newest row (audit D5).
                if self._checkpointer is not None and owns_run:
                    write_run_end(
                        self._checkpointer,
                        execution_id=exec_id,
                        chain_name=self.name,
                        status="completed",
                        agent_path=new_path,
                        middleware_state=_end_state(mw_ctx, strict=True),
                    )
                    run_ended = True
                return result
            except _AgentInterrupted as susp:
                # Deliberately NO run-end marker: a paused run has not ended, and
                # a row written after the ``interrupted`` one would hide the pause
                # from ``aresume``'s status guard (audit D5).
                latency = int((time.monotonic() - start) * 1000)
                # A paused run has already spent what it spent; reporting 0.0
                # would make a pause look free. Same for its tokens, which the
                # paused branch simply never set — a HITL run that burned three
                # turns before the approval gate reported none of them.
                paused_cost, paused_cost_known = run_cost()
                return AgentResult(
                    output="",
                    tokens_used=run_tokens(),
                    cost=paused_cost,
                    cost_known=paused_cost_known,
                    execution_id=exec_id,
                    status="paused",
                    pending_interrupt={
                        "reason": susp.reason,
                        "context": susp.context,
                        "node_id": susp.node_id,
                        "agent_path": susp.agent_path,
                    },
                    latency_ms=latency,
                    # Input and tool guardrails already ran before the interrupt;
                    # a pause is not a reason to drop what they reported.
                    guardrails=collected_firings(),
                    messages=_conversation(llm_messages, prefix),
                )
            except BaseException as exc:
                # The run died. Mark it, then re-raise untouched — the marker is
                # best-effort inside ``write_run_end`` precisely so a checkpointer
                # problem here can never replace the exception the caller needs.
                if self._checkpointer is not None and owns_run and not run_ended:
                    if loop_done:
                        # The tool loop ENDED before this failure, so the loop is
                        # not the place to re-enter. Without this row the newest
                        # resumable checkpoint is the last *pre-tool* one, and
                        # ``aresume`` re-invokes that tool: measured as one charge
                        # becoming two. A turn-boundary row past the loop's last
                        # turn re-issues the model instead, which is the documented
                        # (and idempotent-by-default) resume shape.
                        try:
                            from fastaiagent.agent.executor import _put_turn_checkpoint

                            _put_turn_checkpoint(
                                checkpointer=self._checkpointer,
                                execution_id=exec_id,
                                agent_name=self.name,
                                iteration=_loop_end_turn(tool_calls, _start_iteration),
                                messages=llm_messages,
                                middleware_state=(
                                    mw_ctx.saved_state() if mw_ctx is not None else None
                                ),
                            )
                        except Exception:  # pragma: no cover - best effort, like the marker
                            logger.debug("loop-end checkpoint failed", exc_info=True)
                    stopped_by = run_control.stop_of(exc, control)
                    write_run_end(
                        self._checkpointer,
                        execution_id=exec_id,
                        chain_name=self.name,
                        status="failed",
                        # A cancel or timeout arrives as ``CancelledError``; the
                        # row records what it was (1.91.0, decision B2-8).
                        error=control.stop_error() if stopped_by and control else exc,
                        agent_path=new_path,
                        middleware_state=_end_state(mw_ctx, strict=False),
                        reason=stopped_by,
                    )
                raise

        except BaseException as exc:
            # ``on_error`` (1.91.0, 1A/1E): once, whatever killed the run — an
            # input guardrail, the loop, a re-ask, a hook. It observes; the run
            # still fails with ``exc`` (``notify_error`` never raises).
            if mw_ctx is not None and not isinstance(exc, StopAgent):
                stopped = run_control.stop_of(exc, control)
                await self._mw_pipeline.notify_error(
                    mw_ctx, control.stop_error() if stopped and control else exc
                )
            raise
        finally:
            unbind_cache(cache_tokens)
            _agent_path.reset(ap_token)
            _execution_id.reset(exec_token)
            reset_active_run_context(rc_token)
            reset_prompt_provenance(pv_token)
            stop_firing_collection(gf_token)
            stop_run_cost(rcost_token)

    async def astream(
        self,
        input: AgentInput,
        *,
        context: RunContext[Any] | None = None,
        trace: bool = True,
        execution_id: str | None = None,
        messages: list[Message] | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Async streaming execution — yields StreamEvent objects as tokens arrive.

        ``input`` accepts the same shapes as :py:meth:`arun` (string,
        ``Image``, ``PDF``, or a mixed list).

        Runs input guardrails before streaming begins. Output guardrails
        run after streaming completes. Memory is updated at the end.

        When the agent has middleware or a checkpointer configured, those
        are forwarded to :func:`stream_tool_loop` so middleware hooks fire
        and checkpoints are written during streaming — matching the behavior
        of :meth:`arun`.

        ``trace`` (default ``True``) opens an ``agent.<name>`` root span around
        the whole streamed run, exactly as :meth:`arun` does. Before 1.67.0 the
        parameter was accepted and ignored: a streamed run produced ``llm.*``
        and tool spans with no agent root to hang them on, and the
        :class:`AgentResult` :meth:`stream` built from it carried no
        ``trace_id``, no ``tokens_used`` and an empty ``execution_id``. All
        three come from this span. Pass ``trace=False`` to stream inside a
        workflow that already owns the root span.

        Example:
            async for event in agent.astream("Hello"):
                if isinstance(event, TextDelta):
                    print(event.text, end="", flush=True)
        """
        from contextlib import nullcontext

        from fastaiagent.trace.otel import get_tracer

        exec_id = execution_id or str(uuid.uuid4())

        input_text = (
            input if isinstance(input, str) else _input_summary_text(normalize_input(input))
        )

        span_cm = (
            get_tracer().start_as_current_span(f"agent.{self.name}") if trace else nullcontext(None)
        )
        # ``stream()`` drives this generator, so the span opened here is current
        # in the driver's context — the same property the firing collection
        # relies on. That is what lets ``stream()`` read the run's identity out
        # of ``_stream_outcome`` below without ``astream`` having to return one.
        outcome = _stream_outcome.get()
        with span_cm as span:
            if span is not None:
                span.set_attribute("agent.name", self.name)
                span.set_attribute("fastaiagent.framework", "fastaiagent")
                span.set_attribute("agent.input", input_text)
                span.set_attribute("agent.streamed", True)
                if self._mw_pipeline:
                    span.set_attribute("fastaiagent.middleware", self._middleware_names())
                # Same link ``arun`` stamps: the run's checkpoints and any
                # resume of it are found by this id.
                span.set_attribute("agent.execution_id", exec_id)
                if outcome is not None:
                    outcome["trace_id"] = format(span.get_span_context().trace_id, "032x")
            if outcome is not None:
                outcome["execution_id"] = exec_id
            from fastaiagent.prompt.provenance import (
                reset_prompt_provenance,
                set_prompt_provenance,
            )

            # The inner generator leaves its middleware context here, so the
            # ``on_stream_event`` hook below sees the same run context.
            mw_holder: dict[str, MiddlewareContext] = {}
            filters_events = bool(self._mw_pipeline) and self._mw_pipeline.overrides(
                "on_stream_event"
            )
            inner = self._astream_inner(
                input,
                context=context,
                execution_id=exec_id,
                messages=messages,
                input_text=input_text,
                span=span,
                outcome=outcome,
                _mw_holder=mw_holder,
                **kwargs,
            )
            # This run's spend, which ``cost_limit`` reads (1.91.0, X3): before,
            # a streamed run had no scope, its spend read as unknown and the
            # fail-closed rule blocked every one. ``stream()`` opens a scope to
            # build its result, so share it when there is one.
            run_usage = current_run_usage() or _RunUsage()
            # Limits and cancel (1.91.0, 2I) for a top-level run. A generator is
            # driven by its consumer, so it cannot be stopped from outside
            # mid-call the way ``arun`` is: it stops at its next event. A nested
            # stream (a Swarm hop) runs under its parent's control.
            control: run_control.RunControl = self._run_control(exec_id, usage_bucket=run_usage)
            nested = _agent_path.get() is not None
            if not nested:
                run_control.add(control)
            try:
                while True:
                    # Bind this run's prompt provenance (see _arun_core), spend
                    # scope and stop control around each step, never across a
                    # yield: a generator runs in its consumer's context, so a
                    # value held between yields would stamp another stream
                    # advanced in the same task.
                    pv_token = set_prompt_provenance(self._prompt_provenance)
                    usage_token = bind_run_cost(run_usage)
                    ctl_token = run_control.bind(control)
                    try:
                        top = control.root()
                        if top.reason is None and top.overdue():
                            top.reason = run_control.TIMEOUT
                        if top.reason is not None:
                            # Ends the run where it stands: its run-end marker
                            # records the stop, as on ``arun``.
                            await inner.athrow(control.stop_error())
                        event = await inner.__anext__()
                    except StopAsyncIteration:
                        break
                    finally:
                        run_control.unbind(ctl_token)
                        stop_run_cost(usage_token)
                        reset_prompt_provenance(pv_token)
                    if filters_events and "ctx" in mw_holder:
                        # ``on_stream_event`` (1.91.0, 1A): rewrite or drop.
                        filtered = await self._mw_pipeline.apply_stream_event(
                            mw_holder["ctx"], event
                        )
                        if filtered is None:
                            continue
                        event = filtered
                    yield event
            except run_control._RunCancelled:
                if nested:
                    raise  # the top-level run ends it
                # A cancelled stream ends, it does not fail its consumer.
                if outcome is not None:
                    outcome["status"] = run_control.CANCELLED
                    outcome["stop_reason"] = run_control.CANCELLED
                    outcome["output"] = ""
                    outcome["messages"] = _conversation(
                        control.llm_messages or [], control.prefix_ids
                    )
                yield StreamDone(finish_reason=run_control.CANCELLED)
            finally:
                if not nested:
                    run_control.remove(control)
                await inner.aclose()

    async def _astream_inner(
        self,
        input: AgentInput,
        *,
        context: RunContext[Any] | None = None,
        execution_id: str,
        messages: list[Message] | None = None,
        input_text: str,
        span: Any = None,
        outcome: dict[str, Any] | None = None,
        _mw_holder: dict[str, MiddlewareContext] | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamEvent, None]:
        """The streamed run itself. Split out of :meth:`astream` only so the
        root span opened there wraps the whole generator body."""
        exec_id = execution_id
        for key, value in self.config._llm_kwargs().items():
            kwargs.setdefault(key, value)

        # Middleware context shared across the whole run.
        mw_ctx: MiddlewareContext | None = None
        if self._mw_pipeline:
            mw_ctx = MiddlewareContext(
                run_context=context, agent_name=self.name, execution_id=exec_id
            )
            if _mw_holder is not None:
                _mw_holder["ctx"] = mw_ctx
        stopped_at_start: StopAgent | None = None

        # Expose the RunContext before anything reads memory: a per-user Memory
        # resolves its user from it while the prompt is built below. It is
        # reset on every exit, including an input guardrail block.
        rc_token = set_active_run_context(context)
        try:
            # ``before_run`` before the input guardrails, as on ``arun``.
            if mw_ctx is not None:
                try:
                    input = await self._mw_pipeline.apply_before_run(mw_ctx, input)
                    parts: list[ContentPart] = (
                        [input] if isinstance(input, str) else normalize_input(input)  # type: ignore[arg-type]
                    )
                    input_text = input if isinstance(input, str) else _input_summary_text(parts)
                except StopAgent as stop:
                    stopped_at_start = stop

            # Effective guardrails = local + plane-authored (see arun for details).
            eff_guardrails = self._effective_guardrails()

            # Execute input guardrails (blocking) on the text portion.
            guarded_input: AgentInput = input
            if eff_guardrails and stopped_at_start is None:
                in_outcome = await execute_guardrails(
                    eff_guardrails, input_text, GuardrailPosition.input, agent_name=self.name
                )
                if in_outcome.modified:
                    guarded_input, input_text = _apply_input_rewrite(input, in_outcome)

            llm_messages = self._build_messages(
                guarded_input, context=context, history=messages, mw_ctx=mw_ctx
            )
            # ``AgentResult.messages`` (1.91.0, 2J) — and a cancelled run's.
            prefix = _prefix_ids(llm_messages, messages)
            # Only the top-level run's (a nested one shares its parent's control).
            stream_control = run_control.active_control() if _agent_path.get() is None else None
            if stream_control is not None:
                stream_control.llm_messages, stream_control.prefix_ids = llm_messages, prefix

            # Structured output, as ``output_mode`` asks (1.91.0, 2G).
            self._apply_structured(kwargs)
        except BaseException as exc:
            reset_active_run_context(rc_token)
            if mw_ctx is not None and not isinstance(exc, StopAgent):
                await self._mw_pipeline.notify_error(mw_ctx, exc)
            raise

        # Set up execution-scoped ContextVars so interrupt() / @idempotent
        # can find the active execution + checkpointer.
        pause_run_id = self._pause_owner_run_id(exec_id)  # before the rebind below
        exec_token = _execution_id.set(exec_id)
        parent_path = _agent_path.get()
        new_path = (
            f"{parent_path}/{self._agent_path_label}" if parent_path else self._agent_path_label
        )
        ap_token = _agent_path.set(new_path)
        cache_tokens = bind_cache(exec_id, self._checkpointer)
        # The run-end marker, as ``arun`` writes it (1.91.0, X1): without it a
        # finished stream left a ``completed`` step as its newest row, looked
        # crashed, and ``aresume`` re-ran it — model and tools again. Only the
        # run's owner marks it; a worker streamed inside a Swarm / Supervisor
        # does not. ``settled`` is set once nothing more may be written: the run
        # ended, or it paused (a marker after the ``interrupted`` row would hide
        # the pause from ``aresume``).
        marks_run = self._checkpointer is not None and parent_path is None
        settled = False
        loop_done = False

        try:
            if self._checkpointer is not None:
                self._checkpointer.setup()

            # Stream tool loop — yields events to caller
            accumulated_text = ""
            final_reply: dict[str, str] = {}
            streamed_tokens = 0
            try:
                # ``before_run`` stopped the run: its message is the whole reply.
                loop_events: AsyncGenerator[StreamEvent, None] = (
                    _one_event(_stream_stop(stopped_at_start, final_reply))
                    if stopped_at_start is not None
                    else stream_tool_loop(
                        llm=self.llm,
                        messages=llm_messages,
                        tools=self._run_tools(),
                        max_iterations=self.config.max_iterations,
                        tool_choice=self.config.tool_choice,
                        context=context,
                        guardrails=eff_guardrails or None,
                        mw_pipeline=self._mw_pipeline if self._mw_pipeline else None,
                        mw_ctx=mw_ctx,
                        checkpointer=self._checkpointer,
                        execution_id=exec_id,
                        agent_name=self.name,
                        agent_id=self.agent_id,
                        parallel_tools=self.config.parallel_tools,
                        max_parallel_tools=self.config.max_parallel_tools,
                        governance_run_id=pause_run_id,
                        final_reply=final_reply,
                        final_tool=self._final_tool(),
                        **kwargs,
                    )
                )
                # ``PartialOutput`` (1.91.0, 2G): the structured reply so far, from
                # the current turn's text — prose before a tool call is not JSON
                # and yields none; a turn ends at its tool calls or its usage.
                structured = self._output_spec is not None
                final_tool = self._final_tool()
                turn_text = ""
                last_partial: Any = None
                async for event in loop_events:
                    partial: Any = None
                    if isinstance(event, TextDelta):
                        accumulated_text += event.text
                        if structured:
                            turn_text += event.text
                            partial = partial_json(turn_text)
                    elif isinstance(event, Usage):
                        # The only place a stream reports token counts. Summed
                        # across every turn of the loop so a streamed run reports
                        # the same kind of number ``arun`` does.
                        streamed_tokens += event.prompt_tokens + event.completion_tokens
                        turn_text = ""
                    elif isinstance(event, ToolCallStart):
                        turn_text = ""
                    elif isinstance(event, ToolCallEnd) and event.tool_name == final_tool:
                        partial = dict(event.arguments)
                    yield event
                    if partial is not None and partial != last_partial:
                        last_partial = partial
                        yield PartialOutput(data=partial)
            except _AgentInterrupted as susp:
                # A tool paused the run (a managed approval policy, or interrupt())
                # and the pause is already checkpointed. Before 1.77.0 this escaped
                # the generator as a private exception (audit M4). End the stream
                # the way ``arun`` ends the run: one ``Paused`` event, then stop —
                # no output guardrails, no memory write, because there is no output.
                if span is not None:
                    span.set_attribute("agent.status", "paused")
                if outcome is not None:
                    outcome["tokens_used"] = streamed_tokens
                    outcome["output"] = ""
                    outcome["messages"] = _conversation(llm_messages, prefix)
                # Before the yield: a consumer that stops reading at ``Paused``
                # closes the generator right here.
                settled = True
                yield Paused(
                    reason=susp.reason,
                    context=susp.context,
                    execution_id=exec_id,
                    node_id=susp.node_id,
                    agent_path=susp.agent_path,
                )
                return
            loop_done = True

            # The answer is the last turn's reply after middleware — what ``run``
            # returns and stores — not every delta streamed: those include text
            # said before a tool call, and precede any ``after_model`` rewrite
            # (e.g. RedactPII). The deltas are the fallback when the loop ended
            # without a final reply.
            output = final_reply.get("text", accumulated_text)

            # Execute output guardrails. A streamed reply has already left the
            # building: every TextDelta was yielded to the caller as it arrived,
            # so a mask cannot un-emit the span it was meant to redact and a
            # re-ask cannot retract the text the caller already read. Both
            # degrade to a block here — the fail-closed reading, and the reason
            # a team that authors a rewriting rule should not stream.
            if eff_guardrails:
                # ``allow_reask=True`` only so the refusal below can explain
                # itself; nothing here can actually re-drive the model.
                out_outcome = await execute_guardrails(
                    eff_guardrails,
                    output,
                    GuardrailPosition.output,
                    allow_reask=True,
                    agent_name=self.name,
                )
                _refuse_stream_rewrite(out_outcome)

            # Store in memory (text summary for multimodal inputs).
            # Wrapped in a ``memory.write`` span (+ per-block children).
            if self.memory:
                from fastaiagent.agent._memory_tracing import traced_add
                from fastaiagent.llm.message import AssistantMessage

                # Re-assert this run's context for the write. A generator runs
                # in its consumer's context, so another stream advanced in the
                # same task between our yields may have replaced it — and a
                # per-user Memory writes to whichever user it resolves now.
                wr_token = set_active_run_context(context)
                try:
                    traced_add(self.memory, UserMessage(input_text))
                    traced_add(self.memory, AssistantMessage(output))
                finally:
                    reset_active_run_context(wr_token)

            if span is not None:
                span.set_attribute("agent.output", output)
                span.set_attribute("agent.tokens_used", streamed_tokens)
            if outcome is not None:
                outcome["tokens_used"] = streamed_tokens
                outcome["output"] = output
                if final_reply.get("stop_reason"):
                    outcome["stop_reason"] = final_reply["stop_reason"]
            # The end-of-stream marker, once, after output guardrails and the
            # memory write — so a consumer that sees it knows the run is done and
            # was not blocked. The tool loop drops the per-turn ``StreamDone``s so
            # a multi-turn run reads as one stream; until 1.88.0 nothing emitted
            # the final one, and a consumer waiting for it waited forever. A
            # paused run ends with ``Paused`` instead (returned above).
            # ``after_run`` (1.91.0, 1A), before the marker as on ``arun``. The
            # reply has been streamed already; what it returns is what
            # ``stream()`` returns.
            if mw_ctx is not None and self._mw_pipeline.overrides("after_run"):
                streamed_cost, streamed_cost_known = run_cost()
                finished = await self._mw_pipeline.apply_after_run(
                    mw_ctx,
                    AgentResult(
                        output=output,
                        tokens_used=streamed_tokens,
                        cost=streamed_cost,
                        cost_known=streamed_cost_known,
                        execution_id=exec_id,
                        status="completed",
                        trace_id=(outcome or {}).get("trace_id"),
                        guardrails=collected_firings(),
                        stop_reason=final_reply.get("stop_reason"),
                    ),
                )
                if outcome is not None:
                    outcome["output"] = finished.output
                    outcome["stop_reason"] = finished.stop_reason
            if outcome is not None:
                outcome["messages"] = _conversation(
                    llm_messages, prefix, str(outcome.get("output", output))
                )
            # The run is done before the caller hears so: mark it first, so a
            # consumer that stops reading at ``StreamDone`` leaves a finished run.
            if marks_run:
                write_run_end(
                    self._checkpointer,
                    execution_id=exec_id,
                    chain_name=self.name,
                    status="completed",
                    agent_path=new_path,
                    middleware_state=_end_state(mw_ctx, strict=True),
                )
            settled = True
            yield StreamDone()
        except BaseException as exc:
            # The run died — including a consumer that stopped reading mid-run
            # (``GeneratorExit``): it did not finish, and crash recovery is how
            # it can. Same rules as ``_arun_core``: re-raise untouched, the marker
            # is best-effort, and a failure AFTER the loop first gets a re-entry
            # row past the loop's last turn, so recovery re-asks the model rather
            # than re-running a tool that already ran. ``on_error`` hears it too.
            if mw_ctx is not None and not settled and not isinstance(exc, StopAgent):
                await self._mw_pipeline.notify_error(mw_ctx, exc)
            if marks_run and not settled:
                if loop_done:
                    self._stream_loop_end_checkpoint(exec_id, llm_messages, mw_ctx)
                write_run_end(
                    self._checkpointer,
                    execution_id=exec_id,
                    chain_name=self.name,
                    status="failed",
                    error=exc,
                    agent_path=new_path,
                    middleware_state=_end_state(mw_ctx, strict=False),
                    # A cancel or timeout (1.91.0, decision B2-8).
                    reason=run_control.stop_of(exc, stream_control),
                )
            raise
        finally:
            unbind_cache(cache_tokens)
            _agent_path.reset(ap_token)
            _execution_id.reset(exec_token)
            reset_active_run_context(rc_token)

    def _stream_loop_end_checkpoint(
        self,
        exec_id: str,
        messages: list[Message],
        mw_ctx: MiddlewareContext | None = None,
    ) -> None:
        """A turn row past the streamed loop's last turn, when it ended on a tool.

        ``arun`` derives that turn from its tool-call records; the stream path
        has none, so it reads the run's newest row instead. A pre-tool row
        (``turn:N/tool:X``) is what crash recovery would re-invoke — one charge
        becoming two — so it gets a ``turn:N+1`` row holding the full history. A
        turn row is already a safe re-entry point and is left alone.
        """
        store = self._checkpointer
        if store is None:
            return
        try:
            from fastaiagent.agent.executor import _put_turn_checkpoint

            last = store.get_last(exec_id)
            if last is None or "/tool:" not in last.node_id:
                return
            _put_turn_checkpoint(
                checkpointer=store,
                execution_id=exec_id,
                agent_name=self.name,
                iteration=int(last.state_snapshot.get("turn", 0)) + 1,
                messages=messages,
                middleware_state=mw_ctx.saved_state() if mw_ctx is not None else None,
            )
        except Exception:  # pragma: no cover - best effort, like the marker
            logger.debug("stream loop-end checkpoint failed", exc_info=True)

    def resume(
        self,
        execution_id: str,
        *,
        resume_value: Resume | None = None,
        context: RunContext[Any] | None = None,
        trace: bool = True,
        **kwargs: Any,
    ) -> AgentResult:
        """Synchronous resume wrapper around :meth:`aresume`."""
        return run_sync(
            self.aresume(
                execution_id,
                resume_value=resume_value,
                context=context,
                trace=trace,
                **kwargs,
            )
        )

    async def aresume(
        self,
        execution_id: str,
        *,
        resume_value: Resume | None = None,
        context: RunContext[Any] | None = None,
        agent_path_prefix: str | None = None,
        trace: bool = True,
        **kwargs: Any,
    ) -> AgentResult:
        """Resume a paused or crashed agent run.

        Three resume shapes:

        1. **Interrupted** (``status="interrupted"`` checkpoint, pending row
           in ``pending_interrupts``): pass ``resume_value=Resume(...)``.
           The pending row is atomically claimed; concurrent resumers see
           :class:`AlreadyResumed`. The suspended tool is re-invoked with
           ``_resume_value`` in scope so :func:`interrupt` returns the value,
           then the agent loop continues. **Exception:** a managed-policy
           approval pause (``policy_approval_required``) resumed with
           ``approved=False`` does not re-invoke the tool — the model receives
           the refusal as the tool's result, exactly as for a ``deny`` verdict.
        2. **Tool-boundary crash** (latest checkpoint is ``turn:N/tool:X``,
           ``status="completed"``): the saved tool is re-invoked with the
           saved args; the agent loop continues afterwards. The LLM is NOT
           re-called — the assistant's tool_calls are already in messages.
        3. **Turn-boundary crash** (latest checkpoint is ``turn:N``): the
           agent loop re-enters at iteration N, re-issuing the LLM call.
           Wrap side-effectful tool functions with ``@idempotent`` to avoid
           double-execution.

        ``agent_path_prefix`` (advanced): when this Agent is a worker inside
        a Supervisor / Swarm, multiple agents share one ``execution_id`` and
        the global ``get_last`` returns whichever sibling wrote most
        recently. Passing the agent's own path prefix
        (e.g. ``"supervisor:planner/worker:worker_b"``) scopes resume to
        the worker's own subtree.

        ``trace`` (default ``True``) opens an ``agent.<name>`` root span around
        the re-invoked tool and the continued loop — a **new trace per resume**,
        linked to the run by ``agent.execution_id`` and
        ``agent.resumed_execution_id``, and described by ``agent.resume.node_id``,
        ``agent.resume.kind`` (``approval`` / ``interrupt`` / ``tool_boundary`` /
        ``turn_boundary``) and, with a ``resume_value``, ``agent.resume.approved``,
        ``agent.resume.resolver`` (``metadata["resolver"]`` or ``["approver"]``)
        and ``agent.resume.reason`` (``metadata["reason"]``, a payload key). The
        result's ``trace_id`` names it, including when the run pauses again. A
        claim that fails (:class:`AlreadyResumed`, ``ChainCheckpointError``)
        opens no span. Before 1.89.0 a resume opened none at all: its LLM calls
        were orphan ``llm.*`` traces and the result's ``trace_id`` was ``None``.
        """
        from fastaiagent._internal.errors import ChainCheckpointError

        store: Checkpointer = self._checkpointer or SQLiteCheckpointer()
        store.setup()

        # Restore-anywhere (audit D4): when this machine has never seen the run
        # but the plane is holding it, pull it down first. It has to happen
        # BEFORE the pending-interrupt claim below — for an ``interrupted``
        # checkpoint the restore re-creates the pending row, and without it the
        # claim finds nothing and reports AlreadyResumed on a run nobody resumed.
        from fastaiagent.checkpointers.platform_replica import restore_if_missing

        restore_if_missing(store, execution_id)

        def _scoped_latest(*, status: str | None = None) -> Checkpoint | None:
            """Return the most recently committed checkpoint, optionally
            filtered to those whose ``agent_path`` starts with the prefix
            and (if given) whose ``status`` matches.

            Run-end markers are never returned: they are tombstones, not
            re-entry points (audit D5). The unfiltered branch routes through
            :func:`latest_resumable`, which is also what refuses to resume a run
            that already finished. The filtered branch skips them inline —
            a ``status=`` filter would already exclude a ``failed`` marker, but
            not a ``completed`` one.
            """
            if agent_path_prefix is None and status is None:
                return latest_resumable(store, execution_id, runner="Agent execution")
            for cp in reversed(store.list(execution_id, limit=500)):
                if is_run_end(cp):
                    continue
                if status is not None and cp.status != status:
                    continue
                if agent_path_prefix is not None and not (
                    (cp.agent_path or "").startswith(agent_path_prefix)
                ):
                    continue
                return cp
            return None

        if resume_value is not None:
            # Caller is resuming an interrupted run. Atomically claim the
            # pending row before doing anything else — concurrent resumers
            # / second clicks of "Approve" see AlreadyResumed.
            claimed = store.delete_pending_interrupt_atomic(execution_id)
            if claimed is None:
                raise AlreadyResumed(
                    f"Agent execution '{execution_id}' has no pending interrupt to "
                    "claim — either it was never suspended or another resumer won."
                )
            # Find the interrupted checkpoint corresponding to this pending
            # row (most recent ``interrupted`` row for this execution).
            interrupted = _scoped_latest(status="interrupted")
            if interrupted is None:
                raise AlreadyResumed(
                    f"Agent execution '{execution_id}' had a pending interrupt row "
                    "but no matching interrupted checkpoint — storage is inconsistent."
                )
            latest = interrupted
            # Best-effort: report the resolution to a connected plane (no-op when
            # not connected; never blocks/raises). Emitted only on the winning
            # claim, so a losing AlreadyResumed race reports no phantom resolution.
            try:
                from fastaiagent.governance import hitl_kind, resolution_context
                from fastaiagent.trace.hitl_export import record_resolution_event

                record_resolution_event(
                    run_id=execution_id,
                    node=claimed.node_id,
                    approved=resume_value.approved,
                    resolver=resume_value.metadata.get("resolver"),
                    reason=claimed.reason,
                    agent_id=self.name,
                    kind=hitl_kind(claimed.reason),
                    context=resolution_context(claimed.reason, claimed.context),
                )
            except Exception:
                logger.debug("HITL resolution emit failed", exc_info=True)
        else:
            latest_opt = _scoped_latest()
            if latest_opt is None:
                raise ChainCheckpointError(
                    f"No checkpoint found for agent execution '{execution_id}'"
                )
            if latest_opt.status == "interrupted":
                raise ChainCheckpointError(
                    f"Agent execution '{execution_id}' is suspended on interrupt(); "
                    "pass resume_value=Resume(...) to agent.resume()."
                )
            latest = latest_opt

        raw_msgs = latest.state_snapshot.get("messages", [])
        # Use the multimodal-aware deserializer so messages whose content
        # contains base64-encoded Image / PDF parts are rehydrated back into
        # dataclass instances before the LLM client sees them on resume.
        from fastaiagent.agent.executor import _deserialize_messages

        messages: list[Message] = _deserialize_messages(raw_msgs)
        start_iteration = int(latest.state_snapshot.get("turn", 0))
        is_tool_boundary = "/tool:" in latest.node_id

        # The resumed run's input — re-passed only so ``_arun_core`` can record
        # it in memory at the end of the run. The saved messages already carry
        # any multimodal content; memory only needs the text.
        original_input = _resumed_input_text(messages)

        def continuation() -> Coroutine[Any, Any, AgentResult]:
            return self._resume_from(
                latest,
                store=store,
                execution_id=execution_id,
                messages=messages,
                original_input=original_input,
                start_iteration=start_iteration,
                is_tool_boundary=is_tool_boundary,
                resume_value=resume_value,
                context=context,
                **kwargs,
            )

        if not trace:
            return await self._controlled(execution_id, continuation)

        # The claim above has been won (or there was none to win), so this IS a
        # run now — open its root span. Everything before this point can still
        # refuse, and a refused resume is not a run.
        from fastaiagent import governance

        if resume_value is not None:
            kind = (
                "approval" if latest.interrupt_reason == governance.APPROVAL_REASON else "interrupt"
            )
        else:
            kind = "tool_boundary" if is_tool_boundary else "turn_boundary"
        link: dict[str, Any] = {
            "agent.resumed_execution_id": execution_id,
            "agent.resume.node_id": latest.node_id,
            "agent.resume.kind": kind,
        }
        if resume_value is not None:
            meta = resume_value.metadata or {}
            resolver = meta.get("resolver") or meta.get("approver")
            reason = meta.get("reason")
            link["agent.resume.approved"] = bool(resume_value.approved)
            link["agent.resume.resolver"] = str(resolver) if resolver else None
            link["agent.resume.reason"] = str(reason) if reason else None
        with self._open_run_span(
            original_input, context, execution_id=execution_id, extra=link
        ) as span:
            # Limits and cancel hold for a resumed run too (1.91.0, 2I).
            result = await self._controlled(execution_id, continuation)
            self._close_run_span(span, result)
            return result

    async def _resume_from(
        self,
        latest: Checkpoint,
        *,
        store: Checkpointer,
        execution_id: str,
        messages: list[Message],
        original_input: str,
        start_iteration: int,
        is_tool_boundary: bool,
        resume_value: Resume | None,
        context: RunContext[Any] | None,
        **kwargs: Any,
    ) -> AgentResult:
        """Continue a claimed resume from ``latest``. Split out of :meth:`aresume`
        only so its root span wraps the tool re-invocation and the loop."""
        from fastaiagent._internal.errors import ChainCheckpointError

        # Match _arun_core's nestable ``_agent_path`` semantics so a Swarm /
        # Supervisor prefix is preserved; only the run's owner marks its end.
        parent_path = _agent_path.get()
        new_path = (
            f"{parent_path}/{self._agent_path_label}" if parent_path else self._agent_path_label
        )

        # One middleware context for the whole resume — the re-run tool call and
        # the continued loop share it (before 1.91.0 each made its own, so
        # ``scratch`` did not carry across). A resume is a run starting again, so
        # ``before_run`` fires, with the run's original input; its return is not
        # used — the saved history is what the run continues from.
        mw_ctx: MiddlewareContext | None = None
        stopped: StopAgent | None = None
        if self._mw_pipeline:
            mw_ctx = MiddlewareContext(
                run_context=context, agent_name=self.name, execution_id=execution_id
            )
            mw_ctx.turn = start_iteration
            # Middleware state as the run left it (1.91.0, 1B): a budget or a
            # loaded-tools list keeps counting across the pause or crash.
            mw_ctx.restore_state(latest.state_snapshot.get("middleware_state"))

        async def _resume_failed(exc: BaseException) -> None:
            # The pending row is already claimed, so a failure here must leave a
            # ``failed`` marker — before 1.91.0 the run kept ``interrupted`` as
            # its newest row: it looked paused and could never be resumed (X2).
            # Mark it on the store that held the pause; the continued loop's own
            # failures are marked (and reported) by ``_arun_core``.
            # A cancel or timeout during the re-invoked call (1.91.0, 2I).
            control = run_control.active_control()
            stop = run_control.stop_of(exc, control)
            error = control.stop_error() if stop and control is not None else exc
            if mw_ctx is not None and not isinstance(exc, StopAgent):
                await self._mw_pipeline.notify_error(mw_ctx, error)
            if parent_path is None:
                write_run_end(
                    store,
                    execution_id=execution_id,
                    chain_name=self.name,
                    status="failed",
                    error=error,
                    agent_path=new_path,
                    middleware_state=_end_state(mw_ctx, strict=False),
                    reason=stop,
                )

        if mw_ctx is not None:
            try:
                await self._mw_pipeline.apply_before_run(mw_ctx, original_input)
            except StopAgent as stop:
                stopped = stop
            except BaseException as exc:
                await _resume_failed(exc)
                raise

        if not is_tool_boundary or stopped is not None:
            # Pure turn-boundary resume — re-issue LLM at this iteration. Or
            # ``before_run`` stopped the resume: end it without the saved tool.
            return await self._arun_core(
                original_input,
                context=context,
                execution_id=execution_id,
                _resumed_messages=messages,
                _start_iteration=start_iteration,
                _stopped=stopped,
                _mw_ctx=mw_ctx,
                **kwargs,
            )

        # Tool-boundary or interrupted: re-invoke the saved tool, append
        # its ToolMessage, then continue the loop at iteration+1.
        tool_name = latest.state_snapshot.get("tool_name", "")
        tool_call_id = latest.state_snapshot.get("tool_call_id", "")
        tool_args = dict(latest.node_input or {})

        tool: Tool | None = None
        for t in self._run_tools():
            if t.name == tool_name:
                tool = t
                break
        if tool is None:
            raise ChainCheckpointError(
                f"Agent execution '{execution_id}' references tool {tool_name!r} "
                "which is not registered on this Agent. Pass the same tool list "
                "you used at run time."
            )

        rv_token = None
        if resume_value is not None:
            # Pending row already claimed above; just bind the resume value
            # so the suspended tool's interrupt() returns it on re-entry.
            rv_token = _resume_value.set(resume_value)
        # Read BEFORE ``_execution_id`` is rebound below (see the method).
        pause_run_id = self._pause_owner_run_id(execution_id)

        # Bind execution-scoped ContextVars exactly like _arun_core would —
        # the suspended tool may itself call interrupt() / @idempotent.
        exec_token = _execution_id.set(execution_id)
        ap_token = _agent_path.set(new_path)
        cache_tokens = bind_cache(execution_id, self._checkpointer)
        # A managed approval policy paused this call and the answer is no — the
        # app rejected it, or the blocking wait expired. The decision lives in
        # ``governance.gate_tool_call``, which this re-entry never passes through,
        # so without this the saved tool ran anyway (audit H1). Refuse the way a
        # ``deny`` verdict does: the model is told, the tool never runs.
        from fastaiagent import governance

        refused = (
            resume_value is not None
            and not resume_value.approved
            and latest.interrupt_reason == governance.APPROVAL_REASON
        )
        # Which turn's assistant message declared this call, and where in it —
        # what ``wrap_tool`` middleware sees as ``ctx.tool_call_index``.
        call_index = 0
        for m in reversed(messages):
            calls = getattr(m, "tool_calls", None) or []
            ids = [tc.id for tc in calls]
            if tool_call_id in ids:
                call_index = ids.index(tool_call_id)
                break
        from fastaiagent._internal.pricing import current_run_usage
        from fastaiagent.llm.message import ToolMessage

        # Collect what the re-run call records — guardrail firings, judge spend —
        # so the continued loop's scope can start from it (the result covers it).
        gf_token = start_firing_collection()
        rc_token = start_run_cost()
        resumed_firings: list[Any] = []
        resumed_usage: Any = None
        try:
            try:
                if refused:
                    tool_content: str | list[Any] = governance.denied(tool_name)
                else:
                    # The same governed path the loop uses (1.90.0, F5): guardrails
                    # on the saved — pre-guardrail — arguments and on the result,
                    # ``wrap_tool``, the tool span, timeout and retries. A Resume
                    # value means the pause was the decision, so governance is
                    # not asked again; crash recovery has none, and the pre-tool
                    # checkpoint is written before the gate, so it asks.
                    record: dict[str, Any] = {}
                    try:
                        tool_content = (
                            await _governed_tool_call(
                                tool,
                                tool_name,
                                tool_args,
                                governance_mode="resolved" if resume_value is not None else "gate",
                                agent_id=self.agent_id,
                                run_id=pause_run_id,
                                context=context,
                                guardrails=self._effective_guardrails() or None,
                                mw_pipeline=self._mw_pipeline if self._mw_pipeline else None,
                                mw_ctx=mw_ctx,
                                tool_call_index=call_index,
                                record=record,
                                agent_name=self.name,
                            )
                        ).content
                    except StopAgent as stop:
                        # As in the loop: answer the in-flight call with what the
                        # terminal produced, if anything; siblings are balanced below.
                        stopped = stop
                        in_flight = record.get("error") or record.get("output")
                        tool_content = "" if in_flight is None else str(in_flight)
            except _AgentInterrupted:
                # interrupt() was called again inside the resumed tool —
                # _record_agent_interrupt already wrote the new pending
                # row. Bubble paused state up.
                raise
            except InterruptSignal as sig:
                # The re-invoked tool paused. That is the normal case after a
                # crash: the process died inside a tool that had not reached its
                # ``interrupt()`` yet, so crash recovery re-runs it and it pauses
                # now. The tool loop turns this signal into a recorded pause;
                # until 1.88.0 this direct re-invocation did not, so the signal
                # escaped ``resume()`` with nothing checkpointed and every later
                # resume re-ran the tool and raised again. Record it exactly as
                # the loop would — same node, same snapshot — and return a pause.
                from fastaiagent.agent.executor import _record_agent_interrupt

                susp = _record_agent_interrupt(
                    checkpointer=store,
                    execution_id=execution_id,
                    agent_name=self.name,
                    iteration=start_iteration,
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                    tool_args=tool_args,
                    messages=messages,
                    sig=sig,
                    # As of before the re-run call, like the loop's own pauses.
                    middleware_state=latest.state_snapshot.get("middleware_state"),
                )
                return AgentResult(
                    output="",
                    execution_id=execution_id,
                    status="paused",
                    pending_interrupt={
                        "reason": susp.reason,
                        "context": susp.context,
                        "node_id": susp.node_id,
                        "agent_path": susp.agent_path,
                    },
                )
            except BaseException as exc:
                # The re-invoked call failed — a ``tool_call`` guardrail blocked
                # it, say (X2; see ``_resume_failed``).
                await _resume_failed(exc)
                raise
            if stopped is None or tool_content:
                messages.append(ToolMessage(content=tool_content, tool_call_id=tool_call_id))
        finally:
            resumed_firings = collected_firings()
            resumed_usage = current_run_usage()
            stop_run_cost(rc_token)
            stop_firing_collection(gf_token)
            unbind_cache(cache_tokens)
            _agent_path.reset(ap_token)
            _execution_id.reset(exec_token)
            if rv_token is not None:
                _resume_value.reset(rv_token)

        # The pre-tool checkpoint is written BEFORE dispatch, so when the pause
        # landed on the first of several parallel calls the snapshot holds an
        # assistant message declaring all of them and no results at all. We have
        # just answered the one that was suspended; its siblings are still
        # unanswered, and restarting at ``start_iteration + 1`` skips them
        # forever — a history OpenAI and Anthropic both 400 on, and this one is
        # persisted, so it survives the process that made it.
        #
        # They are answered, NOT re-dispatched: the resume machinery re-enters a
        # run at a checkpoint, not in the middle of a turn, and firing them here
        # would run side effects the model never saw a first result for.
        from fastaiagent.agent.executor import STOPPED_NOTE
        from fastaiagent.llm.message import balance_tool_messages

        messages = balance_tool_messages(
            messages, note=STOPPED_NOTE if stopped is not None else SIBLING_SKIPPED_NOTE
        )

        # Continue the loop at the next iteration with the updated
        # message history — or, when middleware stopped the run during the
        # resumed call, finish it the way the loop finishes a stopped run.
        try:
            return await self._arun_core(
                original_input,
                context=context,
                execution_id=execution_id,
                _resumed_messages=messages,
                _start_iteration=start_iteration + 1,
                _stopped=stopped,
                _seed_firings=resumed_firings,
                _seed_usage=resumed_usage,
                _mw_ctx=mw_ctx,
                **kwargs,
            )
        except _AgentInterrupted as susp:
            return AgentResult(
                output="",
                execution_id=execution_id,
                status="paused",
                pending_interrupt={
                    "reason": susp.reason,
                    "context": susp.context,
                    "node_id": susp.node_id,
                    "agent_path": susp.agent_path,
                },
            )

    async def afork(
        self,
        execution_id: str,
        *,
        checkpoint_id: str | None = None,
        input: AgentInput | None = None,
        context: RunContext[Any] | None = None,
        trace: bool = True,
        **kwargs: Any,
    ) -> AgentResult:
        """Fork an agent run from a saved checkpoint into a NEW execution.

        The sibling of :meth:`aresume`: instead of continuing the *same*
        ``execution_id``, ``afork`` branches under a **fresh** id so the
        original run is left intact.

        * Pass ``input`` to re-ask the run with a different request — the
          counterfactual "what if the user had asked X". The branch runs as a
          fresh, traced conversation under the new id.
        * Omit ``input`` to faithfully re-run the checkpoint's restored
          conversation *forward* under the new id (e.g. with a modified
          ``context``).

        ``checkpoint_id`` selects the step to branch from
        (``checkpointer.list(execution_id)`` lists ids); omit it for the last
        **executed** step — the run-end marker a finished run has written since
        1.65.0 is stepped over, because it is a tombstone rather than a step.
        Before 1.67.0 the branch inherited that marker: its origin row pointed at
        the tombstone and carried ``run_status`` into the new run's state.
        Returns an :class:`AgentResult` whose ``execution_id`` is the new forked
        id, linked to the source via ``parent_checkpoint_id``.

        A run held only by the plane is pulled down first, so a fork works on a
        machine that never saw the original run.

        Checkpoint-fork is the SDK primitive; trace-based counterfactual replay
        is the Enterprise plane's job (see :mod:`fastaiagent.trace.replay`).
        """
        from fastaiagent._internal.errors import ChainCheckpointError
        from fastaiagent.chain.checkpoint import Checkpoint

        store: Checkpointer = self._checkpointer or SQLiteCheckpointer()
        store.setup()
        # Restore-anywhere (audit D4) — ``aresume`` has done this since 1.65.0
        # and ``afork`` never did.
        from fastaiagent.checkpointers.platform_replica import restore_if_missing

        restore_if_missing(store, execution_id)
        # NOT ``latest_resumable``: that one raises AlreadyResumed on a finished
        # run, which is precisely what people fork. See ``latest_forkable``.
        base = latest_forkable(store, execution_id, checkpoint_id=checkpoint_id)
        if base is None:
            raise ChainCheckpointError(
                f"No checkpoint found to fork for agent execution '{execution_id}'"
                + (f" / checkpoint '{checkpoint_id}'" if checkpoint_id else "")
                + " — the run holds no executed step to branch from."
            )

        fork_id = str(uuid.uuid4())
        # Lineage marker linking the branch back to the source checkpoint.
        store.put(
            Checkpoint(
                checkpoint_id=str(uuid.uuid4()),
                parent_checkpoint_id=base.checkpoint_id or None,
                chain_name=self.name,
                execution_id=fork_id,
                node_id="__fork_origin__",
                node_index=base.node_index,
                step_type="fork_origin",
                status="completed",
                state_snapshot=dict(base.state_snapshot),
                agent_path=base.agent_path,
            )
        )

        # A fork is a run, and a run that ends leaves a tombstone (audit D5).
        # ``_arun_core`` writes one — but only when the AGENT carries a
        # checkpointer, while ``afork`` runs off ``self._checkpointer or
        # SQLiteCheckpointer()``. So a fork could be durable while the agent was
        # not: the lineage row landed in a real store and the branch was closed
        # by nobody, leaving a dead fork indistinguishable from one that simply
        # stopped. ``write_run_end_once`` reads the store rather than guessing,
        # so the delegating case still ends with exactly one row.
        #
        # ``agent_path`` is this agent's own label, not ``base.agent_path``: the
        # row describes the RUN, so it carries the path of whoever owns it. The
        # source checkpoint may have been written by an agent nested under a
        # swarm, and inheriting that path would name the branch after a topology
        # it is no longer part of. Same rule ``write_run_end`` documents.
        try:
            result = await self._fork_branch(
                base, fork_id, input=input, context=context, trace=trace, **kwargs
            )
        except BaseException as exc:
            write_run_end_once(
                store,
                execution_id=fork_id,
                chain_name=self.name,
                status="failed",
                error=exc,
                agent_path=self._agent_path_label,
            )
            raise
        # A paused branch has not ended, and a row after its ``interrupted`` one
        # would hide the pause from ``aresume``'s status guard.
        if result.status == "completed":
            write_run_end_once(
                store,
                execution_id=fork_id,
                chain_name=self.name,
                status="completed",
                agent_path=self._agent_path_label,
            )
        return result

    async def _fork_branch(
        self,
        base: Checkpoint,
        fork_id: str,
        *,
        input: AgentInput | None,
        context: RunContext[Any] | None,
        trace: bool = True,
        **kwargs: Any,
    ) -> AgentResult:
        """Run one forked branch. Split out of :meth:`afork` only so the marker
        above wraps both of its shapes.

        Both shapes are traced as a run of the fork id, linked back to the source
        by ``agent.forked_execution_id`` / ``agent.fork.of_execution`` /
        ``agent.fork.checkpoint_id``. Until 1.89.0 the no-input shape called the
        loop directly and opened no span at all."""
        link: dict[str, Any] = {
            "agent.forked_execution_id": fork_id,
            "agent.fork.of_execution": base.execution_id,
            "agent.fork.checkpoint_id": base.checkpoint_id or None,
        }
        if input is not None:
            # Counterfactual input: re-run with the new request under the fork id.
            if not trace:
                return await self.arun(
                    input, context=context, execution_id=fork_id, trace=False, **kwargs
                )
            return await self._arun_traced(
                input, context=context, execution_id=fork_id, extra=link, **kwargs
            )

        # No input change: re-run the restored conversation forward.
        from fastaiagent.agent.executor import _deserialize_messages

        messages = _deserialize_messages(base.state_snapshot.get("messages", []))
        # A pre-tool checkpoint's snapshot holds the assistant message that
        # declares a set of tool calls and none of their results — a history both
        # providers reject. Commit 2 caught it at the provider boundary and
        # warned; repairing it here is repairing it at source, and a fork needs
        # its own note: nothing was interrupted, the branch just starts earlier.
        from fastaiagent.llm.message import balance_tool_messages

        messages = balance_tool_messages(messages, note=FORK_SKIPPED_NOTE)
        start_iteration = int(base.state_snapshot.get("turn", 0))
        original_input = _resumed_input_text(messages)
        # The branch continues from the checkpoint, middleware state included
        # (a budget spent before the fork point stays spent).
        mw_state = base.state_snapshot.get("middleware_state")
        if not trace:
            return await self._arun_core(
                original_input,
                context=context,
                execution_id=fork_id,
                _resumed_messages=messages,
                _start_iteration=start_iteration,
                _mw_state=mw_state,
                **kwargs,
            )
        with self._open_run_span(original_input, context, execution_id=fork_id, extra=link) as span:
            result = await self._arun_core(
                original_input,
                context=context,
                execution_id=fork_id,
                _resumed_messages=messages,
                _start_iteration=start_iteration,
                _mw_state=mw_state,
                **kwargs,
            )
            self._close_run_span(span, result)
            return result

    def fork(
        self,
        execution_id: str,
        *,
        checkpoint_id: str | None = None,
        input: AgentInput | None = None,
        context: RunContext[Any] | None = None,
        trace: bool = True,
        **kwargs: Any,
    ) -> AgentResult:
        """Sync wrapper for :meth:`afork`."""
        return run_sync(
            self.afork(
                execution_id,
                checkpoint_id=checkpoint_id,
                input=input,
                context=context,
                trace=trace,
                **kwargs,
            )
        )

    def stream(
        self,
        input: str,
        *,
        context: RunContext[Any] | None = None,
        trace: bool = True,
        **kwargs: Any,
    ) -> AgentResult:
        """Synchronous streaming — collects stream into AgentResult.

        For true streaming, use ``astream()`` in an async context.
        """

        async def _collect() -> AgentResult:
            start = time.monotonic()
            text_parts: list[str] = []
            # Opened here rather than inside ``astream``: an async generator's
            # body runs in the context of whoever drives it, so firings recorded
            # while iterating land in this list — and ``astream`` itself returns
            # events, not an ``AgentResult`` to hang them on.
            gf_token = start_firing_collection()
            # Same trick, same reason: the streamed run's trace id, execution id
            # and token count are published into this dict by ``astream`` while
            # we iterate. Before 1.67.0 all three were simply absent from the
            # result a streamed run returned.
            outcome: dict[str, Any] = {}
            so_token = _stream_outcome.set(outcome)
            rcost_token = start_run_cost()
            try:
                async for event in self.astream(input, context=context, trace=trace, **kwargs):
                    if isinstance(event, TextDelta):
                        text_parts.append(event.text)
                latency = int((time.monotonic() - start) * 1000)
                # The run's answer — the last turn's reply after middleware,
                # what ``run`` returns (1.91.0, X13). Joining every delta also
                # took in text the model said before a tool call ("Let me
                # check."), so ``stream`` and ``run`` returned different answers
                # for the same run. The deltas stay the fallback.
                output = str(outcome.get("output", "".join(text_parts)))
                parsed = self._parse_output(output)
                cost, cost_known = run_cost()
                return AgentResult(
                    output=output,
                    parsed=parsed,
                    tokens_used=int(outcome.get("tokens_used", 0) or 0),
                    cost=cost,
                    cost_known=cost_known,
                    latency_ms=latency,
                    trace_id=outcome.get("trace_id"),
                    execution_id=str(outcome.get("execution_id", "") or ""),
                    guardrails=collected_firings(),
                    stop_reason=outcome.get("stop_reason"),
                    # A cancelled stream (1.91.0, 2I); the run's messages (2J).
                    status=str(outcome.get("status", "completed")),
                    messages=list(outcome.get("messages", [])),
                )
            finally:
                stop_run_cost(rcost_token)
                _stream_outcome.reset(so_token)
                stop_firing_collection(gf_token)

        return run_sync(_collect())

    def _resolve_system_prompt(self, context: RunContext[Any] | None = None) -> str:
        """Resolve system_prompt to a string. Calls it if callable."""
        if callable(self.system_prompt):
            return self.system_prompt(context)
        return self.system_prompt

    def _build_messages(
        self,
        input: AgentInput,
        context: RunContext[Any] | None = None,
        history: list[Message] | None = None,
        mw_ctx: MiddlewareContext | None = None,
    ) -> list[Message]:
        """Build the message array for the LLM.

        Middleware ``instructions`` (1.91.0, 1B) become one system message right
        after the system prompt and before memory — a leading system message, so
        ``TrimLongMessages`` keeps it.

        Accepts string, ``Image``, ``PDF``, or a list of those parts. When
        the input is multimodal, the trailing user message carries a
        ``list[ContentPart]`` that ``LLMClient`` later renders per provider.

        ``history`` (optional) is a list of prior-turn messages — e.g. a
        multi-turn conversation from :func:`fastaiagent.simulate`. When given,
        the turns are inserted after the system prompt + memory context and
        before the current user input. Default ``None`` reproduces the
        single-input behavior exactly.
        """
        messages: list[Message] = []

        system_text = self._resolve_system_prompt(context)
        if system_text:
            messages.append(SystemMessage(system_text))
        if mw_ctx is not None:
            instructions = self._mw_pipeline.instructions_text(mw_ctx)
            if instructions:
                messages.append(SystemMessage(instructions))
        structured = self._structured_instruction()
        if structured:
            messages.append(SystemMessage(structured))

        if isinstance(input, str):
            user_content: str | list[ContentPart] = input
            query_text = input
        else:
            parts = normalize_input(input)
            # Single-string lists collapse back to plain strings so the wire
            # shape is identical to the legacy text-only path.
            if len(parts) == 1 and isinstance(parts[0], str):
                user_content = parts[0]
                query_text = parts[0]
            else:
                user_content = parts
                query_text = _input_summary_text(parts)

        # Add memory context. ComposableMemory uses the query text to run
        # query-conditioned blocks (e.g. VectorBlock); AgentMemory ignores
        # it. Multimodal queries fall back to their text summary. The read is
        # wrapped in a ``memory.read`` span (+ per-block children) for observability.
        if self.memory:
            from fastaiagent.agent._memory_tracing import traced_get_context

            messages.extend(traced_get_context(self.memory, query_text))

        # Prior conversation turns (multi-turn simulation / chat history).
        if history:
            messages.extend(history)

        messages.append(UserMessage(user_content))
        return messages

    def as_mcp_server(
        self,
        transport: str = "stdio",
        expose_tools: bool = False,
        expose_system_prompt: bool = True,
        tool_name: str | None = None,
        tool_description: str | None = None,
    ) -> Any:
        """Expose this agent as an MCP server.

        Returns a :class:`fastaiagent.tool.mcp_server.FastAIAgentMCPServer`.
        Call ``await server.run()`` to start the stdio loop.

        Requires ``pip install 'fastaiagent[mcp]'``.

        Example::

            asyncio.run(agent.as_mcp_server(transport="stdio").run())  # run() is async
        """
        from fastaiagent.tool.mcp_server import FastAIAgentMCPServer

        return FastAIAgentMCPServer(
            target=self,
            transport=transport,  # type: ignore[arg-type]
            expose_tools=expose_tools,
            expose_system_prompt=expose_system_prompt,
            tool_name=tool_name,
            tool_description=tool_description,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to canonical format for platform push."""
        if self.prompt_slug:
            # A registry reference is the source of truth: send an empty inline
            # prompt so the plane links the agent to the governed slug (a runtime
            # callable system_prompt is fine here since it isn't serialized).
            system_prompt = ""
        else:
            if callable(self.system_prompt):
                raise ValueError(
                    f"Agent '{self.name}' has a callable system_prompt which cannot be "
                    f"serialized. Use a static string for agents pushed to the platform."
                )
            system_prompt = self.system_prompt
        d: dict[str, Any] = {
            "name": self.name,
            "agent_type": "single",
            "system_prompt": system_prompt,
            "llm_endpoint": self.llm.to_dict(),
            "tools": [t.to_dict() for t in self.tools],
            # Only *locally authored* guardrails belong in the agent's definition.
            # A plane-authored one (pulled via plane_guardrails_for_agent and then
            # passed in guardrails=[...]) must not be echoed back: the plane
            # upserts pushed guardrails by name and attaches them to this agent,
            # which would turn a domain-wide rule into one scoped to whichever
            # agents happened to echo it — silently dropping it for everyone else.
            "guardrails": [
                g.to_dict() for g in self.guardrails if getattr(g, "origin", "local") != "plane"
            ],
            "config": self.config._to_dict(),
        }
        # Governed-input fields — emitted only when configured so an agent with
        # neither is byte-identical to earlier versions (regression-safe).
        if self.prompt_slug:
            d["prompt_slug"] = self.prompt_slug
        if self.memory is not None:
            d["memory_enabled"] = True
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
            prompt_slug=data.get("prompt_slug"),
            llm=LLMClient.from_dict(data.get("llm_endpoint", {})),
            tools=[Tool.from_dict(t) for t in data.get("tools", [])],
            guardrails=[Guardrail.from_dict(g) for g in data.get("guardrails", [])],
            config=AgentConfig(**data.get("config", {})),
        )
