"""``with_guardrails`` keeps its guard through builders and composition (1.89.0, Phase 9).

Before 1.89.0 every Runnable builder on the guarded proxy (``with_config``,
``bind``, ``with_retry``, ``with_fallbacks``, ``with_listeners``,
``with_types``, ``bind_tools``, ``configurable_fields``, ``model_copy``, ...)
fell through ``__getattr__`` and returned an **unguarded** runnable: the
guard silently came off. Composition (``guarded | x``, ``x | guarded``,
``RunnableSequence`` / ``RunnableParallel`` / a LangGraph node containing it)
raised ``TypeError``, because the proxy was not a Runnable.

Now a derived runnable is wrapped again with the same guardrails, and in
composition the proxy is the step itself, so the guard stays at that step's
own input and output.

Sweep: builder × call path × input block / output block / pass, on real
``langchain_core`` runnables, a real LangGraph graph and a real ``ChatOpenAI``
against a localhost server, on both LangChain lines.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

import pytest

pytest.importorskip("langchain_core")

from langchain_core.globals import get_debug, get_verbose  # noqa: E402

try:
    get_debug(), get_verbose()
except AttributeError:
    # langchain_core 0.3 reads ``langchain.debug`` when a ``langchain`` package is
    # importable; a stray langchain 1.x has none, and every Runnable call raises.
    pytest.skip(
        "installed 'langchain' package does not match langchain-core", allow_module_level=True
    )

from langchain_core.runnables import (  # noqa: E402
    RunnableLambda,
    RunnableParallel,
    RunnableSequence,
)

from fastaiagent.integrations._registry import GuardrailBlocked  # noqa: E402
from tests.test_langchain_guarded_surface import (  # noqa: E402
    IN_RAIL,
    OUT_RAIL,
    call_abatch,
    call_ainvoke,
    call_astream,
    call_astream_events,
    call_batch,
    call_invoke,
    call_stream,
)

AGENT = "guarded-builders-agent"


def _guard(runnable: Any) -> Any:
    from fastaiagent.integrations.langchain import with_guardrails

    return with_guardrails(
        runnable, name=AGENT, input_guardrails=[IN_RAIL], output_guardrails=[OUT_RAIL]
    )


def echo(x: str) -> str:
    return f"echo {x}"


def echo_dict(x: str) -> dict[str, str]:
    return {"text": f"echo {x}"}


def _ident() -> Any:
    return RunnableLambda(lambda x: x)


def _graph_node(g: Any) -> Any:
    lg = pytest.importorskip("langgraph.graph")

    graph = lg.StateGraph(dict)
    graph.add_node("guarded", g)
    graph.add_edge(lg.START, "guarded")
    graph.add_edge("guarded", lg.END)
    return graph.compile()


#: (id, body, build(guarded) -> runnable, input(text) -> call input)
BUILDERS: list[tuple[str, Callable[..., Any], Callable[[Any], Any], Callable[[str], Any]]] = [
    ("with_config", echo, lambda g: g.with_config(tags=["t"]), lambda t: t),
    ("bind", echo, lambda g: g.bind(), lambda t: t),
    ("with_retry", echo, lambda g: g.with_retry(stop_after_attempt=2), lambda t: t),
    (
        "with_fallbacks",
        echo,
        lambda g: g.with_fallbacks([RunnableLambda(lambda x: "fallback")]),
        lambda t: t,
    ),
    ("with_listeners", echo, lambda g: g.with_listeners(on_end=lambda run: None), lambda t: t),
    ("with_types", echo, lambda g: g.with_types(input_type=str), lambda t: t),
    (
        "chained_builders",
        echo,
        lambda g: g.with_config(tags=["t"]).bind().with_retry(),
        lambda t: t,
    ),
    ("pipe", echo, lambda g: g.pipe(_ident()), lambda t: t),
    ("or_right", echo, lambda g: g | _ident(), lambda t: t),
    ("or_left", echo, lambda g: _ident() | g, lambda t: t),
    ("function_left", echo, lambda g: (lambda x: x) | g, lambda t: t),
    ("sequence", echo, lambda g: RunnableSequence(_ident(), g, _ident()), lambda t: t),
    ("parallel", echo, lambda g: RunnableParallel(out=g), lambda t: t),
    ("map", echo, lambda g: g.map(), lambda t: [t]),
    ("as_tool", echo, lambda g: g.as_tool(name="echo", description="Echo."), lambda t: t),
    ("assign", echo_dict, lambda g: g.assign(n=lambda d: len(d["text"])), lambda t: t),
    ("pick", echo_dict, lambda g: g.pick("text"), lambda t: t),
]
B_IDS = [b[0] for b in BUILDERS]

CALL_PATHS = [
    call_invoke,
    call_ainvoke,
    call_stream,
    call_astream,
    call_batch,
    call_abatch,
    call_astream_events,
]
C_IDS = [c.__name__.removeprefix("call_") for c in CALL_PATHS]


def _built(builder: Any, body: Callable[..., Any] | None = None) -> Any:
    _, default_body, build, _ = builder
    return build(_guard(RunnableLambda(body or default_body)))


#: Every builder × call path, except map × astream_events: LangChain's
#: ``RunnableEach`` has no event stream on either line, so there is no such path
#: to guard. Left out rather than skipped (CI fails on any LangChain skip).
PAIRS = [
    pytest.param(builder, call, id=f"{bid}-{cid}")
    for builder, bid in zip(BUILDERS, B_IDS)
    for call, cid in zip(CALL_PATHS, C_IDS)
    if not (builder[0] == "map" and call is call_astream_events)
]


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize(("builder", "call"), PAIRS)
def test_input_block_survives_the_builder(builder: Any, call: Any) -> None:
    ran: list[Any] = []
    body = builder[1]

    def spy(x: str) -> Any:
        ran.append(x)
        return body(x)

    spy.__annotations__ = body.__annotations__
    with pytest.raises(GuardrailBlocked, match="input blocked by no_block_in"):
        call(_built(builder, spy), builder[3]("please BLOCK_IN now"))
    assert ran == [], "the wrapped runnable ran although its input was blocked"


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize(("builder", "call"), PAIRS)
def test_output_block_survives_the_builder(builder: Any, call: Any) -> None:
    with pytest.raises(GuardrailBlocked, match="output blocked by no_forbidden"):
        call(_built(builder), builder[3]("say FORBIDDEN"))


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize(("builder", "call"), PAIRS)
def test_a_passing_call_through_the_builder_returns_the_output(builder: Any, call: Any) -> None:
    out = call(_built(builder), builder[3]("hello"))
    assert "echo hello" in repr(out)


# -- the guard sits where it should ---------------------------------------------------


def test_a_step_keeps_the_guard_at_its_own_output() -> None:
    """A later step that scrubs the word must not hide the guarded step's output."""
    scrub = RunnableLambda(lambda s: s.replace("FORBIDDEN", "ok"))
    with pytest.raises(GuardrailBlocked, match="output blocked"):
        (_guard(RunnableLambda(echo)) | scrub).invoke("say FORBIDDEN")


def test_a_step_keeps_the_guard_at_its_own_input() -> None:
    """An earlier step that injects the word is caught at the guarded step's input."""
    inject = RunnableLambda(lambda s: f"{s} BLOCK_IN")
    with pytest.raises(GuardrailBlocked, match="input blocked"):
        (inject | _guard(RunnableLambda(echo))).invoke("hello")


def test_a_fallback_never_answers_for_a_blocked_call() -> None:
    fallback_calls: list[Any] = []

    def fallback(x: str) -> str:
        fallback_calls.append(x)
        return "fallback FORBIDDEN-free"

    with pytest.raises(GuardrailBlocked):
        _guard(RunnableLambda(echo)).with_fallbacks([RunnableLambda(fallback)]).invoke(
            "say FORBIDDEN"
        )
    assert fallback_calls == []


def test_a_retry_does_not_rerun_a_blocked_output() -> None:
    calls: list[Any] = []

    def body(x: str) -> str:
        calls.append(x)
        return f"echo {x}"

    with pytest.raises(GuardrailBlocked):
        _guard(RunnableLambda(body)).with_retry(stop_after_attempt=3).invoke("say FORBIDDEN")
    assert len(calls) == 1


def test_a_guarded_sequence_is_not_flattened_away() -> None:
    inner = RunnableSequence(RunnableLambda(echo), _ident())
    chain = _guard(inner) | _ident()
    assert len(chain.steps) == 2
    with pytest.raises(GuardrailBlocked):
        chain.invoke("say FORBIDDEN")


def test_builder_results_are_guarded_proxies_with_the_same_rails_and_name() -> None:
    from fastaiagent.integrations.langchain import _GuardedRunnable

    g = _guard(RunnableLambda(echo))
    for derived in (
        g.with_config(tags=["t"]),
        g.bind(),
        g.with_retry(),
        g.with_fallbacks([_ident()]),
        g.with_listeners(),
        g.with_types(input_type=str),
    ):
        assert isinstance(derived, _GuardedRunnable)
        assert derived._input_guardrails == [IN_RAIL]
        assert derived._output_guardrails == [OUT_RAIL]
        assert derived._fastaiagent_name == AGENT


def test_with_alisteners_keeps_the_guard() -> None:
    async def on_end(run: Any) -> None:
        return None

    derived = _guard(RunnableLambda(echo)).with_alisteners(on_end=on_end)
    with pytest.raises(GuardrailBlocked):
        asyncio.run(derived.ainvoke("say FORBIDDEN"))


# -- LangGraph: a guarded runnable as a node ----------------------------------------


def _state_echo(state: dict[str, Any]) -> dict[str, Any]:
    return {"output": f"echo {state['input']}"}


@pytest.mark.parametrize(
    "call", [call_invoke, call_ainvoke, call_stream, call_astream], ids=C_IDS[:4]
)
def test_a_guarded_langgraph_node_blocks_and_passes(call: Any) -> None:
    graph = _graph_node(_guard(RunnableLambda(_state_echo)))
    with pytest.raises(GuardrailBlocked, match="input blocked"):
        call(graph, {"input": "BLOCK_IN"})
    with pytest.raises(GuardrailBlocked, match="output blocked"):
        call(graph, {"input": "FORBIDDEN"})
    assert "echo hi" in repr(call(graph, {"input": "hi"}))


# -- a real chat model's own builders ---------------------------------------------------


@pytest.fixture
def chat_server() -> Any:
    from tests._sse_provider import SSEProvider

    server = SSEProvider().start()
    try:
        yield server
    finally:
        server.stop()


def _completion(text: str) -> dict[str, Any]:
    return {
        "id": "chatcmpl-b",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "gpt-4o-mini",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 3, "total_tokens": 6},
    }


CHAT_BUILDERS: list[tuple[str, Callable[[Any], Any]]] = [
    ("bind_tools", lambda m: m.bind_tools([echo])),
    ("configurable_fields", lambda m: m.configurable_fields()),
    ("model_copy", lambda m: m.model_copy()),
    ("with_config", lambda m: m.with_config(tags=["t"])),
    ("with_retry", lambda m: m.with_retry()),
]


@pytest.mark.parametrize("build", [b[1] for b in CHAT_BUILDERS], ids=[b[0] for b in CHAT_BUILDERS])
def test_chat_model_builders_keep_the_guard(chat_server: Any, build: Any) -> None:
    lco = pytest.importorskip("langchain_openai")
    from tests._sse_provider import Reply

    model = lco.ChatOpenAI(
        model="gpt-4o-mini",
        api_key="sk-test-local",
        base_url=f"{chat_server.url}/v1",
        max_retries=0,
    )
    derived = build(_guard(model))
    chat_server.script(Reply(body=_completion("this is FORBIDDEN")))
    with pytest.raises(GuardrailBlocked, match="output blocked"):
        derived.invoke("hi")
    chat_server.script(Reply(body=_completion("all fine")))
    assert derived.invoke("hi").content == "all fine"
