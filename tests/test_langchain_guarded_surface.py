"""``with_guardrails`` covers the whole Runnable call surface (1.89.0, Phase 9).

Until 1.89.0 ``_GuardedRunnable`` overrode only ``invoke``, ``ainvoke``,
``stream``, ``astream`` and ``batch``. Everything else (``abatch``,
``astream_events``, ``stream_events``, ``astream_log``, ``transform``,
``atransform``, ``batch_as_completed``, ``abatch_as_completed``) fell through
``__getattr__`` to the wrapped runnable: no input check, no output check, no
``agent.name`` on the root span, and no signal.

The sweep runs every method × input guardrail failing / output guardrail
failing / both passing / no output guardrail, on real ``langchain_core``
runnables (a ``RunnableLambda`` and a compiled LangGraph graph), on both
LangChain lines. Where the output cannot be determined and output guardrails
are set, the call must fail closed.

Run it in the compat venvs: ``.langgraph-compat/matrix/l1/bin/python -m pytest``
(LangChain 1.x) and ``.../l0/bin/python`` (0.x).
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
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

from langchain_core.runnables import RunnableLambda  # noqa: E402

from fastaiagent.guardrail.guardrail import (  # noqa: E402
    Guardrail,
    GuardrailPosition,
    GuardrailType,
)
from fastaiagent.integrations._registry import GuardrailBlocked  # noqa: E402

AGENT = "guarded-surface-agent"


def _regex(name: str, pattern: str, position: GuardrailPosition) -> Guardrail:
    return Guardrail(
        name=name,
        guardrail_type=GuardrailType.regex,
        position=position,
        config={"pattern": pattern},
    )


IN_RAIL = _regex("no_block_in", r"BLOCK_IN", GuardrailPosition.input)
OUT_RAIL = _regex("no_forbidden", r"FORBIDDEN", GuardrailPosition.output)


def _echo() -> Any:
    return RunnableLambda(lambda x: f"echo {x}", name="echo")


def _guard(runnable: Any, *, output: bool = True) -> Any:
    from fastaiagent.integrations.langchain import with_guardrails

    return with_guardrails(
        runnable,
        name=AGENT,
        input_guardrails=[IN_RAIL],
        output_guardrails=[OUT_RAIL] if output else None,
    )


@pytest.fixture
def traced_db(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    from fastaiagent._internal import project as _project
    from fastaiagent._internal.config import reset_config
    from fastaiagent.integrations import langchain as lc
    from fastaiagent.trace import otel

    db_path = tmp_path / "local.db"
    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(db_path))
    for name in ("FASTAIAGENT_API_KEY", "FASTAIAGENT_TARGET"):
        monkeypatch.delenv(name, raising=False)
    reset_config()
    _project.set_project_id("test-proj")
    otel.reset()
    lc.enable()
    try:
        yield db_path
    finally:
        otel.get_tracer_provider().force_flush()
        otel.reset()
        _project.reset_for_testing()
        reset_config()


def _root_agent_names(db: Any) -> list[Any]:
    from fastaiagent.trace import otel

    otel.get_tracer_provider().force_flush()
    with sqlite3.connect(db) as conn:
        rows = conn.execute(
            "SELECT attributes FROM spans WHERE parent_span_id IS NULL OR parent_span_id = ''"
        ).fetchall()
    return [json.loads(r[0]).get("agent.name") for r in rows]


# -- one adapter per call path: (runnable, input) -> output ---------------------


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


def call_invoke(g: Any, x: Any) -> Any:
    return g.invoke(x)


def call_ainvoke(g: Any, x: Any) -> Any:
    return _run(g.ainvoke(x))


def call_stream(g: Any, x: Any) -> Any:
    return list(g.stream(x))


def call_astream(g: Any, x: Any) -> Any:
    async def go() -> list[Any]:
        return [c async for c in g.astream(x)]

    return _run(go())


def call_batch(g: Any, x: Any) -> Any:
    return g.batch([x])


def call_abatch(g: Any, x: Any) -> Any:
    return _run(g.abatch([x]))


def call_batch_as_completed(g: Any, x: Any) -> Any:
    return list(g.batch_as_completed([x]))


def call_abatch_as_completed(g: Any, x: Any) -> Any:
    async def go() -> list[Any]:
        return [p async for p in g.abatch_as_completed([x])]

    return _run(go())


def call_astream_events(g: Any, x: Any) -> Any:
    async def go() -> list[Any]:
        return [e async for e in g.astream_events(x, version="v2")]

    return _run(go())


def call_astream_events_v1(g: Any, x: Any) -> Any:
    async def go() -> list[Any]:
        return [e async for e in g.astream_events(x, version="v1")]

    return _run(go())


def call_astream_log(g: Any, x: Any) -> Any:
    async def go() -> list[Any]:
        return [p async for p in g.astream_log(x)]

    return _run(go())


def call_transform(g: Any, x: Any) -> Any:
    return list(g.transform(iter([x])))


def call_atransform(g: Any, x: Any) -> Any:
    async def source() -> Any:
        yield x

    async def go() -> list[Any]:
        return [c async for c in g.atransform(source())]

    return _run(go())


CALLS: list[Callable[[Any, Any], Any]] = [
    call_invoke,
    call_ainvoke,
    call_stream,
    call_astream,
    call_batch,
    call_abatch,
    call_batch_as_completed,
    call_abatch_as_completed,
    call_astream_events,
    call_astream_events_v1,
    call_astream_log,
    call_transform,
    call_atransform,
]
IDS = [c.__name__.removeprefix("call_") for c in CALLS]


@pytest.mark.parametrize("call", CALLS, ids=IDS)
def test_input_guardrail_blocks_every_call_path(call: Any) -> None:
    calls: list[Any] = []

    def body(x: Any) -> str:
        calls.append(x)
        return f"echo {x}"

    with pytest.raises(GuardrailBlocked, match="input blocked by no_block_in"):
        call(_guard(RunnableLambda(body)), "please BLOCK_IN now")
    assert calls == [], "the wrapped runnable ran although its input was blocked"


@pytest.mark.parametrize("call", CALLS, ids=IDS)
def test_output_guardrail_blocks_every_call_path(call: Any) -> None:
    with pytest.raises(GuardrailBlocked, match="output blocked by no_forbidden"):
        call(_guard(_echo()), "say FORBIDDEN")


@pytest.mark.parametrize("call", CALLS, ids=IDS)
def test_passing_guardrails_return_the_real_output(call: Any) -> None:
    out = call(_guard(_echo()), "hello")
    assert "echo hello" in repr(out)


@pytest.mark.parametrize("call", CALLS, ids=IDS)
def test_input_only_wrapper_never_fails_closed(call: Any) -> None:
    """No output guardrails: nothing to fail closed on, the output is not judged."""
    out = call(_guard(_echo(), output=False), "say FORBIDDEN")
    assert "FORBIDDEN" in repr(out)


@pytest.mark.parametrize("call", CALLS, ids=IDS)
def test_every_call_path_stamps_agent_name(call: Any, traced_db: Any) -> None:
    call(_guard(_echo()), "hello")
    assert AGENT in _root_agent_names(traced_db), (
        f"{call.__name__}: no root span carried agent.name={AGENT!r}"
    )


# -- events: the final event is held back, every other event is yielded ---------


def test_astream_events_holds_back_a_blocked_final_event() -> None:
    seen: list[str] = []

    async def go() -> None:
        async for ev in _guard(_echo()).astream_events("say FORBIDDEN", version="v2"):
            seen.append(ev["event"])

    with pytest.raises(GuardrailBlocked):
        _run(go())
    assert seen == ["on_chain_start", "on_chain_stream"], seen


def test_astream_events_yields_every_event_when_it_passes() -> None:
    async def go(g: Any) -> list[str]:
        return [ev["event"] async for ev in g.astream_events("hi", version="v2")]

    assert _run(go(_guard(_echo()))) == _run(go(_echo()))


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_a_filter_that_hides_the_root_fails_closed(version: str) -> None:
    async def go(g: Any) -> list[Any]:
        return [e async for e in g.astream_events("hello", version=version, include_names=["nope"])]

    with pytest.raises(GuardrailBlocked, match="could not run on astream_events"):
        _run(go(_guard(_echo())))
    # Without output guardrails the same filtered stream is fine.
    assert _run(go(_guard(_echo(), output=False))) == []


@pytest.mark.parametrize("call", ["batch", "abatch"])
def test_return_exceptions_items_are_not_judged(call: str) -> None:
    """A failed item has no output: its exception text is not judged as one."""

    def body(x: str) -> str:
        if x == "boom":
            raise ValueError("FORBIDDEN boom")
        return f"echo {x}"

    guarded = _guard(RunnableLambda(body))
    if call == "batch":
        outs = guarded.batch(["ok", "boom"], return_exceptions=True)
    else:
        outs = _run(guarded.abatch(["ok", "boom"], return_exceptions=True))
    assert outs[0] == "echo ok" and isinstance(outs[1], ValueError)


# -- a real LangGraph graph --------------------------------------------------------


def _graph() -> Any:
    lg = pytest.importorskip("langgraph.graph")
    from langchain_core.messages import AIMessage

    def node(state: dict[str, Any]) -> dict[str, Any]:
        return {"messages": [AIMessage(content=f"reply to {state['messages'][-1].content}")]}

    g = lg.StateGraph(lg.MessagesState)
    g.add_node("answer", node)
    g.add_edge(lg.START, "answer")
    g.add_edge("answer", lg.END)
    return g.compile()


def _msgs(text: str) -> dict[str, Any]:
    return {"messages": [("user", text)]}


@pytest.mark.parametrize(
    "call",
    [call_invoke, call_ainvoke, call_stream, call_astream, call_abatch, call_astream_events],
    ids=["invoke", "ainvoke", "stream", "astream", "abatch", "astream_events"],
)
def test_graph_output_is_checked(call: Any) -> None:
    with pytest.raises(GuardrailBlocked, match="output blocked"):
        call(_guard(_graph()), _msgs("FORBIDDEN"))
    assert call(_guard(_graph()), _msgs("fine")) is not None


def test_graph_astream_events_input_block_and_name(traced_db: Any) -> None:
    with pytest.raises(GuardrailBlocked, match="input blocked"):
        call_astream_events(_guard(_graph()), _msgs("BLOCK_IN"))
    call_astream_events(_guard(_graph()), _msgs("fine"))
    assert AGENT in _root_agent_names(traced_db)


def _v3_graph() -> Any:
    graph = _graph()
    if not hasattr(graph, "stream_events"):
        pytest.skip("not on this LangChain line: stream_events(version='v3') needs 1.x")
    return graph


def test_stream_events_v3_with_output_guardrails_fails_closed() -> None:
    graph = _v3_graph()
    with pytest.raises(GuardrailBlocked, match="version='v3'"):
        _guard(graph).stream_events(_msgs("fine"), version="v3")
    with pytest.raises(GuardrailBlocked, match="version='v3'"):
        _run(_guard(graph).astream_events(_msgs("fine"), version="v3"))


def test_stream_events_v3_input_only_checks_input_runs_and_names(traced_db: Any) -> None:
    graph = _v3_graph()
    with pytest.raises(GuardrailBlocked, match="input blocked"):
        _guard(graph, output=False).stream_events(_msgs("BLOCK_IN"), version="v3")
    run = _guard(graph, output=False).stream_events(_msgs("fine"), version="v3")
    for _ in run:
        pass
    assert run.output["messages"][-1].content == "reply to fine"
    assert AGENT in _root_agent_names(traced_db)


def test_astream_events_v3_input_only_runs_and_names(traced_db: Any) -> None:
    graph = _v3_graph()

    async def go() -> Any:
        run = await _guard(graph, output=False).astream_events(_msgs("fine"), version="v3")
        async for _ in run:
            pass
        return await run.output()

    assert _run(go())["messages"][-1].content == "reply to fine"
    assert AGENT in _root_agent_names(traced_db)


# -- a real chat model's stream: the output is the sum of its chunks ---------------


@pytest.fixture
def sse_provider() -> Any:
    from tests._sse_provider import SSEProvider

    server = SSEProvider().start()
    try:
        yield server
    finally:
        server.stop()


def _chat(sse_provider: Any, *texts: str) -> Any:
    lco = pytest.importorskip("langchain_openai")
    from tests._sse_provider import Reply, openai_stream

    sse_provider.script(Reply(events=openai_stream(*texts)))
    return lco.ChatOpenAI(
        model="gpt-4o-mini",
        api_key="sk-test-local",
        base_url=f"{sse_provider.url}/v1",
        max_retries=0,
        stream_usage=True,
    )


@pytest.mark.parametrize(
    "call", [call_stream, call_astream, call_astream_events], ids=IDS[2:4] + ["astream_events"]
)
def test_a_chat_stream_is_judged_on_its_whole_text(sse_provider: Any, call: Any) -> None:
    """The word is split across chunks and the last chunk is an empty usage delta:
    judging the last chunk alone (pre-1.89.0 ``stream``) saw ``""`` and passed."""
    with pytest.raises(GuardrailBlocked, match="output blocked"):
        call(_guard(_chat(sse_provider, "this is FORB", "IDDEN text")), "hi")
