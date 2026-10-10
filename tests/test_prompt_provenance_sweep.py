"""Registry-prompt provenance on every run path (1.87.0).

An agent built from a registry ``Prompt`` stamps ``fastaiagent.prompt.*`` on each
``llm.*`` span however it runs: ``run``/``arun``, ``stream``/``astream``,
``aresume`` and ``afork``. A local registry prompt stamps ``name`` + ``version``
(the Local UI's prompt lineage reads ``prompt.name``); a control-plane prompt
adds ``slug`` + ``environment`` (Prompt Analytics). Before 1.87.0 only ``arun``
stamped, and only for control-plane prompts, so the lineage panel on the Local
UI's Prompts page was always empty.

No mocks: real OTel spans through the SDK's provider, a real ``PromptRegistry``
on a temp SQLite file, and the offline models in ``fastaiagent.testing``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

import fastaiagent as fa
from fastaiagent import FunctionTool, interrupt
from fastaiagent.chain.interrupt import Resume
from fastaiagent.checkpointers.sqlite import SQLiteCheckpointer
from fastaiagent.prompt import Prompt, PromptRegistry
from fastaiagent.testing.models import FunctionModel, TestModel
from fastaiagent.trace.otel import get_tracer_provider

PROMPT_KEYS = (
    "fastaiagent.prompt.name",
    "fastaiagent.prompt.version",
    "fastaiagent.prompt.slug",
    "fastaiagent.prompt.environment",
)


class _Collector(SpanExporter):
    def __init__(self) -> None:
        self.spans: list[tuple[str, dict[str, Any]]] = []

    def export(self, spans):  # type: ignore[override]
        for s in spans:
            self.spans.append((s.name, dict(s.attributes)))
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:  # pragma: no cover
        pass

    def llm(self) -> list[dict[str, Any]]:
        return [a for n, a in self.spans if n.startswith("llm.")]


@pytest.fixture()
def collector() -> _Collector:
    col = _Collector()
    get_tracer_provider().add_span_processor(SimpleSpanProcessor(col))
    return col


def _system_prompt(kind: str, tmp_path: Path) -> tuple[Any, dict[str, Any]]:
    """The agent's system prompt for ``kind`` and the attributes its llm spans carry."""
    if kind == "local":
        reg = PromptRegistry(path=str(tmp_path / "prompts.db"))
        reg.register("ticket-triage", "You triage tickets.")
        reg.register("ticket-triage", "You triage support tickets.")
        return reg.load("ticket-triage", version=2), {
            "fastaiagent.prompt.name": "ticket-triage",
            "fastaiagent.prompt.version": 2,
        }
    if kind == "platform":
        prompt = Prompt(
            name="acme",
            template="You are support.",
            version=7,
            slug="acme-support-system",
            source="platform",
            environment="production",
        )
        return prompt, {
            "fastaiagent.prompt.name": "acme",
            "fastaiagent.prompt.slug": "acme-support-system",
            "fastaiagent.prompt.version": 7,
            "fastaiagent.prompt.environment": "production",
        }
    return "You triage tickets.", {}


def _prompt_attrs(attrs: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in attrs.items() if k in PROMPT_KEYS}


# --------------------------------------------------------------------------- #
# Run paths
# --------------------------------------------------------------------------- #
def _run(agent: fa.Agent, tmp_path: Path) -> None:
    agent.run("hello")


def _arun(agent: fa.Agent, tmp_path: Path) -> None:
    asyncio.run(agent.arun("hello"))


def _stream(agent: fa.Agent, tmp_path: Path) -> None:
    agent.stream("hello")


def _astream(agent: fa.Agent, tmp_path: Path) -> None:
    async def _consume() -> None:
        async for _ in agent.astream("hello"):
            pass

    asyncio.run(_consume())


def _route(messages: list[Any]) -> Any:
    """Call the approval tool first; answer once its result is in."""
    if messages[-1].role.value == "tool":
        return f"done: {messages[-1].content}"
    return "", [{"name": "refund", "arguments": {"amount": 40}}]


def _asks_manager(amount: int) -> str:
    interrupt("manager_approval", {"amount": amount})
    return f"refunded {amount}"


def _paused_agent(system_prompt: Any, tmp_path: Path) -> fa.Agent:
    return fa.Agent(
        name="prov-resume",
        system_prompt=system_prompt,
        llm=FunctionModel(_route),
        tools=[FunctionTool(name="refund", fn=_asks_manager)],
        checkpointer=SQLiteCheckpointer(str(tmp_path / "ckpt.db")),
    )


PATHS = {"run": _run, "arun": _arun, "stream": _stream, "astream": _astream}


@pytest.mark.parametrize("kind", ["local", "platform", "plain"])
@pytest.mark.parametrize("path", sorted(PATHS))
def test_llm_span_carries_the_registry_prompt(
    kind: str, path: str, tmp_path: Path, collector: _Collector
) -> None:
    system_prompt, expected = _system_prompt(kind, tmp_path)
    agent = fa.Agent(name=f"prov-{kind}-{path}", system_prompt=system_prompt, llm=TestModel("ok"))

    PATHS[path](agent, tmp_path)

    llm = collector.llm()
    assert llm, "the run produced an llm span"
    assert _prompt_attrs(llm[-1]) == expected


@pytest.mark.parametrize("kind", ["local", "platform", "plain"])
def test_resumed_run_carries_the_registry_prompt(
    kind: str, tmp_path: Path, collector: _Collector
) -> None:
    system_prompt, expected = _system_prompt(kind, tmp_path)
    agent = _paused_agent(system_prompt, tmp_path)
    paused = agent.run("refund me")
    assert paused.status == "paused"

    collector.spans.clear()
    resumed = asyncio.run(agent.aresume(paused.execution_id, resume_value=Resume(approved=True)))

    assert resumed.output == "done: refunded 40"
    llm = collector.llm()
    assert llm, "the resumed run called the model again"
    assert _prompt_attrs(llm[-1]) == expected


@pytest.mark.parametrize("kind", ["local", "platform", "plain"])
def test_forked_run_carries_the_registry_prompt(
    kind: str, tmp_path: Path, collector: _Collector
) -> None:
    system_prompt, expected = _system_prompt(kind, tmp_path)
    agent = fa.Agent(
        name=f"prov-fork-{kind}",
        system_prompt=system_prompt,
        llm=TestModel("ok"),
        checkpointer=SQLiteCheckpointer(str(tmp_path / "ckpt.db")),
    )
    first = agent.run("hello")

    collector.spans.clear()
    asyncio.run(agent.afork(first.execution_id))  # no input → re-runs forward

    llm = collector.llm()
    assert llm, "the forked branch called the model"
    assert _prompt_attrs(llm[-1]) == expected


# --------------------------------------------------------------------------- #
# Boundaries: no leaking, no linking
# --------------------------------------------------------------------------- #
def test_an_agent_run_inside_another_does_not_inherit_its_prompt(
    tmp_path: Path, collector: _Collector
) -> None:
    """A plain agent used as a tool runs with no prompt, not its caller's."""
    system_prompt, expected = _system_prompt("local", tmp_path)
    inner = fa.Agent(
        name="inner", system_prompt="You look things up.", llm=TestModel("42", model="inner-model")
    )

    async def lookup(q: str) -> str:
        return (await inner.arun(q)).output

    def _outer_route(messages: list[Any]) -> Any:
        if messages[-1].role.value == "tool":
            return f"the answer is {messages[-1].content}"
        return "", [{"name": "lookup", "arguments": {"q": "x"}}]

    outer = fa.Agent(
        name="outer",
        system_prompt=system_prompt,
        llm=FunctionModel(_outer_route, model="outer-model"),
        tools=[FunctionTool(name="lookup", fn=lookup)],
    )
    assert outer.run("what is it?").output == "the answer is 42"

    by_model = {}
    for name, attrs in collector.spans:
        if name.startswith("llm."):
            by_model.setdefault(name, []).append(_prompt_attrs(attrs))
    assert by_model["llm.test.inner-model"] == [{}]
    assert all(a == expected for a in by_model["llm.test.outer-model"])


def test_interleaved_streams_keep_their_own_prompt(tmp_path: Path, collector: _Collector) -> None:
    """Two streams advanced in turn in ONE task never stamp each other's prompt."""
    system_prompt, expected = _system_prompt("local", tmp_path)
    a = fa.Agent(name="a", system_prompt=system_prompt, llm=TestModel("a", model="model-a"))
    b = fa.Agent(name="b", system_prompt="plain", llm=TestModel("b", model="model-b"))

    async def _interleave() -> None:
        sa, sb = a.astream("x"), b.astream("y")
        done = set()
        while len(done) < 2:
            for key, gen in (("a", sa), ("b", sb)):
                if key in done:
                    continue
                try:
                    await gen.__anext__()
                except StopAsyncIteration:
                    done.add(key)

    asyncio.run(_interleave())

    spans = {n: _prompt_attrs(at) for n, at in collector.spans if n.startswith("llm.")}
    assert spans["llm.test.model-a"] == expected
    assert spans["llm.test.model-b"] == {}


def test_a_local_prompt_never_links_the_pushed_agent(tmp_path: Path) -> None:
    """Only a control-plane prompt sets prompt_slug — the plane has no local slug."""
    local, _ = _system_prompt("local", tmp_path)
    platform, _ = _system_prompt("platform", tmp_path)

    local_agent = fa.Agent(name="l", system_prompt=local, llm=TestModel())
    platform_agent = fa.Agent(name="p", system_prompt=platform, llm=TestModel())

    assert local_agent.prompt_slug is None
    assert "prompt_slug" not in local_agent.to_dict()
    assert local_agent.to_dict()["system_prompt"] == "You triage support tickets."
    assert platform_agent.prompt_slug == "acme-support-system"


# --------------------------------------------------------------------------- #
# The Local UI's prompt lineage, end to end
# --------------------------------------------------------------------------- #
def test_prompts_page_lineage_finds_a_real_run(isolated_local_db: Path) -> None:
    """A real run + eval of a registry-prompt agent shows up on the Prompts page.

    The UI's own route tests seed ``fastaiagent.prompt.name`` by hand; this one
    lets the SDK write it — the only way to know the page is fed in practice.
    """
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from fastaiagent.eval import evaluate
    from fastaiagent.trace import otel
    from fastaiagent.ui.server import build_app

    # The span store captures its db path when the provider is built, and the
    # provider is a module singleton — rebuild it on the isolated local.db.
    otel.reset()
    try:
        reg = PromptRegistry()  # the isolated local.db the UI reads
        reg.register("ticket-triage", "You triage tickets.")
        agent = fa.Agent(
            name="triage", system_prompt=reg.load("ticket-triage"), llm=TestModel("billing")
        )

        run = agent.run("I was charged twice")
        results = evaluate(
            agent.arun,
            [{"input": "charged twice", "expected_output": "billing"}],
            ["exact_match"],
            persist=True,
            run_name="lineage",
        )
        get_tracer_provider().force_flush()
    finally:
        otel.reset()

    client = TestClient(build_app(db_path=str(isolated_local_db), no_auth=True))
    rows = client.get("/api/prompts").json()["rows"]
    assert {r["name"]: r["linked_trace_count"] for r in rows}["ticket-triage"] == 2

    lineage = client.get("/api/prompts/ticket-triage/lineage").json()
    assert run.trace_id in lineage["trace_ids"]
    assert results.run_id in lineage["eval_run_ids"]


def test_prompts_page_lineage_finds_a_langchain_run(isolated_local_db: Path) -> None:
    """``prompt_from_registry`` runs show up too — LangChain stamped only the slug,
    and the page matches on ``prompt.name``. Real LangChain, its own fake chat model."""
    pytest.importorskip("fastapi")
    pytest.importorskip("langchain_core")
    from fastapi.testclient import TestClient
    from langchain_core.globals import get_debug
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    from fastaiagent.integrations import langchain as lc
    from fastaiagent.trace import otel
    from fastaiagent.ui.server import build_app

    try:
        get_debug()
    except AttributeError:
        # langchain_core 0.3 reads ``langchain.debug`` when a ``langchain``
        # package is importable; a stray langchain 1.x (not one of our
        # dependencies) has none, and every LangChain call raises.
        pytest.skip("installed 'langchain' package does not match langchain-core")

    otel.reset()
    try:
        PromptRegistry().register("support-system", "Answer questions about {{topic}}.")
        chain = lc.prompt_from_registry("support-system") | FakeListChatModel(responses=["blue"])
        chain.invoke({"topic": "the sky"}, config={"callbacks": [lc.get_callback_handler()]})
        get_tracer_provider().force_flush()
    finally:
        otel.reset()

    client = TestClient(build_app(db_path=str(isolated_local_db), no_auth=True))
    lineage = client.get("/api/prompts/support-system/lineage").json()
    assert len(lineage["trace_ids"]) == 1
