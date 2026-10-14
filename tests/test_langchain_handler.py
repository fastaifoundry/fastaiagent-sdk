"""The LangChain handler on streamed calls and content-block output (1.89.0, WS6).

A streamed LangChain call (``.stream()``, LangGraph ``stream_mode="messages"``,
``astream_events``, and the v3 streaming protocol) ends with an ``LLMResult``
that carries **no** ``llm_output``. The handler took the model name only from
``llm_output``, so it priced nothing: no ``fastaiagent.cost.total_usd`` on any
streamed call, on either LangChain line, and the plane priced those calls
itself (the over-report class ``trace/normalize.py`` documents).

Content-block output (``content=[{"type": "text", "text": ...}]``, which 1.x
models return more and more) came back from ``as_evaluable`` as the list's
repr, not its text.

Real ``langchain_core`` classes drive the real handler; runs on both lines.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any
from uuid import uuid4

import pytest

pytest.importorskip("langchain_core")

from langchain_core.messages import AIMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, LLMResult  # noqa: E402

SERIALIZED = {
    "id": ["langchain", "chat_models", "openai", "ChatOpenAI"],
    "kwargs": {"model_name": "gpt-4o-mini"},
}
USAGE = {"input_tokens": 1000, "output_tokens": 500, "total_tokens": 1500}


@pytest.fixture
def traced_db(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    from fastaiagent._internal import project as _project
    from fastaiagent._internal.config import reset_config
    from fastaiagent.trace import otel

    db_path = tmp_path / "local.db"
    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(db_path))
    for name in ("FASTAIAGENT_API_KEY", "FASTAIAGENT_TARGET"):
        monkeypatch.delenv(name, raising=False)
    reset_config()
    _project.set_project_id("test-proj")
    otel.reset()
    try:
        yield db_path
    finally:
        otel.reset()
        _project.reset_for_testing()
        reset_config()


def _llm_attrs(db: Any) -> dict[str, Any]:
    with sqlite3.connect(db) as conn:
        (attrs,) = conn.execute("SELECT attributes FROM spans WHERE name LIKE 'llm.%'").fetchone()
    return json.loads(attrs)


def _streamed_call(response_metadata: dict[str, Any] | None) -> None:
    """A chat call as streaming ends it: generations, no ``llm_output``."""
    from fastaiagent.integrations.langchain import get_callback_handler

    handler = get_callback_handler()
    run_id = uuid4()
    handler.on_chat_model_start(SERIALIZED, [[]], run_id=run_id)
    message = AIMessage(
        content="hello", usage_metadata=USAGE, response_metadata=response_metadata or {}
    )
    handler.on_llm_end(LLMResult(generations=[[ChatGeneration(message=message)]]), run_id=run_id)


def test_a_streamed_call_is_priced_from_the_response_model(traced_db: Any) -> None:
    _streamed_call({"model_name": "gpt-4o-mini-2024-07-18"})
    attrs = _llm_attrs(traced_db)
    assert attrs.get("gen_ai.usage.input_tokens") == 1000
    cost = attrs.get("fastaiagent.cost.total_usd")
    assert cost is not None and cost > 0, f"a streamed call carried no cost: {attrs}"


def test_a_streamed_call_is_priced_from_the_start_model(traced_db: Any) -> None:
    """No model anywhere in the response: the model seen at start prices it."""
    _streamed_call(None)
    cost = _llm_attrs(traced_db).get("fastaiagent.cost.total_usd")
    assert cost is not None and cost > 0


def test_a_completed_call_is_still_priced_from_llm_output(traced_db: Any) -> None:
    from fastaiagent.integrations.langchain import get_callback_handler

    handler = get_callback_handler()
    run_id = uuid4()
    handler.on_chat_model_start(SERIALIZED, [[]], run_id=run_id)
    message = AIMessage(content="hello", usage_metadata=USAGE)
    handler.on_llm_end(
        LLMResult(
            generations=[[ChatGeneration(message=message)]],
            llm_output={"model_name": "gpt-4o-mini"},
        ),
        run_id=run_id,
    )
    cost = _llm_attrs(traced_db).get("fastaiagent.cost.total_usd")
    assert cost is not None and cost > 0


def test_content_blocks_come_back_as_text(traced_db: Any) -> None:
    from langchain_core.globals import get_debug
    from langchain_core.runnables import RunnableLambda

    from fastaiagent.integrations import langchain as lc

    try:
        get_debug()
    except AttributeError:
        # langchain-core 0.3 under a langchain 1.x package: every runnable call
        # raises. Our pins no longer allow this mix (CI's langchain-lines job
        # fails if this skip ever fires); a hand-broken venv can still have it.
        pytest.skip("installed 'langchain' package does not match langchain-core")

    graph = RunnableLambda(
        lambda _: {"messages": [AIMessage(content=[{"type": "text", "text": "Paris"}])]}
    )
    assert lc.as_evaluable(graph)("q").output == "Paris"


def test_content_block_text_reaches_the_span(traced_db: Any) -> None:
    from fastaiagent.integrations.langchain import get_callback_handler

    handler = get_callback_handler()
    run_id = uuid4()
    handler.on_chat_model_start(SERIALIZED, [[]], run_id=run_id)
    message = AIMessage(content=[{"type": "text", "text": "Paris"}], usage_metadata=USAGE)
    handler.on_llm_end(LLMResult(generations=[[ChatGeneration(message=message)]]), run_id=run_id)
    attrs = _llm_attrs(traced_db)
    contents = [
        v for k, v in attrs.items() if "completion" in k or "response" in k or "output" in k
    ]
    assert any("Paris" in str(v) and "'type'" not in str(v) for v in contents), attrs
