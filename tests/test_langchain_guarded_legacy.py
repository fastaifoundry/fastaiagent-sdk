"""``with_guardrails`` guards the legacy model call methods (1.89.0, Phase 9).

Chat models and LLMs still carry their pre-Runnable call methods: ``generate``,
``agenerate``, ``generate_prompt``, ``agenerate_prompt`` and, on LangChain
0.x, ``predict``, ``apredict``, ``predict_messages``, ``apredict_messages``,
``call_as_llm`` and ``__call__``. On the guarded proxy they fell through
``__getattr__`` and called the model with no guardrail at all. Now each
prompt's text is checked before the call and each generation's text after it,
and the return type is unchanged. A prompt or result with no readable text
fails closed.

No mocks: real ``ChatOpenAI`` and ``OpenAI`` (completions) model classes talk
to a localhost server (``tests/_sse_provider.py``). Sweep: model class ×
every legacy method it has on the installed line × input block / output block /
pass. Run on both lines.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

pytest.importorskip("langchain_core")
lco = pytest.importorskip("langchain_openai")

from langchain_core.globals import get_debug, get_verbose  # noqa: E402

try:
    get_debug(), get_verbose()
except AttributeError:
    # langchain_core 0.3 reads ``langchain.debug`` when a ``langchain`` package is
    # importable; a stray langchain 1.x has none, and every Runnable call raises.
    pytest.skip(
        "installed 'langchain' package does not match langchain-core", allow_module_level=True
    )

from langchain_core.messages import HumanMessage  # noqa: E402
from langchain_core.prompt_values import StringPromptValue  # noqa: E402

from fastaiagent.integrations._registry import GuardrailBlocked  # noqa: E402
from tests._sse_provider import Reply, SSEProvider  # noqa: E402
from tests.test_langchain_guarded_surface import IN_RAIL, OUT_RAIL  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore")


@pytest.fixture(scope="module")
def server() -> Any:
    s = SSEProvider().start()
    try:
        yield s
    finally:
        s.stop()


def _chat_body(text: str) -> dict[str, Any]:
    return {
        "id": "chatcmpl-legacy",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "gpt-4o-mini",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 3, "total_tokens": 6},
    }


def _completion_body(text: str) -> dict[str, Any]:
    return {
        "id": "cmpl-legacy",
        "object": "text_completion",
        "created": 1700000000,
        "model": "gpt-3.5-turbo-instruct",
        "choices": [{"text": text, "index": 0, "finish_reason": "stop", "logprobs": None}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 3, "total_tokens": 6},
    }


def _chat(server: Any) -> Any:
    return lco.ChatOpenAI(
        model="gpt-4o-mini", api_key="sk-test-local", base_url=f"{server.url}/v1", max_retries=0
    )


def _llm(server: Any) -> Any:
    return lco.OpenAI(
        model="gpt-3.5-turbo-instruct",
        api_key="sk-test-local",
        base_url=f"{server.url}/v1",
        max_retries=0,
    )


def _args(kind: str, method: str, text: str) -> tuple[Any, ...]:
    """The first argument each legacy method takes, carrying ``text``."""
    if method in ("generate", "agenerate"):
        return ([[HumanMessage(content=text)]],) if kind == "chat" else ([text],)
    if method in ("generate_prompt", "agenerate_prompt"):
        return ([StringPromptValue(text=text)],)
    if method in ("predict_messages", "apredict_messages"):
        return ([HumanMessage(content=text)],)
    if method == "__call__" and kind == "chat":
        return ([HumanMessage(content=text)],)
    return (text,)


def _call(guarded: Any, method: str, args: tuple[Any, ...]) -> Any:
    fn = guarded if method == "__call__" else getattr(guarded, method)
    out = fn(*args)
    return asyncio.run(out) if asyncio.iscoroutine(out) else out


def _text(result: Any) -> str:
    if isinstance(result, str):
        return result
    if hasattr(result, "generations"):
        return result.generations[0][0].text
    return str(result.content)


METHODS = [
    "generate",
    "agenerate",
    "generate_prompt",
    "agenerate_prompt",
    "predict",
    "apredict",
    "predict_messages",
    "apredict_messages",
    "call_as_llm",
    "__call__",
]
CASES = [(kind, m) for kind in ("chat", "llm") for m in METHODS]


def _setup(server: Any, kind: str, method: str, reply_text: str) -> Any:
    from fastaiagent.integrations.langchain import with_guardrails

    model = _chat(server) if kind == "chat" else _llm(server)
    has = callable(model) if method == "__call__" else hasattr(model, method)
    if not has:
        pytest.skip(f"not on this LangChain line: {type(model).__name__}.{method} does not exist")
    body = _chat_body if kind == "chat" else _completion_body
    server.script(Reply(body=body(reply_text)))
    return with_guardrails(
        model, name="legacy-agent", input_guardrails=[IN_RAIL], output_guardrails=[OUT_RAIL]
    )


@pytest.mark.parametrize(("kind", "method"), CASES, ids=[f"{k}-{m}" for k, m in CASES])
def test_input_block_stops_the_legacy_call(server: Any, kind: str, method: str) -> None:
    guarded = _setup(server, kind, method, "all fine")
    with pytest.raises(GuardrailBlocked, match="input blocked by no_block_in"):
        _call(guarded, method, _args(kind, method, "please BLOCK_IN now"))
    assert server.requests == [], "the model was called although its input was blocked"


@pytest.mark.parametrize(("kind", "method"), CASES, ids=[f"{k}-{m}" for k, m in CASES])
def test_output_block_on_the_legacy_call(server: Any, kind: str, method: str) -> None:
    guarded = _setup(server, kind, method, "this is FORBIDDEN")
    with pytest.raises(GuardrailBlocked, match="output blocked by no_forbidden"):
        _call(guarded, method, _args(kind, method, "hello"))


@pytest.mark.parametrize(("kind", "method"), CASES, ids=[f"{k}-{m}" for k, m in CASES])
def test_a_passing_legacy_call_returns_the_models_own_type(
    server: Any, kind: str, method: str
) -> None:
    guarded = _setup(server, kind, method, "all fine")
    unguarded_type = {
        "generate": "LLMResult",
        "agenerate": "LLMResult",
        "generate_prompt": "LLMResult",
        "agenerate_prompt": "LLMResult",
        "predict": "str",
        "apredict": "str",
        "call_as_llm": "str",
    }
    out = _call(guarded, method, _args(kind, method, "hello"))
    assert _text(out) == "all fine"
    expected = unguarded_type.get(method)
    if expected is not None:
        assert type(out).__name__ == expected
    elif method in ("predict_messages", "apredict_messages") or kind == "chat":
        assert type(out).__name__ in ("AIMessage", "BaseMessage")


def test_every_generation_of_a_batched_generate_is_judged(server: Any) -> None:
    """Two prompts, the second answer blocked: the whole call is blocked."""
    from fastaiagent.integrations.langchain import with_guardrails

    server.script(Reply(body=_chat_body("all fine")), Reply(body=_chat_body("FORBIDDEN here")))
    guarded = with_guardrails(_chat(server), output_guardrails=[OUT_RAIL])
    with pytest.raises(GuardrailBlocked, match="output blocked"):
        guarded.generate([[HumanMessage(content="a")], [HumanMessage(content="b")]])


def test_an_unreadable_prompt_fails_closed_and_points_to_invoke(server: Any) -> None:
    from fastaiagent.integrations.langchain import with_guardrails

    server.script(Reply(body=_completion_body("all fine")))
    guarded = with_guardrails(_llm(server), input_guardrails=[IN_RAIL])
    with pytest.raises(GuardrailBlocked, match=r"could not run on generate\(\).*invoke\(\)"):
        guarded.generate([object()])
    assert server.requests == []


def test_a_legacy_method_the_model_lacks_still_raises_attribute_error(server: Any) -> None:
    from fastaiagent.integrations.langchain import with_guardrails

    model = _chat(server)
    if hasattr(model, "predict"):
        pytest.skip("not on this LangChain line: 0.x still has predict()")
    with pytest.raises(AttributeError):
        with_guardrails(model, input_guardrails=[IN_RAIL]).predict("hi")


def test_a_non_callable_wrapped_runnable_is_still_not_callable() -> None:
    from langchain_core.runnables import RunnableLambda

    from fastaiagent.integrations.langchain import with_guardrails

    with pytest.raises(TypeError, match="not callable"):
        with_guardrails(RunnableLambda(lambda x: x))("hi")
