"""Regression + behavior tests for the additive ``Agent(...).run(messages=...)`` param.

These use ``TestModel``/``FunctionModel`` (real ``LLMClient`` subclasses, not
``unittest.mock``) so the multi-turn wiring is exercised end-to-end with no
network and no mocking.
"""

from __future__ import annotations

from fastaiagent.agent.agent import Agent
from fastaiagent.llm.message import AssistantMessage, MessageRole, UserMessage
from fastaiagent.testing.models import FunctionModel, TestModel


def test_default_path_unchanged() -> None:
    """``messages=None`` (default) must reproduce the single-input behavior:
    exactly [system?, user] reaches the model — no extra turns."""
    llm = TestModel(response="hi")
    agent = Agent(name="t", llm=llm, system_prompt="You are a bot.")

    result = agent.run("hello", trace=False)
    assert result.output == "hi"

    sent = llm.calls[-1]["messages"]
    roles = [m.role for m in sent]
    assert roles == [MessageRole.system, MessageRole.user]
    assert sent[-1].content == "hello"


def test_no_system_prompt_default_path() -> None:
    """Without a system prompt, the default path sends just the user message."""
    llm = TestModel(response="ok")
    agent = Agent(name="t", llm=llm)

    agent.run("hello", trace=False)

    sent = llm.calls[-1]["messages"]
    assert [m.role for m in sent] == [MessageRole.user]
    assert sent[-1].content == "hello"


def test_prior_turns_reach_the_model_in_order() -> None:
    """Prior turns passed via ``messages=`` land after the system prompt and
    before the current user input, in order."""
    captured: dict[str, list] = {}

    def responder(messages):
        captured["messages"] = list(messages)
        return "answer"

    agent = Agent(name="t", llm=FunctionModel(responder), system_prompt="sys")

    history = [
        UserMessage("first user turn"),
        AssistantMessage("first assistant turn"),
        UserMessage("second user turn"),
        AssistantMessage("second assistant turn"),
    ]
    result = agent.run("current input", messages=history, trace=False)
    assert result.output == "answer"

    sent = captured["messages"]
    roles = [m.role for m in sent]
    contents = [m.content for m in sent]

    # system, then the 4 history turns in order, then the current input last.
    assert roles == [
        MessageRole.system,
        MessageRole.user,
        MessageRole.assistant,
        MessageRole.user,
        MessageRole.assistant,
        MessageRole.user,
    ]
    assert contents == [
        "sys",
        "first user turn",
        "first assistant turn",
        "second user turn",
        "second assistant turn",
        "current input",
    ]


def test_empty_history_equals_default() -> None:
    """An empty ``messages=[]`` behaves like the default (no extra turns)."""
    llm = TestModel(response="ok")
    agent = Agent(name="t", llm=llm)

    agent.run("hi", messages=[], trace=False)

    sent = llm.calls[-1]["messages"]
    assert [m.role for m in sent] == [MessageRole.user]


async def test_arun_async_path() -> None:
    """The async entrypoint forwards ``messages=`` identically."""
    captured: dict[str, list] = {}

    def responder(messages):
        captured["messages"] = list(messages)
        return "answer"

    agent = Agent(name="t", llm=FunctionModel(responder))
    history = [UserMessage("hi"), AssistantMessage("hello")]

    result = await agent.arun("next", messages=history, trace=False)
    assert result.output == "answer"
    assert [m.content for m in captured["messages"]] == ["hi", "hello", "next"]


async def test_astream_threads_history() -> None:
    """``astream`` builds messages with history before streaming begins."""
    captured: dict[str, list] = {}

    def responder(messages):
        captured["messages"] = list(messages)
        return "streamed"

    agent = Agent(name="t", llm=FunctionModel(responder))
    history = [UserMessage("earlier")]

    chunks = []
    async for event in agent.astream("now", messages=history, trace=False):
        from fastaiagent.llm.stream import TextDelta

        if isinstance(event, TextDelta):
            chunks.append(event.text)

    assert "".join(chunks) == "streamed"
    assert [m.content for m in captured["messages"]] == ["earlier", "now"]
