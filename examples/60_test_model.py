"""Example 60: Deterministic agent tests with TestModel + FunctionModel.

Use ``fastaiagent.testing.TestModel`` and ``FunctionModel`` to swap a
real LLM for a canned response — no network, no API key, no flake.

Run via ``pytest``:
    pytest examples/60_test_model.py -v
"""

from fastaiagent import Agent, FunctionTool
from fastaiagent.testing import FunctionModel, TestModel

# --- Test 1: TestModel returns canned text ---------------------------------


def test_simple_answer() -> None:
    agent = Agent(name="hello", llm=TestModel(response="hi there"))
    assert agent.run("greet me").output == "hi there"


# --- Test 2: TestModel round-robins multiple responses ---------------------


def test_round_robin() -> None:
    llm = TestModel(response=["one", "two", "three"])
    agent = Agent(name="counter", llm=llm)
    outs = [agent.run("step").output for _ in range(3)]
    assert outs == ["one", "two", "three"]


# --- Test 3: FunctionModel for two-turn tool flows -------------------------


def test_tool_then_answer() -> None:
    state = {"calls": 0}

    def responder(messages):
        state["calls"] += 1
        if state["calls"] == 1:
            return ("", [{"name": "echo", "arguments": {"text": "boo"}}])
        return ("got: boo", [])

    def echo(text: str) -> str:
        """Echo the input back."""
        return text

    agent = Agent(
        name="echo-bot",
        llm=FunctionModel(responder),
        tools=[FunctionTool(name="echo", fn=echo)],
    )
    result = agent.run("say it")
    assert result.output == "got: boo"
    assert state["calls"] == 2  # exactly two LLM iterations
