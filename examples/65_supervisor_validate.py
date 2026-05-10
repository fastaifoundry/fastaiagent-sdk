"""Example 65: Hierarchical Supervisor — manager validates worker outputs.

When ``Supervisor(validate_outputs=True)`` is set, the supervisor LLM
inspects each worker's output before accepting it. If the output is
incomplete or off-topic, the worker is re-invoked once with the
manager's feedback appended to the original task.

This file is runnable as a pytest:
    pytest examples/65_supervisor_validate.py -v
No API keys required — uses ``fastaiagent.testing.FunctionModel`` to
drive both the worker and the validator deterministically.
"""

from fastaiagent.agent import Agent
from fastaiagent.agent.team import Supervisor, Worker
from fastaiagent.llm.message import Message, MessageRole
from fastaiagent.testing import FunctionModel


def test_supervisor_rejects_then_accepts_second_try() -> None:
    # Worker returns a sloppy answer first, a polished one second.
    state = {"worker_calls": 0}

    def worker_responder(_messages):
        state["worker_calls"] += 1
        if state["worker_calls"] == 1:
            return ("nope", [])
        return ("Paris is the capital of France.", [])

    worker = Worker(
        agent=Agent(name="researcher", llm=FunctionModel(worker_responder)),
        role="researcher",
        description="Looks up factual answers.",
    )

    # Supervisor LLM does two jobs:
    #   - On its tool-calling turn: delegate to the researcher.
    #   - On its validator turn:    approve or reject the worker's output.
    # We distinguish via the system prompt the validator path uses.
    sup_state = {"sup_calls": 0, "val_calls": 0}

    def supervisor_responder(messages: list[Message]):
        is_validator = any(
            m.role == MessageRole.system
            and "output reviewer" in (m.content or "").lower()
            for m in messages
        )
        if is_validator:
            sup_state["val_calls"] += 1
            if sup_state["val_calls"] == 1:
                # First try is too short — reject with feedback.
                return '{"approved": false, "feedback": "answer in a full sentence"}'
            return '{"approved": true}'

        sup_state["sup_calls"] += 1
        if sup_state["sup_calls"] == 1:
            return (
                "",
                [
                    {
                        "name": "delegate_to_researcher",
                        "arguments": {"task": "What is the capital of France?"},
                    }
                ],
            )
        return ("Final answer: Paris is the capital of France.", [])

    sup = Supervisor(
        name="manager",
        llm=FunctionModel(supervisor_responder),
        workers=[worker],
        validate_outputs=True,
        max_validation_retries_per_worker=1,
    )
    result = sup.run("What is the capital of France?")

    assert "Paris" in result.output
    assert state["worker_calls"] == 2  # worker re-ran exactly once
    assert sup_state["val_calls"] == 2  # validator was consulted both times
