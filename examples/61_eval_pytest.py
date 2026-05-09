"""Example 61: First-class pytest evals with @case and @pytest_dataset.

The fastaiagent pytest plugin (registered automatically when fastaiagent
is installed) lets you express evals in the same file as your unit
tests. Each tagged test:

- runs the agent against the case input,
- scores the output against ``expected`` using a configurable scorer,
- persists one ``eval_runs`` row to ``./.fastaiagent/local.db`` so the
  Local UI's ``/evals`` page picks it up automatically.

Run via ``pytest``:
    pytest examples/61_eval_pytest.py -v

The local UI then surfaces these as ``run_name="pytest::..."`` rows.
"""

from pathlib import Path

from fastaiagent import Agent
from fastaiagent.eval import case
from fastaiagent.eval import pytest_dataset as dataset
from fastaiagent.testing import TestModel

# --- @case: a single expected-output check ---------------------------------


@case(input="hello", expected="hi")
def test_greet(evaluate_one):  # type: ignore[no-untyped-def]
    agent = Agent(name="greeter", llm=TestModel(response="hi"))
    evaluate_one(agent.run, scorers=["exact_match"])


@case(input="capital of france", expected="paris")
def test_capital(evaluate_one):  # type: ignore[no-untyped-def]
    # Lowercase normalised in the responder so exact_match passes.
    agent = Agent(
        name="capital-bot",
        llm=TestModel(response="paris"),
    )
    evaluate_one(agent.run, scorers=["exact_match"])


# --- @pytest_dataset: parametrise over a JSONL file -----------------------

_DATASET_PATH = Path(__file__).with_name("_eval_pytest_data.jsonl")
_DATASET_PATH.write_text(
    '{"input": "ping", "expected_output": "pong"}\n'
    '{"input": "ack",  "expected_output": "ack"}\n',
    encoding="utf-8",
)


@dataset(_DATASET_PATH)
def test_dataset_cases(eval_case, evaluate_one):  # type: ignore[no-untyped-def]
    # The TestModel echoes whatever we tell it to per case via a small
    # FunctionModel-style closure.
    expected = eval_case["expected_output"]
    agent = Agent(name="dataset-bot", llm=TestModel(response=expected))
    evaluate_one(agent.run, scorers=["exact_match"])
