"""Proof for "6 · Between a fix and the future" on docs/replay/replay-boundaries.md.

Offline: the SDK's own FunctionModel, no API key, a throwaway local.db.

A rerun is saved as a regression case in the JSONL shape evaluate() reads,
with the trail back to the trace it came from. evaluate() then runs the agent
against that case and scores it.
"""

import json
import os
import tempfile

TMP = tempfile.mkdtemp()
os.environ["FASTAIAGENT_LOCAL_DB"] = os.path.join(TMP, "local.db")

import fastaiagent as fa  # noqa: E402
from fastaiagent.testing import FunctionModel  # noqa: E402
from fastaiagent.trace import Replay  # noqa: E402

POLICY = "You are a support agent. Refunds are issued within 14 days."


def responder(messages):
    system = messages[0].content if messages[0].role.value == "system" else ""
    return "Refunds are issued within 14 days." if "14 days" in system else "Refunds take 30 days."


agent = fa.Agent(name="support", system_prompt=POLICY, llm=FunctionModel(responder))
r = agent.run("What is your refund policy?")

rerun = Replay.load(r.trace_id).fork_at(0).with_determinism("recorded", on_miss="error").rerun()
dataset = os.path.join(TMP, "regression_tests.jsonl")
rerun.save_as_test(
    dataset,
    input="What is your refund policy?",
    expected_output=str(rerun.new_output),
    source_trace_id=r.trace_id,
    fork_step=0,
    modifications={"prompt": POLICY},
)
record = json.loads(open(dataset).read().splitlines()[-1])
print("the saved case:")
for k, v in record.items():
    print(f"  {k:<16} {v!r}")

results = fa.evaluate(
    agent_fn=lambda text: agent.run(text).output,
    dataset=dataset,
    scorers=["exact_match"],
    persist=False,
)
score = results.scores["exact_match"][0]
print(f"evaluate() on the saved case: exact_match passed={score.passed} score={score.score}")

broken = fa.Agent(
    name="support", system_prompt="You are a support agent.", llm=FunctionModel(responder)
)
results = fa.evaluate(
    agent_fn=lambda text: broken.run(text).output,
    dataset=dataset,
    scorers=["exact_match"],
    persist=False,
)
score = results.scores["exact_match"][0]
print(f"evaluate() with the old prompt:  exact_match passed={score.passed} score={score.score}")
