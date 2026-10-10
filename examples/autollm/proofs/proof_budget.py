"""Proof for "6 · The bill has a ceiling" on docs/evaluation/autollm-how-it-works.md.

Offline: a FunctionModel agent and a FunctionModel proposer, in a temporary local.db.

max_eval_runs that can't cover the baseline and the guard is refused before a
single call. A cap of 5 ends the run as "budget" — and the number of evaluation
passes actually written to local.db is at most 5.
"""

import json
import os
import sqlite3
import tempfile

tmp = tempfile.mkdtemp()
os.environ["FASTAIAGENT_LOCAL_DB"] = os.path.join(tmp, "local.db")

import fastaiagent as fa  # noqa: E402
from fastaiagent.testing import FunctionModel  # noqa: E402

cases = [{"input": f"q{i}", "expected_output": "yes"} for i in range(16)]


def answer(messages):
    system = " ".join(m.content for m in messages if m.role.value == "system")
    return "yes" if "Say yes." in system else "maybe"


proposer = FunctionModel(
    lambda m: json.dumps({"proposals": [{"system_prompt": "Say yes.", "rationale": "be decisive"}]})
)
agent = fa.Agent(name="decider", system_prompt="Decide.", llm=FunctionModel(answer))

try:
    fa.optimize(
        agent,
        cases,
        ["exact_match"],
        config=fa.OptimizeConfig(max_eval_runs=1),
        proposer_llm=proposer,
        persist=False,
    )
except ValueError as e:
    print("max_eval_runs=1 →", e, "\n")

report = fa.optimize(
    agent,
    cases,
    ["exact_match"],
    config=fa.OptimizeConfig(max_eval_runs=5, max_iterations=8, candidates_per_iteration=1),
    proposer_llm=proposer,
    run_name="budget-proof",
)
print(report.summary())
db = sqlite3.connect(os.environ["FASTAIAGENT_LOCAL_DB"])
(n,) = db.execute("select count(*) from eval_runs where run_name like 'budget-proof:%'").fetchone()
print(f"\nevaluation passes written to local.db: {n}  (cap was 5)")
