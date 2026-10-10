"""Proof for "5 · The judge that picks doesn't grade" on docs/evaluation/autollm-how-it-works.md.

Offline: both checks happen before anything is scored.

Two judges that share a name are refused before anything runs; an unset audit
judge falls back to the selection judge, with a warning that says why that is
not a number to quote.
"""

import warnings

import fastaiagent as fa
from fastaiagent.eval import GEval
from fastaiagent.testing import FunctionModel

agent = fa.Agent(name="writer", system_prompt="Answer.", llm=FunctionModel(lambda m: "ok"))
cases = [{"input": f"q{i}"} for i in range(16)]

try:
    fa.optimize(
        agent,
        cases,
        [GEval(criteria="answer quality")],
        config=fa.OptimizeConfig(audit_judge=GEval(criteria="answer quality, audited")),
        persist=False,
    )
except ValueError as e:
    print("same name   →", e, "\n")

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    fa.OptimizeConfig(selection_judge=GEval(criteria="answer quality")).resolve_audit_judge()
print("audit unset →", caught[0].message)
