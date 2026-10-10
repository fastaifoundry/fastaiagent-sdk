"""Proof for "4 · A crash is not a pass" on docs/evaluation/autollm-how-it-works.md.

Offline, no model at all.

Two candidates on the same six cases: one answers three and raises on the three
"hard" ones; the other answers all six and gets four right. evaluate() alone
leaves errored cases out of its pass rate; AutoLLM's candidate score counts them
as failures.
"""

import asyncio

from fastaiagent.eval.evaluate import aevaluate
from fastaiagent.eval.results import Scorecard
from fastaiagent.optimize import CandidateScore

cases = [{"input": f"q{i}", "expected_output": "yes"} for i in range(6)]


async def crashes_on_hard(q: str) -> str:
    if int(q[1:]) >= 3:
        raise RuntimeError("tool timeout")
    return "yes"


async def answers_all(q: str) -> str:
    return "yes" if int(q[1:]) < 4 else "no"


for name, fn in (
    ("crashes on 3 of 6, right on 3", crashes_on_hard),
    ("answers all 6, right on 4   ", answers_all),
):
    r = asyncio.run(aevaluate(fn, cases, ["exact_match"], persist=False))
    s = CandidateScore.from_eval("c", "dev", r, primary_metric=None)
    print(
        f"{name}  evaluate() pass-rate {Scorecard.from_eval_results(r).overall_pass_rate:.3f}"
        f"  |  AutoLLM score {s.score:.3f}  ({s.errored} errored of {s.n})"
    )
