"""Can you trust your LLM judge? Calibrate it on human labels, then let it tune an agent.

An LLM judge is a prompt, and a prompt can be wrong. "Is this good customer
service?" passes a warm reply that promises a refund support can't grant, and
fails a blunt one that does exactly the right thing. Your reviewers know the
difference; the judge's prompt doesn't. So treat the judge like any other agent:

  Stage 1 — calibrate the judge. The judge is an ``Agent`` whose answer is a
  verdict; the dataset is replies your reviewers labelled pass/fail, each with a
  one-line note; the scorer is agreement with the reviewers. ``optimize()``
  rewrites the judge's prompt until it agrees with them. Then the tuned prompt
  becomes an ``LLMJudge(prompt_template=…)`` and is re-checked on 20 labelled
  replies the calibration never saw — the move from agent to scorer has to keep
  the gain, so it is measured, not assumed.

  Stage 2 (``--tune-agent``) — the calibrated judge tunes a support agent.
  No reference answers exist for "reply to this ticket", so the judge is the
  only signal: it is the ``selection_judge``. A different judge — a ``GEval``
  on a different model, with the policy as its steps — is the ``audit_judge``
  on the holdout, so the agent can't win by gaming the judge that picked it.

    zsh -lc 'python calibrate_judge.py'                # stage 1, ~3 minutes
    zsh -lc 'python calibrate_judge.py --tune-agent'   # both stages, ~10 minutes
    zsh -lc 'python calibrate_judge.py --try "My card was charged twice." \\
                                             "Sorry! I have refunded you."'
                                       # naive vs calibrated judge on any reply

Data: ``data/support_replies.jsonl`` — 80 replies to Acme Cloud tickets, half
passing the review policy, half failing it; labels and notes are the reviewers'.
The policy (a reply passes only if ALL hold): it answers the question or gives a
concrete next step; it promises no refund, credit or compensation (Billing
decides those); it commits to no date or time for a fix or a delivery; it never
asks for a password, card number or 2FA code. Tone doesn't matter.

UI: AutoLLM (both runs) · Eval Runs (every candidate, per case, with the
reviewers' note as the expected value).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import fastaiagent as fa
from fastaiagent.eval import GEval, LLMJudge
from fastaiagent.eval.scorer import Scorer, ScorerResult

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "support_replies.jsonl"
OUT = HERE / "out"
MODEL = os.environ.get("JUDGE_MODEL", "gpt-4.1-mini")
PROPOSER_MODEL = os.environ.get("JUDGE_PROPOSER_MODEL", "gpt-5")
AUDIT_MODEL = os.environ.get("JUDGE_AUDIT_MODEL", "gpt-4.1")

# The judge a team writes first: reasonable, and blind to the house policy.
NAIVE_JUDGE = """You review replies written by Acme Cloud support agents.
Read the customer's question and the agent's reply, and decide whether the reply
is good customer service.
Reply with JSON only: {"score": <1 if it is good, 0 if not>, "reasoning": "<one sentence>"}"""

# The template an LLMJudge renders per case — the judge agent's input, verbatim.
CASE = "Question: {input}\nReply: {output}"

POLICY_STEPS = [
    "Does the reply answer the question or give a concrete next step?",
    "Does it promise no refund, credit or compensation (Billing decides those)?",
    "Does it commit to no date or time for a fix or a delivery?",
    "Does it never ask for a password, a card number or a 2FA code?",
    "Score high only if every answer is yes; tone does not matter.",
]

SUPPORT_V1 = (
    "You are a support agent for Acme Cloud. Answer the customer's question helpfully "
    "and warmly in two or three sentences."
)


def llm(model: str) -> fa.LLMClient:
    return fa.LLMClient(provider="openai", model=model)


def rows() -> list[dict[str, Any]]:
    return [json.loads(line) for line in DATA.read_text().splitlines() if line.strip()]


def split(all_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Every fourth reply is held back from calibration entirely: 60 to tune on
    (AutoLLM splits those again), 20 fresh ones for the LLMJudge check."""
    fresh = [r for r in all_rows if int(r["id"][1:]) % 4 == 0]
    return [r for r in all_rows if r not in fresh], fresh


def verdict(text: str | None) -> int | None:
    try:
        score = json.loads(text or "")["score"]
        return 1 if float(score) >= 0.5 else 0
    except (ValueError, KeyError, TypeError):
        return None


class AgreesWithReviewers(Scorer):
    """Pass when the judge's verdict matches the reviewers'.

    The reason carries the reviewers' note — "promises a refund", "commits to a
    fix time" — which is what lets AutoLLM's proposer work out the policy.
    """

    name = "agrees_with_reviewers"

    def score(
        self, input: str, output: str, expected: str | None = None, **kw: Any
    ) -> ScorerResult:
        want = json.loads(expected or "{}")
        got = verdict(output)
        if got is None:
            return ScorerResult(score=0.0, passed=False, reason=f"no verdict: {output!r:.60}")
        if got == want["score"]:
            return ScorerResult(score=1.0, passed=True)
        said = {1: "pass", 0: "fail"}
        return ScorerResult(
            score=0.0,
            passed=False,
            reason=f"judge said {said[got]}; reviewers said {said[want['score']]}: {want['note']}",
        )


def as_llm_judge(rules: str, name: str) -> LLMJudge:
    return LLMJudge(
        prompt_template=f"{rules}\n\n{CASE}", scale="0-1", threshold=0.5, llm=llm(MODEL), name=name
    )


def agreement(judge: LLMJudge, cases: list[dict[str, Any]]) -> tuple[int, list[str]]:
    hits, misses = 0, []
    for r in cases:
        got = int(judge.score(input=r["question"], output=r["reply"]).passed)
        if got == r["human"]:
            hits += 1
        else:
            misses.append(f"{r['id']} {r['reply'][:60]!r} — reviewers: {r['note']}")
    return hits, misses


def calibrate() -> str:
    calibration, fresh = split(rows())
    judge = fa.Agent(name="reply-judge", system_prompt=NAIVE_JUDGE, llm=llm(MODEL))
    cases = [
        {
            "input": CASE.format(input=r["question"], output=r["reply"]),
            "expected_output": json.dumps({"score": r["human"], "note": r["note"]}),
        }
        for r in calibration
    ]
    report = fa.optimize(
        judge,
        cases,
        [AgreesWithReviewers()],
        config=fa.OptimizeConfig(max_iterations=5, patience=2, seed=0, max_eval_runs=30),
        proposer_llm=llm(PROPOSER_MODEL),
        run_name="reply judge — calibrated on reviewer labels",
    )
    print(report.summary())
    rules = report.best_candidate.system_prompt or NAIVE_JUDGE

    print(f"\nAs an LLMJudge, on {len(fresh)} labelled replies the calibration never saw:")
    for label, text in (("naive judge", NAIVE_JUDGE), ("calibrated judge", rules)):
        hits, misses = agreement(as_llm_judge(text, label.replace(" ", "_")), fresh)
        print(f"  {label:<17} agrees with the reviewers on {hits}/{len(fresh)}")
        for m in misses[:4]:
            print(f"      ✗ {m}")

    OUT.mkdir(exist_ok=True)
    (OUT / "policy_judge.txt").write_text(rules)
    print(f"\ncalibrated judge prompt → {OUT / 'policy_judge.txt'}")
    return rules


def tune_agent(rules: str) -> None:
    questions = sorted({r["question"] for r in rows()})
    agent = fa.Agent(name="acme-support", system_prompt=SUPPORT_V1, llm=llm(MODEL))
    report = fa.optimize(
        agent,
        [{"input": q} for q in questions],
        [],  # no reference answers: the judges are the only signal
        config=fa.OptimizeConfig(
            max_iterations=4,
            patience=2,
            seed=0,
            selection_judge=as_llm_judge(rules, "policy_judge"),
            audit_judge=GEval(
                name="policy_audit",
                criteria="Does the support reply follow Acme Cloud's support policy?",
                evaluation_steps=POLICY_STEPS,
                llm=llm(AUDIT_MODEL),
            ),
            max_eval_runs=20,
        ),
        proposer_llm=llm(PROPOSER_MODEL),
        run_name="support agent — tuned by the calibrated judge",
    )
    print(report.summary())
    if report.improved:
        print("\nthe agent's new prompt:\n" + (report.best_candidate.system_prompt or SUPPORT_V1))


def try_reply(question: str, reply: str) -> None:
    """The naive and the calibrated judge on one reply, side by side."""
    saved = OUT / "policy_judge.txt"
    if not saved.exists():
        print("no calibrated judge yet — run `python calibrate_judge.py` first.")
        return
    for label, rules in (("naive judge", NAIVE_JUDGE), ("calibrated judge", saved.read_text())):
        result = as_llm_judge(rules, label.replace(" ", "_")).score(input=question, output=reply)
        print(f"{label:<17} {'PASS' if result.passed else 'FAIL'}  — {result.reason}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tune-agent", action="store_true", help="also run stage 2")
    parser.add_argument(
        "--try", nargs=2, metavar=("QUESTION", "REPLY"), dest="try_", help="judge one reply"
    )
    args = parser.parse_args()
    if args.try_:
        try_reply(*args.try_)
        return
    rules = calibrate()
    if args.tune_agent:
        print("\n" + "=" * 60 + "\nStage 2 — the calibrated judge tunes a support agent\n")
        tune_agent(rules)


if __name__ == "__main__":
    main()
