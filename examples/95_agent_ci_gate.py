"""Example 95: Agent CI — gate a build on agent quality.

Your pytest run *is* the quality gate. Every ``evaluate_one`` case in a
session aggregates into ONE persisted eval run, thresholds gate the
aggregate, and a baseline catches regressions against a previous run.

Run it three ways to see all three outcomes:

    # 1. Green — the aggregate clears the bar (exit 0)
    pytest examples/95_agent_ci_gate.py --eval-fail-under "overall.pass_rate=0.5"

    # 2. Quality gate FAILS — 2 of 3 cases pass, bar is 0.9 (exit 1)
    pytest examples/95_agent_ci_gate.py --eval-fail-under "overall.pass_rate=0.9"

    # 3. Publish a baseline, then gate a later run against it
    pytest examples/95_agent_ci_gate.py --eval-run-name main
    pytest examples/95_agent_ci_gate.py --eval-baseline main --eval-tolerance 0.02

Everything runs on ``TestModel`` — deterministic, no API keys, no network.

The gate's terminal summary names exactly what missed::

    =========================== fastaiagent eval ===========================
    Scorecard
    exact_match            avg=0.67  pass_rate=67%  (n=3)
    --------------------------------------------------
    overall pass_rate=67%
    persisted run_id=8f2c… (Local UI: /evals/8f2c…)
    overall.pass_rate: 0.6667 < 0.9 required
    GATE: FAILED — quality threshold(s) missed

See docs/evaluation/agent-ci.md for the GitHub Actions recipe and the
production-failure → regression-test loop.
"""

from fastaiagent import Agent
from fastaiagent.eval import case
from fastaiagent.testing import TestModel

# --- Quality cases: these pass -------------------------------------------


@case(input="greet", expected="hi")
def test_greeting(evaluate_one):  # type: ignore[no-untyped-def]
    agent = Agent(name="support", llm=TestModel(response="hi"))
    evaluate_one(agent.run, scorers=["exact_match"])


@case(input="farewell", expected="bye")
def test_farewell(evaluate_one):  # type: ignore[no-untyped-def]
    agent = Agent(name="support", llm=TestModel(response="bye"))
    evaluate_one(agent.run, scorers=["exact_match"])


# --- A quality miss: the agent answers, but wrongly ------------------------
#
# ``assert_pass=False`` keeps the individual test green so the *aggregate*
# gate is what fails the build — the point of Agent CI. Drop the flag and
# this case also fails on its own, which is what you'd normally want.


@case(input="refund policy", expected="30 days")
def test_refund_policy(evaluate_one):  # type: ignore[no-untyped-def]
    agent = Agent(name="support", llm=TestModel(response="I'm not sure"))
    evaluate_one(agent.run, scorers=["exact_match"], assert_pass=False)


# --- Infra failure is NOT a quality miss ----------------------------------
#
# When the agent *raises* (provider 500, timeout, auth), the case is recorded
# as ERRORED: unscored, excluded from pass/fail, counted separately. Add
# ``--eval-max-error-rate 0.1`` and this run becomes INVALID rather than
# "failed" — during an outage, threshold misses are noise, and an outage must
# never report a green build either.
#
# Uncomment to see it:
#
# @case(input="lookup order", expected="shipped")
# def test_order_lookup(evaluate_one):
#     def broken_agent(_input):
#         raise TimeoutError("provider 500")
#     evaluate_one(broken_agent, scorers=["exact_match"])
