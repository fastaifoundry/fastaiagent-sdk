# Agent Hardening & Scorecard

A closed loop for making agents better: **auto-generate** test scenarios →
**run** them → roll up a **Scorecard** → ask `harden()` for **concrete fixes**.
It builds on [`simulate()`](../simulation/index.md) and the
[scorers](index.md) you already have.

## Auto-generate test scenarios

`generate_scenarios()` introspects an agent's name, system prompt, and tools and
asks an LLM to propose diverse multi-turn `Scenario`s — each with a simulated-user
persona and success / failure criteria — ready to pass straight to `simulate()`.

```python
from fastaiagent import Agent, LLMClient, generate_scenarios, simulate

llm = LLMClient(provider="openai", model="gpt-4o-mini")
agent = Agent(
    name="support",
    system_prompt="You are a support agent for an online store. Help with orders, refunds, shipping.",
    llm=llm,
)

scenarios = generate_scenarios(agent, n=5, llm=llm, focus="frustrated customers")
results = simulate(scenarios, agent)
print(results.summary())
```

`focus` optionally steers generation (e.g. `"adversarial users"`, `"edge cases"`).
Async variant: `agenerate_scenarios(...)`.

## Named metrics

Three metrics round out the AgentEval-style set (alongside the existing
`faithfulness`, `context_precision`/`recall`, `toxicity`, `bias`, `pii_leakage`,
`prompt_injection`, `moderation`, tool-call accuracy, and the LLM judge):

| Scorer name | What it measures | Needs |
| --- | --- | --- |
| `task_completion` | Did the response accomplish the user's task/goal? | input + output |
| `hallucination` | Fraction of output claims supported by the context (reuses the groundedness engine) | output + `context` |
| `reflection_quality` | Internal consistency / sound reasoning / appropriate hedging | input + output |

Use them by name in `evaluate()` or directly:

```python
from fastaiagent.eval import TaskCompletion, evaluate

evaluate(agent.run, dataset="cases.jsonl",
         scorers=["task_completion", "hallucination", "reflection_quality"])

# or directly
TaskCompletion(llm=llm).score(input="Book a table for 2 at 7pm.", output="Booked — confirmation #A12.")
```

## Scorecard

`Scorecard` rolls up any `EvalResults` or `SimulationResults` into a compact
per-metric panel (avg score + pass-rate) plus an overall pass-rate. Aggregation
only — no LLM calls.

```python
from fastaiagent import Scorecard, evaluate

results = evaluate(agent.run, dataset="cases.jsonl",
                   scorers=["task_completion", "faithfulness"])
card = Scorecard.from_eval_results(results, label="support-v2")
print(card.summary())
# Scorecard — support-v2
# ==================================================
# task_completion        avg=0.82  pass_rate=80%  (n=20)
# faithfulness           avg=0.91  pass_rate=95%  (n=20)
# --------------------------------------------------
# overall pass_rate=88%

card.to_dict()   # programmatic form
Scorecard.from_simulation(simulate(scenarios, agent))   # also works on sim runs
```

## Hardening — turn failures into fixes

`harden()` reads the **failures** from a `simulate()` / `evaluate()` run, inspects
the agent's config (system prompt, tools, guardrails), and returns a structured
`HardeningReport` of concrete, actionable recommendations.

```python
from fastaiagent import harden

results = simulate(scenarios, agent)
report = harden(agent, results, llm=llm)
print(report.summary())
# Hardening Report — support (3 failing case(s))
# ============================================================
# 1. [instructions] State the 30-day refund window explicitly and cite the policy section.
#      ↳ The agent answered "I don't know" to refund questions.
# 2. [tools] Add a `lookup_order(order_id)` tool.
#      ↳ Several scenarios needed live order status the agent couldn't provide.

for rec in report.recommendations:
    print(rec.target, "→", rec.recommendation)   # target ∈ instructions|model|tools|guardrails|memory
report.to_dict()
```

!!! note "v1 is recommend-only"
    `harden()` **never mutates your agent** — it returns recommendations for you
    to apply, then re-run `simulate()` / `evaluate()` to confirm the fixes. Auto-apply
    is a deliberate future step (keeps the agent immutable and the changes reviewable).

## The full loop

```python
scenarios = generate_scenarios(agent, n=8, llm=llm)   # 1. generate
results   = simulate(scenarios, agent)                # 2. run
print(Scorecard.from_simulation(results).summary())   # 3. score
report    = harden(agent, results, llm=llm)           # 4. get fixes
print(report.summary())                               # 5. apply + repeat
```

Everything runs **in-process** in the open-source SDK and persists to the Local
UI's Simulations / Evals pages — no hosted runtime required.
