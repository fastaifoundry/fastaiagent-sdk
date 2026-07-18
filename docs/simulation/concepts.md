# Concepts & Mental Model

This page is the mental model for simulation — *why* it exists, *how* a
simulated conversation actually runs turn by turn, how scenarios are written or
generated, how transcripts are judged, and how simulation closes the loop into a
better agent. Read it first, then use the [Simulation reference](index.md) for
the API.

## Why simulate

Single-turn evaluation answers "given this input, is the output good?" But real
agents hold *conversations* — they ask follow-ups, handle a user who changes
their mind, and can go wrong on turn three even when turn one looked fine. You
can't capture that with a fixed `input → output` dataset, and you can't wait for
real users to hit every edge case (adversarial, confused, impatient) before you
ship.

**Simulation** fills that gap: an LLM role-plays a user, holds a multi-turn
conversation with your agent, and a judge scores the whole transcript against
natural-language criteria. It's how you stress-test *behavior over a dialogue*,
with coverage you generate rather than wait for.

## Where it fits

Simulation is one of three testing tools, each for a different shape of test:

| Tool | Scores | Shape |
|------|--------|-------|
| `evaluate()` | fixed `input → output` pairs | single-turn, over a dataset |
| **`simulate()`** | a **multi-turn conversation** vs. criteria | synthetic dialogue |
| `Replay` | a past trace, fork-and-rerun | debugging a real run |

Reach for **evaluate** when you have golden input/output cases, **simulate** when
the behavior only emerges over a conversation, and **replay** when you're
debugging something that already happened. Simulation reuses your existing
`Agent`, `LLMClient`, and `LLMJudge` — there's no new infrastructure.

## The simulated-conversation model

A scenario is a conversation between a **simulated user** and your **agent**,
judged at the end. The loop (`_run_scenario` in `fastaiagent/eval/simulate.py`):

1. **The user opens.** The simulated user speaks first (turn 0) — agents are
   reactive, so someone has to start.
2. **The agent replies.** Your agent responds to the transcript so far (fed via
   the additive `messages=` param, so multi-turn context is preserved).
3. **The user responds.** The simulated user produces the next message from the
   full transcript — or ends the conversation.
4. **Repeat** until the user stops or `max_turns` is hit.
5. **Judge once.** An `LLMJudge` scores the *whole transcript* against the
   scenario's criteria — one judge call per criterion.

```
user (turn 0) ─▶ agent (1) ─▶ user (2) ─▶ agent (3) ─▶ … ─▶ [max_turns]
                                                              │
                                              judge full transcript vs criteria
```

!!! info "Verified against a live run"
    A persona scenario (`max_turns=4`) produced exactly `user → agent → user →
    agent` and a scripted scenario ran its 3 fixed user messages across 6 turns
    — user on the even turns, agent on the odd. The judge returned one verdict
    per criterion, and the overall pass was "every success criterion holds AND
    no failure criterion holds."

### Two kinds of simulated user

A `SimulatedUser` takes **exactly one** of:

- **`persona`** — a natural-language description; an LLM role-plays the user,
  generating each turn from the transcript so far. It's prompted to write only
  the next user message, and to reply `END` when its goal is met — which stops
  the conversation early.
- **`script`** — a fixed list of user messages, returned one per turn; the
  conversation ends when the script is exhausted.

Use a **persona** for realistic, open-ended, adversarial behavior; use a
**script** for a deterministic, repeatable test (pair it with a `TestModel` for
fully offline runs).

### How judging works

The judge scores the transcript **once at the end**, one `LLMJudge` call per
criterion:

- **Success criteria** — things the agent *should* do; each must hold.
- **Failure criteria** — things the agent must *not* do; the criterion is
  inverted (the judge scores "did this bad thing happen?"), so absence is a pass.

Overall pass = all success criteria hold **and** no failure criterion occurred.

## Writing vs. generating scenarios

- **Hand-write** a `Scenario` when you have a specific behavior in mind —
  a known edge case, a regression you want to lock down.
- **Generate** with `generate_scenarios(agent, n=..., focus=...)` when you want
  breadth: it introspects the agent's name, system prompt, and tools and asks an
  LLM to propose diverse multi-turn scenarios (each with a persona and
  success/failure criteria), ready to pass straight to `simulate()`. `focus`
  steers it (e.g. `"adversarial users"`, `"edge cases"`).

!!! info "Verified against a live run"
    `generate_scenarios(agent, n=2, focus="adversarial users")` returned two
    named scenarios, each with success criteria, built from the agent's own
    system prompt and tools.

## Closing the loop

Simulation is a producer in the [evaluation](../evaluation/concepts.md) improve
step, not an island:

```
generate_scenarios ─▶ simulate ─▶ SimulationResults ─▶ Scorecard.from_simulation
                                          │
                                          └─▶ harden()  ─▶ recommended fixes ─▶ re-simulate
```

- **`SimulationResults`** carries per-scenario transcripts, pass/fail, and
  per-criterion verdicts; `.summary()`, `.export()`, `.persist_local()` land it
  in the Local UI (`persist=True` by default).
- **`Scorecard.from_simulation(results)`** rolls scenarios up into a pass-rate.
- **`harden(agent, results, llm)`** reads the *failing transcripts* and proposes
  concrete fixes to instructions/tools/guardrails/memory (recommend-only — you
  apply and re-simulate to confirm).

Runs are traced like everything else, so each scenario's turns and judge calls
nest under one trace in the UI for debugging.

## A guided path

1. [`examples/agent-simulation/scenario_test.py`](https://github.com/fastaiagent/fastaiagent-sdk/blob/main/examples/agent-simulation/scenario_test.py) — deterministic scripted + persona scenarios with offline models.
2. [`examples/74_agent_hardening.py`](https://github.com/fastaiagent/fastaiagent-sdk/blob/main/examples/74_agent_hardening.py) — the full loop: `generate_scenarios → simulate → Scorecard → harden`.

## Next steps

- [Simulation reference](index.md) — the full API: `Scenario`, `SimulatedUser`, `simulate` / `asimulate`, results, determinism
- [Evaluation](../evaluation/concepts.md) — where simulation sits in the improve loop
- [Agent Hardening](../evaluation/agent-hardening.md) — turn failures into fixes
- [Replay](../replay/concepts.md) — debug a specific past run instead of a synthetic one
