# Agent Simulation

Simulation stress-tests **multi-turn** agent behavior. A `Scenario` drives a
conversation between a **simulated user** (an LLM persona or a fixed script) and
the agent under test, then a **judge** scores the whole transcript against
natural-language **success criteria** (and optional **failure criteria**).

Where it fits among the testing tools:

| Tool | Scores | Shape |
|---|---|---|
| `evaluate()` | fixed `input → output` pairs | single-turn, dataset |
| **`simulate()`** | a **multi-turn conversation** vs. criteria | synthetic dialogue |
| **Replay** | a past trace, fork-and-rerun | debugging |

It reuses existing primitives — your `Agent`, `LLMClient`, and `LLMJudge` — so
there's no new infrastructure to learn, and runs land in the same Local UI.

## Quickstart

```python
from fastaiagent import Agent, LLMClient, Scenario, SimulatedUser, simulate

llm = LLMClient(provider="openai", model="gpt-4o-mini")

agent = Agent(
    name="support",
    system_prompt="You are a friendly support agent. Explain the 30-day refund policy when asked.",
    llm=llm,
)

scenario = Scenario(
    name="refund-request",
    user=SimulatedUser(persona="A frustrated customer who wants a refund for shoes bought 10 days ago."),
    success_criteria=["The agent explains the refund policy clearly and politely."],
    failure_criteria=["The agent is rude or refuses to help."],
    max_turns=6,
)

results = simulate(scenario, agent)
print(results.summary())
```

`simulate()` persists each run to the Local UI by default — open the
[Simulations page](../ui/simulations.md) to read the transcript and per-criterion
verdicts.

## The pieces

### `Scenario`

```python
@dataclass
class Scenario:
    name: str
    user: SimulatedUser
    success_criteria: list[str]          # all must hold to pass
    failure_criteria: list[str] = ()     # if any holds → fail
    max_turns: int = 6                   # hard cap on total user + agent turns
```

### `SimulatedUser`

Provide **exactly one** of `persona` or `script`:

```python
# Persona — an LLM role-plays the user, generating each turn from the
# transcript so far. It can end the conversation early by replying "END".
SimulatedUser(persona="A confused first-time user.", llm=llm)

# Script — fixed user turns, returned one per turn; the conversation ends
# when the script is exhausted.
SimulatedUser(script=["Hi, I need help", "How do I reset my password?", "Thanks"])
```

### Judge

By default each criterion is judged by a `LLMJudge` call over the full
transcript: the scenario **passes** when every `success_criteria` holds and no
`failure_criteria` holds. Pass your own `judge=LLMJudge(llm=...)` to control the
model used for judging.

## Native agents and adapters

`simulate()` accepts either a native `Agent` or any callable adapter — so you
can simulate LangChain / CrewAI / custom agents too:

```python
# Native Agent — auto-wrapped to use the additive messages= param
simulate(scenario, agent)

# Callable adapter: (messages: list[Message]) -> str | AgentResult
def my_agent(messages):
    return my_framework.respond(messages[-1].content)

simulate(scenario, my_agent)
```

Native agents are driven via the additive
[`messages=` parameter](../agents/index.md#multi-turn-with-messages) on
`Agent.arun`, so prior turns flow into the model exactly as a real conversation
would.

## Determinism in tests

Inject `TestModel` / `FunctionModel` (real `LLMClient` subclasses) as the `llm`
for the agent, the simulated user, **and** the judge — the whole multi-turn run
becomes fully deterministic with no network and no mocks:

```python
from fastaiagent.testing.models import TestModel
from fastaiagent.eval import LLMJudge
import json

agent = Agent(name="bot", llm=TestModel(response="canned reply"))
judge = LLMJudge(llm=TestModel(response=json.dumps({"score": 1.0, "reasoning": "ok"})))
scenario = Scenario(
    name="smoke",
    user=SimulatedUser(script=["hi", "thanks"]),
    success_criteria=["The agent replied."],
)
results = simulate(scenario, agent, judge=judge, persist=False)
assert results.results[0].passed
```

## Results

`simulate()` returns `SimulationResults`:

- `.results` — a `SimulationResult` per scenario (`transcript`, `passed`,
  per-criterion `verdicts`, root `trace_id`).
- `.summary()` — a printable pass/fail table.
- `.persist_local() -> run_id` — writes one `sim_runs` row + one `sim_cases`
  row per scenario to the Local UI DB (done automatically when `persist=True`).
- `.export(path)` — dump the full results (transcripts + verdicts) to JSON.

`asimulate(...)` is the async entrypoint with the same signature; both run
scenarios concurrently (bounded by `concurrency`, default 4).
