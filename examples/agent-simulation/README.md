# Agent Simulation

Multi-turn scenario testing with `simulate()`. A `Scenario` drives a
conversation between a `SimulatedUser` (an LLM persona or a fixed script) and the
agent under test, then a judge scores the full transcript against
natural-language success / failure criteria.

`scenario_test.py` is a runnable pytest that exercises two scenarios — one
scripted, one persona-driven — fully deterministically using `TestModel` /
`FunctionModel` (no network, no API key).

## Run

```bash
pytest examples/agent-simulation/scenario_test.py -v
```

## Run against a real model

Swap the `TestModel` instances for a real `LLMClient` and drop the canned judge:

```python
from fastaiagent import Agent, LLMClient, Scenario, SimulatedUser, simulate

llm = LLMClient(provider="openai", model="gpt-4o-mini")
agent = Agent(name="support", system_prompt="You are a support agent.", llm=llm)

scenario = Scenario(
    name="refund-policy",
    user=SimulatedUser(persona="A customer asking about refunds.", llm=llm),
    success_criteria=["The agent explains the 30-day refund policy."],
    failure_criteria=["The agent is rude or refuses to help."],
)

results = simulate(scenario, agent)        # persists to the Local UI by default
print(results.summary())
```

Open the **Simulations** page in the Local UI to read the transcript and
per-criterion verdicts, and to deep-link into each turn's trace.

See [docs/simulation](../../docs/simulation/index.md) for the full guide.
