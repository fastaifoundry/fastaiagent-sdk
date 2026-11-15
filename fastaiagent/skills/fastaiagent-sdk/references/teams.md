# Teams: Supervisor, Swarm and Chain

Read this when one agent is not enough: a coordinator that delegates (`Supervisor`), peers that hand off (`Swarm`), or a fixed graph you draw (`Chain`).

## Which to choose

Start with ONE `Agent` and tools. Reach for a team only when one agent's prompt or tool set cannot hold the job.

| Need | Use | Who decides the path |
|---|---|---|
| One goal, open-ended steps | single `Agent` | the model's tool loop |
| A coordinator splits work, calls several specialists, writes ONE combined answer | `Supervisor` (`routing="tools"`, default) | the lead model, at runtime |
| Pick exactly one specialist for a request (triage, support queue), return its reply unchanged | `Supervisor(routing="decisions")` | one Decisions API call |
| Specialists decide for themselves who goes next; loops like writer <-> critic | `Swarm` | the active agent, via `handoff_to_<peer>` |
| The shape is known: routing, retry loops, approval gates, durable resume, one trace | `Chain` | you, with edges and conditions |

Do NOT use a Supervisor to do a Chain's job. A fixed pipeline run through an LLM coordinator costs extra turns and is not reproducible. Do NOT use a Chain when the path really emerges at runtime.

## Supervisor and Worker

```signature
fastaiagent.Worker(agent, role=, description=)
fastaiagent.Supervisor(name, llm=, workers=, system_prompt=, max_delegation_rounds=, checkpointer=, validate_outputs=, validation_prompt=, max_validation_retries_per_worker=, validation_mode=, validation_llm=, validation_threshold=, validation_criteria=, routing=, router_llm=, fallback_worker=, routing_min_confidence=, routing_questions=, routing_instructions=, guardrails=, middleware=, timeout_seconds=, memory=)
fastaiagent.Supervisor.run(input, context=, execution_id=, **kwargs)
fastaiagent.Supervisor.arun(input, context=, execution_id=, **kwargs)
fastaiagent.Supervisor.resume(execution_id, resume_value=, context=, **kwargs)
```

- `role` defaults to `agent.name`; `description` defaults to the first 200 chars of a string system prompt. The lead sees each worker as a tool named `delegate_to_<role>(task)`. Do write a real `description` per worker — it is how the lead (or the router) chooses.
- `llm` defaults to `LLMClient()` when omitted. `max_delegation_rounds` defaults to 3.
- The lead only passes a worker the `task` string it writes. Do give a worker its own `memory=` if it needs history; the lead's `memory=` does not reach it.
- Offline test (the lead is scripted; the worker is a `TestModel`):

```python
from fastaiagent import Agent, Supervisor, Worker
from fastaiagent.testing import FunctionModel, TestModel

researcher = Agent(name="researcher", system_prompt="Find facts.",
                   llm=TestModel(response="Fact: the sky is blue."))
turn = {"n": 0}

def lead(messages):
    turn["n"] += 1
    if turn["n"] == 1:
        return "", [{"name": "delegate_to_researcher", "arguments": {"task": "sky colour"}}]
    return "Summary: the sky is blue."

team = Supervisor(
    name="lead",
    llm=FunctionModel(lead),
    workers=[Worker(agent=researcher, role="researcher", description="Finds facts")],
)
result = team.run("What colour is the sky?")
assert result.output == "Summary: the sky is blue."
assert result.tool_calls[0]["tool_name"] == "delegate_to_researcher"
```

### Optional behaviours

- `validate_outputs=True`: the lead reviews each worker output; on rejection the worker re-runs with feedback, capped by `max_validation_retries_per_worker` (default 1). Failure of the review itself is FAIL-OPEN (treated as approval). After the cap the last output is used and a `supervisor.validate` / `warned` guardrail event is written. `validation_mode="decisions"` swaps the chat review for one Decisions API predicate (`validation_criteria=`, `validation_threshold=` in 0..1, `validation_llm=`).
- `guardrails=` / `middleware=` are the supervisor's OWN (its input, its `delegate_to_*` calls, its answer); each worker keeps its own. Since 1.90.0 / 1.91.0.
- `timeout_seconds=` bounds the whole run, workers included; raises `AgentTimeoutError`.
- `memory=` (since 1.99.0): the lead keeps a conversation, one turn per run. Refused (`ValueError`) with `routing="decisions"`.
- `context=RunContext(...)` flows to every worker and its tools.

## Supervisor routing="decisions" and decide()

`routing="decisions"` replaces the lead chat model with one call to OpenAI's Decisions API. It picks exactly ONE worker and returns that worker's reply unchanged (no synthesis turn). Requires an OpenAI-served decisions endpoint at run time (default router model `gpt-6-luna`; a key is needed: `# needs: OPENAI_API_KEY`) — offline, give `router_llm=TestModel(decisions=...)`.

- Constructor refuses (`ValueError`): fewer than 2 workers, duplicate roles, `fallback_worker` not one of the roles, a `routing_questions` key named `"worker"`, `routing_min_confidence` outside 0..1, or `memory=`.
- A refusal, or confidence below `routing_min_confidence`, goes to `fallback_worker`. The router never guesses a specialist. `llm=` is not used for routing.
- `routing_questions={"urgent": Predicate(...)}` ride in the same call; the worker gets their answers as a short note.
- `result.route` is a `SupervisorRoute`: `worker`, `chosen`, `confidence`, `fallback`, `answers`, `latency_ms`, `cost_usd`, `reviews`.

```python
from fastaiagent import Agent, Supervisor, Worker
from fastaiagent.testing import TestModel

def worker(role, reply):
    return Worker(agent=Agent(name=f"w-{role}", llm=TestModel(response=reply)),
                  role=role, description=f"Handles {role} requests.")

router = TestModel(decisions={"answers": [
    {"type": "choice", "name": "worker", "choice": "billing", "confidence": 0.9}]})
desk = Supervisor(
    name="desk",
    workers=[worker("billing", "Refund issued."), worker("other", "Let me help.")],
    routing="decisions",
    router_llm=router,
    fallback_worker="other",
    routing_min_confidence=0.6,
)
result = desk.run("Refund my order")
assert result.output == "Refund issued."
assert (result.route.worker, result.route.fallback) == ("billing", False)
```

`decide()` is the underlying primitive on any `LLMClient`; use it directly for a one-off classification instead of a chat turn. Questions are `Predicate` (probability), `Choice` (one of N options, with confidence) and `Score` (levels) from `fastaiagent.llm`. Any answer can be a refusal; check before using it.

```signature
fastaiagent.LLMClient.decide(input, questions, safety_identifier=)
fastaiagent.tool.decision_tool(questions, llm=, name=, description=)
```

Chains have the same idea as a `NodeType.condition` node with `decision=Choice(...)` (below).

## Swarm

```signature
fastaiagent.Swarm(name, agents, entrypoint, handoffs=, max_handoffs=, checkpointer=, middleware=, timeout_seconds=)
fastaiagent.Swarm.run(input, context=, execution_id=)
fastaiagent.Swarm.cancel(execution_id)
fastaiagent.Swarm.resume(execution_id, resume_value=, context=, **kwargs)
```

- `agents` are plain `Agent`s with unique names; `entrypoint` names the first. Each hop gets one `handoff_to_<peer>` tool per allowed peer, added to the agent's own tools.
- `handoffs={"a": ["b"], "b": []}` is an allowlist. Omit it for a full mesh. An empty list means that agent terminates the run. `max_handoffs` defaults to 8.
- The run ends when an agent answers with no handoff. The handoff tool accepts `reason` and an optional `context` dict that merges into the shared blackboard and is shown to later agents; tool results from earlier agents are also listed in each briefing (since 1.99.0).
- Errors: `SwarmError` (a subclass of `AgentError`) for a missing entrypoint, duplicate names, unknown peers, a disallowed handoff, or exceeding `max_handoffs`. Do tune `max_handoffs` to the topology; a writer <-> critic loop with no exit hits it.
- `timeout_seconds` bounds the whole run and raises `AgentTimeoutError`.
- A member's output guardrails are not run on its handoff turn (that turn's `stop_reason` is `"handoff"`, since 1.102.0): the reply is a control marker, and an allow-style rule such as `allowed_topics` used to block the swarm on its first handoff. They still judge the member that answers.

```python
from fastaiagent import Agent, Swarm
from fastaiagent.testing import FunctionModel

def triage_llm(messages):
    return "", [{"name": "handoff_to_billing", "arguments": {"reason": "invoice question"}}]

triage = Agent(name="triage", system_prompt="Route.", llm=FunctionModel(triage_llm))
billing = Agent(name="billing", system_prompt="Billing.", llm=FunctionModel(lambda m: "Your invoice is $10."))
swarm = Swarm(name="desk", agents=[triage, billing], entrypoint="triage",
              handoffs={"triage": ["billing"], "billing": []}, max_handoffs=3)
result = swarm.run("Where is my invoice?")
assert result.output == "Your invoice is $10."
assert result.tool_calls[0]["agent"] == "triage"
```

## What a team run returns

`Supervisor.run/arun/resume` and `Swarm.run/arun/resume` return an `AgentResult` for the WHOLE team:

- `output`: the lead's final answer (Supervisor), the last agent's answer (Swarm), the worker's reply (`routing="decisions"`).
- `tool_calls`: every call, each tagged with the `agent` that made it. A Supervisor's list also holds the workers' calls (`worker` key, since 1.99.0).
- `tokens_used`, `cost`, `cost_known`: summed over the whole team. `cost_known` is `False` if any model had no rate.
- `trace_id`: one `supervisor.<name>` / `swarm.<name>` root span. `guardrails`: every firing, workers' included. `status`: `"paused"` after an `interrupt()`, `"cancelled"` after `cancel()`.
- Stream with `astream(input)` (events incl. `HandoffEvent`). `Supervisor.stream()` / `Swarm.stream()` are sync and return an `AgentResult` (no `tool_calls` on a Supervisor's).
- A `Supervisor` has no `cancel()`; call `Agent.cancel(execution_id)` or `Swarm.cancel(...)` — the registry is shared.
- Do pass the same `RunContext` to `run(..., context=ctx)`; it reaches every member and its tools.

## Chain

A `Chain` is a directed graph you build with `add_node` and `connect`; the executor runs it with shared state, checkpoints, and one trace.

```signature
fastaiagent.Chain(name, state_schema=, checkpoint_enabled=, checkpointer=, strict_routing=, middleware=)
fastaiagent.Chain.add_node(id, agent=, tool=, type=, name=, node=, output_key=, input_schema=, output_schema=, **config)
fastaiagent.Chain.connect(source, target, condition=, max_iterations=, exit_condition=, label=)
fastaiagent.Chain.validate()
fastaiagent.Chain.execute(initial_state=, trace=, context=, **kwargs)
fastaiagent.Chain.aexecute(initial_state=, trace=, execution_id=, hitl_handler=, context=, **kwargs)
fastaiagent.Chain.resume(execution_id, modified_state=, resume_value=, context=)
fastaiagent.Chain.cancel(execution_id)
```

- Node type is inferred: `agent=` -> agent node, `tool=` or `node=` -> tool node. Others need `type=NodeType.<condition|transformer|parallel|hitl|start|end>` (import `NodeType` from `fastaiagent.chain`).
- A node with nothing to run FAILS the run (`ChainError`) and `validate()` reports it. Do call `chain.validate()` (returns a list of error strings; `[]` is valid) before running.
- `Chain` checkpoints by default (`checkpoint_enabled=True`) into a local SQLite store. Do pass `checkpoint_enabled=False` in unit tests so nothing is written.
- `ChainResult`: `output`, `final_state`, `node_results`, `execution_id`, `status` (`completed|paused|rejected|cancelled`), `pending_interrupt`, `trace_id`. For an agent node, `node_results[id]` and `output` are dicts shaped `{"output": ..., "tool_calls": [...]}` — read `["output"]` for the text (the docs print the dict).
- `Chain.resume` is `async` only (`aresume` is an alias). Don't call it without `await` / `asyncio.run`.
- Edge routing: all-unconditional edges fan out and all fire. If any edge has a `condition`, the first match in declaration order wins and one unconditional sibling is the default. Conditions are `{{...}}` templates over `input`, `state`, `node_results`, with `== != > < >= <= contains startswith`; a bare word is literal text, not a state lookup. A dead branch is skipped silently unless `strict_routing=True` (then `ChainRoutingError`).
- A tool node's return is wrapped `{"output": ..., "error": None}` and overwrites `state.output`; keep values you must thread on top-level state. A tool node that reports an error fails the run.

```python
from fastaiagent import Agent, Chain, node
from fastaiagent.testing import TestModel

classifier = Agent(name="classifier", system_prompt="Classify.", llm=TestModel(response="billing"))
billing = Agent(name="billing", system_prompt="Billing.", llm=TestModel(response="refund sent"))
general = Agent(name="general", system_prompt="General.", llm=TestModel(response="hello"))

chain = Chain("router", checkpoint_enabled=False)
chain.add_node("classify", agent=classifier)
chain.add_node("billing", agent=billing)
chain.add_node("general", agent=general)
chain.connect("classify", "billing", condition="{{node_results.classify.output}} == billing")
chain.connect("classify", "general")  # the default edge
assert chain.validate() == []
result = chain.execute({"message": "Refund please"})
assert result.status == "completed"
assert "general" not in result.node_results
assert result.node_results["billing"]["output"] == "refund sent"

@node(output_key="shouted")
def shout(text: str) -> str:
    return text.upper()

typed = Chain("typed", checkpoint_enabled=False)
typed.add_node("shout", node=shout, input_mapping={"text": "{{state.text}}"})
assert typed.execute({"text": "hi"}).final_state["shouted"] == "HI"
```

### Typed nodes, cycles, decision nodes, approval gates (brief)

- `@node(output_key=, output_schema=, validate_input=)` turns a function into a tool node; its type hints become `input_schema`. Templates render as STRINGS, so an `int` parameter fed from `{{state.n}}` fails input validation — type it `str` and convert inside.
- Cycle: `chain.connect("evaluate", "research", max_iterations=3, exit_condition="quality >= 0.8")`. A cyclic edge without `max_iterations` fails `validate()`; exceeding it raises `ChainCycleError`.
- Decision node (needs a Decisions endpoint at run time): `chain.add_node("triage", type=NodeType.condition, decision=Choice(instructions=..., options={...}))` with `from fastaiagent.llm import Choice`, then one `connect(..., label=<option>)` per option plus an unlabelled default edge (taken on refusal or low confidence). Its `llm=` takes shareable kwargs only; an `api_key` raises at `add_node`.
- Approval: inside any node call `interrupt(reason, context)` (`from fastaiagent import interrupt, Resume`); `execute` returns `status="paused"`, and later `await chain.resume(execution_id, resume_value=Resume(approved=True))`. Pass your own `execution_id=` to `aexecute` so you can find it. `NodeType.hitl` with `hitl_handler=` is the blocking, in-process variant. A rejected gate gives `status="rejected"` and stops the chain.
- Nodes may re-run on resume; make side-effecting nodes idempotent (`@idempotent`, see `docs/chains/idempotency.md` in the repo).

## Testing teams offline

- Use `TestModel(response=..., tool_calls=[...], decisions=...)` for a fixed reply and `FunctionModel(fn, decide_fn=)` when the reply depends on call order or messages (count calls in a closure, as above). Both are `LLMClient`s: put them on the lead, each worker, each swarm agent, and `router_llm=`.
- A scripted lead calls `delegate_to_<role>` with argument `task`; a scripted swarm agent calls `handoff_to_<peer>` with argument `reason`.
- Assert on `result.output`, `result.tool_calls` (look at `agent` / `worker`), `result.route`, `result.tokens_used` and `ChainResult.node_results`. Don't assert on exact free text from a real model.
- Running a team still emits traces to a local `.fastaiagent/` directory under the current working directory. Run tests from a temp dir if that matters.
- Tests that need a real model belong in `tests/e2e/` marked `e2e`; they need a key (`# needs: OPENAI_API_KEY`).

```signature
fastaiagent.testing.TestModel(response=, tool_calls=, usage=, model=, delay_ms=, decisions=)
fastaiagent.testing.FunctionModel(fn, model=, delay_ms=, decide_fn=)
```

See `guardrails.md` for guardrails on a team member or on the supervisor itself.
