# Registration (make a connected agent visible)

`fa.connect()` gives you a **runtime session** — auth, policy, trace export,
governance enrollment. **Registration** is the separate step that pushes an
agent's *definition* to the plane so it exists as a governed console object
(`managed_by=sdk`, read-only) and its traces group by `agent_id`.

The SDK owns registration end-to-end. **You never write `httpx.post(...)`.**

## The one thing to know

> Call `fa.connect()` once, before your agents run. They register themselves
> automatically.

That's it. There's no "connect at the end" rule and no ordering puzzle:

- Agents that exist when you call `connect()` are registered at connect time.
- Agents created later register themselves on their **first run**.

Either way, `connect()` is the single entry point.

```python
import fastaiagent as fa

fa.connect(api_key="fa_k_…", target="https://app.fastaiagent.net")

agent = fa.Agent(name="Support", system_prompt="Help the customer.")
agent.run("hi")          # auto-registers on first run → appears in the console
fa.disconnect()
```

### Auto-registration is ON by default

While connected, running (or defining-then-connecting) an agent creates a
governed console object and links its traces. This is **best-effort, idempotent
(upsert by name), and never fails your run**. It does create objects in your
console and makes a network call on first run — opt out with:

```python
fa.connect(api_key="…", target="…", auto_register=False)
```

Registration is **once per name per process** — the 2nd, 3rd, … run make no
network call. Change your agent's code and re-run (a new process) and it
re-pushes automatically (the plane upserts by name).

## Explicit registration (CI / deploy)

For deterministic control, register explicitly. Same code path as auto:

```python
result = agent.push()          # or fa.push(agent)
print(result.agent_id, result.version, result.url)   # clickable console URL
```

`push()` returns a `PushResult` (`agent_id`, `name`, `version`, `url`) and always
re-pushes (so a same-process definition change resyncs). Requires the key to have
the `agent:write` scope.

### From the CLI

```bash
fastaiagent push --module my_app.agents        # register every agent in the module
fastaiagent push --module my_app.agents --dry-run   # preview the payloads
```

`--dry-run` prints one JSON array of the request bodies on stdout and its notes on
stderr, so `--dry-run | jq` works (since 1.97.0).

## What gets pushed

`Agent.to_dict()` — `name`, `system_prompt` (or `prompt_slug` when the agent
references a [registry prompt](../prompts/index.md)), `llm_endpoint`, `tools`,
`guardrails`, `config`, and `memory_enabled`. So a **guardrailed** agent's
guardrails are attached on the plane, and a registry-prompt agent shows the
**slug** (not "Inline"). It's all additive — an agent with neither `prompt_slug`
nor `memory` serializes exactly as before.

## If something's off

If an agent runs while connected but isn't registered (auto-register off, missing
`agent:write` scope, or a push error), the SDK logs a **one-time** warning telling
you exactly what to do — enable `auto_register`, call `agent.push()`, or run
`fastaiagent push`. Silence never hides it.

## Naming

The plane keys agents by **name within the project**, so two live agents with the
same name overwrite each other — give them distinct names.

### Agents built in another framework (LangGraph / CrewAI / Pydantic AI)

Push a `fa.Agent` with the agent's name that describes it, and run the
framework call inside a `fa.trace_context(...)` root span carrying the same
`agent.name`. The plane matches the trace to the agent by that name. See
[Use fastaiagent with your existing framework](../integrations/frameworks.md#7-the-control-plane).
The `register_agent()` helpers of the framework integrations were removed in
1.96.0.
