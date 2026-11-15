# Platform connection, prompts, tracing, push, runner, CLI

Read this when the task connects the SDK to a FastAIAgent control plane, loads versioned prompts, inspects or exports traces, registers agents, runs a runner, or uses the `fastaiagent` CLI.

Everything works offline without `connect()`: traces go to local SQLite, prompts to the local registry. `connect()` adds platform backends beside the local ones.

## The address rule

NEVER invent, guess or hard-code a platform address. Take it from the user, from `FASTAIAGENT_TARGET`, or from `~/.fastaiagent/credentials.toml`. Do not use `app.fastaiagent.net` — that host is dead and was the default before 1.100.0. Each account has its own address, shown in the console when an API key is created. If you do not have it, ask the user; do not fill a placeholder that looks real.

## connect / disconnect

```signature
fastaiagent.connect(api_key, target=, project=, governance_fail_mode=, auto_register=, console_url=, export_traces=, export_evals=, export_checkpoints=, export_experiments=)
fastaiagent.disconnect()
fastaiagent.push(agent, force=)
fastaiagent.refresh_policy()
```

```python
# needs: a FastAIAgent API key and program address from the user
import os

import fastaiagent as fa

fa.connect(
    api_key=os.environ["FASTAIAGENT_API_KEY"],
    target=os.environ["FASTAIAGENT_TARGET"],
    project="my-project",  # optional
)
# ... run agents; traces are exported in the background ...
fa.disconnect()  # flushes pending spans first
```

- `target` is REQUIRED since 1.100.0: the argument, else `FASTAIAGENT_TARGET`, else `ValueError` ("No target. Pass target= or set FASTAIAGENT_TARGET ..."). Don't call `fa.connect(api_key=...)` alone — it raises instead of reaching a default host. A failed resolve leaves the previous connection untouched.
- `target` needs an explicit `http://` or `https://` scheme; `"localhost:7842"` raises `ValueError` (message mentions the scheme). A trailing `/` is stripped.
- `api_key` has no default and Python `connect()` does NOT read `FASTAIAGENT_API_KEY`. Read the env var yourself and pass it, as above. Only the CLI reads that variable.
- Invalid key (HTTP 401) or forbidden (403) raises `PlatformAuthError`. An unreachable plane does NOT raise: the connection is stored, a warning is logged, and traces buffer locally until the plane answers. So `connect()` returning proves nothing about reachability.
- An answer that is not 200/401/403 (e.g. wrong host) logs a warning; the key is unverified.
- `auto_register=True` (default): agents that exist at connect time register with the plane, later agents register on their first run (`run`, `arun`, `stream`, `astream`). Best-effort, never fails a run. Use `auto_register=False` to opt out and register with `agent.push()`.
- `export_traces=False`: connect for policy, prompts, scorers and governance without registering a span exporter (for a runtime that owns its own OTel exporter).
- `governance_fail_mode="closed"` (or `FASTAIAGENT_GOVERNANCE_FAIL_MODE=closed`): a governed agent refuses tool calls when the policy cannot be confirmed. Default `"open"`. Any other value logs a warning and stays open.
- `fa.is_connected` is a lazy module attribute (bool), not a function.
- `fa.refresh_policy()` re-pulls the plane policy (returns its version, or `None` when not connected).

## Credentials resolution

Python: arguments, then `FASTAIAGENT_TARGET` (target only). Nothing else is read.

CLI (`fastaiagent push`, `fastaiagent auth status|env`): the helper `_load_saved_credentials` merges, with environment winning over the file:

| Source | Fields |
|---|---|
| Env: `FASTAIAGENT_API_KEY`, `FASTAIAGENT_TARGET`, `FASTAIAGENT_PROJECT` | api key, target, project |
| File `~/.fastaiagent/credentials.toml` (mode 0600, written by `fastaiagent connect`) | `api_key`, `target`, `project` |
| `--api-key` / `--target` on `fastaiagent push` | override both |

- `fastaiagent connect --api-key ... [--target ...] [--project ...]` calls `connect()`, then saves the file. `--api-key` prompts (hidden) when omitted; it does not read `FASTAIAGENT_API_KEY`. Exit code 1 when the plane is unreachable ("Saved, not verified") or answers a non-plane HTTP code (then nothing is saved).
- `fastaiagent disconnect` deletes the file. `fastaiagent auth status` shows source and masked key; `eval "$(fastaiagent auth env)"` exports the saved values as env vars.
- Other optional env vars: `FASTAIAGENT_CONSOLE_URL` (console origin for deep links, when it differs from the target), `FASTAIAGENT_GOVERNANCE_FAIL_MODE`.

## Prompts

```signature
fastaiagent.PromptRegistry(store=, path=)
fastaiagent.PromptRegistry.register(name, template, fragments=, metadata=, version=, project_id=, created_by=)
fastaiagent.PromptRegistry.register_fragment(name, content)
fastaiagent.PromptRegistry.get(slug, version=, source=)
fastaiagent.PromptRegistry.load(name, version=, alias=, resolve_fragments=)
fastaiagent.PromptRegistry.set_alias(name, version, alias)
fastaiagent.PromptRegistry.publish(slug, content, variables=)
fastaiagent.Prompt.format(**kwargs)
fastaiagent.Prompt.trace_attributes()
```

```python
from fastaiagent import PromptRegistry

reg = PromptRegistry(path="/tmp/fa_platform_ref_prompts.db")
reg.register_fragment(name="tone", content="Be professional.")
reg.register(name="greeting", template="Hello {{name}}. {{@tone}}")
reg.set_alias("greeting", 1, "production")

prompt = reg.get("greeting")  # local registry when not connected
print(prompt.format(name="World"))
print(reg.load("greeting", alias="production").version)
```

- Placeholders are `{{var}}`; fragments are `{{@fragment}}`. `Prompt.format(**kwargs)` is plain string replacement: a missing variable is left in the output as `{{var}}` with NO error — check `prompt.variables` yourself.
- `get(slug, version=None, source="auto")`: `"auto"` uses the platform when connected, else local; `"platform"` raises `PromptNotFoundError` if absent; `"local"` never touches the network. A platform fetch that fails for another reason silently falls back to the local registry under `"auto"`. Platform results are cached 5 minutes per `(slug, version)`; `reg.refresh(slug)` drops the cache.
- `load()` is local only and takes `alias=` (e.g. `"production"`). `get()` has no alias argument.
- Versions are immutable. `register()` without `version=` adds the next version; re-registering an existing `version=` with a different template raises `ValueError`.
- `publish(slug, content)` pushes to the platform and raises `PlatformNotConnectedError` when not connected.
- Pass the `Prompt` object itself as `Agent(system_prompt=prompt)` (not `prompt.format()`, which drops the lineage) and the run stamps `fastaiagent.prompt.*` attributes on its LLM span. Around a foreign framework call, use `fa.trace_context("name", **prompt.trace_attributes())`.
- The registry lives in the local DB (see Tracing); `PromptRegistry(path=...)` points it elsewhere.

## Tracing

Tracing is automatic: every `Agent`/`Chain` run writes spans to local SQLite (`.fastaiagent/local.db`, override with `FASTAIAGENT_LOCAL_DB`). `result.trace_id` identifies the run. Local capture is always full fidelity; filtering happens only on the way out.

```signature
fastaiagent.trace_context(name, **attributes)
fastaiagent.TraceStore(db_path=)
fastaiagent.TraceStore.list_traces(last_hours=, limit=, name_filter=, agent_name=, exclude_sources=)
fastaiagent.TraceStore.get_trace(trace_id)
fastaiagent.trace.add_exporter(exporter)
fastaiagent.enable_otel_capture(framework=)
fastaiagent.disable_otel_capture()
```

```python
import os
import tempfile

os.environ["FASTAIAGENT_LOCAL_DB"] = os.path.join(tempfile.mkdtemp(), "local.db")

from fastaiagent import Agent, TraceStore, trace_context
from fastaiagent.testing import TestModel

agent = Agent(name="demo", system_prompt="Be brief.", llm=TestModel(response="hi"))

with trace_context("my-operation", **{"custom.key": "value"}) as span:
    span.set_attribute("step", "start")
    result = agent.run("hello")

store = TraceStore()
print(result.trace_id, len(store.list_traces(last_hours=1)))
```

- `trace_context` is a context manager yielding an OTel span; nested use nests spans. It needs no connection.
- `list_traces(last_hours=24, limit=100)` defaults hide older traces: pass `last_hours=None` for all time. An unknown keyword only warns (`UserWarning`) and is ignored.
- `FASTAIAGENT_TRACE_ENABLED=0` is the master switch: no local capture and no export at all (Replay and the UI then have nothing).
- `FASTAIAGENT_TRACE_PAYLOADS=0` keeps prompts, outputs and tool arguments local: they are stripped before spans reach the plane or any `add_exporter` target. It does not reduce what `local.db` holds. An unparseable value counts as off (fails closed).
- `add_exporter(exporter)` adds any OTel `SpanExporter` (Jaeger, Datadog, ...); it is wrapped so payloads are filtered per the settings above.
- `enable_otel_capture()` is the one opt-in call to capture and render spans from third-party OTel/OpenInference instrumentors (including other agent frameworks) in the local store and UI. Call it once at startup; idempotent. It does nothing if `FASTAIAGENT_TRACE_ENABLED` is off. `disable_otel_capture()` only stops normalisation; an attached processor stays attached.
- Other egress switches (all default on, all fail closed when unparseable): `FASTAIAGENT_EXPORT_EVALS`, `FASTAIAGENT_EXPORT_CHECKPOINTS`. `FASTAIAGENT_EXPORT_EXPERIMENTS` defaults off (replay reruns and Playground calls stay local). `connect()` logs the resolved posture once at INFO.
- Unsent spans buffer in `local.db` while the plane is unreachable and drain later. `fastaiagent traces prune` deletes only spans the plane already has.

## Registering agents (`push`)

```signature
fastaiagent.Agent.push(force=)
```

```python
# needs: a connected plane (fa.connect with a key and FASTAIAGENT_TARGET)
import fastaiagent as fa
from fastaiagent.testing import TestModel

agent = fa.Agent(name="support", system_prompt="Help.", llm=TestModel(response="ok"))
result = agent.push()          # same as fa.push(agent)
print(result.agent_id, result.version, result.url)
```

- Requires an active `connect()` and the `agent:write` scope on the key; raises on platform errors (`PlatformNotConnectedError` when not connected). `force=True` is the default and always re-pushes.
- The plane keys agents by name within a project: two live agents with the same name overwrite each other. Give them distinct names.
- A Supervisor worker or Swarm member registers as the agent it was made from (since 1.99.0), without `handoff_to_*` tools.
- Auto-registration (see connect) already does this on first run; use `push()` for deterministic CI/deploy registration.
- CLI: `fastaiagent push --module my_app.agents [--dry-run]` imports the module and pushes every module-level `Agent`, `Supervisor` (lead and workers) and `Swarm` (members). `--dry-run` prints one JSON array of request bodies on stdout, notes on stderr, and contacts nothing. Exit 1 if no agents are found or any push fails.

## Runner

```bash
fastaiagent runner --connect "$FASTAIAGENT_TARGET" --key "$FASTAIAGENT_API_KEY" \
  [--labels k=v ...] [--max-concurrency 4] [--tools module:callable ...]
```

- A long-lived daemon that registers with the plane, heartbeats, pulls live jobs (`live_playground`, `eval_run`) and runs them in your process with your local tools and keys. Ctrl-C / SIGTERM drains in-flight jobs.
- `--connect` and `--key` are both required. It calls `connect()` with the same key so job traces are exported; a bad key exits 1.
- A non-loopback `--connect` must be `https://`; plain `http` exits 2 unless `FASTAIAGENT_RUNNER_ALLOW_INSECURE=1`. Loopback http is allowed.
- `--tools module:callable` imports and calls the callable, which must register your local tools in `ToolRegistry`; only then does the runner advertise `tool_exec`. Without it, plane-dispatched tool calls are not accepted.
- Each job runs in its own request-scoped `job_scope`, so concurrent jobs do not share a connection.

## CLI commands

Run `fastaiagent <command> --help` for flags. Entry point: `fastaiagent` (also `python -m fastaiagent.cli.main`).

| Command | Purpose |
|---|---|
| `connect` / `disconnect` | Save / remove platform credentials; verify the key. |
| `auth status` / `auth env` | Show saved credentials and their source / print `export` lines. |
| `push --module M [--dry-run]` | Register agents defined in a module with the plane. |
| `runner --connect --key` | Run a registered runner daemon that pulls live jobs. |
| `version` | SDK version and which optional extras are installed. |
| `migrate` | Copy legacy `traces.db`, `checkpoints.db`, `.prompts/` into `local.db`. |
| `resume` | Resume a suspended execution. |
| `list-pending` | List pending interrupts. |
| `inspect` | Show checkpoint history for an execution. |
| `setup-checkpointer` | Provision the durability backend (SQLite or Postgres). |
| `export-trace --trace-id ID [-o FILE]` | Export one trace as a self-contained JSON file from local SQLite (default `trace.json`). |
| `traces list` / `export` | List recent traces / export a trace. |
| `traces prune` | Delete spans the plane already has, to reclaim space. |
| `traces scrub-secrets` | Remove credentials from spans stored by an older SDK. |
| `traces purge` | Delete trace data from `local.db`. |
| `replay show` / `inspect` / `fork` | Show replay steps / inspect one step / fork a trace at a step, modify, rerun. |
| `eval run` | Run an evaluation and gate on it (non-zero exit for CI; 3 = invalid run). |
| `eval compare` | Compare two persisted eval runs, baseline first. |
| `eval export` | Preview what Agent-CI evidence would leave the machine (`--dry-run`). |
| `eval curate` | Build an eval dataset from captured agent traces. |
| `prompts list` / `diff` | List prompts / diff two versions. |
| `kb list` / `status` / `add` / `clear` / `delete` | Manage local knowledge bases. |
| `mcp serve` | Expose an Agent or Chain as an MCP server over stdio (needs the `mcp` extra). |
| `agent serve` | Serve an Agent or Chain as a FastAPI service (`/run`, `/run/stream`, `/resume`, `/cancel`). Set `FASTAIAGENT_SERVE_TOKEN` for any network-exposed deployment; default bind is `0.0.0.0`. |
| `ui start` / `ui reset-password` | Start the local web UI (default `127.0.0.1:7842`) / reset its login. |
| `learn` / `learn list` / `learn supersede` | Extract durable facts from past traces / list active facts / supersede one. |
| `optimize --agent --dataset` | Eval-driven prompt optimisation loop for an agent. |

Don't call `fastaiagent connect` from CI without `--target` or `FASTAIAGENT_TARGET` — it fails with the no-target error, exit 1.

## Agent Studio from code or the command line (1.102.0)

Not for writing agent code by hand. A platform at 0.20.0 or later can build an agent from a spec you
design: `fastaiagent studio …` (`context`, `new`, `spec validate|push|get`, `show`, `try`, `eval`,
`export`, …) and `fastaiagent.studio.StudioClient` (also `fa.studio`). The `fastaiagent-studio` skill
teaches the loop, the spec and the errors; read that skill, not this paragraph, before you use it.
`StudioError(code, message, status, details)` is a `PlatformError`; a platform without Studio raises
`StudioError("studio_not_available")`.
