# What older answers get wrong

Read this when your memory of the SDK predates 1.102.0, or when code or advice you recall fails: each row is something that was removed, renamed or changed in meaning between 1.40 and 1.102.0. Purely additive features are left out.

Rule of thumb: if an answer from memory imports a framework adapter, calls `connect()` with only a key, or waits for an approval inside `arun()`, it is out of date.

| Old (what you may remember) | Now | Since |
|---|---|---|
| An output guardrail such as `allowed_topics` on a `Swarm` member stops the swarm at its first handoff ("Off topic") | The handoff turn is not judged; the member that answers is. Its `stop_reason` is `"handoff"` | 1.102.0 |
| `PlatformAPI` has only `get` and `post` and returns a dict; `PlatformRateLimitError` has only a message | `patch`, `delete`, `post_multipart`, `download` (and async twins); the body is any JSON; `PlatformRateLimitError.retry_after` | 1.102.0 |
| `fa.connect(api_key=...)` with no target, defaulting to `app.fastaiagent.net` | `target=` is required, else `FASTAIAGENT_TARGET`, else `ValueError`. The old host is dead; never write an address you were not given. `PlatformAPI(base_url=...)` is required too. | 1.100.0 |
| `fastaiagent connect` / `push` falling back to a built-in host | `--target`, else `FASTAIAGENT_TARGET`; with neither, nothing is saved and exit code is 1. | 1.100.0 |
| `fastaiagent connect` prints "Connected" and exits 0 whatever happened | Unreachable or 5xx plane: saves, says "not verified", exit 1. A non-plane answer (e.g. 404): saves nothing, exit 1. Python `fa.connect()` still connects offline and only logs a warning. | 1.97.0 |
| `from fastaiagent.integrations.langchain import enable` (also `.crewai`, `.pydanticai`) | Modules removed. Use the framework's own OpenInference instrumentor plus `fa.enable_otel_capture()`. | 1.96.0 |
| `with_guardrails(agent, ...)` around a framework agent | Call `Guardrail.execute(...)` (or `fa.run_guardrail`) around the framework call and act on the result. | 1.96.0 |
| `as_evaluable(agent)` for `fa.evaluate` | Pass a plain function `agent_fn(input) -> output` as the first argument of `fa.evaluate`. | 1.96.0 |
| `kb_as_retriever(kb)` / `kb_as_tool(kb)` | Call `kb.search(...)` inside the framework's own tool. | 1.96.0 |
| `prompt_from_registry(name)` | `PromptRegistry().get(name).format(...)`. For lineage on a foreign root span: `fa.trace_context("agent.x", **prompt.trace_attributes())`. | 1.96.0 / 1.98.0 |
| `register_agent(agent, name=...)` and `integrations.agent_name(name)` | `fa.Agent(name=...).push()` while connected, and `fa.trace_context("agent.<name>", **{"agent.name": name})` around the framework call. | 1.96.0 |
| `pip install "fastaiagent[langchain]"` / `[crewai]` / `[pydanticai]` | Extras no longer exist: pip installs core and warns. `fastaiagent.integrations` keeps only `openai` and `anthropic` (`.enable()`). | 1.96.0 |
| `fastaiagent.guardrail.actions.harness_halts` | Removed. Act on the result's `action_taken`. | 1.96.0 |
| `discover_tools(url)` for MCP | `MCPTool.from_server(...)` / `await MCPTool.afrom_server(...)` return one ready `MCPTool` per server tool. Needs `pip install "fastaiagent[mcp]"` (`[mcp-server]` is an alias). The hand-written JSON-RPC client is gone; it runs on the official `mcp` library. | 1.92.0 |
| `fastaiagent agent serve --reload` | Removed (it never worked). | 1.92.0 |
| `MCPTool`/`RESTTool` reaching a Tailscale or other CGNAT (`100.64.0.0/10`) host | Refused as non-global. Set `FASTAIAGENT_ALLOW_PRIVATE_NETWORKS=1` to allow on purpose. | 1.92.0 |
| `await agent.arun(..., wait_for_approval=True)` blocks for a console approval | Raises `ValueError` before anything runs. Default is `False`: a policy pause returns `status="paused"` and your app resumes it with `aresume(run_id, resume_value=Resume(approved=...))`. `wait_for_approval=False` is accepted and does nothing. | 1.74.0 / 1.76.0 |
| An agent without a checkpointer swallowing a pause as `output=""` | Callers that cannot hold a pause report it (MCP error, serve response) instead of treating `""` as the answer. | 1.78.0 |
| `execute_guardrails(...)` returns `list[GuardrailResult]` | Returns `GuardrailOutcome` (verdicts plus the value to carry forward after `mask`/`override`). Iteration, indexing, `len()` and `list()` still work; only `isinstance(x, list)` breaks. Imported from `fastaiagent.guardrail.executor`. | 1.57.0 |
| A failed guardrail always stops the run | Each rule has an `action`: `block`, `warn`, `mask`, `override`, `reask`. `mask`/`override` rewrite the payload; where a rewrite cannot be applied faithfully (streamed output, multimodal input) the run blocks. | 1.57.0 |
| A guardrail `llm={"api_key": ...}` config dict | Any credential (or non-shareable field) raises `ValueError` at construction. Use the provider env var, or pass a live `LLMClient(api_key=...)`. | 1.89.0 |
| Misspelt `position=` / `guardrail_type=` builds a guardrail that never runs; `code` guardrail with no `fn`, `regex` with empty pattern pass silently | Misspelt values raise `ValueError` at construction; a `code` rule with no `fn` and an empty `regex` pattern raise instead of passing. | 1.93.0 / 1.64.0 |
| `TraceStore().list_traces()` returns the newest 100 of all time and ignores filters | Honours `last_hours` (default 24; `None` = all time), `limit`, `name_filter`, `agent_name`. Unknown filters only warn. | 1.81.0 |
| Replay `fork_at(...).rerun()` re-runs every tool for real | Tools you did not supply are blocked (model gets an error result). Opt in with `.with_tool_policy("live", allow=[...])`, `.with_tool_override(name, fn)` or `.with_tool_stub(name, returns=...)`, or `FASTAIAGENT_REPLAY_TOOL_POLICY=live`. | 1.89.0 |
| Replay of a chain or supervisor trace "works" | `fork_at()` / `rerun()` raise `ReplayError` on a chain or supervisor root (use `Chain.afork` / `aresume`). Only a standalone agent trace replays. | 1.95.0 |
| Replay reruns and Playground calls counted as production runs and exported | Local experiments: not exported unless `FASTAIAGENT_EXPORT_EXPERIMENTS=1` / `connect(export_experiments=True)`. | 1.89.0 |
| `connect()` uploads up to a week of traces recorded while offline | Only spans recorded while a platform exporter is registered are queued; offline history stays local (`TraceData.publish()` sends one on demand). | 1.95.0 |
| `FASTAIAGENT_TRACE_PAYLOADS=0` stops payloads being captured | Egress-only: `local.db`, UI and Replay stay full fidelity; payloads are stripped on the way to the plane and any `add_exporter` target. | 1.50.0 |
| `FASTAIAGENT_TRACE_ENABLED=0` / `FASTAIAGENT_EXPORT_EVALS=yes` / `FASTAIAGENT_EXPORT_CHECKPOINTS=false` ignored or mis-parsed | All parse as booleans (`1/true/yes/on`, `0/false/no/off`). `TRACE_ENABLED=0` really stops local capture (`result.trace_id` is `None`). An unparseable egress value fails closed. | 1.67.0 |
| `pip install fastaiagent` brings `pymupdf`; PDF text and `LocalKB.add("x.pdf")` just work | Not in core. Install `fastaiagent[kb]` (engine is `pypdfium2` since 1.56.0); `pdf_mode="native"` needs no library. | 1.55.0 / 1.56.0 |
| `PromptRegistry.register(name, text, version=n)` overwrites an existing version | Same text and metadata is a no-op; different text raises `ValueError`. Omit `version=` for the next one. | 1.88.0 |
| `set_alias(name, version, alias)` accepts a missing version; `diff()` prints both templates | `set_alias` raises `PromptNotFoundError`; `diff()` is a unified diff. | 1.89.0 |
| Chain tool node that returns an error leaves the chain `completed` | Raises `ChainError` naming node, tool and error; a checkpointed run is marked `failed` (resumable). A chain approval gate with no handler refuses; a rejection stops the chain. | 1.78.0 / 1.77.0 / 1.67.0 |
| `chain.resume()` / `aresume` on a finished run re-executes it | Raises `AlreadyResumed`; a failed run still resumes. Use `fork()` to branch from a finished run. `PostgresCheckpointer.put` with a repeated `checkpoint_id` raises. | 1.65.0 |
| `Memory(window=0)` keeps everything | Keeps no messages (facts, summary and recall still work); a negative window raises `ValueError`. | 1.97.0 |
| `Memory(learn=)` extracts from assistant replies; a shared window when the user cannot be resolved; `forget(tier="global")` wipes all agents' facts; `Memory(agent_id="")` | Learns from the user's messages only. An unresolvable user gets no conversation memory. `forget(tier="global")` needs `agent_id=` or `id="*"`. `agent_id=""` raises `ValueError`. | 1.80.0 / 1.81.0 |
| `Supervisor` result `tokens_used` / `cost` count only the lead | They count the whole team (workers, `validate_outputs` retries, routing). Team members are now gated by approval policies and plane guardrails on every run. | 1.99.0 |
| `PlatformKB(...).as_tool()` named `search_<uuid>` and cut hits to 200 characters | `PlatformKB(kb_id, name=, description=)` gives `search_<name>`; whole chunks returned. | 1.99.0 |
| `Agent.stream()` returns the joined text of every turn | Returns what `run()` returns (the final reply). Streamed and resumed runs open one root span; pass `trace=False` to opt out. | 1.91.0 / 1.67.0 |
| Anthropic/Bedrock `max_tokens` defaults to 4096; cached tokens priced as input; one rate covers all `gpt-5.x` | Defaults to the model's own maximum; cached and cache-write tokens have their own rates; an unlisted model is unpriced, never mispriced, so cost figures move. | 1.91.0 |
| An HTTP 429 raises a generic `LLMProviderError` | `LLMRateLimitError` (a subclass, so old handlers still catch it) with `retry_after`. | 1.91.0 |
| `ROUGEScore(variant="rouge-l")` result key `"rouge"` | Named `"rouge-l"` (the default stays `"rouge"`). | 1.95.0 |
| Eval errors scored as failures; `eval run` exits 1 for a missing dataset | A check that could not run is `errored` and one that does not apply is `skipped`; neither is a pass or a fail. `eval run` exits 3 ("run invalid") for a missing dataset, an agent that fails to import, or unmeasured gated scorers. | 1.89.0 / 1.97.0 |
| `traces prune` deletes everything already synced; offline spans looked "synced" | Deletes only spans the plane confirmed. Everything recorded before 1.97.0 is kept; use `fastaiagent traces purge` to delete by scope. | 1.97.0 |
| `FASTAIAGENT_UI_ENABLED=1` turns on guardrail event logging | Ignored; guardrail events follow `FASTAIAGENT_TRACE_ENABLED`. | 1.89.0 |
