# Guardrails

Read this when you add input/output/tool checks to an agent, write a custom check, or need to know what blocks, what warns, and what a check that cannot run reports.

## The model

A guardrail is a check on a payload at one position. The SDK runs it locally, inside your process: a block always happens at the edge. A connected control plane can author and distribute rules, and the SDK enforces them next to your local `guardrails=[...]`; it never relies on the plane to stop a run.

```signature
fastaiagent.Guardrail(name, guardrail_type=, position=, config=, blocking=, description=, fn=, on_error=, origin=, action=, severity=, floor=)
fastaiagent.Guardrail.execute(data)
fastaiagent.Guardrail.aexecute(data)
```

- `position` (`GuardrailPosition`): `input` (before the model sees the user message), `output` (the reply), `tool_call` (before a tool runs; the payload is the JSON text `{"tool": name, "arguments": {...}}`), `tool_result` (the tool's output text). Strings work: `position="input"`; a misspelt value raises at construction.
- `guardrail_type` (`GuardrailType`): `code` (default; your `fn=`), `llm_judge`, `regex`, `schema`, `classifier`, `content_safety`, `groundedness`, `topic`, `pii`, `secrets`. Import both enums from `fastaiagent.guardrail`.
- `blocking=True` (default): runs inline, in order, and a failing check halts. `blocking=False`: observe-only, runs in parallel, never halts and never rewrites.
- Attach with `Agent(..., guardrails=[...])`. A `Supervisor` takes its own `guardrails=` too (see `teams.md`).
- `fn(text)` returns a `bool` or a `GuardrailResult(passed, score=, message=, metadata=)`. A dict payload is passed to `fn` as JSON text.
- `execute` is sync. Inside async code use `await guardrail.aexecute(...)`.
- A blocking failure raises `GuardrailBlockedError` out of `agent.run()`. Import it with `from fastaiagent import GuardrailBlockedError` (it has `.guardrail_name`, `.results` and `.trace_id`, the blocked run's trace, `None` when tracing is off). Code written for 1.103.0 or earlier imports it from `fastaiagent._internal.errors`, which still works.

## Built-in factories

All live in `fastaiagent.guardrail`; each returns a `Guardrail` (`responsible_ai` returns a list).

```signature
fastaiagent.guardrail.no_pii(position=, entities=, backend=)
fastaiagent.guardrail.no_secrets(position=)
fastaiagent.guardrail.no_prompt_injection(position=, mode=, llm=, on_error=)
fastaiagent.guardrail.json_valid(position=)
fastaiagent.guardrail.toxicity_check(position=, mode=, llm=, threshold=, on_error=)
fastaiagent.guardrail.openai_moderation(position=, client=, model=, on_error=)
fastaiagent.guardrail.banned_topics(topics, llm=, mode=, position=, on_error=)
fastaiagent.guardrail.allowed_topics(topics, llm=, mode=, position=, on_error=)
fastaiagent.guardrail.grounded(reference, llm=, threshold=, position=, on_error=)
fastaiagent.guardrail.no_hallucination(reference, llm=, threshold=, position=, on_error=)
fastaiagent.guardrail.cost_limit(max_usd, position=, on_error=)
fastaiagent.guardrail.allowed_domains(domains, position=)
fastaiagent.guardrail.responsible_ai(pii=, prompt_injection=, secrets=, toxicity=, moderation=, grounded_to=, banned=, allowed=, llm=, threshold=, on_error=)
```

| Factory | Default position | Needs a model/key? | Notes |
|---|---|---|---|
| `no_pii` | output | no | `entities` default `("email","phone","ssn","credit_card")`; credit cards Luhn-checked; `"ip"`, `"iban"` opt-in; `backend="presidio"` needs `pip install "fastaiagent[safety]"` |
| `no_secrets` | output | no | private keys, AWS/GitHub/Slack/Google/OpenAI/Stripe tokens, JWTs, `api_key = "..."`; matches are masked in metadata |
| `no_prompt_injection` | input | heuristic: no; `mode="llm"`/`"decisions"`: yes | `on_error` default `"allow"` |
| `json_valid` | output | no | output must parse as JSON |
| `toxicity_check` | output | keyword: no; `mode="llm"`/`"decisions"`: yes | blocks when score >= `threshold` (0.5); `on_error` default `"allow"` |
| `openai_moderation` | output | yes: `OPENAI_API_KEY` + `openai` package | `on_error` default `"block"` |
| `banned_topics` | output | `mode="keyword"`: no; `mode="llm"` (default): yes | `topics` = names, `{name, description}` pairs, or a name->description dict; `on_error` default `"allow"` |
| `allowed_topics` | output | `mode="keyword"`: no; `mode="llm"` (default): yes | whitelist; `on_error` default `"block"` (a whitelist that cannot classify must not pass) |
| `grounded` / `no_hallucination` (same function) | output | yes | `reference` is a string or a zero-arg callable evaluated at check time; no reference -> passes with "groundedness skipped"; `on_error` default `"block"`; threshold 0.7 |
| `cost_limit` | output | no | blocks when run spend > `max_usd`; an unpriced model RAISES (so `on_error` decides, default `"block"`) |
| `allowed_domains` | tool_call | no | host must equal or be a subdomain of an entry; `evilexample.com` does not match `example.com` |
| `responsible_ai` | mixed | default bundle: no | list: injection (input) + PII + secrets (output) on by default; LLM-backed rails are opt-in; spread it: `guardrails=responsible_ai(...)` |

`banned_topics` / `allowed_topics` in `mode="llm"` with no live-client `llm=` produce a `topic`-type rule; with `mode="keyword"` or a live `LLMClient` they stay a local `code` rule.

`mode="keyword"` is a word match: it needs no model, and it passes any text that contains a listed word ("tell me a joke about my order" passes an order-topics whitelist). Use it for offline tests or a coarse first line, and tell the user so; use `mode="llm"` when the meaning matters. A blocked run raises `GuardrailBlockedError` to the caller; an application should catch it and answer the person, not crash.

```python
from fastaiagent import Agent, GuardrailBlockedError
from fastaiagent.guardrail import banned_topics, json_valid, no_pii, no_secrets
from fastaiagent.testing import TestModel

assert no_pii().execute("call me on 123-45-6789").passed is False
assert json_valid().execute('{"a": 1}').passed is True
assert banned_topics(["politics"], mode="keyword").execute("let's discuss politics").passed is False
assert no_secrets().execute('api_key = "sk-proj-abcdefghijklmnopqrst"').passed is False

agent = Agent(name="bot", llm=TestModel(response="SSN is 123-45-6789"), guardrails=[no_pii()])
try:
    agent.run("hi")
except GuardrailBlockedError as err:
    assert err.guardrail_name == "no_pii"
```

A tool-call gate runs before the tool executes (the tool never runs on a block):

```python
from fastaiagent import Agent, FunctionTool, GuardrailBlockedError
from fastaiagent.guardrail import allowed_domains
from fastaiagent.testing import FunctionModel

fetched = []
def fetch(url: str) -> str:
    fetched.append(url)
    return "page"

calls = {"n": 0}
def model(messages):
    calls["n"] += 1
    if calls["n"] == 1:
        return "", [{"name": "fetch", "arguments": {"url": "https://evil.com/x"}}]
    return "done"

agent = Agent(name="web", llm=FunctionModel(model),
              tools=[FunctionTool(name="fetch", fn=fetch, description="Fetch a URL")],
              guardrails=[allowed_domains(["api.mycompany.com"])])
try:
    agent.run("go")
except GuardrailBlockedError:
    assert fetched == []
```

## Config-driven types (survive serialization and distribution)

`Guardrail(guardrail_type=..., config={...})`. These round-trip through `to_dict()`/`from_dict()` and are what a control plane distributes. A `code` rule's `fn=` cannot be serialized.

| Type | Key `config` fields |
|---|---|
| `regex` | `pattern`, `should_match` (default False: a match fails), `case_insensitive`, `mask_token`, `timeout_seconds` (ReDoS bound; a timeout FAILS, it does not allow) |
| `schema` | `schema` (or `json_schema`): a non-empty JSON Schema dict; validated with `jsonschema` (a core dependency) |
| `classifier` | `categories` {name: [keywords]}, optional `blocked` list (absent = every detected category blocks) |
| `pii` | `entities` list, `backend` (`regex` or `presidio`) |
| `secrets` | none, by design |
| `topic` | `topics`, `mode` (`deny` default, or `allow`), `backend` |
| `content_safety` | `categories` (S1..S14), `threshold`, `thresholds` per category, `backend` |
| `groundedness` | `context_key`, `answer_key`, `threshold` (0.7) |
| `llm_judge` | `prompt` (or `backend: "decisions"` + `instructions` + `threshold`), `pass_value` (`PASS`), `llm` |

`config["llm"]` takes shareable `LLMClient` kwargs only; an `api_key` there raises at construction. Model-backed judges need a key in the environment (`# needs: OPENAI_API_KEY` or the provider you configured).

## on_error, action, severity

Three independent axes: `blocking` (does it run inline and halt?), `on_error` (what does a check that COULD NOT RUN mean?), `action` (what does a genuine failure cost?).

- `on_error="block"` (default on `Guardrail`): a check that errors fails closed. `"allow"`: fails open. An errored result has `errored=True` either way, so a degraded pass is distinguishable: `passed=True, errored=True` is NOT a genuine pass. Check `result.errored`, not just `result.passed`.
- `action` is one of `block` (default), `warn`, `mask`, `override`, `reask`. An unknown value becomes `block`.
  - `warn`: record and continue. `mask`: replace the offending spans with `config["mask_token"]` (default `[REDACTED]`) and continue; only `regex`, `classifier`, `pii`, `secrets` can mask, and a mask with nothing to redact BLOCKS. `override`: replace the payload with `config["override_message"]` and continue. `reask`: re-prompt the model with the failure, up to `AgentConfig.guardrail_retries` (default 1); exhausting it blocks; at any position other than output, or when streaming, it blocks.
  - An errored check always blocks regardless of `action`. A rewrite the run cannot apply (a streamed reply, multimodal input) blocks instead of passing the payload through.
- Branch on `result.action_taken` (`none|blocked|warned|masked|overridden|reask`), never on the configured `result.action`. `result.modified_data` holds the rewrite.
- `severity` (`low|medium|high|critical`) and `floor` are carried and traced; nothing in local enforcement depends on them (`floor` is plane governance).
- Every run's guardrail firings are on `AgentResult.guardrails` (a `GuardrailFiring` has `name`, `position`, `action_taken`, `effect`, `passed`, `errored`). Use `firing.fired()` to find rules that did anything. `warn`/`mask` are only visible there.

```python
from fastaiagent import Agent, Guardrail
from fastaiagent.guardrail import GuardrailPosition, GuardrailType
from fastaiagent.testing import TestModel

redact = Guardrail(
    name="redact-ssn",
    guardrail_type=GuardrailType.regex,
    position=GuardrailPosition.output,
    config={"pattern": r"\b\d{3}-\d{2}-\d{4}\b", "mask_token": "[SSN]"},
    action="mask",
)
agent = Agent(name="bot", llm=TestModel(response="SSN is 123-45-6789"), guardrails=[redact])
result = agent.run("hi")
assert result.output == "SSN is [SSN]"
assert [(g.name, g.action_taken) for g in result.guardrails] == [("redact-ssn", "masked")]
```

## A check that cannot check must say so

> A guardrail whose configuration cannot check anything reports that it could not run (`errored=True`, decided by `on_error`) — never a clean pass.

Do NOT write a rule that returns `passed=True` when it has nothing to inspect — because a green rule over an unchecked payload reads as a healthy control. The runners already enforce this for config-driven types: an empty schema, `pii` with `entities: []` or an unknown entity, `regex` with no/empty/uncompilable pattern, `classifier` with no categories, `topic` with no topics or a mistyped `mode`, `content_safety` with only unknown categories, `groundedness` with no context, and `code` with no `fn` (for example after `from_dict`) all report `errored=True` and block under the default `on_error`.

```python
from fastaiagent import Guardrail
from fastaiagent.guardrail import GuardrailType

rule = Guardrail(name="empty-pii", guardrail_type=GuardrailType.pii,
                 config={"entities": []}, on_error="block")
result = rule.execute("my ssn is 123-45-6789")
assert result.errored and not result.passed

fail_open = Guardrail(name="empty-pii", guardrail_type=GuardrailType.pii,
                      config={"entities": []}, on_error="allow")
result = fail_open.execute("my ssn is 123-45-6789")
assert result.errored and result.passed  # a degraded pass, not a verdict
```

- When you add a new distributable type or a new config switch, add an unusable-config case to `tests/test_guardrail_unusable_config_sweep.py`; that sweep fails if a distributable type has none (`secrets` and `llm_judge` are exempt).
- Parameterised builtins (`cost_limit`, `allowed_domains`, `grounded`) and your own `fn=` do NOT survive `from_dict`: they come back with no callable and raise at execution rather than pass. Zero-argument builtins (`no_pii`, `no_secrets`, `json_valid`, `toxicity_check`, `no_prompt_injection`, `openai_moderation`) come back armed.

## guardrail_context

An output guardrail sees only the answer. A `groundedness` rule scores that answer against the context it should have used, taken from a run-scoped slot (per task, via `ContextVar`):

```python
# needs: OPENAI_API_KEY
import fastaiagent as fa
from fastaiagent import Agent, Guardrail, LLMClient
from fastaiagent.guardrail import GuardrailPosition, GuardrailType

rule = Guardrail(
    name="grounded-answer",
    guardrail_type=GuardrailType.groundedness,
    position=GuardrailPosition.output,
    config={"context_key": "context", "threshold": 0.7},
)
agent = Agent(name="qa", llm=LLMClient(provider="openai", model="gpt-4o-mini"), guardrails=[rule])
with fa.guardrail_context(context=["Refunds take 5 business days."]):
    reply = agent.run("How long do refunds take?")
```

```signature
fastaiagent.guardrail_context(**values)
fastaiagent.set_guardrail_context(**values)
fastaiagent.get_guardrail_context()
fastaiagent.clear_guardrail_context(token)
```

Do wrap the run in `with fa.guardrail_context(...)`. Don't rely on it being set: a `groundedness` rule that finds no context raises (could-not-run), it does not pass. For a locally written rule `grounded(lambda: latest_docs)` is the callable equivalent.

## Local versus control plane

- The SDK executes: every type above runs in-process. A connected agent also turns plane-authored rules (`regex`, `schema`, `classifier`, `llm_judge`, `content_safety`, `groundedness`, `topic`, `pii`, `secrets`) into `Guardrail(origin="plane")` at all four positions; a plane `code` rule is skipped because its logic is not distributable.
- Plane-authored rules are enforced alongside local ones. `floor` and draft/publish governance are plane concepts; locally `floor` is only displayed.

### Plane-authored rules on a connected agent

You do not fetch them or pass them in. `connect()` pulls the policy once, and every run adds the
rules that apply to that agent to its own `guardrails=[...]`. What you write is the check that they
arrived:

```python
# needs: a connected plane with at least one active guardrail rule
import os

import fastaiagent as fa
from fastaiagent import Agent, GuardrailBlockedError, LLMClient

fa.connect(api_key=os.environ["FASTAIAGENT_API_KEY"], target=os.environ["FASTAIAGENT_TARGET"])
try:
    agent = Agent(name="support", llm=LLMClient(provider="openai", model="gpt-5-mini"))
    agent.push()  # registers it now, so its id is known before you list its rules
    plane_rules = fa.plane_guardrails_for_agent(agent.agent_id)
    print([(g.name, g.position.value) for g in plane_rules])
    try:
        agent.run("an input one of those rules should block")
    except GuardrailBlockedError as err:
        print("blocked by", err.guardrail_name, "trace", err.trace_id)  # read this trace back
finally:
    fa.disconnect()
```

- **When they change.** The policy is not re-pulled by itself. A rule authored or edited after
  `connect()` applies after `fa.refresh_policy()`; the next run rebuilds the list.
- **Which ones apply.** A rule with no agents attached is domain-wide. A rule attached to agents
  applies to those plane agent ids only. `plane_guardrails_for_agent` takes the id, or the name of
  an agent this process has pushed. An agent gets its id when it registers, which a connected
  agent does on its first run; that run waits for it (up to 5 s) when rules are attached to agents.
  If registration takes longer, a warning says the attached rules were not applied to that run.
  The plane identifies an agent by name, so a new agent that reuses an existing name in the
  domain gets that agent's rules: pick a name that is not taken.
- **An empty list is not a pass.** It means none apply, the rule was authored after `connect()`,
  it is a `code` rule (skipped), or the policy pull failed. A failed pull is logged as a warning
  and keeps the last policy, so read the log and say what you saw; do not say the rules are
  enforced.
- **Do not copy them into code.** Do not rebuild a plane rule locally or pass the result of
  `plane_guardrails_for_agent` into `guardrails=[...]`. Plane rules are never pushed back with the
  agent's definition.
- **Testing.** Two tests, two different questions.
  *How does my app behave when a plane rule blocks?* Offline, with
  `fastaiagent.testing.plane_rules([...])` (1.105.0; `references/testing.md`): the rules go through
  the same cache and builder as the real policy. Do not write to `_connection.policy_cache`, pass
  a local guardrail (`no_pii()`, a regex) in their place, or add a `guardrails=` parameter to fake it.
  *Does my platform's rule reach this agent and fire?* Only a connected test answers that: in
  `tests/e2e/`, marked `e2e`, skipped when `FASTAIAGENT_TARGET` is not set. A passing
  `plane_rules` test is not proof that the platform enforces anything.
- **Evidence.** Each check is a guardrail span on the run's trace, which reaches the plane with the
  trace. The plane's own guardrail views are read in the console, not through the SDK.
- SDK-only: `action="reask"`, `guardrail_context`, `timeout_seconds` and the regex ReDoS bound, and `code` rules via `fn=`.
- Traces: local capture is full fidelity. Guardrail span detail leaving the process is an allowlist per type; setting `FASTAIAGENT_TRACE_PAYLOADS=0` stops payload-bearing attributes from being exported.
- Test offline with `TestModel` / `FunctionModel` (see `teams.md`); keep tests of model-backed judges in `tests/e2e/` marked `e2e`.
