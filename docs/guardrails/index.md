# Guardrails

Guardrails validate data at every stage of agent execution — before the LLM sees user input, after the LLM responds, and around tool calls. They can block unsafe content, enforce schemas, detect PII, or run any custom validation logic.

> **Guardrails vs Middleware.** Guardrails **assert** (pass/fail, block/allow). [Middleware](../agents/middleware.md) **transforms** (trim history, redact, rewrite). Use a guardrail when you want a policy check that raises on failure; use middleware when you want to change the data flowing through the pipeline. Input guardrails run before middleware's `before_model`; output guardrails run after middleware's `after_model`.

## How Guardrails Work

```
User Input → [Input Guardrails] → LLM → [Output Guardrails] → Response
                                    ↕
                            [Tool Call Guardrails]
                                    ↕
                            [Tool Result Guardrails]
```

1. **Input guardrails** run before the LLM sees the user's message
2. **Output guardrails** run on the LLM's response before returning to the user
3. **Tool call guardrails** run on tool arguments before execution
4. **Tool result guardrails** run on tool output before sending back to the LLM

If a **blocking** guardrail fails, execution stops immediately with `GuardrailBlockedError`.

## Built-in Guardrails

Ready-to-use factories cover common safety needs:

### no_pii()

Detects SSNs, email addresses, phone numbers, and credit card numbers. Credit
cards are **validated with the Luhn checksum** so random 16-digit strings don't
trip a false positive. Shares its detector with the
[`PIILeakage`](../evaluation/safety-metrics.md#piileakage) scorer.

```python
from fastaiagent.guardrail import no_pii, GuardrailPosition

# On output (default) — catches PII in LLM responses
agent = Agent(guardrails=[no_pii()])

# On input — blocks users from sending PII to the LLM
agent = Agent(guardrails=[no_pii(position=GuardrailPosition.input)])

# Opt into extra entity types, or the Presidio backend (needs [safety] extra)
agent = Agent(guardrails=[no_pii(entities=("email", "phone", "ssn", "credit_card", "ip"))])
```

**Detected patterns:**
| Type | Pattern Example |
|------|----------------|
| SSN | `123-45-6789` |
| Email | `user@example.com` |
| Phone | `555-123-4567` |
| Credit Card | `4111 1111 1111 1111` (Luhn-validated) |
| `ip` / `iban` | opt-in via `entities=` |

### no_prompt_injection()

Blocks prompt-injection / jailbreak attempts — input that tries to override,
ignore, or extract the system instructions ("ignore all previous instructions",
"reveal your system prompt", DAN, role-overrides, delimiter attacks). Defaults
to the `input` position. Zero-dependency heuristic mode by default; opt into an
LLM classifier with `mode="llm"`. Shares its detector with the
[`PromptInjection`](../evaluation/safety-metrics.md#promptinjection) scorer.

```python
from fastaiagent.guardrail import no_prompt_injection

# Blocks malicious user input before the LLM ever sees it
agent = Agent(guardrails=[no_prompt_injection()])

# Opt into the LLM-classifier mode (costs a call, catches more)
agent = Agent(guardrails=[no_prompt_injection(mode="llm")])
```

### openai_moderation()

Blocks content flagged by the OpenAI moderation endpoint. Defaults to the
`output` position. Requires the `openai` package and an API key.

```python
from fastaiagent.guardrail import openai_moderation

agent = Agent(guardrails=[openai_moderation()])
```

### json_valid()

Ensures the LLM's output is valid JSON. Useful for agents that must return structured data.

```python
from fastaiagent.guardrail import json_valid

agent = Agent(
    system_prompt="Always respond with valid JSON.",
    guardrails=[json_valid()],
    llm=LLMClient(provider="openai", model="gpt-4.1"),
)
```

### toxicity_check()

Keyword-based detection of toxic or harmful language.

```python
from fastaiagent.guardrail import toxicity_check

agent = Agent(guardrails=[toxicity_check()])
```

### cost_limit()

Policy marker for enforcing cost limits on agent execution.

```python
from fastaiagent.guardrail import cost_limit

agent = Agent(guardrails=[cost_limit(max_usd=0.50)])
```

### allowed_domains()

Restricts URLs in tool calls to a whitelist of domains. Position defaults to `tool_call`.

```python
from fastaiagent.guardrail import allowed_domains

agent = Agent(
    guardrails=[allowed_domains(["api.mycompany.com", "internal.service.local"])],
    tools=[my_rest_tool],
)

# Tool calls to https://evil.com will be blocked
# Tool calls to https://api.mycompany.com/data will pass
```

## Custom Guardrails

### Inline Function

The simplest way — pass a function that returns `True` (pass) or `False` (block):

```python
from fastaiagent.guardrail import Guardrail, GuardrailPosition

# Block responses longer than 500 characters
length_guard = Guardrail(
    name="max_length",
    position=GuardrailPosition.output,
    blocking=True,
    fn=lambda text: len(text) < 500,
)

# Block input containing specific keywords
keyword_guard = Guardrail(
    name="no_competitor_names",
    position=GuardrailPosition.input,
    blocking=True,
    fn=lambda text: not any(name in text.lower() for name in ["competitor_a", "competitor_b"]),
)
```

### Returning GuardrailResult

For richer feedback, return a `GuardrailResult` with score and message:

```python
from fastaiagent.guardrail import Guardrail, GuardrailResult

def check_quality(text: str) -> GuardrailResult:
    word_count = len(text.split())
    if word_count < 10:
        return GuardrailResult(
            passed=False,
            score=word_count / 10,
            message=f"Response too short ({word_count} words, minimum 10)",
        )
    return GuardrailResult(passed=True, score=1.0)

quality_guard = Guardrail(name="quality_check", fn=check_quality)
```

## Eight Implementation Types

Beyond inline functions, guardrails support seven more implementation types for configuration-driven validation. The last three — `content_safety`, `groundedness` and `topic` — are model-backed judges with structure, and are documented in full under [Actions, severity & floor](actions.md#three-model-backed-check-types):

### Code (default)

Python function execution, as shown above. **Always** pass the callable
via `fn=`. The legacy `config={"code": "..."}` string-execution path was
removed in 1.10.0 because its sandbox was bypassable; supplying a `code`
string now fails closed without executing anything.

```python
Guardrail(
    name="custom_check",
    guardrail_type=GuardrailType.code,
    fn=lambda text: "confidential" not in text.lower(),
)
```

#### Migrating from `config={"code": "..."}` (1.9.0 → 1.10.0)

If you previously wrote a guardrail by passing a Python *string* through
config — e.g. loading guardrail definitions from a YAML/JSON file — move
the logic into a real function and pass it via `fn=`. Three common shapes:

```python
from fastaiagent.guardrail import Guardrail, GuardrailType, GuardrailResult

# 1. One-liner: lambda is enough.
#
#    BEFORE (no longer executes — fails closed):
#      Guardrail(
#          name="no_secret",
#          guardrail_type=GuardrailType.code,
#          config={"code": "result = 'secret' not in data"},
#      )
#
#    AFTER:
no_secret = Guardrail(
    name="no_secret",
    fn=lambda text: "secret" not in text.lower(),
)


# 2. Multiple checks + custom message: use a named function.
#
#    BEFORE:
#      Guardrail(
#          name="length_band",
#          guardrail_type=GuardrailType.code,
#          config={"code": "result = 10 <= len(data) <= 500"},
#      )
#
#    AFTER:
def length_band(text: str) -> GuardrailResult:
    n = len(text)
    if n < 10:
        return GuardrailResult(passed=False, message=f"Too short ({n} chars)")
    if n > 500:
        return GuardrailResult(passed=False, message=f"Too long ({n} chars)")
    return GuardrailResult(passed=True, score=1.0)

length_guard = Guardrail(name="length_band", fn=length_band)


# 3. Loading guardrails from config files: import the callable by name
#    instead of embedding source code in YAML/JSON.
#
#    Recommended: ship a small registry module the loader can resolve,
#    e.g. ``my_app.guardrails:no_secret``. The loader looks up the
#    callable and passes it via ``fn=``.
```

If you cannot move the logic into Python (e.g. the rules genuinely live
in user-supplied configuration), reach for a non-code guardrail type
instead — `GuardrailType.regex`, `.schema`, or `.classifier` cover the
same cases declaratively without executing arbitrary code:

```python
# Pattern check expressed declaratively — no code execution at all.
no_secret = Guardrail(
    name="no_secret",
    guardrail_type=GuardrailType.regex,
    config={"pattern": r"\bsecret\b", "should_match": False, "case_insensitive": True},
)
```

### Regex

Pattern matching without writing a function:

```python
from fastaiagent.guardrail import Guardrail, GuardrailType

# Block output containing URLs
no_urls = Guardrail(
    name="no_urls",
    guardrail_type=GuardrailType.regex,
    config={
        "pattern": r"https?://[^\s]+",
        "should_match": False,       # Fail if pattern IS found
        "case_insensitive": True,
    },
)

# Require output to contain a reference number
has_ref = Guardrail(
    name="has_reference",
    guardrail_type=GuardrailType.regex,
    config={
        "pattern": r"REF-\d{6}",
        "should_match": True,        # Fail if pattern is NOT found
    },
)
```

Regex guardrails run on a ReDoS-resistant engine under a hard timeout, so a
catastrophic-backtracking pattern fails closed instead of hanging the run. The
default is 2 seconds; adjust with `"timeout_seconds"` in the config (clamped to
0.1–10s so a plane-supplied policy can't disable the protection):

```python
config={"pattern": r"...", "should_match": False, "timeout_seconds": 1.0}
```

### Schema

JSON Schema validation — useful for structured agent output.

!!! warning "A rule with no schema errors, it does not pass"
    An empty `schema` validates *everything*: the validator finds no violations
    in `{}`, so the rule reported every payload as valid while showing as an
    active control — worse than no rule, because it looks like one. Since 1.59.0
    a missing, empty or non-object schema **raises**, so `on_error` decides what
    it costs and the result is marked `errored`. `json_schema` is accepted as an
    alias for `schema`.

```python
schema_guard = Guardrail(
    name="response_schema",
    guardrail_type=GuardrailType.schema,
    config={
        "schema": {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "sources": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["answer", "confidence"],
        }
    },
)
```

### LLM Judge

Use an LLM to evaluate quality. The judge LLM responds with PASS or FAIL:

```python
judge_guard = Guardrail(
    name="relevance_judge",
    guardrail_type=GuardrailType.llm_judge,
    blocking=False,  # Log but don't block
    config={
        "prompt": "Is this response relevant and helpful? Respond PASS or FAIL.\n\nResponse: {data}",
        "pass_value": "PASS",
        "llm": {"provider": "openai", "model": "gpt-4.1"},
    },
)
```

### Classifier

Keyword-based category detection with blocked category lists:

```python
content_filter = Guardrail(
    name="content_filter",
    guardrail_type=GuardrailType.classifier,
    config={
        "categories": {
            "financial_advice": ["invest", "stock", "portfolio", "buy shares"],
            "medical_advice": ["diagnosis", "prescribe", "treatment plan"],
            "legal_advice": ["sue", "liable", "legal action"],
        },
        "blocked": ["financial_advice", "medical_advice", "legal_advice"],
    },
)
```

### Model-backed judges

Three types are judges with *structure* — a model call whose question, and whose
answer, have a fixed shape, so an operator can say which harm, which bar, or
which topic they mean and read from the audit row which one tripped. Their
prompts and parsers are mirrored from the plane so a rule reaches the same
verdict at the edge as it does centrally. Each is documented in full under
[Actions, severity & floor](actions.md#three-model-backed-check-types):

| Type | Decides | Configured with |
|------|---------|-----------------|
| `content_safety` | Which MLCommons hazard categories (S1–S14) the payload hits, each against its own bar | `categories`, `threshold`, `thresholds` |
| `groundedness` | Whether an answer is supported by the context it was given | `threshold`, `context_key`, `answer_key` |
| `topic` | Which named topics the payload discusses, then `deny` or `allow` on that | `topics` (`{name, description}`), `mode` |

```python
no_competitors = Guardrail(
    name="no_competitors",
    guardrail_type=GuardrailType.topic,
    config={
        "topics": [
            {"name": "Competitor products",
             "description": "Any mention or comparison of a competing vendor's product."},
        ],
        "mode": "deny",   # "allow" makes the same list an on-topic gate
    },
)
```

Unlike `classifier` above, this is not substring matching: the *description* is
what lets the judge catch "the other vendor's offering" without the word
"competitor" appearing anywhere.

## Positions

All four guardrail positions are fully wired and operational:

| Position | When it runs | Use case |
|----------|-------------|----------|
| `GuardrailPosition.input` | Before LLM sees user message | Block PII, profanity, prompt injection |
| `GuardrailPosition.output` | After LLM responds | Block PII leaks, validate format, quality check |
| `GuardrailPosition.tool_call` | Before tool executes | Restrict URLs, validate arguments, audit |
| `GuardrailPosition.tool_result` | After tool returns (success only) | Validate tool output, filter sensitive data |

```python
from fastaiagent import Agent, LLMClient
from fastaiagent.guardrail import Guardrail, GuardrailPosition, allowed_domains

agent = Agent(
    name="safe-agent",
    llm=LLMClient(provider="openai", model="gpt-4.1"),
    tools=[my_api_tool],
    guardrails=[
        # Block tool calls to unapproved domains
        allowed_domains(["api.mycompany.com"]),
        # Block sensitive data in tool results
        Guardrail(
            name="no-secrets-in-results",
            position=GuardrailPosition.tool_result,
            blocking=True,
            fn=lambda text: "sk-" not in text,
        ),
    ],
)
```

Tool-position guardrails work in both `arun()` and `astream()` execution modes.

```python
from fastaiagent.guardrail import GuardrailPosition

# Same guardrail logic, different positions
agent = Agent(
    guardrails=[
        Guardrail(name="input_check", position=GuardrailPosition.input, fn=check_fn),
        Guardrail(name="output_check", position=GuardrailPosition.output, fn=check_fn),
    ],
)
```

## Blocking vs Non-Blocking

**Blocking** (default): Execution stops immediately if the guardrail fails. A `GuardrailBlockedError` is raised.

```python
# Blocking — raises exception on failure
strict = Guardrail(name="strict", blocking=True, fn=my_check)
```

**Non-blocking**: Failure is recorded but execution continues. Useful for monitoring and logging without interrupting the user.

```python
# Non-blocking — logs failure, continues execution
monitor = Guardrail(name="quality_monitor", blocking=False, fn=quality_check)
```

When multiple guardrails are attached to an agent:
1. **Blocking** guardrails run first, sequentially — first failure stops everything
2. **Non-blocking** guardrails run in parallel after all blocking guardrails pass

## Guardrail Executor

For advanced use cases, call the executor directly:

```python
from fastaiagent.guardrail import execute_guardrails, GuardrailPosition

outcome = await execute_guardrails(
    guardrails=[guard1, guard2, guard3],
    data="text to validate",
    position=GuardrailPosition.output,
)

for r in outcome:
    print(f"Passed: {r.passed}, Time: {r.execution_time_ms}ms, Message: {r.message}")

# A `mask` or `override` rule rewrites the payload, so the outcome also carries
# the value to carry forward. `outcome` iterates and indexes like the list it
# used to be, so existing code keeps working.
print(outcome.data, outcome.modified)
```

| Field | Description |
|-------|-------------|
| `results` | The verdicts, one per applicable guardrail |
| `data` | The payload to carry forward — rewritten when something rewrote it, otherwise the original |
| `modified` | Did anything rewrite it? |
| `reask` | The first rule that asked to re-prompt the model, if any |

## GuardrailResult

| Field | Type | Description |
|-------|------|-------------|
| `passed` | `bool` | Whether validation passed |
| `score` | `float \| None` | Optional quality score (0.0-1.0) |
| `message` | `str \| None` | Human-readable explanation |
| `execution_time_ms` | `int` | How long the check took |
| `metadata` | `dict` | Extra data (e.g., detected PII types, blocked categories) |
| `errored` | `bool` | True when the check itself failed to run; `passed` then reflects the `on_error` policy, not a verdict |
| `action` | `str` | What the rule was *configured* to cost: `block` / `warn` / `mask` / `override` / `reask` |
| `action_taken` | `str` | What it *actually did*: `none` / `blocked` / `warned` / `masked` / `overridden` / `reask`. Branch on this, never on `action` |
| `modified_data` | `str \| dict \| None` | The rewritten payload, when the action produced one |

See [Actions, severity & floor](actions.md) for what each action does and the
two cases where a rewrite degrades to a block.

## Fail policy: `on_error`

Model-judged guardrails depend on an LLM call that can fail. `on_error` decides
what a *failed check* means, independent of the verdict it would have returned:

```python
from fastaiagent.guardrail import toxicity_check, grounded

toxicity_check(mode="llm", on_error="allow")  # fail open — an error passes through
grounded(reference, on_error="block")         # fail closed — an error blocks

# Works on any guardrail you build yourself:
Guardrail(name="my_judge", guardrail_type=GuardrailType.llm_judge,
          config={...}, on_error="block")
```

| `on_error` | On a check error | Built-ins that default to it |
|------------|------------------|------------------------------|
| `"allow"` (fail open) | Content passes through | `toxicity_check`, `no_prompt_injection`, `banned_topics` |
| `"block"` (fail closed) | Content is blocked | `grounded`, `openai_moderation`, `allowed_topics`, custom `Guardrail`/`llm_judge` |

Whichever you choose, the failure is **visible**: the result is `errored=True`,
the trace span carries `fastaiagent.guardrail.errored`, and the Local UI logs an
[`errored` outcome](../ui/guardrail-events.md). Deterministic guardrails
(`no_pii`, `no_secrets`, `json_valid`, `allowed_domains`) can't make a fallible
call, so `on_error` doesn't apply to them. Override a whole `responsible_ai(...)`
bundle at once with `responsible_ai(on_error="block", ...)`.

## Serialization

Guardrails serialize to JSON for platform push:

```python
data = guardrail.to_dict()
# {
#   "name": "no_urls",
#   "guardrail_type": "regex",
#   "position": "output",
#   "config": {"pattern": "https?://...", "should_match": false},
#   "blocking": true,
#   "description": "Blocks URLs in output",
#   "on_error": "block",
#   "action": "block",
#   "severity": null,
#   "floor": false
# }

restored = Guardrail.from_dict(data)
```

> **Note:** Inline functions (`fn=`) are NOT serialized. After `from_dict()`, code guardrails with inline functions will have no executable logic. Use config-driven types (regex, schema, classifier) for guardrails that need to survive serialization.

## Error Handling

```python
from fastaiagent._internal.errors import GuardrailBlockedError

try:
    result = agent.run("Some input")
except GuardrailBlockedError as e:
    print(f"Guardrail: {e.guardrail_name}")  # Which guardrail blocked
    print(f"Message: {e}")                    # Why it blocked
    print(f"Results: {e.results}")            # All guardrail results up to the failure
```

---

## Next Steps

- [Agents](../agents/index.md) — Attach guardrails to agents
- [Tools](../tools/index.md) — Guard tool calls and results
- [Platform Sync](../platform/index.md) — Push guardrails to the platform
