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

Five ready-to-use factories cover common safety needs:

### no_pii()

Detects SSNs, email addresses, phone numbers, and credit card numbers using regex patterns.

```python
from fastaiagent.guardrail import no_pii, GuardrailPosition

# On output (default) — catches PII in LLM responses
agent = Agent(guardrails=[no_pii()])

# On input — blocks users from sending PII to the LLM
agent = Agent(guardrails=[no_pii(position=GuardrailPosition.input)])

# Both directions
agent = Agent(guardrails=[
    no_pii(position=GuardrailPosition.input),
    no_pii(position=GuardrailPosition.output),
])
```

**Detected patterns:**
| Type | Pattern Example |
|------|----------------|
| SSN | `123-45-6789` |
| Email | `user@example.com` |
| Phone | `555-123-4567` |
| Credit Card | `4111 1111 1111 1111` |

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

## Five Implementation Types

Beyond inline functions, guardrails support four more implementation types for configuration-driven validation:

### Code (default)

Python function execution, as shown above.

```python
Guardrail(
    name="custom_check",
    guardrail_type=GuardrailType.code,
    fn=lambda text: "confidential" not in text.lower(),
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

### Schema

JSON Schema validation — useful for structured agent output:

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

results = await execute_guardrails(
    guardrails=[guard1, guard2, guard3],
    data="text to validate",
    position=GuardrailPosition.output,
)

for r in results:
    print(f"Passed: {r.passed}, Time: {r.execution_time_ms}ms, Message: {r.message}")
```

## GuardrailResult

| Field | Type | Description |
|-------|------|-------------|
| `passed` | `bool` | Whether validation passed |
| `score` | `float \| None` | Optional quality score (0.0-1.0) |
| `message` | `str \| None` | Human-readable explanation |
| `execution_time_ms` | `int` | How long the check took |
| `metadata` | `dict` | Extra data (e.g., detected PII types, blocked categories) |

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
#   "description": "Blocks URLs in output"
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
