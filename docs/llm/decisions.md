# OpenAI Decisions API

*New in 1.84.0.*

OpenAI's Decisions API (`POST /v1/decisions`, public beta) is not a chat endpoint.
You hand it **evidence** (text and/or images) and **questions whose answers you
fixed in advance**, and it returns a probability distribution for each question
instead of generated text. There's no prompt to engineer and no JSON to parse.

OpenAI's guide says it answers about **10× faster** than asking the same model through
the Responses API. It bills **input tokens only**. The endpoint serves one
model today, `gpt-6-luna`.

```python
from fastaiagent.llm import Choice, LLMClient, Predicate, Score

llm = LLMClient(model="gpt-6-luna")
r = llm.decide(
    "I was charged twice for the same subscription renewal.",
    {
        "department": Choice(
            instructions="Which department should handle this?",
            options={"billing": "Payments and refunds", "technical": "Product bugs", "other": None},
        ),
        "duplicate": Predicate(instructions="The customer reports a duplicate charge."),
        "severity": Score(instructions="How severe is this?", levels=["Cosmetic", "Inconvenient", "Blocking"]),
    },
)
r.choices["department"].choice        # "billing"
r.predicates["duplicate"].probability  # 1.0
r.scores["severity"].level             # "Inconvenient"
r.cost_usd                             # 5e-05
```

Runnable versions: [`102_decisions_basics.py`](https://github.com/fastaifoundry/fastaiagent-sdk/blob/main/examples/102_decisions_basics.py)
through [`105_decision_replay.py`](https://github.com/fastaifoundry/fastaiagent-sdk/blob/main/examples/105_decision_replay.py).

## When to use it

| You need | Use |
|---|---|
| A yes/no probability, one label from a fixed set, or a rating on a fixed scale | **`decide()`** |
| Extracted fields, or an object matching your own schema | [Structured output](../structured-output/index.md) (`acomplete(output_type=...)`) |
| The model to call tools with arguments | An `Agent` with tools |
| Generated text | `acomplete()` / `Agent.run()` |

Classification, routing, triage, moderation, relevance and "did this answer meet
the bar?" checks all fit `decide()`. Anything that has to *write* something doesn't.

## Questions

| Question | You supply | Main result |
|---|---|---|
| `Predicate(instructions)` | a condition about the input | `probability` (0..1) that it holds |
| `Choice(instructions, options)` | 2+ unordered options | `choice`, every option's `probabilities`, `confidence` |
| `Score(instructions, levels)` | 2+ ordered levels, lowest first | `score`, per-level `probabilities`, `confidence` |

- **Phrase a predicate as a statement**, not a question to answer in prose:
  `"The customer reports a duplicate charge."`
- **`options`** accepts a list (`["billing", "other"]`), a mapping of value to
  description (`{"billing": "Payments and refunds", ...}`), or `Option(value,
  description)` objects. Include a catch-all such as `"other"` when the options
  may not cover every input.
- **Choice values are typed.** `options=[True, False]` returns the boolean `True`, and
  the string `"true"` would be a *different* option.
- **`levels`** accepts labels or `Level(label, description)`.
- Pass **one question**, a **list**, or a **`{name: question}` mapping**. The key
  becomes the question's name.

A question that can't decide anything raises `ValueError` when you build it,
before any call is made. That covers empty instructions, fewer than two
options or levels, a repeated option, and two questions with the same name.

## Answers

`decide()` returns a `DecisionResult`. Answers come back **in question order**.

```python
r[0]                        # by position
r["department"]             # by name
r.predicates / r.choices / r.scores   # {name: answer}, filtered by type
r.refusals, r.refused       # see below
r.usage, r.cost_usd, r.latency_ms, r.request_id, r.model
```

**`score` is not 0..1.** It's the probability-weighted level index, so with
three levels it runs from `0` to `2` and can fall between levels: probabilities
`0.1 / 0.7 / 0.2` give `1.1`. Use `ScoreAnswer.normalized` for a 0..1 value, and
`ScoreAnswer.level` for the single most likely label.

OpenAI doesn't publish how `confidence` is computed. Pick thresholds from labelled
examples of your own traffic, weighing what a false positive and a false negative
each cost you.

### Refusals

Any **single** answer can come back as `Refusal(name)`, and the other questions in
the same request are still answered. The SDK never turns a refusal into a score.
Every integration below treats one as "could not decide":

| Where | A refusal means |
|---|---|
| `decide()` | a `Refusal` in `r.answers`; `r.refused` is `True` |
| Guardrails (`backend="decisions"`) | the rule **could not run**: `errored=True`, and `on_error` decides |
| `DecisionJudge` | score `0.0`, not passed, with a reason saying so |
| Chain decision node | the **default** edge |
| Supervisor `routing="decisions"` | the `fallback_worker` |
| `decision_tool` | `{"refused": true}` for that question |
| Supervisor validation | fail-open (approved, with a warning), the same as a failed call |

An answer the SDK can't read raises `LLMError` naming what arrived. That covers a
count that doesn't match the questions, or an answer type it doesn't know (the API
is in beta). The SDK never guesses, and never silently drops an answer.

## Input

- A `str`.
- An `Image`, or a list mixing `str` and `Image` (sent as one user message).
- A list of `UserMessage` objects.

```python
from fastaiagent.multimodal.image import Image

llm.decide(
    ["What colour fills this image?", Image.from_file("photo.png")],
    Choice(name="colour", instructions="Dominant colour", options=["red", "green", "blue"]),
)
```

Images are always sent **inline as base64 data URLs**, because OpenAI's types reject
external URLs and file ids. An `Image.from_url(...)` is therefore sent as its bytes.
`detail` passes through. PDFs, files, audio, and system/assistant/tool messages
aren't accepted by the endpoint, so they raise `LLMError` rather than being
silently converted.

## Client, providers, errors

`decide()` lives on `LLMClient`, so it reuses that client's API key
(`OPENAI_API_KEY`), `base_url`, TLS `verify`, `max_retries` (429 and 5xx are
retried, 4xx are not) and the image size cap.

| Client | Works? |
|---|---|
| `LLMClient(provider="openai", model="gpt-6-luna")` | yes, the default |
| `LLMClient(provider="custom", base_url=...)` | yes, for a gateway or proxy that serves `/decisions` |
| `LLMClient(openai_client=OpenAI(...))` / `AsyncOpenAI` / `AzureOpenAI` | yes, see below |
| `anthropic`, `ollama`, `bedrock`, any preset | no: `LLMError` says the endpoint is OpenAI-only |

**No `openai` upgrade needed.** The SDK calls the endpoint itself over `httpx`.
With an injected `openai_client`, it uses `client.decisions.create` on openai
≥ 3.26. On the 1.x/2.x most installs pin, it falls back to the client's generic
`client.post("/decisions")`, which still uses the client's own base URL, auth
(including Azure AD token refresh) and `http_client`.

A non-200 raises `LLMProviderError` with OpenAI's message and `.status_code`.
For example, a model your key can't use comes back as a 404 `model_not_found`.

## Cost

Priced at the **decisions rate**: `gpt-6-luna` is $0.10 per 1M input tokens, with
no output charge. That's separate from the chat price table, because the same
model id is billed differently there. Each call:

- sets `DecisionResult.cost_usd`;
- adds to the run's accumulated cost, so `AgentResult.cost` and `cost_limit`
  guardrails include decisions;
- stamps `fastaiagent.cost.total_usd` on its span, so the Local UI's per-model cost
  shows it.

A model with no known decision rate leaves `cost_usd=None`, and the run's cost is
reported as unknown rather than `$0.00`.

![Analytics cost by model: the Decisions API next to the chat models](../ui/screenshots/decisions-07-analytics.png)

## Tracing and egress

Each call emits one span named **`llm.<provider>.decisions.<model>`**: provider,
then the API, then the model. In a trace it sits next to chat spans such as
`llm.openai.gpt-5.1`:

```text
supervisor.call-center
├── llm.openai.decisions.gpt-6-luna    ← a Decisions API call, made with gpt-6-luna
├── agent.complaints
│   ├── llm.openai.gpt-5.1             ← a chat completion
│   ├── tool.lookup_order
│   └── llm.openai.gpt-5.1
└── llm.openai.decisions.gpt-6-luna    ← the review
```

`gpt-6-luna` is the model. The **Decisions API** is how it's called: the same
model also answers through the Responses API, and OpenAI's guide compares the two
paths. Through a gateway (`provider="custom"`) the span reads
`llm.custom.decisions.<model>`.

The span carries the SDK's own keys:

- `fastaiagent.decision.questions`, `fastaiagent.decision.answers` and
  `fastaiagent.decision.refusals`;
- `fastaiagent.cost.total_usd`;
- `fastaiagent.decision.input`, a summary of the evidence. Images appear by type
  and size, never their bytes.

The evidence is user content, so **`fastaiagent.decision.input` is stripped on export
when `FASTAIAGENT_TRACE_PAYLOADS=0`**. It still stays in your local trace store for
the UI and Replay. The questions and answers are your own structure plus
probabilities, so they're exported the same way guardrail scores are.

In the Local UI, a Decisions span sits in the trace tree like any LLM call. Its
**Attributes** tab shows the standard keys, the questions, and the answers with
their probabilities:

![A Decisions API span in the Local UI](../ui/screenshots/decisions-04-decisions-span.png)

![The same span's answers, usage and cost](../ui/screenshots/decisions-04b-decision-answers.png)

### OpenTelemetry and OpenInference

A Decisions call is a standard **`CLIENT`-kind GenAI span**, so any OTel backend
you register with `fastaiagent.trace.otel.add_exporter(...)` (Datadog, Jaeger,
Grafana, Langfuse, Phoenix and others) reads it as an LLM call with no
SDK-specific mapping:

| Convention | Attributes |
|---|---|
| OTel GenAI semantic conventions | `gen_ai.operation.name="decisions"` (named after the API, as `chat` is for `/chat/completions`), `gen_ai.provider.name="openai"`, `gen_ai.request.model`, `gen_ai.response.model`, `gen_ai.response.id`, `gen_ai.usage.input_tokens` / `output_tokens`, plus `gen_ai.system` as on chat spans |
| OpenInference (Phoenix / Arize) | `openinference.span.kind="LLM"`, `llm.provider`, `llm.model_name`, `llm.token_count.prompt` / `completion` / `total` |

The exporter goes through the same egress filter as the plane, so with
`FASTAIAGENT_TRACE_PAYLOADS=0` the user's text never reaches it.

## Replay

A recorded rerun (`with_determinism("recorded")`) serves each captured decision
back from the trace, in order, exactly as it serves captured chat turns. A
decision-driven agent therefore replays **with no model calls at all**:

```python
from fastaiagent.trace.replay import Replay

rerun = await (
    Replay.load(trace_id)
    .fork_at(step=0)
    .with_determinism("recorded", on_miss="error")
    .arerun()
)
```

A rerun that asks a decision the original didn't is a **miss**, governed by
`on_miss`: `"error"` raises `ReplayError`, and `"live"` warns and calls. A miss
covers more decisions than were captured, or a captured decision that answered
different questions. See [`105_decision_replay.py`](https://github.com/fastaifoundry/fastaiagent-sdk/blob/main/examples/105_decision_replay.py).

![A recorded rerun: the Decisions span tagged replay.mode="recorded"](../ui/screenshots/decisions-06-replay-recorded.png)

## Testing without the network

```python
from fastaiagent.testing import FunctionModel, TestModel

# Canned answers, served in order (the last repeats).
llm = TestModel(decisions={"answers": [{"type": "predicate", "name": "spam", "probability": 0.97}]})

# Or compute them.
def decide_fn(input, questions):
    p = 0.95 if "WIN $$$" in input else 0.05
    return {"answers": [{"type": "predicate", "name": questions[0].name, "probability": p}]}

llm = FunctionModel(lambda messages: "ok", decide_fn=decide_fn)
```

Both emit the same Decisions span a live call does (`llm.test.decisions.<model>`). A `TestModel` with no
canned decision **raises** rather than inventing one.

## Built on `decide()`

| Feature | Page |
|---|---|
| Guardrails with `backend="decisions"` (`topic`, `content_safety`, `llm_judge`) and `mode="decisions"` builtins | [Guardrails](../guardrails/index.md#decisions-api-backend) |
| `DecisionJudge`, an eval judge that returns a probability | [LLM judge](../evaluation/llm-judge.md#decisionjudge-decisions-api) |
| Chain `condition` nodes that route on a `Choice` | [Chains](../chains/index.md#routing-on-meaning-decision-nodes) |
| `decision_tool`, classification as an agent tool | [Function tools](../tools/function-tools.md#decision_tool) |
| Supervisor `routing="decisions"`: the Decisions API routes each request to one worker | [Teams](../agents/teams.md#routing-with-the-decisions-api-routingdecisions) |
| Supervisor `validation_mode="decisions"` | [Teams](../agents/teams.md#reviewing-workers-with-the-decisions-api) |

## Status

The Decisions API is in **public beta** (OpenAI expects GA "in the coming weeks").
The SDK follows OpenAI's published types (`openai` 3.26.0) and was checked against
the live endpoint. If the response shape changes, parsing fails loudly with
`LLMError` rather than misreading it.
