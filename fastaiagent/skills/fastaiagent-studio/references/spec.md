# Writing a spec

A **spec** is the design of one agent, as one JSON object. Studio checks it, renders the SDK
code from it (the spec is the only input to the renderer), and keeps every version. You send it
whole each time (`studio spec push`); there is no partial update. Print the exact JSON Schema
with `fastaiagent studio spec schema`; this page is what a schema cannot say.

Validate before you push: `fastaiagent studio spec validate <draft> spec.json --json`. A
**problem** stops a version; a **warning** does not. Both name their place in `where`.

## The top level

| Field | Rule |
|---|---|
| `spec_version` | `1` |
| `name` | The agent's name in code: `^[a-z][a-z0-9_-]{2,63}$`. Not the draft's display name. |
| `description` | 1 to 1,000 characters. |
| `architecture` | `single`, `supervisor` or `swarm`. (`chain` is in the schema but is refused.) |
| `system_prompt` | A single agent's prompt, or a supervisor's routing prompt. At least 40 characters. A swarm has none: its workers own theirs. |
| `llm` | `{"endpoint_id": "<id>"}`, an LLM endpoint of the project (`studio context`). The generated code is written for that endpoint's provider and model. |
| `tools` | A single agent's tools. For a team it must be empty: every tool belongs to a worker. |
| `guardrails` | Safety checks on the whole agent (see below). |
| `eval_cases` | **At least 3.** For a team these are the end-to-end cases. |
| `config` | Optional: `max_iterations` (1 to 25, default 10), `temperature` (0 to 2), `max_tokens` (1 to 32000). |
| `workers` | Teams only: **2 to 6**, names unique. |
| `max_delegation_rounds` | Supervisor only, 1 to 8 (default 3). |
| `entrypoint`, `handoffs`, `max_handoffs` | Swarm only. `max_handoffs` is 1 to 16 (default 8). |

A team size a person fixed in the console binds a spec sent here: a different worker count is a
problem that names the fixed size.

## Tools

Every tool has `name` (a lowercase Python identifier, **unique across the whole spec**),
`description` (1 to 500 characters; the agent's model reads it), `parameters`, and optionally
`fixture_ref` (the sample-data file that answers for it in a trial).

`parameters` is flat. Each entry is an object, never a bare string:

```json
{ "order_id": { "type": "string", "description": "the order number" } }
```

`type` is `string`, `integer`, `number` or `boolean`. Parameter names are lowercase identifiers.

| `type` | Extra fields | In the exported code | In a trial |
|---|---|---|---|
| `mock` | `fixture_ref` (required), `match_keys` (each must be one of `parameters`), `default_response` | A lookup over the bundled sample data | The same lookup, served by Studio |
| `rest` | `url` (http or https), `method` (`GET` default, `POST`, `PUT`, `DELETE`), `auth` (`none`, `bearer_env`, `api_key_env`) | A `RESTTool`; the secret is read from the environment | Answered from `fixture_ref` if set |
| `mcp` | `server_url`, `tool_name` | An `MCPTool` | Answered from `fixture_ref` if set |
| `kb` | `kb_id` (a knowledge base from `studio context`) | A `PlatformKB` search; needs a draft made with `--harness` | With `kb_id`: a search of that knowledge base. With `fixture_ref`: sample data |
| `function` | `returns` (`str`, `int`, `float`, `bool`) | A typed stub the developer finishes | Answered from `fixture_ref` if set |

**No tool ever carries a secret.** With `auth: bearer_env` the exported code reads
`TOOL_API_TOKEN`; with `api_key_env` it reads `TOOL_API_KEY`. Never put a key in a spec.

**Sample data must be reachable.** If you uploaded data, at least one tool must read it, and a
`mock` tool names it with `fixture_ref` and the sample file's name (`orders.csv` is `orders`). A
`mock` tool matches rows on `match_keys` (for `order_id`, the row whose `order_id` equals what the
agent passed) and answers `default_response` when no row matches.

## Guardrails

`{"factory": "<name>", "kwargs": {...}}`. An unknown name, a missing required argument or an
extra one is a problem. `studio context` lists the names; these are the arguments:

| Factory | Arguments | Notes |
|---|---|---|
| `no_pii` | none | |
| `no_secrets` | none | |
| `no_prompt_injection` | none | |
| `responsible_ai` | none | |
| `openai_moderation` | none | Calls OpenAI's moderation API. The exported agent needs the `openai` package and an OpenAI key, and **blocks every answer without them** |
| `toxicity_check` | optional `threshold` | |
| `cost_limit` | required `max_usd` | |
| `allowed_domains` | required `domains` | |
| `banned_topics` | required `topics` | |
| `allowed_topics` | required `topics` | In a **swarm** it is generated to check the question, not the answer (warning `guardrail_checks_question`) |
| `json_valid` | none | **Warns.** Blocks every answer that is not a JSON document; only for an agent told to answer in JSON |
| `grounded` | required `reference`, optional `threshold` | **Warns.** Checks the answer against a fixed reference text, so answers built from tool results fail |
| `no_hallucination` | required `reference`, optional `threshold` | **Warns.** Same as `grounded` |

Guardrails are **not applied in a sandbox trial**: only in the exported code. A worker with its
own non-empty `guardrails` replaces the team's for that worker; an empty list inherits them.

## Eval cases

```json
{
  "name": "found_1003",
  "input": "What is the status of order 1003?",
  "expected_output": "Order 1003 has shipped.",
  "criteria": { "must_mention": ["1003", "shipped"], "must_not_mention": ["refund"] }
}
```

`name` and `input` are required. `criteria` may hold `must_mention` (a list), `must_not_mention`
(a list) and `behaviour` (a sentence).

**Scoring is by word rules, not a judge.** Every `must_mention` value must appear; no
`must_not_mention` value may; and a case that does not use those rules passes on a 60% word overlap
with `expected_output`. A case with only a `behaviour` can be neither passed nor failed: it reads
**needs review**, and a person reads the answer. A value too short to score (one character, say)
is a warning (`eval_value_unusable`) and is ignored. Prefer specific words ("shipped", "1003") over
common ones. Write at least one case that should be refused or answered "not found".

## Workers

```json
{
  "name": "orders",
  "description": "Order status and delivery questions",
  "system_prompt": "You answer order status questions. Use orders_lookup with the order id and answer briefly.",
  "tools": [],
  "guardrails": [],
  "eval_cases": []
}
```

`name` is a lowercase identifier, unique in the spec; `description` (1 to 500) is what a supervisor
or a peer reads to choose; `system_prompt` is at least 40 characters. Optional: `llm_endpoint_id`
(another endpoint of the project), `max_iterations`. A worker's own `eval_cases` are scored in a
sandbox, one worker at a time; give each worker at least one, or `studio eval --worker` refuses
that worker (`worker_has_no_cases`).

## Swarms

Peers hand control to each other. Name an `entrypoint` (the first worker; it defaults to the first
listed) and list each allowed handoff as `{"from_worker", "to_worker", "when"}`. A worker cannot
hand off to itself, and **every worker must be reachable from the entrypoint**; an unreachable one is
a problem. A swarm has no `system_prompt`.

## Warnings

| Code | Meaning |
|---|---|
| `guardrail_blocks_output` | A guardrail that will block every answer this agent gives (`json_valid`, `grounded`, `no_hallucination`) |
| `guardrail_checks_question` | In a swarm, `allowed_topics` is generated to check the question |
| `eval_value_unusable` | An eval value too short to score reliably; it is ignored |
| `eval_dataset_not_updated` | The draft has a linked eval dataset and the spec carries different cases; this version uses the spec's cases |

Nothing is changed behind your back: the spec that is checked is the spec that is rendered.

## Example: one agent

Sample data: `orders.csv` with `order_id`, `status`, `total`.

```json spec
{
  "spec_version": 1,
  "name": "order-status-bot",
  "description": "Answers questions about customer orders from the uploaded orders data.",
  "architecture": "single",
  "system_prompt": "You answer order-status questions for customers. Use the orders_lookup tool with the order id. If the order is not found, say so plainly. Never reveal one customer's order to another. Answer in one or two sentences.",
  "llm": { "endpoint_id": "ENDPOINT_ID" },
  "tools": [
    {
      "type": "mock",
      "name": "orders_lookup",
      "description": "Look up one order by its id",
      "parameters": { "order_id": { "type": "string", "description": "the order number" } },
      "fixture_ref": "orders",
      "match_keys": ["order_id"],
      "default_response": "No order with that id."
    }
  ],
  "guardrails": [{ "factory": "no_pii", "kwargs": {} }],
  "eval_cases": [
    { "name": "found_1003", "input": "What is the status of order 1003?", "criteria": { "must_mention": ["1003", "shipped"] } },
    { "name": "missing_9999", "input": "Where is order 9999?", "criteria": { "behaviour": "says plainly that no such order was found" } },
    { "name": "delivered_1004", "input": "Has order 1004 arrived?", "criteria": { "must_mention": ["1004", "delivered"] } }
  ]
}
```

## Example: a supervisor team

A supervisor routes each request to one worker and gives the answer. Every tool is a worker's.

```json spec
{
  "spec_version": 1,
  "name": "support-triage-team",
  "description": "Routes a support request to the orders or the billing specialist.",
  "architecture": "supervisor",
  "system_prompt": "You route each customer request to the right specialist, orders or billing, then give the customer their answer in one or two sentences.",
  "llm": { "endpoint_id": "ENDPOINT_ID" },
  "max_delegation_rounds": 3,
  "tools": [],
  "guardrails": [],
  "workers": [
    {
      "name": "orders",
      "description": "Order status and delivery questions",
      "system_prompt": "You answer order status questions. Use orders_lookup with the order id and answer briefly.",
      "tools": [
        {
          "type": "mock",
          "name": "orders_lookup",
          "description": "Look up one order by its id",
          "parameters": { "order_id": { "type": "string", "description": "the order number" } },
          "fixture_ref": "orders",
          "match_keys": ["order_id"],
          "default_response": "No order with that id."
        }
      ],
      "eval_cases": [
        { "name": "orders_1003", "input": "Where is order 1003?", "criteria": { "must_mention": ["1003", "shipped"] } }
      ]
    },
    {
      "name": "billing",
      "description": "What an order cost",
      "system_prompt": "You answer questions about what an order cost. Use billing_lookup with the order id and answer briefly.",
      "tools": [
        {
          "type": "mock",
          "name": "billing_lookup",
          "description": "Look up what one order cost",
          "parameters": { "order_id": { "type": "string", "description": "the order number" } },
          "fixture_ref": "orders",
          "match_keys": ["order_id"],
          "default_response": "No order with that id."
        }
      ],
      "eval_cases": [
        { "name": "billing_1002", "input": "What did order 1002 cost?", "criteria": { "must_mention": ["89"] } }
      ]
    }
  ],
  "eval_cases": [
    { "name": "end_to_end_status", "input": "Where is order 1003?" },
    { "name": "end_to_end_cost", "input": "What did order 1002 cost?" },
    { "name": "end_to_end_unknown", "input": "Where is order 9999?" }
  ]
}
```

## Example: a swarm

```json spec
{
  "spec_version": 1,
  "name": "support-swarm",
  "description": "Peers that hand a customer to whoever can help, starting with orders.",
  "architecture": "swarm",
  "llm": { "endpoint_id": "ENDPOINT_ID" },
  "entrypoint": "orders",
  "max_handoffs": 6,
  "tools": [],
  "guardrails": [{ "factory": "no_pii", "kwargs": {} }],
  "workers": [
    {
      "name": "orders",
      "description": "Order status. Hands billing questions to billing.",
      "system_prompt": "You answer order status questions with orders_lookup. If the customer asks what an order cost, hand off to billing.",
      "tools": [
        {
          "type": "mock",
          "name": "orders_lookup",
          "description": "Look up one order by its id",
          "parameters": { "order_id": { "type": "string", "description": "the order number" } },
          "fixture_ref": "orders",
          "match_keys": ["order_id"],
          "default_response": "No order with that id."
        }
      ],
      "eval_cases": [
        { "name": "orders_1003", "input": "Where is order 1003?", "criteria": { "must_mention": ["1003", "shipped"] } }
      ]
    },
    {
      "name": "billing",
      "description": "What an order cost. Hands status questions back to orders.",
      "system_prompt": "You answer questions about what an order cost with billing_lookup. If the customer asks where an order is, hand off to orders.",
      "tools": [
        {
          "type": "mock",
          "name": "billing_lookup",
          "description": "Look up what one order cost",
          "parameters": { "order_id": { "type": "string", "description": "the order number" } },
          "fixture_ref": "orders",
          "match_keys": ["order_id"],
          "default_response": "No order with that id."
        }
      ],
      "eval_cases": [
        { "name": "billing_1002", "input": "What did order 1002 cost?", "criteria": { "must_mention": ["89"] } }
      ]
    }
  ],
  "handoffs": [
    { "from_worker": "orders", "to_worker": "billing", "when": "the customer asks what an order cost" },
    { "from_worker": "billing", "to_worker": "orders", "when": "the customer asks where an order is" }
  ],
  "eval_cases": [
    { "name": "end_to_end_status", "input": "Where is order 1003?" },
    { "name": "end_to_end_cost", "input": "What did order 1002 cost?" },
    { "name": "end_to_end_unknown", "input": "Where is order 9999?" }
  ]
}
```

## Choosing a shape

Start with a single agent. Use a supervisor when distinct specialists have distinct tools and one
router should pick; use a swarm when the specialists hand the same conversation back and forth.
Do not add workers to look thorough: each one is a prompt, tools and cases someone has to keep true.
