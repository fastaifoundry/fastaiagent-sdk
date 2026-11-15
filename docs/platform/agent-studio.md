# Agent Studio from the command line and from code

Agent Studio is the FastAIAgent platform's builder. You describe an agent, Studio checks the design,
renders real SDK code from it, tries it against sample data, scores eval cases, keeps every version
and exports a project. The console drives it with Studio's own model. This page is the other door:
**`fastaiagent studio …` and `fastaiagent.studio.StudioClient`**, an API-key route for coding tools
(Claude Code, GitHub Copilot, Cursor, Google Antigravity) and scripts.

The one idea: **this route never calls Studio's model.** The caller designs the agent and writes the
design as a **spec** (one JSON object). Studio checks the spec, renders the code, runs trials and
evals in its sandbox and hands the code back. After the export the code is yours; Studio assists, it
is not where agents live.

## What you need

- A platform at **0.20.0 or later** (wire protocol v1.12). An older one has no such routes, and says
  so (see [An older platform](#an-older-platform)).
- An API key with `studio:read` and `studio:write`. In the console's key screen, the **Agent Studio**
  preset has both. Neither is in a new key's default permissions. The key's owner must still be a
  member of the workspace (`key_has_no_owner` otherwise).
- At least one **LLM endpoint** in the project. A spec names the endpoint the agent runs on; with none,
  no version can be made.
- On a production or acceptance install Studio is off: every write is refused with `environment_rule`.
  Reads and the export still work.

Credentials are resolved as `fastaiagent push` resolves them: `--api-key` and `--target`, else
`FASTAIAGENT_API_KEY` and `FASTAIAGENT_TARGET`, else what `fastaiagent connect` saved. A key tied to a
whole domain must also say which project (`--project-id`).

## The loop

```bash
fastaiagent studio context --json                 # endpoints, knowledge bases, guardrail names, limits
fastaiagent studio new --name "Order desk" --brief "Answer order-status questions." \
    --data orders.csv --endpoint <endpoint-id>
fastaiagent studio spec validate "Order desk" spec.json
fastaiagent studio spec push "Order desk" spec.json --note "first design" --wait
fastaiagent studio show "Order desk"              # the blueprint; --files for the generated code
fastaiagent studio try "Order desk" "What is the status of order 1003?"
fastaiagent studio eval "Order desk"              # exit 0 only when no case failed
fastaiagent studio spec get "Order desk" --out spec.json   # to change it: edit, validate, push again
fastaiagent studio export "Order desk" --out .    # unpacks the project; run its tests
```

`<draft>` is a draft's id or its name. Every command takes `--json`: one JSON document on stdout, notes
and errors on stderr. Exit 0 means it happened; exit 1 means it did not (a refusal, a spec with
problems, an eval case that failed). With `--json`, a failure is also printed on stdout as
`{"error": {"code", "message", "hint", "status", "details"}}`. The full list is in the
[CLI reference](../cli/index.md#fastaiagent-studio).

There is no `studio plan`, `build`, `refine` or `reset`: planning and building from a description
need Studio's own model, and starting over is a console action. To change an agent, send a new spec;
it makes a new version and the old one is kept. A spec is always sent whole.

### What a spec is

A spec names the agent (`name`, `description`), its shape (`architecture`: `single`, `supervisor` or
`swarm`), its prompt, the project's LLM endpoint, its tools, guardrails and at least three eval cases;
a team adds 2 to 6 `workers`. Tools are `mock` (a lookup over uploaded sample data), `rest`, `mcp`, `kb`
(a platform knowledge base) or `function` (a stub you finish). The
[Studio skill](../getting-started/coding-assistants.md#the-studio-skill) carries the whole reference,
with a worked example for each shape, and `fastaiagent studio spec schema` prints the JSON Schema.

Checking a spec returns **problems**, which stop a version, and **warnings**, which do not. Both name
their place in the spec. Warnings matter: `json_valid` on an agent that answers in sentences would block
every answer, and the check says so instead of changing your spec.

### Sample data, and what a trial is

Upload `.csv`, `.json`, `.jsonl`, `.md`, `.txt`, `.pdf` or `.docx` files with `--data`. A scan, a photo or
any image has no text layer and is refused (`no_text_layer`): read it yourself (your model, or an OCR
service) and send the text with `--data-text name=file`. Studio never reads one for you on this route.

A **trial** asks the agent one question in the project's sandbox. Know what it is:

- Tools answer from your sample data, never from a live system; a knowledge-base tool searches its
  knowledge base. A passing trial does not prove a REST tool works.
- **Guardrails are not applied in a trial.** They apply in the exported code.
- A team is tried **one worker at a time** (`--worker`); its end-to-end eval cases run only from the
  exported code.
- Eval cases are scored by word rules, not a judge: every `must_mention` value must appear, no
  `must_not_mention` value may. A case with only a `behaviour` reads **needs review** and is not a failure.
- A trial is a real trace, linked in the answer.

## From Python

```python
from fastaiagent.studio import StudioClient, StudioError

studio = StudioClient(api_key="fa_k_…", target="https://your-program-address")
context = studio.context()
endpoint = context.llm_endpoints[0].id

draft = studio.create_draft("Order desk", "Answer order-status questions.", endpoint)
studio.upload_fixtures(draft.id, [("orders.csv", open("orders.csv", "rb").read())])

spec = {...}  # see the Studio skill's spec reference
check = studio.validate_spec(draft.id, spec)
for warning in check.warnings:
    print(warning.code, warning.where, warning.message)

try:
    accepted = studio.create_version(draft.id, spec, note="first design")
except StudioError as err:
    print(err.code, err.problems)       # spec_invalid, with each problem's place
else:
    studio.wait_for_version(draft.id, accepted.version_id)
    trial = studio.run_trial(draft.id, "What is the status of order 1003?")
    print(trial.output, [(c.tool, c.matched) for c in trial.mock_calls])
    runs = studio.run_evals(draft.id)
    studio.export(draft.id, "project.zip")
```

`fa.studio` is the same package, loaded lazily. Every route has a plain method and an `a`-prefixed
async twin (`acreate_version`, `arun_trial`, `aexport`); the waiting helpers are `wait_for_version`,
`wait_for_sandbox`, `wait_for_trial` (async: `await_for_version` …), and `run_trial` and `run_evals`
start the sandbox, ask, wait and recover. Answers are typed pydantic models that keep keys they do not
know, so a field the plane adds later does not break an older client.

### Waiting

Making a version, starting the sandbox and running a trial or an eval each answer at once and finish
later. The plane says how long to wait: every such read carries `retry_after_seconds`, and `0` means
stop. The waiting helpers poll with a **monotonic deadline** (`timeout=`), never by counting sleeps, and
raise `StudioError("wait_timeout")` past it.

- A sandbox can read `ready` while its runner has just gone (a restart, a crash). A trial is then
  refused with `no_active_sandbox`, or ends `failed` naming the runner. `run_trial` and `run_evals` start
  the sandbox again and ask once more, up to three attempts.
- The key's rate limit is 60 calls a minute by default. A refusal for rate (HTTP 429) happens before
  anything runs, so the client waits out a short `Retry-After` (up to 30 seconds) and asks again, twice
  at most; a longer one reaches you as `PlatformRateLimitError` with `.retry_after`. Inside a wait it
  keeps waiting until the deadline.
- One dropped connection while waiting is retried; two in a row are raised.

### Errors

| Raised | When |
|---|---|
| `StudioError` (`.code`, `.message`, `.status`, `.details`, `.problems`) | The plane refused with one of its codes (`spec_invalid`, `name_taken`, `no_llm_endpoint`, `environment_rule`, `insufficient_scope` …), or the client could not finish (`wait_timeout`, `studio_not_available`, `ambiguous_draft`) |
| `PlatformAuthError` | No usable key: 401 `invalid_api_key`, or on a server that holds several workspaces, 400 `customer_required` |
| `PlatformRateLimitError` | 429, after the short retries |
| `PlatformConnectionError` | The platform could not be reached, or answered with a server error |
| `PlatformNotFoundError` | A route that is not there, on a platform that does have Studio |

`StudioError` is a `PlatformError`. The codes and what each means are in the Studio skill's
`references/errors.md`.

### An older platform

The first Studio call that gets a bare 404 reads `GET /public/v1/version`. If the platform does not
list the `/public/v1/agent-studio` routes you get `StudioError("studio_not_available")`: *this plane
does not have headless Agent Studio (it needs 0.20.0 or later).* A platform that has them costs no
extra request. Since 1.102.0 the SDK announces wire protocol **v1.12** on every call to the platform.
The platform's version negotiation serves a client that announces a newer minor than it knows at its
own newest minor instead of refusing it, so an older platform keeps working for everything else.

## See also

- [Use fastaiagent with a coding assistant](../getting-started/coding-assistants.md): install the
  Studio skill so a coding tool follows this loop (it shows you the design and waits for a yes before it
  pushes anything).
- [Platform overview](index.md), [CLI reference](../cli/index.md#fastaiagent-studio),
  [the platform API layer](../internals/platform-api.md).
