# The commands

Every command takes `--json` (one JSON document on stdout), and these three, which are left out
of the lines below:

- `--api-key KEY` and `--target ADDRESS` override the saved credentials;
- `--project-id ID` is only for a key that is not tied to one project.

Exit 0 means it happened. Exit 1 means it did not; with `--json` stdout then holds an `error`
document (`references/errors.md`). Exit 2 is a usage error (a missing argument, a bad option).
`<draft>` is a draft's id or its name. JSON below is trimmed to the keys you will use; the plane
may add more.

## context

```command
fastaiagent studio context
```

What the project has. Read it first.

```json
{
  "project": { "id": "7fe6…", "name": "Agent Studio", "domain_id": "18a5…" },
  "environment": { "tier": "development", "studio_writes_allowed": true },
  "llm_endpoints": [{ "id": "7d95…", "name": "OpenAI cheap", "provider": "openai", "model": "gpt-5.4-mini" }],
  "knowledge_bases": [{ "id": "kb1…", "name": "Support KB" }],
  "sdk": { "sdk_pin": "1.88.0", "renderer_version": 16, "dependency_range": "fastaiagent>=1.88.0,<2.0.0" },
  "limits": { "max_fixtures_per_draft": 20, "max_workers": 6, "min_eval_cases": 3 },
  "spec_version": 1,
  "architectures": ["single", "supervisor", "swarm"],
  "tool_types": ["rest", "mcp", "kb", "function", "mock"],
  "eval_criteria_keys": ["must_mention", "must_not_mention", "behaviour"],
  "guardrails": [{ "factory": "cost_limit", "required": ["max_usd"], "optional": [] }],
  "guardrails_that_warn": { "json_valid": "blocks every answer that is not a JSON document" },
  "guardrails_on_the_question_in_a_swarm": { "allowed_topics": "in a swarm it checks the question, not the answer" }
}
```

The generated code targets `sdk.dependency_range`: that is the SDK the exported project asks for,
which may be older than the one installed here.

## list

```command
fastaiagent studio list [--status STATUS] [--query TEXT] [--limit N]
```

The project's drafts (`status` is `drafting`, `generating`, `ready`, `failed` or `exported`). A JSON
array; each draft has `id`, `name`, `status`, `architecture`, `created_via`, `console_url` and
`steps_summary` (`current` step, `stale`, `version`, `evals`).

## new

```command
fastaiagent studio new --name NAME --brief TEXT [--data FILE]... [--data-text NAME=FILE]... [--harness] [--endpoint ID]
```

Makes the draft and adds the sample data, one file per request. `--brief` is what the person asked
for (at least 10 characters). `--endpoint` is required when the project has more than one LLM
endpoint. `--data-text name=file` is for text you read yourself (a scan, an image, an OCR result):
`name` is the sample-data name a tool's `fixture_ref` will use. Exit 1 if any file was not added;
the draft still exists and the JSON says which.

```json
{
  "draft": { "id": "c6e2…", "name": "Order desk", "status": "drafting", "created_via": "api" },
  "fixtures": [{ "id": "9bc8…", "name": "orders", "kind": "csv" }, { "id": "a1f0…", "name": "receipt", "kind": "doc" }]
}
```

## status

```command
fastaiagent studio status <draft>
```

Where the draft is and what to do next. `steps.stale` true means someone changed it in the console:
stop and ask. `steps.next.action` is one of `push_spec`, `wait`, `start_sandbox`, `trial`, `eval`,
`export`, `none`.

```json
{
  "id": "c6e2…", "name": "Order desk", "status": "ready", "architecture": "single",
  "steps": {
    "current": "try", "stale": false,
    "steps": [{ "key": "build", "state": "done", "summary": "v2, made through the API" }],
    "next": { "action": "trial", "reason": "Not tried yet", "hint": "fastaiagent studio try … \"<a question>\"" }
  },
  "versions": [{ "id": "79c9…", "version": 2, "generation_status": "completed", "note": "repeat the id" }],
  "fixtures": [{ "id": "9bc8…", "name": "orders", "kind": "csv" }]
}
```

## spec schema

```command
fastaiagent studio spec schema [--full]
```

Prints the spec's JSON Schema (always JSON). `--full` also prints the guardrail names, tool types
and limits.

## spec validate

```command
fastaiagent studio spec validate <draft> <file>
```

Checks a spec and makes nothing. `<file>` is a JSON file, or `-` for stdin. **Exit 0 only when
there are no problems.** Warnings do not change the exit code: read them out.

```json
{
  "ok": false,
  "problems": [{ "where": "tools[0]", "message": "tool 'orders_lookup' references fixture 'nope' but uploaded fixtures are: ['orders']" }],
  "warnings": [{ "where": "guardrails[0]", "code": "guardrail_blocks_output", "message": "json_valid blocks every answer that is not a JSON document" }]
}
```

## spec push

```command
fastaiagent studio spec push <draft> <file> [--note TEXT] [--wait]
```

Makes the next version from a whole spec. Problems make nothing: exit 1 with `spec_invalid`.
`--wait` waits until the version is made (a couple of seconds) and exits 1 if it failed
(`version_failed`).

```json
{
  "version_id": "2a83…", "version": 1, "warnings": [],
  "generation": { "id": "2a83…", "version": 1, "generation_status": "completed", "created_via": "api" }
}
```

Without `--wait` there is no `generation`; the version is still being made.

## spec get

```command
fastaiagent studio spec get <draft> [--version N] [--out FILE]
```

The spec of a version (the current one by default), to edit and push again. Prints it, or writes
it to `--out`. Always JSON.

## show

```command
fastaiagent studio show <draft> [--version N] [--files] [--file PATH]
```

The design at a glance, or the generated code. `--files` lists the generated files; `--file
agent.py` prints one.

```json
{
  "version": { "id": "79c9…", "version": 2, "note": "repeat the id", "generation_status": "completed" },
  "blueprint": {
    "name": "order-status-bot", "architecture": "single", "description": "…",
    "guardrails": ["no_pii"], "eval_cases": 3, "tools": ["orders_lookup (mock)"]
  }
}
```

A team's blueprint has `workers` (each with `name`, `tools`, `eval_cases`) instead of `tools`; a
swarm also has `entrypoint` and `handoffs`.

## diff

```command
fastaiagent studio diff <draft> A B
```

What changed from version A to version B (numbers).

```json
{
  "from": 1, "to": 2,
  "spec_changes": [{ "where": "system_prompt", "kind": "changed" }, { "where": "tools.escalate", "kind": "added" }],
  "files": [{ "path": "agent.py", "kind": "changed" }]
}
```

## try

```command
fastaiagent studio try <draft> QUESTION [--worker NAME] [--version N] [--wait | --no-wait]
```

Asks the agent one question in the sandbox, starting the sandbox first if needed. A team needs
`--worker`. It waits for the answer by default (there is no command to read a trial later). Exit 1
if the trial failed. A guardrail is not applied.

```json
{
  "id": "7652…", "kind": "chat", "worker": "orders", "status": "completed",
  "output": "Order 1003 is shipped.",
  "mock_calls": [{ "tool": "orders_lookup", "args": { "order_id": "1003" }, "matched": true }],
  "trace_id": "0008…", "trace_url": "https://<console>/next/traces/0008…"
}
```

Use `output` for the answer. `matched` says whether a sample-data row (or a knowledge-base passage)
came back for that call; a false one is worth reporting.

## eval

```command
fastaiagent studio eval <draft> [--worker NAME] [--case NAME] [--wait | --no-wait]
```

Runs the eval cases in the sandbox. A single agent is one run; a team with `--worker` runs that
worker's cases; a team without it runs every worker that has cases. The team's own end-to-end
cases run only from the exported code. **Exit 0 only when no case failed**; a case that "needs
review" is not a failure.

```json
{
  "runs": [{ "id": "93a7…", "kind": "eval_preview", "status": "completed" }],
  "rollup": {
    "version": 2,
    "totals": { "passed": 2, "failed": 0, "needs_review": 1, "not_run": 0, "runnable_here": 3, "on_export_only": 0 },
    "groups": [{
      "scope": "agent",
      "cases": [
        { "name": "found_1003", "status": "passed", "output": "Order 1003 is shipped.", "detail": "all mentions present" },
        { "name": "missing_9999", "status": "needs_review", "criteria": { "behaviour": "says plainly that no such order was found" } }
      ]
    }]
  }
}
```

A case's `status` is `passed`, `failed`, `needs_review`, `not_run` or `on_export`. A group's `scope`
is a worker's name, `agent` (a single agent) or `end_to_end` (a team's own cases). Results are
for one version.

## export

```command
fastaiagent studio export <draft> [--out DIR] [--version N] [--force]
```

Unpacks the project (code, tests, eval cases, sample data) into `DIR` (default: the current
folder), under one folder named after the draft. It writes **nothing** if any file already exists,
unless `--force`. Needs only `studio:read`. After it, the code is the person's.

```json
{ "out": "/work/repo", "folders": ["order-desk"], "files": ["order-desk/agent.py", "order-desk/tests/test_agent.py"] }
```

## promote

```command
fastaiagent studio promote <draft> --agent ID
```

Turns the eval cases into a real eval suite on an agent the exported code has already registered
(run it once with a key). A team's cases cannot be promoted. Optional, after the export.

```json
{ "suite_id": "s1…", "dataset_id": "ds1…", "cases": 3 }
```

## open

```command
fastaiagent studio open <draft>
```

Prints the draft's console address. It does not open a browser.

```json
{ "id": "c6e2…", "name": "Order desk", "console_url": "https://<console>/next/studio/c6e2…" }
```

## delete

```command
fastaiagent studio delete <draft> [--yes]
```

Deletes a draft and its versions. It cannot be undone. It asks only at a terminal; `--yes` skips
that. Delete only what you made, and only when the person asks.

```json
{ "deleted": "c6e2…", "name": "Order desk" }
```
