# When a command fails

A failed command exits 1. With `--json` stdout holds one document:

```json
{
  "error": {
    "code": "spec_invalid",
    "message": "The spec has 1 problem; nothing was made.",
    "hint": "…",
    "status": 422,
    "details": { "problems": [{ "where": "tools[0]", "message": "…" }] }
  }
}
```

Branch on `code`. `message` is the platform's own words (quote it to the person); `hint` is what to
do; `status` is the HTTP status; `details` carries what the code needs (`problems`, `workers`,
`dataset_id`). Without `--json` the same words go to stderr and stdout stays empty.

**A refusal is an answer.** Say what was refused and why, and stop or change the request. Do not
retry the same call, and do not look for a way around a rule.

## Stop and tell the person

| Code | Meaning | What to say or do |
|---|---|---|
| `insufficient_scope` | The key lacks `studio:read` or `studio:write` (`details.required_scope`) | Tell them which permission to add; the console's "Agent Studio" key preset has both |
| `feature_not_available` | The plan does not include Agent Studio | Tell them |
| `environment_rule` | Studio is off on this install (production or acceptance rules). Reads and `export` still work | Tell them. Do not retry |
| `key_has_no_owner` | The person who made the key is no longer a member | Tell them to make a new key |
| `no_llm_endpoint` | The project has no LLM endpoint, or the draft's was removed | Tell them to add one in the console; nothing can be made without it |
| `auth_error` | No usable key (`invalid_api_key`, or on a multi-workspace server `customer_required`) | Tell them to connect (`fastaiagent connect`) or set `FASTAIAGENT_API_KEY` |
| `studio_not_available` | This platform has no headless Agent Studio | It needs 0.20.0 or later |
| `topology_promote_unsupported` | A team's eval cases cannot be promoted | Export instead |
| `not_sdk_owned` | `promote` onto an agent the exported code did not register | Run the exported agent once with a key, so it registers, then promote |

## Fix the request

| Code | Meaning | What to do |
|---|---|---|
| `spec_invalid` | The spec has problems (`details.problems`, each with `where`) | Fix every one and push again. `studio spec validate` first. Nothing was made |
| `invalid_request` | The request itself is malformed (`details.problems`) | Fix the request |
| `invalid_input` | The command was used wrongly, or a local file could not be read | Read the message (a missing file, a spec that is not JSON, a bad `--data-text`) |
| `project_required` | The key is not tied to one project | Pass `--project-id` |
| `name_taken` | A draft with that name exists (case ignored) | Choose another name, or use that draft |
| `worker_required` | A team was tried without `--worker` (`details.workers`) | Name one |
| `unknown_worker` | No worker by that name (`details.workers`; empty for a single agent) | Use a name from the list; for a single agent leave `--worker` out |
| `worker_has_no_cases` | That worker has no eval cases of its own | Add cases to the spec |
| `unknown_case` | No eval case by that name | Check the name with `studio eval` (no `--case`) |
| `no_version` | The draft has no completed version | Push a spec first |
| `not_found` | No such draft, version, fixture or trial in reach. A draft in another project reads as not found too | Check the id or name with `studio list` |
| `ambiguous_draft` | More than one draft has that name | Use an id |
| `unsupported_file` | Not `.csv`, `.json`, `.jsonl`, `.md`, `.txt`, `.pdf` or `.docx` | Convert it, or read it yourself and send the text with `--data-text` |
| `no_text_layer` | A PDF or image with no readable text (a scan) | Read it yourself with your own model, write the text to a file, pass `--data-text name=file` |
| `too_many_fixtures` | More than 20 sample-data files on one draft | Combine or drop some |
| `dataset_already_linked` | An eval dataset is already linked (`details.dataset_id`) | Use the dataset import instead |
| `no_linked_dataset` | Dataset import on a draft with no linked dataset | Link one first |

## Wait, then continue

| Code | Meaning | What to do |
|---|---|---|
| `build_in_progress` | A version of this draft is still being made | Wait for it (`studio status`), then retry |
| `no_active_sandbox` | The trial sandbox is not ready | `try` and `eval` already start it and retry; if you see this, run the command again once |
| `rate_limited` | Over the key's limit (60 a minute by default) | The command already waits out a short `Retry-After`; if this reaches you, wait the time it names |
| `wait_timeout` | A version, sandbox or trial did not finish in time | It may still be going: read `studio status`, then ask again |
| `version_failed` | `spec push --wait` ended with the version failed (`message` says why) | Read the message; fix the spec and push again |
| `connection_error` | The platform could not be reached (or answered with a server error) | Check the address (`FASTAIAGENT_TARGET`); try once more later |

## Writing to the draft's files

| Code | Meaning | What to do |
|---|---|---|
| `export_would_overwrite` | A file the export would write already exists; nothing was written | Choose another `--out`, or pass `--force` if the person agrees to overwrite |
| `unsafe_export` | The export holds a path outside the target folder; nothing was written | Do not use it; tell the person |
