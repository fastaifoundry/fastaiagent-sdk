---
name: fastaiagent-studio
description: Use only when the person wants a NEW agent built, or an agent Studio made changed, inside FastAIAgent Agent Studio from the command line — "build me an order-status agent with Studio", "make a support team in Studio", "change the agent Studio made". You design the agent and write it as a spec; the `fastaiagent studio` commands check it, render the code, try it in Studio's sandbox, keep versions and export a project. Needs a FastAIAgent platform (0.20.0 or later) and an API key with Agent Studio permissions. Not for an agent whose code is in this project, and not for evaluating against a platform dataset, using platform guardrails or connecting an app to the platform: those are the fastaiagent-sdk skill.
---

# Agent Studio, from the command line

Agent Studio is the FastAIAgent platform's builder: it checks a design, renders real SDK code
from it, tries the agent against sample data, scores eval cases and exports a project. From the
command line **you are the planner**. Studio's own model is never called: you design the agent
with your model and send the design as a **spec**. Studio checks it and does the rest.

After the export the code is the person's. Studio assists; it is not where agents live.

**Is this the right skill?** If the project already has `fastaiagent` code and the person did not
say "Studio", it is not: evaluating that agent (including against a dataset on the platform),
enforcing the platform's guardrails or connecting it to the platform is done in the code. Stop and
use the fastaiagent-sdk skill. Do not ask for Agent Studio permissions for that work.

## Before you start

- A platform **0.20.0 or later** and an API key with `studio:read` and `studio:write` (the
  console's "Agent Studio" key preset has both). Credentials come from `fastaiagent connect`,
  or `FASTAIAGENT_API_KEY` and `FASTAIAGENT_TARGET`. Never ask for a key in chat, never print
  one, never write one into a file.
- Every command takes `--json` and prints one JSON document on stdout; notes and errors go to
  stderr. **Always pass `--json`** and read the document. Exit 0 means it happened; exit 1 means
  it did not (a refusal, a spec with problems, an eval that failed).
- `<draft>` is a draft's id or its name.

## The loop

Do the steps in order. Do not skip step 3.

1. **`fastaiagent studio context --json`** — the project's LLM endpoints, knowledge bases, the
   guardrail names and tool types a spec may use, and the limits. **No LLM endpoint: stop** and
   tell the person to add one in the console; nothing can be made without it. Writes off
   (`environment.studio_writes_allowed` is false): stop and say so.
2. **Read what the person wants.** Read any requirements document yourself. Read any scan or image
   yourself, and say plainly what you could not read.
3. **Lay the design out for the person and wait for a yes.** Say: the shape (one agent, a
   supervisor team, or a swarm), each worker and what it does, each tool and where its data comes
   from, the guardrails, and the eval cases. There is no plan step in the API, so this is where
   the person approves. Never push a spec they have not agreed to.
4. **`studio new --name … --brief "…" --data orders.csv --data-text scan=scan.txt --endpoint <id>`**
   — makes the draft and adds the sample data. Pass `--endpoint` when the project has more than
   one. A PDF or Word file with a text layer can be uploaded as it is; **a scan, a photo or any
   image is refused** (`no_text_layer`): read it yourself, write the text to a file, and pass it
   with `--data-text name=file`. Add `--harness` only when the person wants the prompts and eval
   cases wired to the platform.
5. **Write the spec** (`references/spec.md`), then **`studio spec validate <draft> spec.json
   --json`**. Fix every problem it lists, each names its place. **Read every warning out to the
   person**: a warning does not stop a version, and the person may not want what it says.
6. **`studio spec push <draft> spec.json --note "…" --wait --json`** — makes the next version.
7. **Look, then try, then evaluate:** `studio show <draft> --json` (the blueprint), `studio show
   <draft> --files`, `studio try <draft> "a real question" --json`, `studio eval <draft> --json`.
   Report what happened as it happened: failed cases and their reasons, and cases that "need
   review" because the word rules cannot score them.
8. **To change anything:** `studio spec get <draft> --out spec.json`, edit it, `validate`, `push`.
   A new version is made and the old one is kept. A spec is always sent whole, never a patch.
   Results count for one version: a pass on v1 is not carried to v2, so evaluate again.
9. **`studio export <draft> --out <repo folder>`** — unpacks the project. Run its tests (they run
   offline). Tell the person how to run the agent with their own model key. After the export,
   the code is theirs.

## The rules

- **Never push a spec the person has not agreed to.** The platform cannot know they agreed.
- **Never print or store a key.**
- **A refusal is an answer, not an obstacle.** When a command fails, read `error.code`,
  `error.message` and `error.hint`, tell the person, and stop. Do not retry a refusal, and do not
  look for another way round it (`environment_rule`, `insufficient_scope`, `no_llm_endpoint`,
  `key_has_no_owner`). `references/errors.md` says what each code means.
- **Stale: stop and ask.** If `studio status` says `steps.stale` is true, someone changed the
  draft in the console after the current version was made. Ask the person before changing anything.
- **A team is tried one worker at a time** (`studio try … --worker orders`). Its end-to-end eval
  cases run only from the exported code.
- **Guardrails are not applied in a sandbox trial.** They apply in the exported code. Never tell
  the person a guardrail "passed" because a trial succeeded.
- **Tools answer from sample data in a trial,** never from a live system (except a knowledge-base
  tool, which searches its knowledge base). A passing trial does not prove a REST tool works.
- **Do not invent commands.** There is no `studio plan`, `build`, `refine` or `reset`: planning
  and building from a description, and starting over, are console-only. To start over, push a new
  spec; it makes a new version.
- **Do not ask a draft's model to design for you.** You design; Studio checks and renders.
- Use `--json`, branch on `error.code`, and do not parse the human text.

## Where to read

| You are doing | Read |
|---|---|
| Writing the spec: every field, the tool types, guardrails, eval criteria, worked examples for a single agent, a supervisor team and a swarm | `references/spec.md` |
| Running a command: its options, and the JSON it prints | `references/commands.md` |
| A command failed | `references/errors.md` |

To write the SDK code the exported project is built from, or to change it by hand afterwards, use
the `fastaiagent-sdk` skill.
