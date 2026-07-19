# Concepts & Mental Model

This page is the mental model for prompts — *why* a registry exists, *when* to
use one instead of an inline string, *how* a prompt resolves at load time, and
how prompts compose with agents and the platform. Read it first, then use the
[Prompt Registry reference](index.md) for the full API.

## Why a prompt registry exists

The simplest way to give an agent instructions is an inline string:

```python
Agent(name="support", system_prompt="You are a helpful support agent...")
```

That's fine for a one-off script. It stops being fine the moment a prompt is
something you *iterate on and ship*: you can't roll back a bad edit, the same
tone/format instructions get copy-pasted across agents, you can't see what
changed between two versions, and you can't point "production" at a tested
prompt while you experiment on "staging."

A **PromptRegistry** turns a prompt into a managed artifact — **versioned**,
**composable** from reusable fragments, **aliasable** for deployment control,
and **diffable** — stored locally and optionally pushed to the platform for
team governance.

## When to use the registry

| Use | When |
|-----|------|
| **Inline string** | One-off scripts, prototypes, a prompt you won't iterate on or reuse. |
| **PromptRegistry (local)** | You iterate on a prompt, reuse fragments across agents, or want version history and diffs. |
| **Registry + platform slug** | You want a governed, auditable prompt that versions independently of code deploys — referenced by `prompt_slug` on a pushed agent. |

Rule of thumb: reach for the registry when a prompt has a *lifecycle* — it will
change, be reused, or need to be governed.

## The resolution model

A registry entry is a **template** with two kinds of placeholders, resolved at
different times:

- `{{variable}}` — a runtime value, filled by `prompt.format(name="World")`.
- `{{@fragment}}` — a reusable block, resolved from the registry **at load
  time** (before you ever call `format`).

The lifecycle of a prompt:

1. **Register fragments** — `reg.register_fragment(name="tone", content="...")`.
2. **Register the prompt** — `reg.register(name="greeting", template="Hello {{name}}. {{@tone}}")`. Each `register` of an existing name creates a **new auto-incrementing version**; `{{variable}}` names are auto-extracted.
3. **Load** — `reg.load("greeting")` (or `version=` / `alias=`). Loading resolves every `{{@fragment}}` against the registry and returns a `Prompt` whose `.variables` are known.
4. **Format** — `prompt.format(name="World")` substitutes the runtime `{{variable}}` placeholders and returns the final string.

!!! info "Verified against a live run"
    Registering `tone` then `greeting="Hello {{name}}. {{@tone}}"`,
    `load().format(name="World")` produced
    `"Hello World. Be professional and concise."` — the fragment resolved at
    load, the variable at format. A second `register` bumped the entry to
    version 2; `set_alias("greeting", 1, "production")` kept `production`
    pinned to v1; and `diff("greeting", 1, 2)` returned a unified diff of the
    two templates (fragments shown unresolved, as `{{@tone}}`).

### Under the hood

The two placeholder kinds are two different operations. Fragment resolution is a
**regex substitution** at load time — `{{@name}}` is matched and replaced with
the fragment's content pulled from the store, so a `load()` returns a template
that still has `{{variable}}` holes but no `{{@fragment}}` markers. `format()` is
then a **literal string replacement** of each `{{variable}}`. That's why a
fragment can carry variables of its own: it's spliced in *before* formatting, so
its `{{...}}` holes get filled in the same `format()` pass.

Everything lives in the unified local SQLite store as ordinary rows — separate
tables for prompts, versions, aliases, and fragments — which is what makes
versions immutable, aliases repointable, and diffs a row-to-row comparison. A
connected registry adds a platform lookup in front (`source="auto"`), with a
5-minute cache so repeated `get()`s don't re-fetch.

### Versions, aliases, and diff

- **Versions** are immutable snapshots — every `register` adds one, so you can
  always roll back.
- **Aliases** are movable pointers — `set_alias(name, version, "production")`
  lets you ship "production" while iterating elsewhere, then repoint it when a
  new version is tested. `load(name, alias="production")` follows the pointer.
- **Diff** — `diff(name, v1, v2)` returns a unified diff between two versions.

## Where prompts sit relative to agents and the platform

An agent's system prompt can come from three places — pick by how much
governance you need:

```
Agent.system_prompt  ◄── inline string        (no history, no reuse)
                     ◄── reg.load(...).format(...)   (versioned + composable, resolved locally)
Agent.prompt_slug    ◄── platform slug         (governed: agent links to the prompt, resolved at runtime)
```

`prompt.format(...)` **inlines** the resolved text into the agent — good for
local runs. If you also push the agent to a connected control plane and want it
linked to the governed prompt (not stored as "Inline"), set
`Agent(prompt_slug="support-agent")` instead: the platform sees the slug, owns
the versioning, and can gate which version deploys — independently of your code.

When connected, the registry resolves with `source="auto"` — **platform first
(if connected), else local** — and caches platform fetches for 5 minutes.

## A guided path

1. Register a fragment and a prompt, then `load().format(...)` — see the [Quick Start](index.md#quick-start).
2. Add a second version and `diff` them — see [Versioning](index.md#versioning).
3. Set a `production` alias and load through it — see [Aliases](index.md#aliases).
4. Reference a governed prompt from a pushed agent via `prompt_slug` — see [Platform Prompt Registry](index.md#platform-prompt-registry).

## Next steps

- [Prompt Registry](index.md) — the full API: fragments, versioning, aliases, diff, CLI, storage
- [Agents](../agents/concepts.md) — where the resolved prompt is used (step 2 of the run loop)
- [Platform Sync](../platform/index.md) — push prompts and agents to the control plane
