# Use fastaiagent with a coding assistant

A coding assistant — Claude Code, GitHub Copilot, Cursor, Google Antigravity and the
rest — was not trained on this SDK's current API. Left alone it guesses: it calls
`connect()` without a target, imports the LangChain adapter that was removed in 1.96.0,
or reaches for `discover_tools()`. The code looks right and fails when you run it.

The SDK ships two **Agent Skills**. A skill is a folder with a `SKILL.md` that the assistant
reads when the work calls for it.

- **`fastaiagent-sdk`** is read when the assistant is about to write `fastaiagent` code. It holds
  the ten mistakes assistants make most, a reference per area (agents, tools, teams, guardrails,
  evals, platform, testing, what changed), a script that checks what the assistant wrote, and a
  short recipe for starting a project: a layout where plain `pytest` works and an entry point that
  connects and always ends with `disconnect()`.

  It also asks the assistant to do more than write code that passes its offline tests. Those tests
  use scripted models, so they cannot see a wrong model id, a missing key or a dead server. When
  your model key is in the environment the assistant should run the real entry point once and say
  what it ran and what it did not. It should keep an address you give it in `FASTAIAGENT_TARGET`
  instead of in code, test an MCP tool against a real local server instead of a fake, and not add
  tools or helpers you did not ask for.
- **`fastaiagent-studio`** is read when you ask for an agent to be built through
  [Agent Studio](../platform/agent-studio.md). It teaches the `fastaiagent studio` commands and how
  to write a spec (see [The Studio skill](#the-studio-skill) below). It needs a platform.

## Install it

```bash
pip install fastaiagent
cd your-project
fastaiagent skills install
```

That copies both skills into the two folders assistants read:

| Folder | Read by |
|---|---|
| `.claude/skills/` | Claude Code. GitHub Copilot and Cursor read it too. |
| `.agents/skills/` | GitHub Copilot, Cursor, Google Antigravity, Codex, Gemini CLI and others |

The command prints what it did as JSON and exits non-zero if it could not do what you
asked. It also offers to add a three-line pointer to `AGENTS.md` (created if missing,
appended to otherwise, never rewritten); it asks only when it has a terminal. Pass
`--agents-md` or `--no-agents-md` to decide up front, and `--claude-md` to add the pointer
to `CLAUDE.md` too — it never edits that file otherwise.

| Command | Does |
|---|---|
| `fastaiagent skills list` | The packaged skills, and any installed copies that are out of date |
| `fastaiagent skills install` | Install into the project. `--tool claude\|agents\|all`, `--user` for every project, `--force` to replace |
| `fastaiagent skills path` | The folder the skills ship in |

Each installed copy is stamped with the SDK version that installed it. After you upgrade the
SDK, `skills list` says which copies are stale and `skills install --force` refreshes them.

## Install it without the SDK

The same skills are in a public repository,
[`fastaifoundry/fastaiagent-skills`](https://github.com/fastaifoundry/fastaiagent-skills):

=== "Claude Code"

    ```text
    /plugin marketplace add fastaifoundry/fastaiagent-skills
    /plugin install fastaiagent@fastaiagent-skills
    ```

=== "Any tool (`skills` CLI)"

    ```bash
    npx skills add fastaifoundry/fastaiagent-skills
    ```

=== "GitHub CLI"

    ```bash
    gh skill install fastaifoundry/fastaiagent-skills fastaiagent-sdk --agent github-copilot
    ```

=== "By hand"

    Copy a folder from `skills/` (`fastaiagent-sdk/`, `fastaiagent-studio/`) into `.claude/skills/`
    or `.agents/skills/` in your project, or into `~/.claude/skills/` or `~/.agents/skills/` for
    every project.

The repository is generated from the SDK at each release; it does not take pull requests. The
`gh skill install` line above names one skill; use `fastaiagent-studio` for the other.

## What the assistant does with it

1. You ask for `fastaiagent` code.
2. The assistant loads `SKILL.md` and reads the reference for the area — for example
   `references/tools.md` before it writes a tool.
3. It writes the code and an offline test using `fastaiagent.testing.TestModel`.
4. It runs the checker from the skill's folder:

```bash
python .agents/skills/fastaiagent-sdk/scripts/check_imports.py agent.py tests/
```

The checker resolves every `fastaiagent` import and every `fa.something` attribute against
the installed SDK, and names the replacement for anything that was removed:

```text
agent.py:3: error: cannot import discover_tools from fastaiagent.tool (removed. Use MCPTool.from_server(...) (needs the [mcp] extra).)
agent.py:7: warn: connect() needs target=... (or FASTAIAGENT_TARGET in the environment); there is no default since 1.100.0.
checked 1 file(s): 1 error(s), 1 warning(s)
```

It exits `0` when clean, `1` on an error (or on a warning with `--strict`), and `2` if the SDK
is not importable from that Python. You can run it yourself, or in CI.

## The Studio skill

[Agent Studio](../platform/agent-studio.md) can build an agent for you from a spec. With the
`fastaiagent-studio` skill installed, ask your assistant for an agent ("build me an order-status agent
with Studio, here is the orders data") and it follows this loop:

1. `fastaiagent studio context` — which LLM endpoints and knowledge bases the project has. With no
   endpoint it stops and tells you to add one in the console.
2. It reads what you want, and any requirements document or scan itself, and says what it could not read.
3. **It lays the design out for you — the shape, each worker, the tools, the guardrails, the eval
   cases — and waits for a yes.** The API has no plan step, so this is where you approve; nothing is
   pushed before it.
4. `studio new` with your sample data, then it writes the spec and runs `studio spec validate`. Every
   warning is read out to you.
5. `studio spec push --wait`, then `studio show`, `studio try` and `studio eval`.
6. To change something it reads the spec back, edits it and pushes a new version; the old one is kept.
7. `studio export` into your repository, then it runs the exported tests and tells you how to run
   the agent with your own key.

It never prints or stores a key, reports a refusal and its reason instead of retrying it, tries a
team one worker at a time, and stops and asks if someone changed the draft in the console. It needs a
platform at 0.20.0 or later and an API key with the **Agent Studio** permissions.

This skill has not been tried by hand in GitHub Copilot, Cursor or Google Antigravity yet, so this page
does not claim it works there: it is in the folder each of them reads, and the checklist for trying it
is in the SDK repository.

## What it does not do

- It does not make an assistant reliable on its own. It removes one cause of wrong code — a guessed
  API — and gives you a check for it. Read what the assistant writes.
- It does not call the model, the platform or the network. The checker runs none of your code.

## Related

- [CLI reference](../cli/index.md#fastaiagent-skills)
- [Testing without a model](../testing/index.md)
