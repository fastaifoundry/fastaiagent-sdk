# Use fastaiagent with a coding assistant

A coding assistant — Claude Code, GitHub Copilot, Cursor, Google Antigravity and the
rest — was not trained on this SDK's current API. Left alone it guesses: it calls
`connect()` without a target, imports the LangChain adapter that was removed in 1.96.0,
or reaches for `discover_tools()`. The code looks right and fails when you run it.

The SDK ships an **Agent Skill** that fixes this. A skill is a folder with a `SKILL.md`
that the assistant reads when it is about to write `fastaiagent` code. This one holds the
ten mistakes assistants make most, a reference per area (agents, tools, teams, guardrails,
evals, platform, testing, what changed), and a script that checks what the assistant wrote.

## Install it

```bash
pip install fastaiagent
cd your-project
fastaiagent skills install
```

That copies the skill into the two folders assistants read:

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

    Copy `skills/fastaiagent-sdk/` into `.claude/skills/` or `.agents/skills/` in your
    project, or into `~/.claude/skills/` or `~/.agents/skills/` for every project.

The repository is generated from the SDK at each release; it does not take pull requests.

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

## What it does not do

- It does not make an assistant reliable on its own. It removes one cause of wrong code — a guessed
  API — and gives you a check for it. Read what the assistant writes.
- It does not call the model, the platform or the network. The checker runs none of your code.

## Related

- [CLI reference](../cli/index.md#fastaiagent-skills)
- [Testing without a model](../testing/index.md)
