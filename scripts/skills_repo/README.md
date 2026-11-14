# FastAIAgent skills

Agent Skills that teach a coding assistant the [FastAIAgent SDK](https://github.com/fastaifoundry/fastaiagent-sdk),
so it writes `fastaiagent` code against the real API instead of guessing one from older versions.

Built from SDK **{{sdk_version}}**. The same skills ship inside the SDK
(`fastaiagent skills install`); this repository is a generated copy for tools that install from GitHub.

## Skills

{{skill_list}}

## Install

Pick the line for your tool. Every one of them installs the same folders.

### Claude Code

```text
/plugin marketplace add fastaifoundry/fastaiagent-skills
/plugin install fastaiagent@fastaiagent-skills
```

### Any tool, with the `skills` CLI

```bash
npx skills add fastaifoundry/fastaiagent-skills
# one tool only:
npx skills add fastaifoundry/fastaiagent-skills -a claude-code
npx skills add fastaifoundry/fastaiagent-skills -a github-copilot
npx skills add fastaifoundry/fastaiagent-skills -a cursor
npx skills add fastaifoundry/fastaiagent-skills -a antigravity
```

### GitHub Copilot, Cursor, Antigravity and others, with the GitHub CLI

```bash
gh skill install fastaifoundry/fastaiagent-skills fastaiagent-sdk --agent github-copilot
gh skill install fastaifoundry/fastaiagent-skills fastaiagent-sdk --agent cursor
gh skill install fastaifoundry/fastaiagent-skills fastaiagent-sdk --agent antigravity
```

Add `--scope user` to install for every project. Run `gh skill install fastaifoundry/fastaiagent-skills`
with no skill name to pick from the list.

### With the SDK

```bash
pip install fastaiagent
fastaiagent skills install          # this project: .claude/skills and .agents/skills
fastaiagent skills install --user   # every project: ~/.claude/skills and ~/.agents/skills
```

### By hand

Copy a folder from `skills/` into the folder your tool reads:

| Tool | Project folder | Personal folder |
|---|---|---|
| Claude Code | `.claude/skills/` | `~/.claude/skills/` |
| GitHub Copilot | `.agents/skills/` (also reads `.github/skills/`, `.claude/skills/`) | `~/.copilot/skills/`, `~/.agents/skills/` |
| Cursor | `.agents/skills/` (also reads `.cursor/skills/`, `.claude/skills/`) | `~/.agents/skills/`, `~/.cursor/skills/` |
| Google Antigravity | `.agents/skills/` | see the Antigravity docs |

Folder locations are from each tool's own documentation. Claude Code's documentation lists only
`.claude/skills/`, so use that one for it.

## Use

Ask your assistant to write or change `fastaiagent` code. The skill tells it to read the matching
reference first and to run `scripts/check_imports.py` on what it wrote, which resolves every
`fastaiagent` import against the installed SDK.

## Contributing

This repository is generated from the SDK and does not take pull requests. Report a problem with a
skill in the [SDK's issues](https://github.com/fastaifoundry/fastaiagent-sdk/issues).

## License

Apache-2.0, the same as the SDK. See `LICENSE`.
