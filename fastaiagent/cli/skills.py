"""``fastaiagent skills`` — install the packaged Agent Skills for coding assistants.

A coding assistant that was not trained on this SDK guesses its API. The skills
under ``fastaiagent/skills/`` teach it the real one. They follow the open Agent
Skills format (a folder with a ``SKILL.md``), which the tools read from two
folders:

* ``.claude/skills/``  — Claude Code (GitHub Copilot and Cursor read it too)
* ``.agents/skills/``  — GitHub Copilot, Cursor, Codex, Gemini CLI, Google
  Antigravity and others

``skills install`` copies the packaged skills into both. Output on stdout is JSON;
notes go to stderr; the exit code is non-zero when what you asked for did not
happen.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

import typer

skills_app = typer.Typer(
    name="skills",
    help="Install Agent Skills that teach coding assistants this SDK.",
    no_args_is_help=True,
)

_STAMP = ".fastaiagent-installed.json"
_POINTER_MARKER = "<!-- fastaiagent-skills -->"

# --tool value -> folder under the project (or home, with --user).
_TARGETS: dict[str, str] = {
    "claude": ".claude/skills",
    "agents": ".agents/skills",
}


def packaged_skills_dir() -> Path:
    """The folder inside the installed package that holds the skill folders."""
    return Path(__file__).resolve().parent.parent / "skills"


def _sdk_version() -> str:
    from fastaiagent._version import __version__

    return __version__


def _frontmatter(skill_md: Path) -> dict[str, str]:
    """Read the ``name`` / ``description`` keys of a SKILL.md (flat ``key: value`` lines)."""
    lines = skill_md.read_text(encoding="utf-8").splitlines()
    out: dict[str, str] = {}
    if not lines or lines[0].strip() != "---":
        return out
    for line in lines[1:]:
        if line.strip() == "---":
            break
        if ":" in line and not line.startswith((" ", "\t")):
            key, _, value = line.partition(":")
            out[key.strip()] = value.strip().strip("\"'")
    return out


def _packaged() -> list[dict[str, str]]:
    root = packaged_skills_dir()
    found: list[dict[str, str]] = []
    if not root.is_dir():
        return found
    for folder in sorted(p for p in root.iterdir() if (p / "SKILL.md").is_file()):
        meta = _frontmatter(folder / "SKILL.md")
        found.append(
            {
                "name": meta.get("name", folder.name),
                "description": meta.get("description", ""),
                "path": str(folder),
            }
        )
    return found


def _roots(user: bool, tools: list[str]) -> list[tuple[str, Path]]:
    base = Path.home() if user else Path.cwd()
    return [(tool, base / _TARGETS[tool]) for tool in tools]


def _read_stamp(skill_dir: Path) -> str | None:
    try:
        data = json.loads((skill_dir / _STAMP).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    version = data.get("sdk_version") if isinstance(data, dict) else None
    return version if isinstance(version, str) else None


def _copy(src: Path, dest: Path, name: str, version: str) -> None:
    """Copy one skill folder and stamp the SDK version that installed it."""
    shutil.copytree(src, dest, ignore=shutil.ignore_patterns("__pycache__"))
    (dest / _STAMP).write_text(
        json.dumps({"sdk_version": version, "skill": name}) + "\n", encoding="utf-8"
    )


def _note(message: str) -> None:
    typer.echo(message, err=True)


def _emit(payload: Any) -> None:
    typer.echo(json.dumps(payload, indent=2))


@skills_app.command("path")
def path_cmd() -> None:
    """Print the folder that holds the packaged skills."""
    typer.echo(str(packaged_skills_dir()))


@skills_app.command("list")
def list_cmd() -> None:
    """List the packaged skills and any installed copies (project and user).

    An installed copy is ``stale`` when it was installed by a different SDK version.
    """
    current = _sdk_version()
    packaged = _packaged()
    installed: list[dict[str, Any]] = []
    for scope, user in (("project", False), ("user", True)):
        for tool, root in _roots(user, list(_TARGETS)):
            for skill in packaged:
                dest = root / skill["name"]
                if not dest.is_dir():
                    continue
                stamp = _read_stamp(dest)
                installed.append(
                    {
                        "skill": skill["name"],
                        "scope": scope,
                        "tool": tool,
                        "path": str(dest),
                        "sdk_version": stamp,
                        "stale": stamp != current,
                    }
                )
    _emit({"sdk_version": current, "skills": packaged, "installed": installed})
    for item in installed:
        if item["stale"]:
            _note(
                f"{item['path']} was installed by SDK {item['sdk_version'] or 'unknown'}; "
                f"you have {current}. Run: fastaiagent skills install --force"
                f"{' --user' if item['scope'] == 'user' else ''}"
            )


def _pointer_text(tools: list[str]) -> str:
    where = ".agents/skills" if "agents" in tools else ".claude/skills"
    return (
        f"{_POINTER_MARKER}\n"
        "## FastAIAgent SDK\n"
        f"This project uses the fastaiagent SDK. Before writing agent code, read "
        f"`{where}/fastaiagent-sdk/SKILL.md`, and run "
        f"`python {where}/fastaiagent-sdk/scripts/check_imports.py <file>` on what you write.\n"
    )


def _add_pointer(file: Path, tools: list[str]) -> str:
    """Append the pointer to ``file`` (created if missing, never rewritten)."""
    existing = file.read_text(encoding="utf-8") if file.exists() else ""
    if _POINTER_MARKER in existing:
        return "already present"
    prefix = "" if not existing else ("\n" if existing.endswith("\n") else "\n\n")
    with file.open("a", encoding="utf-8") as fh:
        fh.write(prefix + _pointer_text(tools))
    return "appended" if existing else "created"


@skills_app.command("install")
def install_cmd(
    tool: str = typer.Option(
        "all",
        "--tool",
        help="Folder to install into: claude (.claude/skills), agents (.agents/skills) or all.",
    ),
    user: bool = typer.Option(
        False,
        "--user",
        help="Install for your user (~/.claude/skills, ~/.agents/skills) instead of this project.",
    ),
    force: bool = typer.Option(
        False, "--force", help="Replace a copy that already exists (e.g. from an older SDK)."
    ),
    agents_md: bool | None = typer.Option(
        None,
        "--agents-md/--no-agents-md",
        help="Add a short pointer to AGENTS.md (created if missing, appended otherwise). "
        "Asked about when there is a terminal; skipped when there is none.",
    ),
    claude_md: bool = typer.Option(
        False, "--claude-md", help="Also add the pointer to CLAUDE.md (never edited otherwise)."
    ),
) -> None:
    """Copy the packaged skills into the folders coding assistants read."""
    if tool != "all" and tool not in _TARGETS:
        _note(f"unknown --tool {tool!r}: choose claude, agents or all")
        raise typer.Exit(code=2)
    tools = list(_TARGETS) if tool == "all" else [tool]
    packaged = _packaged()
    if not packaged:
        _note(f"no packaged skills found under {packaged_skills_dir()}")
        raise typer.Exit(code=1)

    current = _sdk_version()
    results: list[dict[str, str]] = []
    failed = False
    for tool_name, root in _roots(user, tools):
        for skill in packaged:
            src = Path(skill["path"])
            dest = root / skill["name"]
            entry = {"skill": skill["name"], "tool": tool_name, "path": str(dest)}
            if dest.is_symlink():
                entry["action"] = "refused: destination is a symbolic link"
                failed = True
            elif dest.exists():
                stamp = _read_stamp(dest)
                if stamp == current and not force:
                    entry["action"] = "unchanged"
                elif not force:
                    entry["action"] = "skipped: exists (use --force to replace)"
                    _note(
                        f"{dest} exists"
                        + (f" (installed by SDK {stamp}; you have {current})" if stamp else "")
                        + ". Re-run with --force to replace it."
                    )
                    failed = True
                else:
                    shutil.rmtree(dest)
                    _copy(src, dest, skill["name"], current)
                    entry["action"] = "replaced"
            else:
                root.mkdir(parents=True, exist_ok=True)
                _copy(src, dest, skill["name"], current)
                entry["action"] = "installed"
            results.append(entry)

    pointers: list[dict[str, str]] = []
    if user:
        if agents_md or claude_md:
            _note("--agents-md / --claude-md are project files; ignored with --user.")
    else:
        want_agents = agents_md
        if want_agents is None and sys.stdin.isatty() and sys.stdout.isatty():
            want_agents = typer.confirm(
                "Add a pointer to the skill in AGENTS.md?", default=True, err=True
            )
        targets = []
        if want_agents:
            targets.append(Path.cwd() / "AGENTS.md")
        if claude_md:
            targets.append(Path.cwd() / "CLAUDE.md")
        for file in targets:
            pointers.append({"file": str(file), "action": _add_pointer(file, tools)})

    _emit({"sdk_version": current, "results": results, "pointers": pointers})
    if failed:
        raise typer.Exit(code=1)
