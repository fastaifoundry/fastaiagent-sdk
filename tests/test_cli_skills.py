"""``fastaiagent skills`` installs the packaged skills where coding assistants read them.

Real folders, real files: each test runs the CLI in a temporary project directory with
a temporary home, then looks at what is on disk.
"""

from __future__ import annotations

import filecmp
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from fastaiagent._version import __version__
from fastaiagent.cli.main import app
from fastaiagent.cli.skills import _POINTER_MARKER, _STAMP, packaged_skills_dir

try:  # Click < 8.2 mixes stderr into stdout unless asked not to; 8.2+ always separates them.
    runner = CliRunner(mix_stderr=False)
except TypeError:
    runner = CliRunner()


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    proj = tmp_path / "proj"
    home.mkdir()
    proj.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.chdir(proj)
    return proj


def _skill_names() -> list[str]:
    return sorted(p.name for p in packaged_skills_dir().iterdir() if (p / "SKILL.md").is_file())


def _same_tree(a: Path, b: Path) -> bool:
    cmp = filecmp.dircmp(a, b, ignore=[_STAMP, "__pycache__"])
    if cmp.left_only or cmp.right_only or cmp.diff_files or cmp.funny_files:
        return False
    return all(_same_tree(a / d, b / d) for d in cmp.common_dirs)


def test_path_prints_the_packaged_folder() -> None:
    result = runner.invoke(app, ["skills", "path"])
    assert result.exit_code == 0, result.output
    assert Path(result.stdout.strip()) == packaged_skills_dir()


def test_list_reports_skills_as_json(project: Path) -> None:
    result = runner.invoke(app, ["skills", "list"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    assert data["sdk_version"] == __version__
    assert [s["name"] for s in data["skills"]] == _skill_names()
    assert all(s["description"] for s in data["skills"])
    assert data["installed"] == []


def test_install_copies_into_both_tool_folders(project: Path) -> None:
    result = runner.invoke(app, ["skills", "install", "--no-agents-md"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    for name in _skill_names():
        for folder in (".claude/skills", ".agents/skills"):
            dest = project / folder / name
            assert (dest / "SKILL.md").is_file(), dest
            assert _same_tree(packaged_skills_dir() / name, dest)
            stamp = json.loads((dest / _STAMP).read_text())
            assert stamp["sdk_version"] == __version__
    assert {r["action"] for r in data["results"]} == {"installed"}
    assert not (project / "AGENTS.md").exists()


def test_install_only_one_tool(project: Path) -> None:
    result = runner.invoke(app, ["skills", "install", "--tool", "agents", "--no-agents-md"])
    assert result.exit_code == 0, result.output
    assert (project / ".agents" / "skills").is_dir()
    assert not (project / ".claude").exists()


def test_install_rejects_unknown_tool(project: Path) -> None:
    result = runner.invoke(app, ["skills", "install", "--tool", "emacs"])
    assert result.exit_code == 2
    assert "unknown --tool" in result.stderr
    assert not (project / ".claude").exists()


def test_second_install_is_unchanged(project: Path) -> None:
    runner.invoke(app, ["skills", "install", "--no-agents-md"])
    again = runner.invoke(app, ["skills", "install", "--no-agents-md"])
    assert again.exit_code == 0, again.output
    assert {r["action"] for r in json.loads(again.stdout)["results"]} == {"unchanged"}


def test_install_over_an_older_copy_needs_force(project: Path) -> None:
    runner.invoke(app, ["skills", "install", "--tool", "claude", "--no-agents-md"])
    name = _skill_names()[0]
    dest = project / ".claude" / "skills" / name
    (dest / _STAMP).write_text(json.dumps({"sdk_version": "0.0.1", "skill": name}))
    (dest / "local-edit.txt").write_text("mine")

    refused = runner.invoke(app, ["skills", "install", "--tool", "claude", "--no-agents-md"])
    assert refused.exit_code == 1
    assert "0.0.1" in refused.stderr and "--force" in refused.stderr
    assert (dest / "local-edit.txt").exists(), "a refused install must not touch the copy"

    forced = runner.invoke(
        app, ["skills", "install", "--tool", "claude", "--no-agents-md", "--force"]
    )
    assert forced.exit_code == 0, forced.output
    assert not (dest / "local-edit.txt").exists()
    assert json.loads((dest / _STAMP).read_text())["sdk_version"] == __version__


def test_list_flags_a_stale_installed_copy(project: Path) -> None:
    runner.invoke(app, ["skills", "install", "--tool", "agents", "--no-agents-md"])
    name = _skill_names()[0]
    (project / ".agents" / "skills" / name / _STAMP).write_text(
        json.dumps({"sdk_version": "0.0.1", "skill": name})
    )
    result = runner.invoke(app, ["skills", "list"])
    data = json.loads(result.stdout)
    (item,) = [i for i in data["installed"] if i["skill"] == name]
    assert item["stale"] is True and item["sdk_version"] == "0.0.1"
    assert "installed by SDK 0.0.1" in result.stderr


def test_user_install_goes_to_home(project: Path, tmp_path: Path) -> None:
    result = runner.invoke(app, ["skills", "install", "--user", "--no-agents-md"])
    assert result.exit_code == 0, result.output
    home = tmp_path / "home"
    for name in _skill_names():
        assert (home / ".claude" / "skills" / name / "SKILL.md").is_file()
        assert (home / ".agents" / "skills" / name / "SKILL.md").is_file()
    assert not (project / ".claude").exists()


def test_agents_md_is_created_with_a_pointer(project: Path) -> None:
    result = runner.invoke(app, ["skills", "install", "--agents-md"])
    assert result.exit_code == 0, result.output
    text = (project / "AGENTS.md").read_text()
    assert _POINTER_MARKER in text
    assert ".agents/skills/fastaiagent-sdk/SKILL.md" in text
    assert json.loads(result.stdout)["pointers"][0]["action"] == "created"


def test_agents_md_is_appended_never_rewritten_and_not_duplicated(project: Path) -> None:
    original = "# My project\n\nExisting rules.\n"
    (project / "AGENTS.md").write_text(original)
    runner.invoke(app, ["skills", "install", "--agents-md"])
    text = (project / "AGENTS.md").read_text()
    assert text.startswith(original)
    assert text.count(_POINTER_MARKER) == 1

    runner.invoke(app, ["skills", "install", "--agents-md"])
    assert (project / "AGENTS.md").read_text() == text


def test_claude_md_is_only_touched_when_asked(project: Path) -> None:
    (project / "CLAUDE.md").write_text("# rules\n")
    runner.invoke(app, ["skills", "install", "--agents-md"])
    assert (project / "CLAUDE.md").read_text() == "# rules\n"
    runner.invoke(app, ["skills", "install", "--claude-md", "--no-agents-md"])
    assert _POINTER_MARKER in (project / "CLAUDE.md").read_text()


def test_no_prompt_without_a_terminal(project: Path) -> None:
    """CliRunner has no TTY: with neither flag the install must not wait for an answer."""
    result = runner.invoke(app, ["skills", "install"])
    assert result.exit_code == 0, result.output
    assert not (project / "AGENTS.md").exists()


def test_symlinked_destination_is_refused(project: Path, tmp_path: Path) -> None:
    name = _skill_names()[0]
    target = tmp_path / "elsewhere"
    target.mkdir()
    dest = project / ".claude" / "skills"
    dest.mkdir(parents=True)
    (dest / name).symlink_to(target, target_is_directory=True)
    result = runner.invoke(
        app, ["skills", "install", "--tool", "claude", "--no-agents-md", "--force"]
    )
    assert result.exit_code == 1
    assert list(target.iterdir()) == [], "must not write through a symlink"
