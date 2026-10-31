"""A ``pkg.module:attr`` target resolves from the project directory (1.95.0, S6).

A console script does not put the working directory on ``sys.path``, so
``fastaiagent push --module myapp.agents`` (and ``resume --runner``, ``eval run
--agent``, ``agent serve``, ``mcp serve``, ``runner --tools``) failed with
``No module named 'myapp'`` when run from the project. Each test strips the cwd
from ``sys.path`` the way a console script has it.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from fastaiagent._internal.target import import_dotted, resolve_target
from fastaiagent.cli.main import app

AGENTS = '''
from fastaiagent import Agent, LLMClient

support = Agent(
    name="s6-support",
    system_prompt="Help.",
    llm=LLMClient(provider="openai", model="gpt-4o-mini", api_key="sk-unused"),
)
'''


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """A project dir holding an uninstalled package, as the cwd, off ``sys.path``."""
    package = f"s6app_{os.getpid()}"
    (tmp_path / package).mkdir()
    (tmp_path / package / "__init__.py").write_text("")
    (tmp_path / package / "agents.py").write_text(AGENTS)
    (tmp_path / "json.py").write_text("raise RuntimeError('a stray json.py shadowed the stdlib')\n")
    monkeypatch.chdir(tmp_path)
    cwd_forms = {"", ".", str(tmp_path), os.path.realpath(tmp_path)}
    monkeypatch.setattr(sys, "path", [p for p in sys.path if p not in cwd_forms])
    yield package
    for name in [m for m in sys.modules if m == package or m.startswith(f"{package}.")]:
        del sys.modules[name]


def test_push_finds_a_package_in_the_project_dir(project: str) -> None:
    result = CliRunner().invoke(app, ["push", "--module", f"{project}.agents", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "s6-support" in result.output


def test_resolve_target_finds_a_package_in_the_project_dir(project: str) -> None:
    agent = resolve_target(f"{project}.agents:support")
    assert agent.name == "s6-support"


def test_the_project_dir_never_shadows_an_installed_module(project: str) -> None:
    import_dotted(f"{project}.agents")
    sys.modules.pop("json", None)
    assert import_dotted("json").dumps({"ok": 1}) == json.dumps({"ok": 1})
