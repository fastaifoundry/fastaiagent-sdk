"""The ``fastaiagent-studio`` skill is kept true by tests, not by care.

Format limits, links, python snippets and quoted signatures are checked for every skill by
``tests/test_skills_package.py``. These are the checks only this skill needs:

* every command in ``commands.md`` is run with ``--help`` against the real CLI, every option
  it lists exists, and no real command is left undocumented;
* every example spec in ``spec.md`` validates against the JSON Schema the plane serves (a
  copy is vendored in ``tests/data``; ``tests/e2e/test_gate_agent_studio.py`` also sends them
  to a live plane, which knows the rules a schema cannot say);

The vendored copy is the platform repository's ``files_claude/agent-studio-spec-schema.json``,
the body of ``GET /spec-schema`` byte for byte (platform PR #268, which keeps it true with a
test of its own; first vendored from the live route, and found identical). To refresh it after
the platform changes the spec, copy that file over ``tests/data/agent_studio_spec_schema.json``.
Never edit one copy.

* the guardrail, tool-type and warning tables in ``spec.md`` agree with that schema answer;
* every error code a command can print is explained in ``errors.md``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from fastaiagent.cli.main import app
from fastaiagent.cli.studio import _HINTS

try:  # Click < 8.2 mixes stderr into stdout unless asked not to; 8.2+ always separates them.
    runner = CliRunner(mix_stderr=False)
except TypeError:
    runner = CliRunner()

ROOT = Path(__file__).resolve().parent.parent
SKILL = ROOT / "fastaiagent" / "skills" / "fastaiagent-studio"
REFS = SKILL / "references"
SCHEMA_ANSWER = json.loads((ROOT / "tests" / "data" / "agent_studio_spec_schema.json").read_text())

_FENCE = re.compile(r"^```([^\n`]*)\n(.*?)^```$", re.S | re.M)


def _fences(path: Path) -> list[tuple[str, str, str]]:
    """``(info string, body, "file:line")`` for every fenced block in a markdown file."""
    text = path.read_text(encoding="utf-8")
    out = []
    for m in _FENCE.finditer(text):
        line = text.count("\n", 0, m.start()) + 2
        out.append((m.group(1).strip(), m.group(2), f"{path.name}:{line}"))
    return out


def test_the_skill_exists_and_is_linked_to_its_three_references() -> None:
    body = (SKILL / "SKILL.md").read_text(encoding="utf-8")
    assert {p.name for p in REFS.glob("*.md")} == {"spec.md", "commands.md", "errors.md"}
    for name in ("spec", "commands", "errors"):
        assert f"references/{name}.md" in body


# -- commands ----------------------------------------------------------------------------


def _documented_commands() -> list[tuple[list[str], str, str]]:
    """``(command words, the whole line, where)`` for every line in a ``command`` fence."""
    found = []
    for info, body, where in _fences(REFS / "commands.md"):
        if info != "command":
            continue
        for raw in body.splitlines():
            line = raw.strip()
            if not line:
                continue
            words: list[str] = []
            for token in line.split():
                if token[0] in "<[-\"'" or token.isupper():
                    break
                words.append(token)
            found.append((words[1:], line, where))  # drop the leading "fastaiagent"
    return found


def _real_commands() -> set[tuple[str, ...]]:
    """Every ``studio`` command the CLI really has, as a tuple of words."""
    import typer.main

    group = typer.main.get_command(app)
    studio = group.commands["studio"]  # type: ignore[attr-defined]
    found: set[tuple[str, ...]] = set()

    def walk(cmd: object, prefix: tuple[str, ...]) -> None:
        children = getattr(cmd, "commands", None)
        if children:
            for name, child in children.items():
                walk(child, (*prefix, name))
        else:
            found.add(prefix)

    walk(studio, ("studio",))
    return found


@pytest.mark.parametrize("case", _documented_commands(), ids=lambda c: " ".join(c[0]) or c[1])
def test_each_documented_command_runs_with_help_and_lists_real_options(
    case: tuple[list[str], str, str],
) -> None:
    words, line, where = case
    assert words[:1] == ["studio"], f"{where}: {line!r} is not a studio command"
    result = runner.invoke(app, [*words, "--help"], terminal_width=200)
    assert result.exit_code == 0, f"{where}: {line!r}\n{result.output}"
    for flag in sorted(set(re.findall(r"--[a-z][a-z-]*", line))):
        assert flag in result.stdout, f"{where}: {line!r} lists {flag}, which --help does not"


def test_no_real_command_is_left_undocumented() -> None:
    documented = {tuple(words) for words, _, _ in _documented_commands()}
    missing = sorted(_real_commands() - documented)
    assert not missing, f"commands.md does not document: {missing}"
    invented = sorted(documented - _real_commands())
    assert not invented, f"commands.md documents commands that do not exist: {invented}"


def test_every_option_of_every_command_is_documented_or_shared() -> None:
    """The three credential options and ``--json`` are explained once, at the top."""
    shared = {"--json", "--api-key", "--target", "--project-id", "--help"}
    text = (REFS / "commands.md").read_text(encoding="utf-8")
    problems = []
    for words, line, where in _documented_commands():
        result = runner.invoke(app, [*words, "--help"], terminal_width=200)
        real = set(re.findall(r"--[a-z][a-z-]*", result.stdout)) - shared
        listed = set(re.findall(r"--[a-z][a-z-]*", line))
        for flag in sorted(real - listed):
            problems.append(f"{where}: `{' '.join(words)}` has {flag}, which its line omits")
    assert not problems, "\n".join(problems)
    for flag in shared - {"--help"}:
        assert flag in text.split("## context")[0]


def test_the_json_in_commands_md_parses() -> None:
    for info, body, where in _fences(REFS / "commands.md"):
        if info == "json":
            json.loads(body)  # a fence the reader copies from must at least be JSON
        elif info == "command":
            assert body.startswith("fastaiagent studio"), where


# -- the spec ------------------------------------------------------------------------------


def _example_specs() -> list[tuple[str, dict]]:
    return [
        (where, json.loads(body))
        for info, body, where in _fences(REFS / "spec.md")
        if info == "json spec"
    ]


def test_there_is_an_example_of_every_shape() -> None:
    shapes = {spec["architecture"] for _, spec in _example_specs()}
    assert shapes == {"single", "supervisor", "swarm"}


@pytest.mark.parametrize("example", _example_specs(), ids=lambda e: e[1]["name"])
def test_example_specs_validate_against_the_planes_schema(example: tuple[str, dict]) -> None:
    jsonschema = pytest.importorskip("jsonschema")
    where, spec = example
    validator = jsonschema.Draft202012Validator(SCHEMA_ANSWER["schema"])
    errors = sorted(validator.iter_errors(spec), key=lambda e: list(e.path))
    assert not errors, f"{where}: " + "; ".join(
        f"{'/'.join(map(str, e.path)) or '<top>'}: {e.message}" for e in errors[:5]
    )
    assert spec["spec_version"] == SCHEMA_ANSWER["spec_version"]


@pytest.mark.parametrize("example", _example_specs(), ids=lambda e: e[1]["name"])
def test_example_specs_follow_the_rules_a_schema_cannot_say(example: tuple[str, dict]) -> None:
    """The plane's own rules, as far as they can be read from the spec alone. A live plane
    checks the rest (``tests/e2e/test_gate_agent_studio.py`` sends every example to one)."""
    _, spec = example
    limits = SCHEMA_ANSWER["limits"]
    assert len(spec["eval_cases"]) >= limits["min_eval_cases"]
    assert re.fullmatch(r"[a-z][a-z0-9_-]{2,63}", spec["name"])
    workers = spec.get("workers", [])
    if spec["architecture"] == "single":
        assert not workers and spec["tools"]
        assert len(spec["system_prompt"]) >= 40
    else:
        assert limits["min_workers"] <= len(workers) <= limits["max_workers"]
        assert not spec["tools"], "a team's tools belong to its workers"
        assert len({w["name"] for w in workers}) == len(workers)
        assert all(len(w["system_prompt"]) >= 40 for w in workers)
    tools = spec["tools"] + [t for w in workers for t in w.get("tools", [])]
    names = [t["name"] for t in tools]
    assert len(names) == len(set(names)), "tool names are unique across the whole spec"
    for tool in tools:
        if tool["type"] == "mock":
            assert set(tool["match_keys"]) <= set(tool["parameters"])
    factories = {g["factory"]: g for g in SCHEMA_ANSWER["guardrails"]}
    for g in spec["guardrails"]:
        assert g["factory"] in factories
        assert set(factories[g["factory"]]["required"]) <= set(g.get("kwargs", {}))
    if spec["architecture"] == "swarm":
        assert "system_prompt" not in spec
        reachable, queue = {spec["entrypoint"]}, [spec["entrypoint"]]
        while queue:
            here = queue.pop()
            for h in spec["handoffs"]:
                if h["from_worker"] == here and h["to_worker"] not in reachable:
                    reachable.add(h["to_worker"])
                    queue.append(h["to_worker"])
        assert reachable == {w["name"] for w in workers}, "every worker must be reachable"


def _table_rows(text: str, heading: str) -> list[list[str]]:
    section = text.split(f"## {heading}")[1].split("\n## ")[0]
    rows = [r for r in section.splitlines() if r.startswith("|")][2:]
    return [[c.strip() for c in r.strip("|").split("|")] for r in rows]


def test_the_guardrail_table_matches_the_planes_schema() -> None:
    text = (REFS / "spec.md").read_text(encoding="utf-8")
    documented: dict[str, tuple[str, str]] = {}
    for factory, args, notes in (r[:3] for r in _table_rows(text, "Guardrails")):
        documented[factory.strip("`")] = (args, notes)
    real = {g["factory"]: g for g in SCHEMA_ANSWER["guardrails"]}
    assert set(documented) == set(real), (
        f"missing: {sorted(set(real) - set(documented))}; "
        f"invented: {sorted(set(documented) - set(real))}"
    )
    for name, (args, notes) in documented.items():
        for required in real[name]["required"]:
            assert f"required `{required}`" in args, f"{name}: required {required} not shown"
        for optional in real[name]["optional"]:
            assert f"optional `{optional}`" in args, f"{name}: optional {optional} not shown"
        if not real[name]["required"] and not real[name]["optional"]:
            assert args == "none", name
        assert ("**Warns.**" in notes) == (name in SCHEMA_ANSWER["guardrails_that_warn"]), name
    for name in SCHEMA_ANSWER["guardrails_on_the_question_in_a_swarm"]:
        assert "swarm" in documented[name][1].lower(), name


def test_the_tool_types_and_architectures_are_all_covered() -> None:
    text = (REFS / "spec.md").read_text(encoding="utf-8")
    tool_rows = {r[0].strip("`") for r in _table_rows(text, "Tools") if r[0].startswith("`")}
    assert tool_rows == set(SCHEMA_ANSWER["tool_types"])
    for arch in SCHEMA_ANSWER["architectures"]:
        assert f"`{arch}`" in text
    for key in SCHEMA_ANSWER["eval_criteria_keys"]:
        assert f"`{key}`" in text


def test_the_limits_in_the_text_are_the_planes() -> None:
    text = (REFS / "spec.md").read_text(encoding="utf-8")
    limits = SCHEMA_ANSWER["limits"]
    assert f"**At least {limits['min_eval_cases']}.**" in text
    assert f"**{limits['min_workers']} to {limits['max_workers']}**" in text


def test_the_warning_codes_in_the_text_are_the_four_the_plane_has() -> None:
    text = (REFS / "spec.md").read_text(encoding="utf-8")
    codes = {r[0].strip("`") for r in _table_rows(text, "Warnings")}
    assert codes == {
        "guardrail_blocks_output",
        "guardrail_checks_question",
        "eval_value_unusable",
        "eval_dataset_not_updated",
    }


# -- errors --------------------------------------------------------------------------------

#: The plane's codes (``docs/Agent_Studio_API.md`` §1.1) and the codes the SDK adds itself.
PLANE_CODES = {
    "insufficient_scope", "feature_not_available", "environment_rule", "key_has_no_owner",
    "project_required", "not_found", "name_taken", "no_llm_endpoint", "build_in_progress",
    "no_version", "no_active_sandbox", "worker_required", "unknown_worker",
    "worker_has_no_cases", "unknown_case", "topology_promote_unsupported",
    "dataset_already_linked", "spec_invalid", "invalid_request", "no_linked_dataset",
    "not_sdk_owned", "unsupported_file", "no_text_layer", "too_many_fixtures",
    "rate_limited",
}  # fmt: skip
SDK_CODES = {
    "auth_error", "connection_error", "invalid_input", "version_failed", "wait_timeout",
    "studio_not_available", "ambiguous_draft", "export_would_overwrite", "unsafe_export",
}  # fmt: skip


def test_every_error_code_is_explained() -> None:
    text = (REFS / "errors.md").read_text(encoding="utf-8")
    documented = set(re.findall(r"^\| `([a-z_]+)` \|", text, re.M))
    expected = PLANE_CODES | SDK_CODES
    assert expected <= documented, f"not explained: {sorted(expected - documented)}"
    assert documented <= expected, f"not real codes: {sorted(documented - expected)}"


def test_every_code_with_a_hint_is_explained() -> None:
    text = (REFS / "errors.md").read_text(encoding="utf-8")
    missing = [code for code in _HINTS if f"| `{code}` |" not in text]
    assert not missing, (
        f"the CLI gives a hint for these but errors.md does not explain them: {missing}"
    )


def test_the_skill_never_asks_for_or_shows_a_key() -> None:
    for path in [SKILL / "SKILL.md", *REFS.glob("*.md")]:
        text = path.read_text(encoding="utf-8")
        assert not re.search(r"fa_k_[A-Za-z0-9_-]{8,}", text), f"{path.name} contains a key"
