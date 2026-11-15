"""``fastaiagent studio`` — every command, driven through the real CLI.

The plane is a real local HTTP server (``CaptureServer``) that answers with the shapes the
plane's contract documents; the CLI is the real Typer app. What is asserted is what a
coding tool relies on: the request that went out, JSON on stdout, notes on stderr, and the
exit code.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from fastaiagent.cli.main import app

try:  # Click < 8.2 mixes stderr into stdout unless asked not to; 8.2+ always separates them.
    runner = CliRunner(mix_stderr=False)
except TypeError:
    runner = CliRunner()

BASE = "/public/v1/agent-studio"
LISTING = {"current": "v1.12", "endpoints": [f"GET {BASE}/context"]}
UUID = "2d70c0de-1111-4222-8333-444455556666"
ENDPOINT = {"id": "ep1", "name": "OpenAI cheap", "provider": "openai", "model": "gpt-5.4-mini"}
CONTEXT = {
    "project": {"id": "p1", "name": "Agent Studio", "domain_id": "dom"},
    "environment": {"tier": "development", "studio_writes_allowed": True},
    "llm_endpoints": [ENDPOINT],
    "knowledge_bases": [{"id": "kb1", "name": "Support KB"}],
    "sdk": {"sdk_pin": "1.88.0", "renderer_version": 16, "dependency_range": "fastaiagent>=1.88.0"},
    "limits": {"max_fixtures_per_draft": 20, "max_workers": 6, "min_eval_cases": 3},
}
SCHEMA = {
    "spec_version": 1,
    "renderer_version": 16,
    "schema": {"title": "AgentSpec", "type": "object"},
    "guardrails": [
        {"factory": "no_pii", "required": [], "optional": []},
        {"factory": "cost_limit", "required": ["max_usd"], "optional": []},
        {"factory": "json_valid", "required": [], "optional": []},
    ],
    "guardrails_that_warn": {"json_valid": "blocks every answer that is not a JSON document"},
    "guardrails_on_the_question_in_a_swarm": {"allowed_topics": "checks the question"},
    "tool_types": ["rest", "mcp", "kb", "function", "mock"],
    "architectures": ["single", "supervisor", "swarm"],
    "eval_criteria_keys": ["must_mention", "must_not_mention", "behaviour"],
    "limits": {},
}
SPEC = {
    "spec_version": 1,
    "name": "order-desk",
    "architecture": "single",
    "description": "Answers order questions.",
    "tools": [{"type": "mock", "name": "orders_lookup"}],
    "guardrails": [{"factory": "no_pii"}],
    "eval_cases": [{"name": "a"}, {"name": "b"}, {"name": "c"}],
}


@pytest.fixture(autouse=True)
def _no_saved_credentials(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The developer's own ``~/.fastaiagent/credentials.toml`` must never reach a test."""
    import fastaiagent.cli.auth as auth

    monkeypatch.setattr(auth, "_CREDENTIALS_PATH", tmp_path / "no-credentials.toml")
    monkeypatch.delenv("FASTAIAGENT_API_KEY", raising=False)
    monkeypatch.delenv("FASTAIAGENT_TARGET", raising=False)
    monkeypatch.delenv("FASTAIAGENT_PROJECT", raising=False)


def _serve(server, *responses: tuple[Any, ...]) -> None:
    server.set_response_sequence(list(responses))


def _serve_runs(server, *runs: list[tuple[Any, ...]]) -> None:
    """For a test that invokes the CLI more than once: ``runs`` holds the answers for each
    invocation, in order."""
    server.set_response_sequence([answer for run in runs for answer in run])


def _studio(server, *args: str, input: str | None = None):
    env = {"FASTAIAGENT_API_KEY": "fa_k_test", "FASTAIAGENT_TARGET": server.url}
    return runner.invoke(app, ["studio", *args], env=env, input=input)


def _calls(server) -> list[dict[str, Any]]:
    return list(server.requests)


def _json(result) -> Any:
    assert result.stdout.strip(), f"stdout was empty; stderr: {result.stderr}"
    return json.loads(result.stdout)


def _draft(**extra: Any) -> dict[str, Any]:
    return {"id": UUID, "name": "Order desk", "status": "ready", "architecture": "single", **extra}


def _version(number: int = 1, **extra: Any) -> dict[str, Any]:
    return {
        "id": f"v{number}",
        "version": number,
        "generation_status": "completed",
        "created_at": "2026-11-15T10:00:00Z",
        "retry_after_seconds": 0,
        **extra,
    }


def _error(code: str, message: str = "words", **extra: Any) -> dict[str, Any]:
    return {"detail": {"error": code, "message": message, **extra}}


def _zip(files: dict[str, str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        for name, content in files.items():
            bundle.writestr(name, content)
    return buffer.getvalue()


# -- every command is there, with help ----------------------------------------------------------

COMMANDS = [
    ["context"],
    ["list"],
    ["new"],
    ["status"],
    ["spec", "schema"],
    ["spec", "validate"],
    ["spec", "push"],
    ["spec", "get"],
    ["show"],
    ["diff"],
    ["try"],
    ["eval"],
    ["export"],
    ["promote"],
    ["open"],
    ["delete"],
]


@pytest.mark.parametrize("command", COMMANDS, ids=[" ".join(c) for c in COMMANDS])
def test_every_command_has_help_and_takes_json(command: list[str]) -> None:
    result = runner.invoke(app, ["studio", *command, "--help"])
    assert result.exit_code == 0, result.output
    assert "--json" in result.stdout


def test_there_is_no_plan_build_refine_or_reset() -> None:
    for name in ("plan", "build", "refine", "reset"):
        result = runner.invoke(app, ["studio", name, "--help"])
        assert result.exit_code == 2, name  # Typer's "no such command"


# -- credentials -------------------------------------------------------------------------------


def test_no_key_says_how_to_get_one(capture_server) -> None:
    result = runner.invoke(app, ["studio", "list"], env={"FASTAIAGENT_TARGET": capture_server.url})
    assert result.exit_code == 1
    assert result.stdout == ""
    assert "No API key" in result.stderr and "fastaiagent connect" in result.stderr


def test_saved_credentials_are_used(tmp_path: Path, capture_server, monkeypatch) -> None:
    import fastaiagent.cli.auth as auth

    saved = tmp_path / "credentials.toml"
    saved.write_text(f'api_key = "fa_k_saved"\ntarget = "{capture_server.url}"\n')
    monkeypatch.setattr(auth, "_CREDENTIALS_PATH", saved)
    _serve(capture_server, (200, []))
    result = runner.invoke(app, ["studio", "list"])
    assert result.exit_code == 0, result.stderr
    assert _calls(capture_server)[0]["headers"]["X-API-Key"] == "fa_k_saved"


def test_a_flag_overrides_the_environment(capture_server) -> None:
    _serve(capture_server, (200, []))
    result = _studio(capture_server, "list", "--api-key", "fa_k_flag")
    assert result.exit_code == 0
    assert _calls(capture_server)[0]["headers"]["X-API-Key"] == "fa_k_flag"


def test_the_key_never_appears_in_output(capture_server) -> None:
    _serve(capture_server, (401, _error("invalid_api_key", "A key is required.")))
    result = _studio(capture_server, "list", "--json")
    assert "fa_k_test" not in result.stdout + result.stderr


# -- context, list ---------------------------------------------------------------------------


def test_context_json_carries_the_project_and_the_guardrails(capture_server) -> None:
    _serve(capture_server, (200, CONTEXT), (200, SCHEMA))
    result = _studio(capture_server, "context", "--json")
    assert result.exit_code == 0, result.stderr
    out = _json(result)
    assert out["llm_endpoints"][0]["id"] == "ep1"
    assert out["sdk"]["sdk_pin"] == "1.88.0"
    assert {"factory": "cost_limit", "required": ["max_usd"], "optional": []} in out["guardrails"]
    assert "json_valid" in out["guardrails_that_warn"]
    assert out["tool_types"] == SCHEMA["tool_types"]
    assert "schema" not in out  # the JSON Schema itself is `spec schema`


def test_context_in_words_and_the_missing_endpoint_note(capture_server) -> None:
    empty = {**CONTEXT, "llm_endpoints": []}
    _serve(capture_server, (200, empty), (200, SCHEMA))
    result = _studio(capture_server, "context")
    assert result.exit_code == 0
    assert "Support KB" in result.stdout and "cost_limit(max_usd)" in result.stdout
    assert "json_valid()  [warns]" in result.stdout
    assert "no LLM endpoint" in result.stderr


def test_context_says_when_writes_are_off(capture_server) -> None:
    off = {**CONTEXT, "environment": {"tier": "production", "studio_writes_allowed": False}}
    _serve(capture_server, (200, off), (200, SCHEMA))
    result = _studio(capture_server, "context")
    assert "WRITES OFF" in result.stdout and "production rules" in result.stderr


def test_list_prints_a_table_and_json(capture_server) -> None:
    drafts = [
        _draft(steps_summary={"current": "evaluate"}),
        _draft(id="d2", name="Team", architecture="swarm"),
    ]
    _serve_runs(capture_server, [(200, drafts)], [(200, drafts)])
    table = _studio(capture_server, "list", "--status", "ready", "-q", "order", "--limit", "5")
    assert table.exit_code == 0
    assert "Order desk" in table.stdout and "evaluate" in table.stdout and "swarm" in table.stdout
    assert _calls(capture_server)[0]["path"] == f"{BASE}/drafts?status=ready&q=order&limit=5"
    assert [d["name"] for d in _json(_studio(capture_server, "list", "--json"))] == [
        "Order desk",
        "Team",
    ]


def test_an_empty_list_says_so(capture_server) -> None:
    _serve(capture_server, (200, []))
    assert "No drafts." in _studio(capture_server, "list").stdout


# -- new -----------------------------------------------------------------------------------------


def test_new_makes_the_draft_with_the_projects_only_endpoint(
    capture_server, tmp_path: Path
) -> None:
    orders = tmp_path / "orders.csv"
    orders.write_text("order_id,status\n1003,shipped\n")
    scan = tmp_path / "receipt.txt"
    scan.write_text("Receipt: order 1003, 42.00")
    _serve(
        capture_server,
        (200, CONTEXT),
        (201, _draft(status="drafting", console_url="http://console/next/studio/x")),
        (201, [{"id": "f1", "name": "orders", "kind": "csv"}]),
        (201, {"id": "f2", "name": "receipt", "kind": "doc"}),
    )
    result = _studio(
        capture_server, "new", "--name", "Order desk", "--brief", "Answer order questions.",
        "--data", str(orders), "--data-text", f"receipt={scan}", "--harness", "--json",
    )  # fmt: skip
    assert result.exit_code == 0, result.stderr
    out = _json(result)
    assert [f["name"] for f in out["fixtures"]] == ["orders", "receipt"]
    create, upload, text = _calls(capture_server)[1:]
    assert create["body"] == {
        "name": "Order desk",
        "requirements_text": "Answer order questions.",
        "llm_endpoint_id": "ep1",
        "use_harness": True,
    }
    assert b'filename="orders.csv"' in upload.raw
    assert text["body"] == {
        "name": "receipt",
        "text": "Receipt: order 1003, 42.00",
        "source_filename": "receipt.txt",
    }


def test_new_with_two_endpoints_asks_which(capture_server) -> None:
    two = {**CONTEXT, "llm_endpoints": [ENDPOINT, {**ENDPOINT, "id": "ep2", "name": "Strong"}]}
    _serve(capture_server, (200, two))
    result = _studio(capture_server, "new", "--name", "x", "--brief", "A long enough brief.")
    assert result.exit_code == 1
    assert "pass --endpoint" in result.stderr and "ep2" in result.stderr
    assert len(_calls(capture_server)) == 1  # no draft was made


def test_new_with_no_endpoint_stops_with_the_planes_code(capture_server) -> None:
    _serve(capture_server, (200, {**CONTEXT, "llm_endpoints": []}))
    result = _studio(
        capture_server, "new", "--name", "x", "--brief", "A long enough brief.", "--json"
    )
    assert result.exit_code == 1
    error = _json(result)["error"]
    assert error["code"] == "no_llm_endpoint" and "console" in error["hint"]


def test_new_checks_its_files_before_it_makes_anything(capture_server) -> None:
    _serve(capture_server)
    result = _studio(
        capture_server,
        "new",
        "--name",
        "x",
        "--brief",
        "A long enough brief.",
        "--data",
        "/nope/orders.csv",
    )
    assert result.exit_code == 1 and "No such file" in result.stderr
    assert capture_server.requests == []  # not even the version check: nothing was sent


def test_new_rejects_a_malformed_data_text(capture_server) -> None:
    result = _studio(
        capture_server,
        "new",
        "--name",
        "x",
        "--brief",
        "A long enough brief.",
        "--data-text",
        "oops",
    )
    assert result.exit_code == 1 and "name=file" in result.stderr


def test_new_reports_a_scan_it_could_not_add_and_still_says_which_draft(
    capture_server, tmp_path: Path
) -> None:
    scan = tmp_path / "receipt.png"
    scan.write_bytes(b"\x89PNG\r\n")
    _serve(
        capture_server,
        (200, CONTEXT),
        (201, _draft(status="drafting")),
        (422, _error("no_text_layer", "Send the text to /fixtures/text.")),
    )
    result = _studio(
        capture_server,
        "new",
        "--name",
        "x",
        "--brief",
        "A long enough brief.",
        "--data",
        str(scan),
        "--json",
    )
    assert result.exit_code == 1
    out = _json(result)
    assert out["draft"]["id"] == UUID
    assert out["errors"][0]["code"] == "no_text_layer"
    assert "was made" in result.stderr and UUID in result.stderr


def test_new_with_a_taken_name_is_refused_with_its_code(capture_server) -> None:
    _serve(capture_server, (200, CONTEXT), (409, _error("name_taken", "A draft named 'x' exists.")))
    result = _studio(
        capture_server, "new", "--name", "x", "--brief", "A long enough brief.", "--json"
    )
    assert result.exit_code == 1
    error = _json(result)["error"]
    assert (error["code"], error["status"]) == ("name_taken", 409)


# -- status ----------------------------------------------------------------------------------


def _detail(stale: bool = False) -> dict[str, Any]:
    return _draft(
        steps={
            "current": "evaluate",
            "stale": stale,
            "steps": [
                {"key": "build", "state": "done", "summary": "v1, made through the API"},
                {"key": "evaluate", "state": "current", "summary": "2 of 3 passing"},
            ],
            "next": {
                "action": "eval",
                "reason": "1 case is failing",
                "hint": "fastaiagent studio eval x",
            },
        },
        messages=[{"id": "m1"}],
    )


def test_status_shows_the_steps_and_the_next_action(capture_server) -> None:
    _serve(capture_server, (200, _detail()))
    result = _studio(capture_server, "status", UUID)
    assert result.exit_code == 0
    assert "evaluate  " in result.stdout and "Next: eval — 1 case is failing" in result.stdout
    assert "STALE" not in result.stderr
    assert _calls(capture_server)[0]["path"] == f"{BASE}/drafts/{UUID}"


def test_status_json_leaves_out_the_message_log(capture_server) -> None:
    _serve(capture_server, (200, _detail()))
    out = _json(_studio(capture_server, "status", UUID, "--json"))
    assert out["steps"]["next"]["action"] == "eval" and "messages" not in out


def test_status_says_stop_and_ask_when_stale(capture_server) -> None:
    _serve(capture_server, (200, _detail(stale=True)))
    result = _studio(capture_server, "status", UUID, "--json")
    assert result.exit_code == 0
    assert _json(result)["steps"]["stale"] is True
    assert "STALE" in result.stderr and "Stop and ask" in result.stderr


def test_a_draft_is_found_by_its_name(capture_server) -> None:
    _serve(capture_server, (200, [_draft()]), (200, _detail()))
    result = _studio(capture_server, "status", "order DESK")
    assert result.exit_code == 0
    first, second = _calls(capture_server)
    assert first["path"].startswith(f"{BASE}/drafts?q=order+DESK")
    assert second["path"] == f"{BASE}/drafts/{UUID}"


def test_an_unknown_name_is_not_found(capture_server) -> None:
    _serve(capture_server, (200, []))
    result = _studio(capture_server, "status", "ghost", "--json")
    assert result.exit_code == 1 and _json(result)["error"]["code"] == "not_found"


def test_the_project_id_is_sent_when_given(capture_server) -> None:
    _serve(capture_server, (200, _detail()))
    _studio(capture_server, "status", UUID, "--project-id", "proj-9")
    assert _calls(capture_server)[0]["path"] == f"{BASE}/drafts/{UUID}?project_id=proj-9"


# -- spec ----------------------------------------------------------------------------------------


def test_spec_schema_prints_the_json_schema_only(capture_server) -> None:
    _serve_runs(capture_server, [(200, SCHEMA)], [(200, SCHEMA)])
    assert _json(_studio(capture_server, "spec", "schema")) == {
        "title": "AgentSpec",
        "type": "object",
    }
    full = _json(_studio(capture_server, "spec", "schema", "--full"))
    assert full["schema"]["title"] == "AgentSpec" and full["tool_types"]


def _spec_file(tmp_path: Path, spec: Any = SPEC) -> str:
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    return str(path)


def test_validate_exits_zero_when_the_spec_checks_clean(capture_server, tmp_path: Path) -> None:
    _serve(capture_server, (200, {"ok": True, "problems": [], "warnings": []}))
    result = _studio(capture_server, "spec", "validate", UUID, _spec_file(tmp_path))
    assert result.exit_code == 0 and "checks clean" in result.stdout
    request = _calls(capture_server)[0]
    assert request["path"] == f"{BASE}/drafts/{UUID}/spec/validate"
    assert request["body"] == {"spec": SPEC}


def test_validate_exits_one_on_a_problem_and_lists_it(capture_server, tmp_path: Path) -> None:
    problems = [{"where": "tools[0]", "message": "fixture 'nope' is not uploaded"}]
    _serve(capture_server, (200, {"ok": False, "problems": problems, "warnings": []}))
    result = _studio(capture_server, "spec", "validate", UUID, _spec_file(tmp_path))
    assert result.exit_code == 1
    assert "tools[0]: fixture 'nope' is not uploaded" in result.stdout


def test_validate_warnings_are_printed_and_do_not_fail_it(capture_server, tmp_path: Path) -> None:
    warning = {
        "where": "guardrails[0]",
        "code": "guardrail_blocks_output",
        "message": "blocks sentences",
    }
    _serve(capture_server, (200, {"ok": True, "problems": [], "warnings": [warning]}))
    result = _studio(capture_server, "spec", "validate", UUID, _spec_file(tmp_path))
    assert result.exit_code == 0
    assert "[guardrail_blocks_output] guardrails[0]: blocks sentences" in result.stdout


def test_validate_json_is_the_planes_answer(capture_server, tmp_path: Path) -> None:
    _serve(
        capture_server,
        (200, {"ok": False, "problems": [{"where": "name", "message": "bad"}], "warnings": []}),
    )
    result = _studio(capture_server, "spec", "validate", UUID, _spec_file(tmp_path), "--json")
    assert result.exit_code == 1
    assert _json(result) == {
        "ok": False,
        "problems": [{"where": "name", "message": "bad"}],
        "warnings": [],
    }


def test_a_spec_can_come_from_stdin(capture_server) -> None:
    _serve(capture_server, (200, {"ok": True, "problems": [], "warnings": []}))
    result = _studio(capture_server, "spec", "validate", UUID, "-", input=json.dumps(SPEC))
    assert result.exit_code == 0
    assert _calls(capture_server)[0]["body"] == {"spec": SPEC}


def test_a_spec_that_is_not_json_is_said_so_and_nothing_is_sent(
    capture_server, tmp_path: Path
) -> None:
    broken = tmp_path / "spec.json"
    broken.write_text("{not json")
    result = _studio(capture_server, "spec", "validate", UUID, str(broken))
    assert result.exit_code == 1 and "not valid JSON" in result.stderr
    assert capture_server.requests == []
    listy = tmp_path / "list.json"
    listy.write_text("[1, 2]")
    again = _studio(capture_server, "spec", "validate", UUID, str(listy))
    assert again.exit_code == 1 and "one JSON object" in again.stderr


def test_push_makes_a_version_and_reads_the_warnings_out(capture_server, tmp_path: Path) -> None:
    warning = {
        "where": "guardrails[0]",
        "code": "guardrail_blocks_output",
        "message": "blocks sentences",
    }
    _serve(capture_server, (202, {"version_id": "v1", "version": 1, "warnings": [warning]}))
    result = _studio(
        capture_server, "spec", "push", UUID, _spec_file(tmp_path), "--note", "first design"
    )
    assert result.exit_code == 0
    assert "Version 1 (v1)" in result.stdout and "read them out" in result.stdout
    assert "blocks sentences" in result.stdout
    assert _calls(capture_server)[0]["body"] == {"spec": SPEC, "note": "first design"}


def test_push_with_a_bad_spec_makes_nothing_and_exits_one(capture_server, tmp_path: Path) -> None:
    problems = [{"where": "tools[0]", "message": "fixture 'nope' is not uploaded"}]
    _serve(
        capture_server, (422, _error("spec_invalid", "The spec has problems.", problems=problems))
    )
    result = _studio(capture_server, "spec", "push", UUID, _spec_file(tmp_path), "--json")
    assert result.exit_code == 1
    error = _json(result)["error"]
    assert error["code"] == "spec_invalid" and error["details"]["problems"] == problems
    assert "tools[0]: fixture 'nope'" in result.stderr


def test_push_wait_polls_until_the_version_is_made(capture_server, tmp_path: Path) -> None:
    _serve(
        capture_server,
        (202, {"version_id": "v1", "version": 1, "warnings": []}),
        (200, _version(generation_status="running", retry_after_seconds=0.01)),
        (200, _version()),
    )
    result = _studio(capture_server, "spec", "push", UUID, _spec_file(tmp_path), "--wait", "--json")
    assert result.exit_code == 0, result.stderr
    out = _json(result)
    assert out["version"] == 1 and out["generation"]["generation_status"] == "completed"


def test_push_wait_exits_one_when_the_version_fails(capture_server, tmp_path: Path) -> None:
    _serve(
        capture_server,
        (202, {"version_id": "v1", "version": 1, "warnings": []}),
        (200, _version(generation_status="failed", error_message="render error")),
    )
    result = _studio(capture_server, "spec", "push", UUID, _spec_file(tmp_path), "--wait", "--json")
    assert result.exit_code == 1
    assert _json(result)["error"]["code"] == "version_failed"
    assert "render error" in result.stderr


def test_spec_get_prints_the_current_spec_or_writes_it(capture_server, tmp_path: Path) -> None:
    versions = [_version(2), _version(1)]
    _serve_runs(
        capture_server,
        [(200, versions), (200, _version(2, spec=SPEC))],
        [(200, versions), (200, _version(1, spec={"name": "old"}))],
    )
    current = _studio(capture_server, "spec", "get", UUID)
    assert json.loads(current.stdout) == SPEC
    assert _calls(capture_server)[1]["path"] == f"{BASE}/drafts/{UUID}/versions/v2"
    out = tmp_path / "old.json"
    older = _studio(capture_server, "spec", "get", UUID, "--version", "1", "--out", str(out))
    assert older.exit_code == 0 and json.loads(out.read_text()) == {"name": "old"}
    assert older.stdout == "" and "version 1 written" in older.stderr


def test_spec_get_of_a_draft_with_no_version_says_so(capture_server) -> None:
    _serve(capture_server, (200, []))
    result = _studio(capture_server, "spec", "get", UUID, "--json")
    assert result.exit_code == 1 and _json(result)["error"]["code"] == "no_version"


def test_asking_for_a_version_that_is_not_there_lists_the_ones_that_are(capture_server) -> None:
    _serve(capture_server, (200, [_version(2), _version(1)]))
    result = _studio(capture_server, "spec", "get", UUID, "--version", "9")
    assert result.exit_code == 1 and "No version 9" in result.stderr and "2, 1" in result.stderr


# -- show, diff ----------------------------------------------------------------------------------


def test_show_prints_the_blueprint(capture_server) -> None:
    team = {
        "name": "triage",
        "architecture": "supervisor",
        "workers": [
            {
                "name": "orders",
                "tools": [{"name": "orders_lookup", "type": "mock"}],
                "eval_cases": [{}],
            }
        ],
        "guardrails": [],
        "eval_cases": [{}, {}, {}],
    }
    _serve(capture_server, (200, [_version()]), (200, _version(spec=team)))
    out = _json(_studio(capture_server, "show", UUID, "--json"))
    assert out["blueprint"]["workers"] == [
        {"name": "orders", "tools": ["orders_lookup (mock)"], "eval_cases": 1}
    ]
    assert "spec" not in out["version"]


def test_show_lists_and_prints_generated_files(capture_server) -> None:
    files = {"agent.py": "print('hi')\n", "tests/test_agent.py": "def test(): ...\n"}
    run = [(200, [_version()]), (200, _version(rendered_files=files))]
    _serve_runs(capture_server, run, run, run)
    listing = _studio(capture_server, "show", UUID, "--files")
    assert listing.stdout.split() == ["agent.py", "tests/test_agent.py"]
    assert _calls(capture_server)[1]["path"].endswith("/versions/v1?include=files")
    one = _studio(capture_server, "show", UUID, "--file", "agent.py")
    assert one.stdout == "print('hi')\n"
    missing = _studio(capture_server, "show", UUID, "--file", "nope.py", "--json")
    assert missing.exit_code == 1 and "agent.py" in _json(missing)["error"]["message"]


def test_diff_compares_the_two_versions_by_number(capture_server) -> None:
    diff = {
        "from": 1, "to": 2,
        "spec_changes": [{"where": "system_prompt", "kind": "changed"}],
        "files": [{"path": "agent.py", "kind": "changed"}],
    }  # fmt: skip
    _serve(capture_server, (200, [_version(2), _version(1)]), (200, diff))
    result = _studio(capture_server, "diff", UUID, "1", "2")
    assert result.exit_code == 0
    assert "changed  system_prompt" in result.stdout and "changed  agent.py" in result.stdout
    assert _calls(capture_server)[1]["path"] == f"{BASE}/drafts/{UUID}/versions/v2/diff?against=v1"


# -- try -----------------------------------------------------------------------------------------

SANDBOX_UP = [
    (201, {"status": "requested", "retry_after_seconds": 3}),
    (200, {"status": "ready", "retry_after_seconds": 0}),
]


def _completed_trial(**extra: Any) -> dict[str, Any]:
    return {
        "id": "t1", "status": "completed", "retry_after_seconds": 0,
        "result": {"output": "Order 1003 has shipped."},
        "mock_calls": [{"tool": "orders_lookup", "args": {"order_id": "1003"}, "matched": True}],
        "trace_id": "tr1", "trace_url": "http://console/next/traces/tr1",
        **extra,
    }  # fmt: skip


def test_try_prints_the_answer_the_tool_calls_and_the_trace(capture_server) -> None:
    _serve(
        capture_server,
        *SANDBOX_UP,
        (201, {**_completed_trial(), "status": "queued"}),
        (200, _completed_trial()),
    )
    result = _studio(capture_server, "try", UUID, "Where is order 1003?")
    assert result.exit_code == 0, result.stderr
    assert "Answer: Order 1003 has shipped." in result.stdout
    assert 'orders_lookup {"order_id": "1003"} -> matched' in result.stdout
    assert "Trace: http://console/next/traces/tr1" in result.stdout
    assert "guardrails are not applied" in result.stderr
    asked = [
        r for r in _calls(capture_server) if r["path"].endswith("/trials") and r["method"] == "POST"
    ]
    assert asked[0]["body"] == {"input": "Where is order 1003?"}


def test_try_a_worker_json_and_a_version(capture_server) -> None:
    _serve(
        capture_server, (200, [_version(2), _version(1)]), *SANDBOX_UP,
        (201, {**_completed_trial(), "status": "queued"}), (200, _completed_trial(worker="orders")),
    )  # fmt: skip
    result = _studio(
        capture_server, "try", UUID, "hi", "--worker", "orders", "--version", "1", "--json"
    )
    assert result.exit_code == 0
    assert _json(result)["worker"] == "orders"
    post = next(r for r in _calls(capture_server) if r["path"].endswith("/trials"))
    assert post["body"] == {"input": "hi", "worker": "orders", "version_id": "v1"}


def test_try_a_failed_trial_exits_one(capture_server) -> None:
    failed = {"id": "t1", "status": "failed", "retry_after_seconds": 0, "error": "model refused"}
    _serve(capture_server, *SANDBOX_UP, (201, {**failed, "status": "queued"}), (200, failed))
    result = _studio(capture_server, "try", UUID, "hi")
    assert result.exit_code == 1 and "Error: model refused" in result.stdout


def test_try_a_team_without_a_worker_names_the_workers(capture_server) -> None:
    refusal = _error("worker_required", "Name a worker.", workers=["orders", "billing"])
    _serve(capture_server, *SANDBOX_UP, (400, refusal))
    result = _studio(capture_server, "try", UUID, "hi")
    assert result.exit_code == 1
    assert "workers: orders, billing" in result.stderr and "--worker" in result.stderr


def test_try_no_wait_returns_the_started_trial(capture_server) -> None:
    _serve(
        capture_server,
        *SANDBOX_UP,
        (201, {"id": "t1", "status": "queued", "retry_after_seconds": 2}),
    )
    result = _studio(capture_server, "try", UUID, "hi", "--no-wait", "--json")
    assert result.exit_code == 0 and _json(result)["status"] == "queued"


# -- eval ----------------------------------------------------------------------------------------


def _rollup(failed: int = 0, review: int = 0) -> dict[str, Any]:
    cases = [{"name": "found", "status": "passed"}]
    if failed:
        cases.append(
            {"name": "privacy", "status": "failed", "detail": "must_not_mention found: Dana"}
        )
    if review:
        cases.append({"name": "polite", "status": "needs_review"})
    return {
        "version": 1,
        "totals": {
            "passed": 1, "failed": failed, "needs_review": review,
            "not_run": 0, "on_export_only": 3,
        },
        "groups": [
            {"scope": "orders", "cases": cases},
            {
                "scope": "end_to_end", "on_export_only": True,
                "cases": [{"name": "e2e", "status": "on_export"}],
            },
        ],
    }  # fmt: skip


def _eval_run() -> tuple[Any, ...]:
    run = {"id": "r1", "status": "completed", "retry_after_seconds": 0, "kind": "eval_preview"}
    return (201, {"runs": [{**run, "status": "queued"}]}), (200, run)


def test_eval_exits_zero_when_nothing_failed_and_needs_review_is_not_a_failure(
    capture_server,
) -> None:
    started, finished = _eval_run()
    _serve(capture_server, *SANDBOX_UP, started, finished, (200, _rollup(review=1)))
    result = _studio(capture_server, "eval", UUID)
    assert result.exit_code == 0, result.stdout
    assert "[passed] found" in result.stdout and "[needs_review] polite" in result.stdout
    assert "(runs only from the exported code)" in result.stdout
    assert "1 passed, 0 failed, 1 need review" in result.stdout


def test_eval_exits_one_when_a_case_failed_and_says_why(capture_server) -> None:
    started, finished = _eval_run()
    _serve(capture_server, *SANDBOX_UP, started, finished, (200, _rollup(failed=1)))
    result = _studio(capture_server, "eval", UUID, "--worker", "orders", "--case", "privacy")
    assert result.exit_code == 1
    assert "[failed] privacy — must_not_mention found: Dana" in result.stdout
    post = next(r for r in _calls(capture_server) if r["path"].endswith("/evals/run"))
    assert post["body"] == {"worker": "orders", "case": "privacy"}


def test_eval_json_carries_the_runs_and_the_rollup(capture_server) -> None:
    started, finished = _eval_run()
    _serve(capture_server, *SANDBOX_UP, started, finished, (200, _rollup()))
    out = _json(_studio(capture_server, "eval", UUID, "--json"))
    assert out["runs"][0]["id"] == "r1" and out["rollup"]["totals"]["passed"] == 1


def test_eval_a_run_that_failed_to_run_exits_one(capture_server) -> None:
    dead = {"id": "r1", "status": "failed", "retry_after_seconds": 0, "error": "model key invalid"}
    _serve(
        capture_server, *SANDBOX_UP,
        (201, {"runs": [{**dead, "status": "queued"}]}), (200, dead), (200, _rollup()),
    )  # fmt: skip
    result = _studio(capture_server, "eval", UUID)
    assert result.exit_code == 1 and "failed: model key invalid" in result.stdout


def test_eval_no_wait_starts_the_runs_and_stops(capture_server) -> None:
    started, _ = _eval_run()
    _serve(capture_server, *SANDBOX_UP, started)
    result = _studio(capture_server, "eval", UUID, "--no-wait")
    assert result.exit_code == 0 and "Started 1 run" in result.stdout


# -- export --------------------------------------------------------------------------------------

PROJECT = {
    "order-desk/agent.py": "print('agent')\n",
    "order-desk/tests/test_agent.py": "def test(): ...\n",
}


def test_export_unpacks_the_project_under_one_folder(capture_server, tmp_path: Path) -> None:
    _serve(capture_server, (200, _zip(PROJECT), "application/zip"))
    out = tmp_path / "repo"
    result = _studio(capture_server, "export", UUID, "--out", str(out), "--json")
    assert result.exit_code == 0, result.stderr
    report = _json(result)
    assert report["folders"] == ["order-desk"]
    assert report["files"] == ["order-desk/agent.py", "order-desk/tests/test_agent.py"]
    assert (out / "order-desk" / "agent.py").read_text() == "print('agent')\n"


def test_export_refuses_to_overwrite_and_writes_nothing(capture_server, tmp_path: Path) -> None:
    (tmp_path / "order-desk").mkdir()
    existing = tmp_path / "order-desk" / "agent.py"
    existing.write_text("MINE")
    run = [(200, _zip(PROJECT), "application/zip")]
    _serve_runs(capture_server, run, run)
    refused = _studio(capture_server, "export", UUID, "--out", str(tmp_path), "--json")
    assert refused.exit_code == 1
    assert _json(refused)["error"]["code"] == "export_would_overwrite"
    assert existing.read_text() == "MINE"
    assert not (tmp_path / "order-desk" / "tests").exists()  # nothing partial
    forced = _studio(capture_server, "export", UUID, "--out", str(tmp_path), "--force")
    assert forced.exit_code == 0 and existing.read_text() == "print('agent')\n"


def test_export_refuses_a_path_that_escapes_the_folder(capture_server, tmp_path: Path) -> None:
    _serve(capture_server, (200, _zip({"ok/a.py": "1", "../evil.py": "2"}), "application/zip"))
    out = tmp_path / "repo"
    result = _studio(capture_server, "export", UUID, "--out", str(out), "--json")
    assert result.exit_code == 1 and _json(result)["error"]["code"] == "unsafe_export"
    assert not (tmp_path / "evil.py").exists() and not (out / "ok").exists()


def test_export_a_chosen_version(capture_server, tmp_path: Path) -> None:
    _serve(
        capture_server, (200, [_version(2), _version(1)]), (200, _zip(PROJECT), "application/zip")
    )
    result = _studio(capture_server, "export", UUID, "--out", str(tmp_path), "--version", "1")
    assert result.exit_code == 0
    assert _calls(capture_server)[1]["path"] == f"{BASE}/drafts/{UUID}/export?version_id=v1"


def test_export_of_a_draft_with_no_version_says_so(capture_server, tmp_path: Path) -> None:
    _serve(capture_server, (409, _error("no_version", "Push a spec first.")))
    result = _studio(capture_server, "export", UUID, "--out", str(tmp_path), "--json")
    assert result.exit_code == 1 and _json(result)["error"]["code"] == "no_version"


# -- promote, open, delete ---------------------------------------------------------------------


def test_promote_sends_the_agent_and_prints_the_suite(capture_server) -> None:
    _serve(capture_server, (201, {"suite_id": "s1", "dataset_id": "ds1", "cases": 3}))
    result = _studio(capture_server, "promote", UUID, "--agent", "agent-9")
    assert result.exit_code == 0 and "Promoted 3 cases: suite s1, dataset ds1" in result.stdout
    assert _calls(capture_server)[0]["body"] == {"agent_id": "agent-9"}


def test_promote_a_team_is_refused_and_says_to_export(capture_server) -> None:
    _serve(
        capture_server,
        (409, _error("topology_promote_unsupported", "A team's cases cannot be promoted.")),
    )
    result = _studio(capture_server, "promote", UUID, "--agent", "a")
    assert result.exit_code == 1 and "Export instead" in result.stderr


def test_open_prints_the_console_address_and_nothing_else(capture_server) -> None:
    _serve(capture_server, (200, _draft(console_url="https://login.example/next/studio/x")))
    result = _studio(capture_server, "open", UUID)
    assert result.stdout.strip() == "https://login.example/next/studio/x"


def test_delete_removes_the_draft_without_asking_when_there_is_no_terminal(capture_server) -> None:
    _serve(capture_server, (200, _draft()), (204, b""))
    result = _studio(capture_server, "delete", UUID)
    assert result.exit_code == 0 and "Deleted Order desk" in result.stdout
    assert _calls(capture_server)[-1]["method"] == "DELETE"


def test_delete_json(capture_server) -> None:
    _serve(capture_server, (200, _draft()), (204, b""))
    assert _json(_studio(capture_server, "delete", UUID, "--yes", "--json")) == {
        "deleted": UUID,
        "name": "Order desk",
    }


# -- failures speak one language ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "body", "code", "hint_has"),
    [
        (
            403,
            _error("insufficient_scope", "lacks studio:write"),
            "insufficient_scope",
            "Agent Studio",
        ),
        (
            403,
            _error("environment_rule", "Studio is off here."),
            "environment_rule",
            "Do not retry",
        ),
        (403, _error("key_has_no_owner", "owner gone"), "key_has_no_owner", "new key"),
        (409, _error("build_in_progress", "wait"), "build_in_progress", "Wait"),
    ],
)
def test_a_refusal_is_one_json_error_on_stdout_and_words_on_stderr(
    capture_server, status, body, code, hint_has
) -> None:
    _serve(capture_server, (status, body))
    result = _studio(capture_server, "status", UUID, "--json")
    assert result.exit_code == 1
    error = json.loads(result.stdout)["error"]  # stdout is exactly one JSON document
    assert error["code"] == code and hint_has in error["hint"] and error["status"] == status
    assert f"error: {code}" in result.stderr


def test_without_json_a_failure_leaves_stdout_empty(capture_server) -> None:
    _serve(capture_server, (403, _error("environment_rule", "Studio is off here.")))
    result = _studio(capture_server, "status", UUID)
    assert result.exit_code == 1 and result.stdout == ""


@pytest.mark.parametrize(
    ("status", "body", "code"),
    [
        (401, _error("invalid_api_key", "A key is required."), "auth_error"),
        (400, _error("customer_required", "No workspace."), "auth_error"),
        (500, {"detail": "boom"}, "connection_error"),
    ],
)
def test_platform_errors_keep_their_own_codes(capture_server, status, body, code) -> None:
    _serve(capture_server, (status, body))
    result = _studio(capture_server, "list", "--json")
    assert result.exit_code == 1 and _json(result)["error"]["code"] == code


def test_a_rate_limit_is_its_own_code(capture_server) -> None:
    # A long wait is told to the caller; a short one the client waits out itself.
    _serve(
        capture_server,
        (
            429,
            _error("rate_limited", "slow", retry_after=60),
            "application/json",
            {"Retry-After": "60"},
        ),
    )
    result = _studio(capture_server, "list", "--json")
    error = _json(result)["error"]
    assert result.exit_code == 1 and error["code"] == "rate_limited" and "60" in error["message"]


def test_a_short_rate_limit_is_waited_out_by_the_command(capture_server) -> None:
    limited = (429, _error("rate_limited", "slow"), "application/json", {"Retry-After": "0.1"})
    _serve(capture_server, limited, (200, []))
    result = _studio(capture_server, "list", "--json")
    assert result.exit_code == 0 and _json(result) == []
    assert len(_calls(capture_server)) == 2


def test_an_older_plane_is_told_it_is_too_old(capture_server) -> None:
    capture_server.set_response_sequence(
        [(404, {"detail": "Not Found"}), (200, {"endpoints": ["GET /public/v1/agents"]})]
    )
    result = _studio(capture_server, "list", "--json")
    error = _json(result)["error"]
    assert result.exit_code == 1 and error["code"] == "studio_not_available"
    assert "0.20.0" in error["message"]


def test_a_dead_plane_is_a_connection_error_not_a_traceback(capture_server) -> None:
    url = capture_server.url
    capture_server.stop()
    result = runner.invoke(
        app,
        ["studio", "list", "--json"],
        env={"FASTAIAGENT_API_KEY": "fa_k_test", "FASTAIAGENT_TARGET": url},
    )
    assert result.exit_code == 1 and _json(result)["error"]["code"] == "connection_error"
