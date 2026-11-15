"""End-to-end quality gate — headless Agent Studio, driven through ``fastaiagent studio``.

The loop a coding tool runs, as a subprocess of the real CLI, against a real plane (0.20.0
or later) with real model calls in its sandbox: context → new (sample data) → check a
spec → a refused spec → push → show → try → eval → change and push again → diff → export
→ the exported tests → delete; then the same for a team, and the refusals a tool must
handle. No mocks.

Needs (the plane's own CI sets them; locally source ``~/.fastaiagent-sdk-studio-e2e.env``):

* ``FASTAIAGENT_TARGET`` — the plane's address;
* ``FASTAIAGENT_API_KEY`` — a key with ``studio:read`` and ``studio:write`` (the console's
  "Agent Studio" preset), whose owner is still a member;
* ``E2E_PLANE_PROJECT_ID`` — only for a key that is not tied to one project.

The SDK's own CI cannot boot a plane (``E2E_SKIP_PLATFORM=1``), so this skips there, with
that reason on the skip line: a skip is not a check. It runs for real where a plane is
named — the plane's ``backend.yml`` installs this branch and runs it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from tests.e2e.conftest import require_platform

pytestmark = pytest.mark.e2e

_STUDIO = [sys.executable, "-m", "fastaiagent.cli.main", "studio"]
_ORDERS = "order_id,status,total\n1003,shipped,42.00\n1004,delivered,17.50\n1002,processing,89.00\n"
_RECEIPT = (
    "Receipt (read from a photo): order 1003, paid by card, total 42.00. Returns within 30 days."
)


def _require_plane() -> None:
    require_platform()
    missing = [k for k in ("FASTAIAGENT_API_KEY", "FASTAIAGENT_TARGET") if not os.environ.get(k)]
    if not missing:
        return
    message = f"Missing env for the Agent Studio gate: {missing}"
    if os.environ.get("E2E_REQUIRED") == "1":
        pytest.fail(message)
    pytest.skip(message)


class Cli:
    """``fastaiagent studio …`` as a subprocess; what a coding tool would run."""

    def __init__(self, workdir: Path) -> None:
        self.workdir = workdir

    def run(self, *args: str, stdin: str | None = None) -> subprocess.CompletedProcess[str]:
        argv = [*_STUDIO, *args]
        project = os.environ.get("E2E_PLANE_PROJECT_ID")
        if project and "--project-id" not in args:
            argv += ["--project-id", project]
        return subprocess.run(
            argv, cwd=self.workdir, capture_output=True, text=True, timeout=600, input=stdin
        )

    def ok(self, *args: str, stdin: str | None = None) -> Any:
        """Run with ``--json``, require exit 0 and exactly one JSON document on stdout."""
        done = self.run(*args, "--json", stdin=stdin)
        assert done.returncode == 0, f"{' '.join(args)} → exit {done.returncode}\n{done.stderr}"
        return json.loads(done.stdout)

    def refused(self, *args: str, code: str) -> dict[str, Any]:
        """Run with ``--json``, require exit 1 and the plane's ``code`` in stdout's error."""
        done = self.run(*args, "--json")
        assert done.returncode == 1, f"{' '.join(args)} → exit {done.returncode}\n{done.stdout}"
        error: dict[str, Any] = json.loads(done.stdout)["error"]
        assert error["code"] == code, error
        return error


@pytest.fixture
def cli(tmp_path: Path) -> Iterator[Cli]:
    _require_plane()
    tool = Cli(tmp_path)
    context = tool.run("context", "--json")
    if context.returncode != 0:
        pytest.fail(
            "the Agent Studio gate could not open the door — a gate that cannot sign in "
            f"checked nothing:\n{context.stdout}{context.stderr}"
        )
    made: list[str] = []
    tool.made = made  # type: ignore[attr-defined]
    try:
        yield tool
    finally:
        for draft_id in made:
            tool.run("delete", draft_id, "--yes")


def _endpoint(cli: Cli) -> str:
    endpoints = cli.ok("context")["llm_endpoints"]
    assert endpoints, "the plane's project has no LLM endpoint; the gate cannot make a version"
    return str(endpoints[0]["id"])


def _new(cli: Cli, label: str, *, harness: bool = False) -> str:
    (cli.workdir / "orders.csv").write_text(_ORDERS)
    (cli.workdir / "receipt.txt").write_text(_RECEIPT)
    tag = time.strftime("%H%M%S")
    args = [
        "new", "--name", f"SDK gate {label} {tag}-{os.getpid()}",
        "--brief", "Answer order-status questions from the uploaded orders data.",
        "--data", "orders.csv", "--data-text", "receipt=receipt.txt",
        "--endpoint", _endpoint(cli),
    ]  # fmt: skip
    if harness:
        args.append("--harness")
    made = cli.ok(*args)
    draft_id = str(made["draft"]["id"])
    cli.made.append(draft_id)  # type: ignore[attr-defined]
    assert made["draft"]["created_via"] == "api"
    assert {f["name"] for f in made["fixtures"]} == {"orders", "receipt"}
    return draft_id


def _lookup(name: str, description: str) -> dict[str, Any]:
    return {
        "type": "mock", "name": name, "description": description,
        "parameters": {"order_id": {"type": "string", "description": "the order number"}},
        "fixture_ref": "orders", "match_keys": ["order_id"],
        "default_response": "No order with that id.",
    }  # fmt: skip


def _single_spec(endpoint: str) -> dict[str, Any]:
    return {
        "spec_version": 1, "name": "gate-order-desk", "architecture": "single",
        "description": "Answers order-status questions from the uploaded orders data.",
        "system_prompt": (
            "You answer order-status questions for customers. Use the orders_lookup tool with "
            "the order id; if it is not found, say so plainly. Answer in one or two sentences."
        ),
        "llm": {"endpoint_id": endpoint},
        "tools": [_lookup("orders_lookup", "Look up one order by its id")],
        "guardrails": [{"factory": "no_pii", "kwargs": {}}],
        "eval_cases": [
            {"name": "found_1003", "input": "What is the status of order 1003?",
             "criteria": {"must_mention": ["1003", "shipped"]}},
            {"name": "missing_9999", "input": "Where is order 9999?",
             "criteria": {"behaviour": "says plainly that no such order was found"}},
            {"name": "delivered_1004", "input": "Has order 1004 arrived?",
             "criteria": {"must_mention": ["1004", "delivered"]}},
        ],
    }  # fmt: skip


def _team_spec(endpoint: str) -> dict[str, Any]:
    return {
        "spec_version": 1, "name": "gate-triage-team", "architecture": "supervisor",
        "description": "Routes a support request to the orders or the billing specialist.",
        "system_prompt": (
            "You route each customer request to the right specialist, orders or billing, "
            "then give the customer their answer in one or two sentences."
        ),
        "llm": {"endpoint_id": endpoint}, "tools": [], "guardrails": [],
        "workers": [
            {"name": "orders", "description": "Order status and delivery questions",
             "system_prompt": (
                 "You answer order status questions. Use orders_lookup with the order id "
                 "and answer briefly."
             ),
             "tools": [_lookup("orders_lookup", "Look up one order by its id")],
             "eval_cases": [{"name": "orders_1003", "input": "Where is order 1003?",
                             "criteria": {"must_mention": ["1003", "shipped"]}}]},
            {"name": "billing", "description": "Order totals and charges",
             "system_prompt": (
                 "You answer questions about what an order cost. Use billing_lookup with "
                 "the order id and answer briefly."
             ),
             "tools": [_lookup("billing_lookup", "Look up what one order cost")],
             "eval_cases": [{"name": "billing_1002", "input": "What did order 1002 cost?",
                             "criteria": {"must_mention": ["89"]}}]},
        ],
        "eval_cases": [
            {"name": "end_to_end_status", "input": "Where is order 1003?"},
            {"name": "end_to_end_cost", "input": "What did order 1002 cost?"},
            {"name": "end_to_end_unknown", "input": "Where is order 9999?"},
        ],
    }  # fmt: skip


def _write(cli: Cli, name: str, spec: dict[str, Any]) -> str:
    (cli.workdir / name).write_text(json.dumps(spec, indent=2))
    return name


def test_single_agent_loop(cli: Cli) -> None:
    context = cli.ok("context")
    assert context["environment"]["studio_writes_allowed"], "Studio writes are off on this install"
    assert context["spec_version"] == 1 and any(
        g["factory"] == "no_pii" for g in context["guardrails"]
    )
    schema = json.loads(cli.run("spec", "schema").stdout)
    assert schema["title"] and "properties" in schema

    draft = _new(cli, "single", harness=True)
    spec = _single_spec(_endpoint(cli))

    # A spec that names sample data that is not there is refused, with the place.
    broken = json.loads(json.dumps(spec))
    broken["tools"][0]["fixture_ref"] = "invoices"
    refusal = cli.refused(
        "spec", "push", draft, _write(cli, "broken.json", broken), code="spec_invalid"
    )
    assert refusal["details"]["problems"][0]["where"] == "tools[0]"
    checked = cli.run("spec", "validate", draft, "broken.json")
    assert checked.returncode == 1 and "tools[0]" in checked.stdout

    ok = cli.run("spec", "validate", draft, _write(cli, "spec.json", spec))
    assert ok.returncode == 0, ok.stdout + ok.stderr

    pushed = cli.ok("spec", "push", draft, "spec.json", "--note", "first design", "--wait")
    assert pushed["version"] == 1 and pushed["generation"]["generation_status"] == "completed"

    status = cli.ok("status", draft)
    assert status["architecture"] == "single" and status["steps"]["stale"] is False
    assert status["steps"]["next"]["action"] in {"trial", "start_sandbox"}

    files = cli.ok("show", draft, "--files")["files"]
    assert {"agent.py", "tools.py", "tests/test_agent.py", "evals/cases.jsonl"} <= set(files)
    agent_py = cli.run("show", draft, "--file", "agent.py").stdout
    compile(agent_py, "agent.py", "exec")
    assert cli.ok("show", draft)["blueprint"]["tools"] == ["orders_lookup (mock)"]

    trial = cli.ok("try", draft, "What is the status of order 1003?")
    assert trial["status"] == "completed", trial.get("error")
    assert "shipped" in str(trial["output"]).lower()
    assert any(c["tool"] == "orders_lookup" and c["matched"] for c in trial["mock_calls"])
    assert trial["trace_url"]

    evals = cli.run("eval", draft, "--json")
    # A model may fail a case, so the exit code is 0 or 1; what the gate checks is that it ran.
    assert evals.returncode in (0, 1), evals.stderr
    rollup = json.loads(evals.stdout)["rollup"]
    totals = rollup["totals"]
    assert rollup["version"] == 1 and totals["not_run"] == 0
    assert totals["passed"] + totals["failed"] + totals["needs_review"] == 3
    # A case the word rules cannot judge reads "needs review", never pass or fail.
    statuses = {c["name"]: c["status"] for g in rollup["groups"] for c in g["cases"]}
    assert statuses["missing_9999"] == "needs_review"

    # Change something: read the spec back, edit it, push it. The old version is kept.
    cli.run("spec", "get", draft, "--out", "spec2.json")
    spec2 = json.loads((cli.workdir / "spec2.json").read_text())
    spec2["system_prompt"] += " Always repeat the order id in the answer."
    spec2["tools"].append(
        {"type": "function", "name": "escalate", "description": "Hand the customer to a person",
         "parameters": {"reason": {"type": "string"}}, "returns": "str"}
    )  # fmt: skip
    _write(cli, "spec2.json", spec2)
    assert cli.run("spec", "validate", draft, "spec2.json").returncode == 0
    second = cli.ok("spec", "push", draft, "spec2.json", "--note", "repeat the id", "--wait")
    assert second["version"] == 2
    changes = cli.ok("diff", draft, "1", "2")["spec_changes"]
    assert {"where": "system_prompt", "kind": "changed"} in changes
    assert {"where": "tools.escalate", "kind": "added"} in changes
    assert "first design" in cli.run("show", draft, "--version", "1").stdout
    # Results count for one version: v1's pass is not carried to v2.
    again = json.loads(cli.run("eval", draft, "--case", "found_1003", "--json").stdout)
    assert again["rollup"]["version"] == 2

    # Export: the project lands under one folder; it refuses to overwrite; its tests pass offline.
    out = cli.workdir / "exported"
    exported = cli.ok("export", draft, "--out", str(out))
    (folder,) = exported["folders"]
    assert f"{folder}/agent.py" in exported["files"]
    cli.refused("export", draft, "--out", str(out), code="export_would_overwrite")
    project = out / folder
    compile((project / "agent.py").read_text(), "agent.py", "exec")
    tests = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"],
        cwd=project, capture_output=True, text=True, timeout=300,
    )  # fmt: skip
    assert tests.returncode == 0, tests.stdout + tests.stderr

    assert cli.ok("open", draft)["console_url"].startswith("http")
    listed = cli.ok("list", "-q", "SDK gate single")
    assert any(d["id"] == draft and d["created_via"] == "api" for d in listed)
    cli.ok("delete", draft, "--yes")
    assert not any(d["id"] == draft for d in cli.ok("list", "-q", "SDK gate single"))


def test_team_loop(cli: Cli) -> None:
    draft = _new(cli, "team")
    pushed = cli.ok(
        "spec", "push", draft, _write(cli, "team.json", _team_spec(_endpoint(cli))), "--wait"
    )
    assert pushed["generation"]["generation_status"] == "completed"
    assert cli.ok("status", draft)["architecture"] == "supervisor"

    # A team is tried one worker at a time.
    error = cli.refused("try", draft, "Where is order 1003?", code="worker_required")
    assert error["details"]["workers"] == ["orders", "billing"]
    one = cli.ok("try", draft, "Where is order 1003?", "--worker", "orders")
    assert one["status"] == "completed" and one["worker"] == "orders"

    # Without a worker, eval runs every worker's cases; the team's own run only after export.
    evals = cli.run("eval", draft, "--json")
    assert evals.returncode in (0, 1), evals.stderr
    payload = json.loads(evals.stdout)
    assert "runs" in payload, evals.stdout + evals.stderr
    assert sorted(r["worker"] for r in payload["runs"]) == ["billing", "orders"]
    rollup = payload["rollup"]
    assert [g["scope"] for g in rollup["groups"]] == ["orders", "billing", "end_to_end"]
    assert rollup["totals"]["on_export_only"] == 3 and rollup["totals"]["not_run"] == 0

    # A team's cases are not promoted: export instead.
    cli.refused(
        "promote", draft, "--agent", "00000000-0000-0000-0000-000000000000",
        code="topology_promote_unsupported",
    )  # fmt: skip


def test_the_refusals_a_tool_must_handle(cli: Cli) -> None:
    draft = _new(cli, "refusals")

    # A scan: the plane has no model for it; the tool reads it and sends the text.
    (cli.workdir / "receipt.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    scan = cli.run("new", "--name", f"SDK gate scan {os.getpid()}", "--brief", "A scan to read.",
                   "--data", "receipt.png", "--endpoint", _endpoint(cli), "--json")  # fmt: skip
    assert scan.returncode == 1
    scan_out = json.loads(scan.stdout)
    cli.made.append(scan_out["draft"]["id"])  # type: ignore[attr-defined]
    assert scan_out["errors"][0]["code"] == "no_text_layer"

    # A taken name, whatever its case.
    listed = cli.ok("list", "-q", "SDK gate refusals")
    taken = next(d["name"] for d in listed if d["id"] == draft)
    cli.refused(
        "new", "--name", taken.upper(), "--brief", "The same name again.",
        "--endpoint", _endpoint(cli), code="name_taken",
    )  # fmt: skip

    # No version yet: there is nothing to try, export or read.
    cli.refused("export", draft, "--out", str(cli.workdir / "nothing"), code="no_version")
    cli.refused("spec", "get", draft, code="no_version")

    # A draft out of reach is not found.
    cli.refused("status", "00000000-0000-0000-0000-000000000000", code="not_found")
    cli.refused("status", "no such draft anywhere", code="not_found")

    # A spec that is not JSON never reaches the plane.
    (cli.workdir / "bad.json").write_text("{nope")
    assert cli.run("spec", "validate", draft, "bad.json").returncode == 1


def test_the_example_specs_in_the_skill_are_accepted_by_the_plane(cli: Cli) -> None:
    """The Studio skill teaches three example specs. The SDK's tests check them against
    the schema; only the plane knows the rest of its rules, so each is sent to it here."""
    import re

    text = (
        Path(__file__).resolve().parents[2]
        / "fastaiagent" / "skills" / "fastaiagent-studio" / "references" / "spec.md"
    ).read_text(encoding="utf-8")  # fmt: skip
    examples = re.findall(r"^```json spec\n(.*?)^```$", text, re.S | re.M)
    assert len(examples) == 3, "the skill should carry one example per shape"

    draft = _new(cli, "examples")
    endpoint = _endpoint(cli)
    shapes = set()
    for index, body in enumerate(examples):
        spec = json.loads(body.replace("ENDPOINT_ID", endpoint))
        shapes.add(spec["architecture"])
        checked = cli.run(
            "spec", "validate", draft, _write(cli, f"example{index}.json", spec), "--json"
        )
        report = json.loads(checked.stdout)
        assert checked.returncode == 0 and report["ok"], (spec["name"], report["problems"])
        assert not report["problems"], (spec["name"], report["problems"])
    assert shapes == {"single", "supervisor", "swarm"}
