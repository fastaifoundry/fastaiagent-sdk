"""Run eval from the Local UI against a real agent with real scorers (1.89.0, E9).

The Datasets page could only send ``{"scorers": ["exact_match"]}`` with no
agent, so every UI-launched run was an echo of the inputs. Now:

* ``GET /run-eval/options`` lists the registered agents (a Chain flagged
  unsupported), the scorer catalog with availability and why not, judge models
  with a key, and what the dataset's rows carry;
* ``POST /run-eval`` takes scorer **specs** (parameters, a judge's criteria and
  model), validated before any agent call, every refusal a 422 with a readable
  reason;
* echo runs only when asked (``mode: "echo"``, D18): a dataset check.

Also the data loss found while building it: the datasets API rewrote the whole
JSONL file on every add / edit / delete and kept only four fields, so
``expected_trajectory``, ``context`` and a replay's ``source_trace_id`` vanished
from every row.

Real FastAPI ``TestClient``, real agents over ``TestModel``/``FunctionModel``,
real files; no mocks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("itsdangerous")

from fastapi.testclient import TestClient  # noqa: E402

from fastaiagent.agent import Agent  # noqa: E402
from fastaiagent.chain import Chain  # noqa: E402
from fastaiagent.llm.message import MessageRole  # noqa: E402
from fastaiagent.testing import FunctionModel  # noqa: E402
from fastaiagent.tool import tool  # noqa: E402
from fastaiagent.ui.db import init_local_db  # noqa: E402
from fastaiagent.ui.server import build_app  # noqa: E402


@tool()
def lookup_order(order_id: str) -> str:
    """Look up an order."""
    return f"order {order_id} shipped"


def _turn(messages: list[Any]) -> Any:
    if messages[-1].role == MessageRole.tool:
        return ("Order A1 shipped.", [])
    return ("", [{"name": "lookup_order", "arguments": {"order_id": "A1"}}])


ROWS = [
    {
        "input": "Where is A1?",
        "expected_output": "shipped",
        "expected_trajectory": ["lookup_order"],
        "context": "A1 shipped Monday.",
        "source_trace_id": "abc123",
    },
    {"input": "Hours?", "expected_output": "9-5"},
    {"input": [{"type": "text", "text": "what is this"}, {"type": "image", "path": "red.png"}]},
]


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    fa = tmp_path / ".fastaiagent"
    fa.mkdir()
    db = fa / "local.db"
    init_local_db(db).close()
    datasets = fa / "datasets"
    datasets.mkdir()
    from tests.test_eval_multimodal_persist import _png

    (datasets / "red.png").write_bytes(_png())
    (datasets / "orders.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ROWS))
    agent = Agent(name="support", llm=FunctionModel(_turn), tools=[lookup_order])
    chain = Chain("pipeline")
    app = build_app(
        db_path=str(db), auth_path=tmp_path / "a.json", no_auth=True, runners=[agent, chain]
    )
    with TestClient(app) as c:
        yield c, datasets / "orders.jsonl"


def test_options_describe_agents_scorers_and_rows(client: Any, monkeypatch) -> None:  # noqa: ANN001
    c, _ = client
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    o = c.get("/api/datasets/orders/run-eval/options").json()
    agents = {a["name"]: a for a in o["agents"]}
    assert agents["support"] == {
        "name": "support",
        "kind": "agent",
        "supported": True,
        "reason": None,
    }
    assert agents["pipeline"]["supported"] is False and "Chain" in agents["pipeline"]["reason"]
    f = o["dataset_features"]
    assert (f["rows"], f["with_expected_output"], f["with_expected_trajectory"]) == (3, 2, 1)
    assert (f["with_context"], f["multimodal"]) == (1, 1)
    scorers = {s["name"]: s for s in o["scorers"]}
    assert scorers["exact_match"]["available"] is True
    assert scorers["exact_match"]["rows_missing"] == {"expected_output": 1}
    assert scorers["regex_match"]["params"][0]["name"] == "pattern"
    assert scorers["tool_usage_accuracy"]["available"] is True
    assert scorers["step_efficiency"]["available"] is False
    assert "expected_steps" in scorers["step_efficiency"]["unavailable_reason"]
    assert all(m["provider"] for m in o["judge_models"])
    assert not any(m["provider"] == "openai" for m in o["judge_models"]), "no key, not offered"
    assert scorers["task_completion"]["judge"] is True
    assert "moderation" not in scorers and "decision_judge" not in scorers


def test_a_real_agent_with_spec_scorers(client: Any) -> None:
    c, _ = client
    r = c.post(
        "/api/datasets/orders/run-eval",
        json={
            "agent_name": "support",
            "scorers": [
                "contains",
                {"type": "builtin", "name": "regex_match", "params": {"pattern": "shipped"}},
                {"type": "builtin", "name": "tool_usage_accuracy"},
            ],
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["mode"] == "agent" and body["kind"] == "ui"
    assert body["cases"]["passed"] + body["cases"]["failed"] + body["cases"]["errored"] == 3
    assert set(body["checks"]) >= {"passed", "failed", "errored", "skipped", "pass_rate"}
    case = c.get(f"/api/evals/{body['run_id']}").json()["cases"][0]
    assert case["per_scorer"]["tool_usage_accuracy"]["passed"] is True, case["per_scorer"]
    assert case["per_scorer"]["regex_match"]["passed"] is True


@pytest.mark.parametrize(
    ("body", "needle"),
    [
        ({"scorers": ["exact_match"]}, "Pick an agent"),
        ({"agent_name": "support", "mode": "echo"}, "runs no agent"),
        ({"agent_name": "pipeline"}, "chain with no run()"),
        ({"agent_name": "support", "scorers": ["nope"]}, "Unknown scorer 'nope'"),
        ({"agent_name": "support", "scorers": ["llm_judge"]}, "needs its criteria"),
        (
            {
                "agent_name": "support",
                "scorers": [{"type": "builtin", "name": "regex_match", "params": {"pattern": "("}}],
            },
            "does not compile",
        ),
        (
            {
                "agent_name": "support",
                "scorers": [{"type": "builtin", "name": "exact_match", "params": {"x": 1}}],
            },
            "takes no parameter x",
        ),
        (
            {
                "agent_name": "support",
                "scorers": [{"type": "llm_judge", "criteria": "c", "scale": "7"}],
            },
            "scale must be one of",
        ),
        (
            {
                "agent_name": "support",
                "scorers": [
                    {
                        "type": "llm_judge",
                        "criteria": "c",
                        "model": {"provider": "not-a-provider", "model": "m"},
                    }
                ],
            },
            "No API key",
        ),
        ({"agent_name": "support", "scorers": ["contains", "contains"]}, "Two scorers"),
        ({"mode": "echo", "scorers": ["latency"]}, "needs an agent"),
        ({"mode": "echo", "scorers": ["tool_usage_accuracy"]}, "needs an agent"),
    ],
    ids=[
        "no-agent-no-mode",
        "echo-with-agent",
        "chain",
        "unknown",
        "judge-as-name",
        "bad-regex",
        "unknown-param",
        "bad-scale",
        "judge-without-key",
        "duplicate",
        "echo-latency",
        "echo-trajectory",
    ],
)
def test_every_refusal_is_a_readable_422(client: Any, body: dict, needle: str) -> None:
    c, _ = client
    r = c.post("/api/datasets/orders/run-eval", json=body)
    assert r.status_code == 422, r.text
    detail = r.json()["detail"]
    assert isinstance(detail, str) and needle in detail, detail


def test_an_unloadable_image_is_a_422_not_a_500(client: Any) -> None:
    c, path = client
    path.write_text(json.dumps({"input": [{"type": "image", "path": "missing.png"}]}) + "\n")
    r = c.post("/api/datasets/orders/run-eval", json={"mode": "echo", "scorers": ["json_valid"]})
    assert r.status_code == 422, r.text
    assert "could not be loaded" in r.json()["detail"]


def test_only_chains_registered_leaves_agent_scorers_unavailable(tmp_path: Path) -> None:
    fa = tmp_path / ".fastaiagent"
    (fa / "datasets").mkdir(parents=True)
    db = fa / "local.db"
    init_local_db(db).close()
    (fa / "datasets" / "d.jsonl").write_text(json.dumps(ROWS[0]) + "\n")
    app = build_app(
        db_path=str(db), auth_path=tmp_path / "a.json", no_auth=True, runners=[Chain("only")]
    )
    with TestClient(app) as c:
        options = c.get("/api/datasets/d/run-eval/options").json()
    scorers = {s["name"]: s for s in options["scorers"]}
    assert scorers["tool_usage_accuracy"]["available"] is False
    assert scorers["latency"]["available"] is False


def test_refused_requests_never_call_the_agent(client: Any) -> None:
    c, _ = client
    before = c.get("/api/evals").json()["total"]
    c.post("/api/datasets/orders/run-eval", json={"agent_name": "support", "scorers": ["nope"]})
    assert c.get("/api/evals").json()["total"] == before


def test_unregistered_agent_stays_404(client: Any) -> None:
    c, _ = client
    r = c.post("/api/datasets/orders/run-eval", json={"agent_name": "ghost"})
    assert r.status_code == 404


def test_editing_a_case_keeps_its_other_fields(client: Any) -> None:
    c, path = client
    added = c.post("/api/datasets/orders/cases", json={"input": "new", "expected_output": "x"})
    assert added.status_code == 201, added.text
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows[0]["expected_trajectory"] == ["lookup_order"], "adding a case wiped other rows"
    assert rows[0]["context"] == "A1 shipped Monday."
    assert rows[0]["source_trace_id"] == "abc123"

    edited = c.put(
        "/api/datasets/orders/cases/0", json={"input": "Where is A1 now?", "expected_output": "y"}
    )
    assert edited.status_code == 200, edited.text
    assert edited.json()["extra"]["expected_trajectory"] == ["lookup_order"]
    row0 = json.loads(path.read_text().splitlines()[0])
    assert row0["input"] == "Where is A1 now?" and row0["expected_trajectory"] == ["lookup_order"]

    c.delete("/api/datasets/orders/cases/1")
    assert json.loads(path.read_text().splitlines()[0])["context"] == "A1 shipped Monday."

    replaced = c.put(
        "/api/datasets/orders/cases/0",
        json={"input": "q", "extra": {"expected_trajectory": ["other_tool"]}},
    )
    assert replaced.json()["extra"] == {"expected_trajectory": ["other_tool"]}
    refused = c.put("/api/datasets/orders/cases/0", json={"input": "q", "extra": {"input": "x"}})
    assert refused.status_code == 422
