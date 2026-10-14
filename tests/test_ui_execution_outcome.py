"""The execution page can say what happened to a pause (1.89.0).

Until 1.89.0 ``GET /api/executions/{id}`` returned the checkpoint rows and
nothing else, so the UI showed every ``interrupted`` row as "waiting" forever —
including after someone approved it — and had no outcome to show: no decision,
no resolver, no reason, no answer, no link to the resumed run (which had no
trace of its own anyway). ``POST …/resume`` dropped the top-level ``reason`` the
body schema accepts, and never recorded who clicked.

Now the payload carries ``pending_interrupt``, a per-checkpoint
``interrupt_state``, the ``runs`` (one per trace: the run, each resume, a fork),
a ``resolution`` and the ``final_output``; the resume route folds ``reason`` into
``Resume.metadata`` and names the signed-in user as the resolver (decision D8).

Real Agent / Chain, real ``SQLiteCheckpointer`` on the same ``local.db`` the
tracer writes to, real FastAPI app through ``TestClient``.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("bcrypt")

from fastapi.testclient import TestClient  # noqa: E402

from fastaiagent import Agent, FunctionTool, SQLiteCheckpointer, interrupt  # noqa: E402
from fastaiagent.llm.client import LLMResponse  # noqa: E402
from fastaiagent.llm.message import ToolCall  # noqa: E402
from fastaiagent.testing import FunctionModel  # noqa: E402
from fastaiagent.ui.auth import create_auth_file  # noqa: E402
from fastaiagent.ui.server import build_app  # noqa: E402

TRACE_ID_RE = re.compile(r"^[0-9a-f]{32}$")


class _CSRFAwareTestClient(TestClient):
    """Sends ``X-CSRF-Token`` from the cookie jar, as the bundled UI does."""

    def request(self, method, url, *args, **kwargs):  # type: ignore[override]
        if method.upper() in {"POST", "PUT", "PATCH", "DELETE"}:
            csrf = self.cookies.get("fastaiagent_csrf")
            if csrf:
                headers = kwargs.get("headers")
                if headers is None:
                    headers = kwargs["headers"] = {}
                if isinstance(headers, dict):
                    headers.setdefault("X-CSRF-Token", csrf)
        return super().request(method, url, *args, **kwargs)


def _env(monkeypatch, tmp_path: Path, *, traced: bool) -> Iterator[Path]:
    from fastaiagent._internal import instance as _instance
    from fastaiagent._internal.config import reset_config
    from fastaiagent.trace import otel

    db = tmp_path / "local.db"
    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(db))
    if not traced:
        monkeypatch.setenv("FASTAIAGENT_TRACE_ENABLED", "0")
    reset_config()
    _instance.reset_for_testing()
    otel.reset()
    try:
        yield db
    finally:
        otel.reset()
        _instance.reset_for_testing()
        reset_config()


@pytest.fixture
def local_db(monkeypatch, tmp_path) -> Iterator[Path]:
    yield from _env(monkeypatch, tmp_path, traced=True)


@pytest.fixture
def untraced_db(monkeypatch, tmp_path) -> Iterator[Path]:
    yield from _env(monkeypatch, tmp_path, traced=False)


def _agent(db: Path, orders: tuple[str, ...] = ("1042",)) -> Agent:
    def refund(order: str) -> str:
        """Refund an order; a manager must approve."""
        decision = interrupt(reason="manager_approval", context={"order": order})
        return f"refunded {order}" if decision.approved else f"declined {order}"

    def responder(messages: list[Any]) -> LLMResponse:
        done = [m.content for m in messages if m.role.value == "tool"]
        if len(done) < len(orders):
            return LLMResponse(
                content=None,
                tool_calls=[
                    ToolCall(
                        id=f"c{len(done)}", name="refund", arguments={"order": orders[len(done)]}
                    )
                ],
                finish_reason="tool_calls",
            )
        return LLMResponse(content=f"Done: {done[-1]}", finish_reason="stop")

    return Agent(
        name="refund-bot",
        llm=FunctionModel(responder),
        tools=[FunctionTool(name="refund", fn=refund)],
        checkpointer=SQLiteCheckpointer(str(db)),
    )


def _paused(db: Path, exec_id: str, **kw: Any) -> Agent:
    agent = _agent(db, **kw)
    assert agent.run("Refund it.", execution_id=exec_id).status == "paused"
    return agent


def _interrupted(body: dict[str, Any]) -> list[dict[str, Any]]:
    return [cp for cp in body["checkpoints"] if cp["status"] == "interrupted"]


# ---------------------------------------------------------------------------


def test_a_paused_execution_shows_one_pending_pause(local_db: Path) -> None:
    agent = _paused(local_db, "ui-1")
    client = TestClient(build_app(db_path=str(local_db), no_auth=True, runners=[agent]))

    body = client.get("/api/executions/ui-1").json()

    assert body["pending_interrupt"]["node_id"] == "turn:0/tool:refund"
    assert body["pending_interrupt"]["context"] == {"order": "1042"}
    assert [cp["interrupt_state"] for cp in _interrupted(body)] == ["pending"]
    assert all(
        cp["interrupt_state"] is None for cp in body["checkpoints"] if cp["status"] != "interrupted"
    )
    assert [(r["role"], r["status"]) for r in body["runs"]] == [("run", "paused")]
    assert body["resolution"] is None
    assert body["final_output"] is None


@pytest.mark.parametrize("approved", [True, False], ids=["approved", "rejected"])
def test_after_a_resume_the_page_shows_the_outcome(local_db: Path, approved: bool) -> None:
    agent = _paused(local_db, "ui-2")
    client = TestClient(build_app(db_path=str(local_db), no_auth=True, runners=[agent]))

    r = client.post(
        "/api/executions/ui-2/resume",
        json={
            "approved": approved,
            "metadata": {"resolver": "alice"},
            "reason": "checked the order",
        },
    )
    assert r.status_code == 200, r.text
    posted = r.json()
    assert posted["reason"] == "checked the order"
    assert TRACE_ID_RE.match(posted["trace_id"] or ""), posted
    assert posted["trace_id"] == posted["result"]["trace_id"]

    body = client.get("/api/executions/ui-2").json()
    assert body["pending_interrupt"] is None
    assert [cp["interrupt_state"] for cp in _interrupted(body)] == ["resolved"]
    assert [r["role"] for r in body["runs"]] == ["run", "resume"]
    resume = body["runs"][-1]
    assert resume["trace_id"] == posted["trace_id"]
    assert resume["status"] == "completed"
    assert resume["resume"]["approved"] is approved
    assert resume["resume"]["resolver"] == "alice"
    assert resume["resume"]["reason"] == "checked the order"
    assert resume["resume"]["node_id"] == "turn:0/tool:refund"
    res = body["resolution"]
    assert (res["approved"], res["resolver"], res["reason"]) == (
        approved,
        "alice",
        "checked the order",
    )
    assert res["trace_id"] == posted["trace_id"] and res["source"] == "trace"
    expected = "Done: refunded 1042" if approved else "Done: declined 1042"
    assert body["final_output"] == expected == posted["result"]["output"]


def test_the_signed_in_user_is_the_resolver(local_db: Path, tmp_path: Path) -> None:
    agent = _paused(local_db, "ui-3")
    auth = tmp_path / "auth.json"
    create_auth_file("upendra", "correct-horse", path=auth)
    client = _CSRFAwareTestClient(
        build_app(db_path=str(local_db), auth_path=auth, no_auth=False, runners=[agent])
    )
    login = client.post(
        "/api/auth/login", json={"username": "upendra", "password": "correct-horse"}
    )
    assert login.status_code == 200

    # The bundled UI sends the reason inside metadata and no resolver at all.
    r = client.post(
        "/api/executions/ui-3/resume",
        json={"approved": True, "metadata": {"reason": "ok"}},
    )
    assert r.status_code == 200, r.text
    assert r.json()["resolver"] == "upendra"

    res = client.get("/api/executions/ui-3").json()["resolution"]
    assert (res["approved"], res["resolver"], res["reason"]) == (True, "upendra", "ok")


def test_an_explicit_resolver_is_kept_and_no_auth_names_nobody(
    local_db: Path, tmp_path: Path
) -> None:
    agent = _paused(local_db, "ui-4")
    client = TestClient(build_app(db_path=str(local_db), no_auth=True, runners=[agent]))
    r = client.post("/api/executions/ui-4/resume", json={"approved": True, "metadata": {}})
    assert r.status_code == 200
    assert r.json()["resolver"] is None  # --no-auth: there is no signed-in user to name
    assert client.get("/api/executions/ui-4").json()["resolution"]["resolver"] is None

    agent5 = _paused(local_db, "ui-5")
    auth = tmp_path / "auth.json"
    create_auth_file("upendra", "correct-horse", path=auth)
    authed = _CSRFAwareTestClient(
        build_app(db_path=str(local_db), auth_path=auth, no_auth=False, runners=[agent5])
    )
    authed.post("/api/auth/login", json={"username": "upendra", "password": "correct-horse"})
    r = authed.post(
        "/api/executions/ui-5/resume",
        json={"approved": True, "metadata": {"resolver": "ops-bot"}},
    )
    assert r.json()["resolver"] == "ops-bot"


def test_a_second_pause_is_pending_and_the_first_resolved(local_db: Path) -> None:
    agent = _paused(local_db, "ui-6", orders=("A", "B"))
    client = TestClient(build_app(db_path=str(local_db), no_auth=True, runners=[agent]))

    r = client.post("/api/executions/ui-6/resume", json={"approved": True, "metadata": {}})
    assert r.json()["result"]["status"] == "paused"

    body = client.get("/api/executions/ui-6").json()
    assert body["pending_interrupt"]["context"] == {"order": "B"}
    assert [cp["interrupt_state"] for cp in _interrupted(body)] == ["resolved", "pending"]
    assert [(r["role"], r["status"]) for r in body["runs"]] == [
        ("run", "paused"),
        ("resume", "paused"),
    ]
    assert body["final_output"] is None


def test_a_chain_resume_reports_its_resolution(local_db: Path) -> None:
    from fastaiagent import Chain

    def gate(amount: str) -> dict[str, Any]:
        decision = interrupt(reason="manager_approval", context={"amount": amount})
        return {"approved": decision.approved}

    chain = Chain(
        "ui-chain", checkpoint_enabled=True, checkpointer=SQLiteCheckpointer(str(local_db))
    )
    chain.add_node(
        "gate",
        tool=FunctionTool(name="gate_tool", fn=gate),
        input_mapping={"amount": "{{state.amount}}"},
    )
    assert chain.execute({"amount": 5000}, execution_id="ui-ch").status == "paused"
    client = TestClient(build_app(db_path=str(local_db), no_auth=True, runners=[chain]))

    r = client.post(
        "/api/executions/ui-ch/resume",
        json={"approved": False, "metadata": {"approver": "dana"}, "reason": "too much"},
    )
    assert r.status_code == 200, r.text
    body = client.get("/api/executions/ui-ch").json()
    assert [r["role"] for r in body["runs"]] == ["run", "resume"]
    res = body["resolution"]
    assert (res["approved"], res["resolver"], res["reason"], res["node_id"]) == (
        False,
        "dana",
        "too much",
        "gate",
    )
    assert [cp["interrupt_state"] for cp in _interrupted(body)] == ["resolved"]


def test_with_tracing_off_the_page_still_resolves_the_pause(untraced_db: Path) -> None:
    agent = _paused(untraced_db, "ui-7")
    client = TestClient(build_app(db_path=str(untraced_db), no_auth=True, runners=[agent]))
    r = client.post("/api/executions/ui-7/resume", json={"approved": True, "metadata": {}})
    assert r.status_code == 200
    assert r.json()["trace_id"] is None

    body = client.get("/api/executions/ui-7").json()
    assert body["runs"] == []
    assert body["pending_interrupt"] is None
    assert [cp["interrupt_state"] for cp in _interrupted(body)] == ["resolved"]
    # Nothing local records the decision when tracing is off and no plane is
    # attached; null is the honest answer.
    assert body["resolution"] is None
