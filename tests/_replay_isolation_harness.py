"""Shared harness for the replay-isolation tests (1.89.0, WS2).

One agent whose every tool leaves an observable trace of having run:

* ``charge_card`` (FunctionTool) appends ``CHARGE <amount>`` to a side file;
* ``issue_refund`` (RESTTool) POSTs to ``/refunds`` on a real local server;
* ``notify`` (MCPTool) calls ``/mcp`` on the same server.

Its model is the established chat stand-in (``tests/_decisions_stub.py``),
scripted to call all three tools and then answer, so live, deterministic and
recorded reruns all run offline. ``snapshot``/``new_rows`` diff every table of
``local.db`` plus the side file and the server's hits, which is how "a rerun
changed nothing but its own trace" is checked rather than assumed.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

AGENT_NAME = "refund-agent"

TOOL_CALL_TURN = {
    "role": "assistant",
    "content": None,
    "tool_calls": [
        {
            "id": "call_charge",
            "type": "function",
            "function": {"name": "charge_card", "arguments": json.dumps({"amount": 42})},
        },
        {
            "id": "call_refund",
            "type": "function",
            "function": {"name": "issue_refund", "arguments": json.dumps({"order": "7"})},
        },
        {
            "id": "call_notify",
            "type": "function",
            "function": {"name": "notify", "arguments": json.dumps({"text": "refunded"})},
        },
    ],
}
FINAL_TURN = {"role": "assistant", "content": "Refunded order 7."}

#: Tables a rerun may add rows to: its own spans (and the FTS index over them).
RERUN_TABLES = ("spans", "span_fts")


@dataclass
class World:
    db: Path
    side_file: Path
    server: Any
    stub: Any
    snapshots: list[dict[str, Any]] = field(default_factory=list)


def script_model(stub: Any) -> None:
    """Make the stand-in call every tool once, then answer. Resets its cursor."""
    stub.chat_replies = [TOOL_CALL_TURN, FINAL_TURN]
    stub.chat_calls = 0


def build_agent(world: World, *, name: str = AGENT_NAME) -> Any:
    import fastaiagent as fa
    from fastaiagent.llm import LLMClient
    from fastaiagent.tool.mcp import MCPTool
    from fastaiagent.tool.rest import RESTTool

    side = world.side_file

    def charge_card(amount: float) -> str:
        """Charge the customer's card."""
        with side.open("a") as f:
            f.write(f"CHARGE {amount}\n")
        return f"charged {amount}"

    tools = [
        fa.FunctionTool(name="charge_card", fn=charge_card),
        RESTTool(
            name="issue_refund",
            url=f"{world.server.url}/refunds",
            method="POST",
            description="Issue a refund.",
            parameters={"type": "object", "properties": {"order": {"type": "string"}}},
        ),
        MCPTool(
            name="notify",
            server_url=f"{world.server.url}/mcp",
            tool_name="notify",
            description="Notify the customer.",
            parameters={"type": "object", "properties": {"text": {"type": "string"}}},
        ),
    ]
    llm = LLMClient(**world.stub.llm_kwargs(model="gpt-4o-mini"))
    return fa.Agent(name=name, system_prompt="You process refunds.", llm=llm, tools=tools)


def side_lines(world: World) -> list[str]:
    return world.side_file.read_text().splitlines() if world.side_file.exists() else []


def hits(world: World, path_fragment: str) -> int:
    return sum(1 for r in world.server.requests if path_fragment in r["path"])


def snapshot(world: World) -> dict[str, Any]:
    """Every row of every table, plus the side file and the server's hits."""
    tables: dict[str, set[str]] = {}
    with sqlite3.connect(world.db) as conn:
        names = [
            r[0]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            if not r[0].startswith("sqlite_")
        ]
        for name in names:
            try:
                rows = conn.execute(f'SELECT * FROM "{name}"').fetchall()  # noqa: S608
            except sqlite3.DatabaseError:
                continue
            tables[name] = {repr(r) for r in rows}
    return {
        "tables": tables,
        "side": side_lines(world),
        "hits": [r["path"] for r in world.server.requests],
    }


def new_rows(before: dict[str, Any], after: dict[str, Any]) -> dict[str, int]:
    """``{table: rows added}`` for every table that gained rows."""
    out: dict[str, int] = {}
    for table, rows in after["tables"].items():
        added = len(rows - before["tables"].get(table, set()))
        if added:
            out[table] = added
    return out


def prepare_world(
    tmp_path: Path, monkeypatch: Any, capture_server: Any, decisions_stub: Any
) -> World:
    """A traced private store, the side effects wired up, and one original run.

    Returns the world with ``original_trace_id`` and ``agent`` set. The caller
    resets tracing afterwards (``otel.reset()``, ``reset_config()``).
    """
    from fastaiagent._internal import project as _project
    from fastaiagent._internal.config import reset_config
    from fastaiagent.trace import otel
    from fastaiagent.ui.db import init_local_db

    db = tmp_path / ".fastaiagent" / "local.db"
    db.parent.mkdir(exist_ok=True)
    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(db))
    monkeypatch.setenv("FASTAIAGENT_ALLOW_PRIVATE_NETWORKS", "1")
    for name in (
        "FASTAIAGENT_API_KEY",
        "FASTAIAGENT_TARGET",
        "FASTAIAGENT_EXPORT_EXPERIMENTS",
        "FASTAIAGENT_REPLAY_TOOL_POLICY",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    reset_config()
    _project.set_project_id("test-proj")
    otel.reset()
    init_local_db(db).close()

    w = World(
        db=db, side_file=tmp_path / "side_effects.log", server=capture_server, stub=decisions_stub
    )
    script_model(decisions_stub)
    agent = build_agent(w)
    result = agent.run("Refund order 7")
    lines = side_lines(w)
    assert len(lines) == 1 and lines[0].startswith("CHARGE 42"), "the original run's tools"
    assert hits(w, "/refunds") == 1 and hits(w, "/mcp") >= 1
    w.original_trace_id = result.trace_id  # type: ignore[attr-defined]
    w.agent = agent  # type: ignore[attr-defined]
    return w


def teardown_world() -> None:
    from fastaiagent._internal import project as _project
    from fastaiagent._internal.config import reset_config
    from fastaiagent.trace import otel

    otel.reset()
    _project.reset_for_testing()
    reset_config()


def spans_of(db: Path, trace_id: str) -> list[dict[str, Any]]:
    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT span_id, parent_span_id, name, attributes, synced FROM spans "
            "WHERE trace_id = ?",
            (trace_id,),
        ).fetchall()
    return [{**dict(r), "attributes": json.loads(r["attributes"] or "{}")} for r in rows]
