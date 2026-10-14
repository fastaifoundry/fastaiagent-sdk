"""Live: a ratchet that regresses records ``failed`` against the previous run (E1, E2).

Two real pytest sessions over a real gpt-4o-mini agent, both run with
``--eval-run-name main --eval-baseline main``. Session A answers correctly and
publishes ``main``. Session B is forced to answer wrongly by its system prompt.
Before 1.89.0, B compared with itself (delta 0), exited 0 and recorded ``passed``.
"""

from __future__ import annotations

import json
import os
import sqlite3
from typing import Any

import pytest

pytestmark = pytest.mark.e2e
pytest_plugins = ["pytester"]

_SUITE = """
import os
os.environ["FASTAIAGENT_LOCAL_DB"] = r"{db}"
from fastaiagent._internal.config import reset_config
reset_config()

from fastaiagent.agent import Agent
from fastaiagent.eval import case
from fastaiagent.llm import LLMClient

AGENT = Agent(
    name="colour-bot",
    system_prompt="{prompt}",
    llm=LLMClient(model="gpt-4o-mini", temperature=0),
)

@case(input="What colour is the sky on a clear day?", expected="BLUE")
def test_sky(evaluate_one):
    evaluate_one(AGENT.run, scorers=["contains"], assert_pass=False)

@case(input="Name the colour of the ocean in one word.", expected="BLUE")
def test_ocean(evaluate_one):
    evaluate_one(AGENT.run, scorers=["contains"], assert_pass=False)
"""

_GOOD = "Answer with the single word BLUE in capital letters and nothing else."
_BAD = "Answer with the single word RED in capital letters and nothing else."


@pytest.fixture(autouse=True)
def _isolate_config():
    from fastaiagent._internal.config import reset_config

    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")
    yield
    os.environ.pop("FASTAIAGENT_LOCAL_DB", None)
    reset_config()


def _latest(db: Any) -> tuple[str, dict[str, Any]]:
    with sqlite3.connect(db) as conn:
        run_id, meta = conn.execute(
            "SELECT run_id, metadata FROM eval_runs ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
    return run_id, json.loads(meta or "{}")


def test_a_regressed_ratchet_records_failed(pytester: pytest.Pytester, tmp_path: Any) -> None:
    from fastaiagent.eval.platform_export import build_payloads

    db = tmp_path / "local.db"
    args = (
        "--no-header",
        "-p",
        "no:randomly",
        "--eval-run-name",
        "main",
        "--eval-baseline",
        "main",
    )

    pytester.makepyfile(_SUITE.format(db=db, prompt=_GOOD))
    first = pytester.runpytest(*args)
    a_id, a_meta = _latest(db)
    assert first.ret == 0, first.stdout.str()
    assert a_meta["gate_outcome"] == "passed"

    pytester.makepyfile(_SUITE.format(db=db, prompt=_BAD))
    second = pytester.runpytest(*args)
    b_id, b_meta = _latest(db)

    assert second.ret == 1, second.stdout.str()
    assert b_meta["gate_outcome"] == "failed"
    assert b_meta["baseline"]["run_id"] == a_id, "compared against the wrong run"
    assert b_id != a_id
    assert build_payloads(run_id=b_id, db_path=db)[0]["gate_outcome"] == "failed"
