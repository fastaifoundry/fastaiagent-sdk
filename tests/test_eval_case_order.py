"""Eval cases keep their dataset position, and comparisons pair the same case (1.87.0).

``evaluate()`` runs cases concurrently. It used to record each case as it
*finished*, so a case's stored ``ordinal`` was its finishing position — and every
comparison pairs cases by ordinal first. Two runs of one dataset that finished in
different orders were compared input-against-wrong-input: the AutoLLM flagship's
v1-vs-v2 compare paired 62 of 120 tickets wrongly, in the UI, the CLI and the
pytest baseline gate alike. The existing compare tests all ran with
``concurrency=1``, where finishing order is dataset order.

Swept across every comparison entry point: ``compare_runs``, the UI route, the
CLI and the pytest ``--eval-baseline`` gate. No mocks: real ``evaluate()`` runs
over real SQLite, real inner pytest sessions.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import pytest

from fastaiagent.eval import evaluate
from fastaiagent.eval.compare import compare_runs, load_run, match_cases

pytest_plugins = ["pytester"]

N = 8
DATASET = [{"input": f"case {i}", "expected_output": f"answer {i}"} for i in range(N)]
# v1 passes the even cases; v2 passes the odd ones and case 0. Paired right:
IMPROVED = {"case 1", "case 3", "case 5", "case 7"}
REGRESSED = {"case 2", "case 4", "case 6"}
BUCKETS = "regressed=3 improved=4 unchanged_pass=1 unchanged_fail=0"


def _index(text: str) -> int:
    return int(text.split()[-1])


def _v2_right(i: int) -> bool:
    return i % 2 == 1 or i == 0


async def v1(text: str) -> str:
    """Right on even cases only; the first case finishes last."""
    i = _index(text)
    await asyncio.sleep(0.02 * (N - i))
    return f"answer {i}" if i % 2 == 0 else "wrong"


async def v2(text: str) -> str:
    """Right on odd cases and case 0; the first case finishes first."""
    i = _index(text)
    await asyncio.sleep(0.02 * i)
    return f"answer {i}" if _v2_right(i) else "wrong"


def _persist(agent_fn: Any, run_name: str) -> str:
    results = evaluate(
        agent_fn, DATASET, ["exact_match"], concurrency=4, persist=True, run_name=run_name
    )
    assert results.run_id
    return results.run_id


@pytest.fixture(autouse=True)
def _isolate_config():
    """Inner pytest sessions run in-process; their env must not leak."""
    from fastaiagent._internal.config import reset_config

    yield
    reset_config()


def test_evaluate_stores_dataset_order_under_concurrency(isolated_local_db: Path) -> None:
    results = evaluate(v1, DATASET, ["exact_match"], concurrency=4, persist=True)

    assert [c.input for c in results.cases] == [d["input"] for d in DATASET]
    # scores line up with cases: the odd cases are the failing ones
    assert [r.passed for r in results.scores["exact_match"]] == [i % 2 == 0 for i in range(N)]
    stored = load_run(results.run_id or "")["cases"]
    assert [(c["ordinal"], c["input"]) for c in stored] == [(i, f"case {i}") for i in range(N)]


def _inputs(entries: list[dict[str, Any]]) -> set[str]:
    for e in entries:
        assert e["a"]["input"] == e["b"]["input"], "a comparison paired two different cases"
    return {e["a"]["input"] for e in entries}


def test_compare_runs_pairs_the_same_case(isolated_local_db: Path) -> None:
    _persist(v1, "v1")
    _persist(v2, "v2")
    cmp = compare_runs("v1", "v2")
    assert _inputs(cmp.improved) == IMPROVED
    assert _inputs(cmp.regressed) == REGRESSED
    assert (cmp.unchanged_pass, cmp.unchanged_fail) == (1, 0)


def test_ui_compare_route_pairs_the_same_case(isolated_local_db: Path) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from fastaiagent.ui.server import build_app

    a, b = _persist(v1, "v1"), _persist(v2, "v2")
    client = TestClient(build_app(db_path=str(isolated_local_db), no_auth=True))
    body = client.get(f"/api/evals/compare?a={a}&b={b}").json()

    assert _inputs(body["improved"]) == IMPROVED
    assert _inputs(body["regressed"]) == REGRESSED


def test_cli_compare_pairs_the_same_case(isolated_local_db: Path) -> None:
    from typer.testing import CliRunner

    from fastaiagent.cli.main import app

    _persist(v1, "v1")
    _persist(v2, "v2")
    result = CliRunner().invoke(
        app, ["eval", "compare", "v1", "v2", "--db", str(isolated_local_db)]
    )

    assert BUCKETS in result.output, result.output


_GATE_SUITE = """
import os
os.environ["FASTAIAGENT_LOCAL_DB"] = r"{db}"
from fastaiagent._internal.config import reset_config
reset_config()
import pytest
from fastaiagent.eval import case

CASES = [("case %d" % i, "answer %d" % i) for i in range({n})]

def v2(text):
    i = int(text.split()[-1])
    return "answer %d" % i if (i % 2 == 1 or i == 0) else "wrong"

@pytest.mark.parametrize("text,expected", CASES)
def test_ticket(text, expected, evaluate_one):
    evaluate_one(v2, input=text, expected=expected, scorers=["exact_match"], assert_pass=False)
"""


def test_pytest_baseline_gate_pairs_the_same_case(
    pytester: pytest.Pytester, isolated_local_db: Path
) -> None:
    """The flagship's shape: a concurrent evaluate() baseline, a sequential gate."""
    _persist(v1, "v1")
    pytester.makepyfile(_GATE_SUITE.format(db=isolated_local_db, n=N))

    result = pytester.runpytest("--no-header", "--eval-baseline", "v1")
    os.environ.pop("FASTAIAGENT_LOCAL_DB", None)

    result.stdout.fnmatch_lines([f"*{BUCKETS}*"])


def test_runs_stored_in_finishing_order_still_pair_by_input() -> None:
    """Runs persisted before 1.87.0 keep their scrambled ordinals; the input check
    pairs them correctly anyway."""

    def case(ordinal: int, i: int, passed: bool) -> dict[str, Any]:
        return {
            "ordinal": ordinal,
            "input": f"case {i}",
            "per_scorer": {"exact_match": {"passed": passed, "score": float(passed)}},
        }

    finished_backwards = [case(o, N - 1 - o, (N - 1 - o) % 2 == 0) for o in range(N)]
    in_order = [case(i, i, True) for i in range(N)]

    pairs = match_cases(finished_backwards, in_order)
    assert len(pairs) == N
    assert all(a["input"] == b["input"] for a, b in pairs)
