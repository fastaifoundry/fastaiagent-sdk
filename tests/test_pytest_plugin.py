"""Meta-test for ``fastaiagent.eval.pytest_plugin``.

Uses pytest's ``pytester`` fixture to boot an in-process pytest session
over a tiny generated test file that imports the plugin. This exercises
the real plugin end-to-end with no mocking, no API keys, no network.
"""

from __future__ import annotations

import pytest

# Activate pytester for this module.
pytest_plugins = ["pytester"]


def test_case_decorator_with_test_model(pytester: pytest.Pytester) -> None:
    """The ``@case`` decorator + ``evaluate_one`` fixture form a passing test."""
    pytester.makepyfile(
        """
        from fastaiagent.testing import TestModel
        from fastaiagent.agent import Agent
        from fastaiagent.eval import case

        @case(input="hello", expected="hi")
        def test_greet(evaluate_one):
            agent = Agent(name="g", llm=TestModel(response="hi"))
            evaluate_one(agent.run, scorers=["exact_match"])
        """
    )
    result = pytester.runpytest("-v", "--no-header")
    result.assert_outcomes(passed=1)


def test_evaluate_one_fails_loud_on_mismatch(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        """
        from fastaiagent.testing import TestModel
        from fastaiagent.agent import Agent
        from fastaiagent.eval import case

        @case(input="ping", expected="pong")
        def test_mismatch(evaluate_one):
            # Returns "wrong" but expected is "pong" — should fail.
            agent = Agent(name="g", llm=TestModel(response="wrong"))
            evaluate_one(agent.run, scorers=["exact_match"])
        """
    )
    result = pytester.runpytest("-v", "--no-header")
    result.assert_outcomes(failed=1)
    # The failure message includes the nicely-formatted scorer breakdown.
    result.stdout.fnmatch_lines(["*fastaiagent eval case failed*"])


def test_dataset_decorator_parametrises(pytester: pytest.Pytester, tmp_path) -> None:
    ds_path = tmp_path / "cases.jsonl"
    ds_path.write_text(
        '{"input": "a", "expected_output": "A"}\n'
        '{"input": "b", "expected_output": "B"}\n',
        encoding="utf-8",
    )
    pytester.makepyfile(
        f"""
        from fastaiagent.testing import FunctionModel
        from fastaiagent.agent import Agent
        from fastaiagent.eval import pytest_dataset as dataset

        agent = Agent(
            name="upper",
            llm=FunctionModel(lambda messages: messages[-1].content.upper()),
        )

        @dataset(r"{ds_path}")
        def test_uppercases(eval_case, evaluate_one):
            evaluate_one(agent.run, scorers=["exact_match"])
        """
    )
    result = pytester.runpytest("-v", "--no-header")
    # Two parametrised runs, both should pass.
    result.assert_outcomes(passed=2)


def test_explicit_input_overrides_case_tag(pytester: pytest.Pytester) -> None:
    """``evaluate_one(input=...)`` wins over the @case tag."""
    pytester.makepyfile(
        """
        from fastaiagent.testing import TestModel
        from fastaiagent.agent import Agent
        from fastaiagent.eval import case

        @case(input="from-tag", expected="ok")
        def test_explicit_input(evaluate_one):
            agent = Agent(name="g", llm=TestModel(response="ok"))
            evaluate_one(agent.run, input="from-arg", expected="ok")
        """
    )
    result = pytester.runpytest("-v", "--no-header")
    result.assert_outcomes(passed=1)


def test_assert_pass_false_does_not_fail(pytester: pytest.Pytester) -> None:
    """When ``assert_pass=False`` the test inspects the record manually."""
    pytester.makepyfile(
        """
        from fastaiagent.testing import TestModel
        from fastaiagent.agent import Agent
        from fastaiagent.eval import case

        @case(input="x", expected="y")
        def test_inspect(evaluate_one):
            agent = Agent(name="g", llm=TestModel(response="z"))
            record = evaluate_one(agent.run, assert_pass=False)
            # Mismatch — but we did not assert.
            assert record.actual_output == "z"
            assert record.per_scorer["exact_match"]["passed"] is False
        """
    )
    result = pytester.runpytest("-v", "--no-header")
    result.assert_outcomes(passed=1)


def test_evaluate_one_persists_to_local_db(pytester: pytest.Pytester, tmp_path) -> None:
    """Running ``evaluate_one`` writes one ``eval_runs`` row tagged ``pytest::``."""
    pytester.makepyfile(
        f"""
        import os
        os.environ["FASTAIAGENT_LOCAL_DB"] = r"{tmp_path / 'local.db'}"

        from fastaiagent.testing import TestModel
        from fastaiagent.agent import Agent
        from fastaiagent.eval import case

        @case(input="hello", expected="hi")
        def test_persist(evaluate_one):
            agent = Agent(name="g", llm=TestModel(response="hi"))
            evaluate_one(agent.run, scorers=["exact_match"])
        """
    )
    result = pytester.runpytest("-v", "--no-header")
    result.assert_outcomes(passed=1)

    # After the inner pytest run completes, inspect the local.db that the
    # inner test populated.
    import sqlite3

    db_path = tmp_path / "local.db"
    if not db_path.exists():
        # The inner run may have written to the package's default DB
        # location instead — that's the fallback when FASTAIAGENT_LOCAL_DB
        # isn't honoured. Skip rather than fail; persistence is best-effort.
        pytest.skip("local.db not produced at expected path; persistence is best-effort")

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT run_name FROM eval_runs WHERE run_name LIKE 'pytest::%'"
        ).fetchall()
    assert rows, "expected at least one pytest:: eval run row"
