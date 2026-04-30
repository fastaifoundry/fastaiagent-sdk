"""Unit tests for the ``@idempotent`` decorator (v1.0 Phase 3).

Spec test #4 — single-execution: same execution_id + args = body executes once.
Spec test #5 — cross-execution: different execution_id = body executes twice
                (cache is execution-scoped).

Plus: outside-chain pass-through, key_fn override, non-serializable result
(IdempotencyError), and a Pydantic-model return (round-trips through
``pydantic_core.to_jsonable_python``).
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from fastaiagent import IdempotencyError, SQLiteCheckpointer, idempotent
from fastaiagent.chain.idempotent import _current_checkpointer
from fastaiagent.chain.interrupt import _execution_id


@pytest.fixture
def store(temp_dir) -> SQLiteCheckpointer:
    cp = SQLiteCheckpointer(db_path=str(temp_dir / "cp.db"))
    cp.setup()
    yield cp
    cp.close()


def _bind(execution_id: str, store: SQLiteCheckpointer):
    """Set up the ContextVars the executor would normally set."""
    return _execution_id.set(execution_id), _current_checkpointer.set(store)


def _unbind(tokens) -> None:
    e_tok, cp_tok = tokens
    _current_checkpointer.reset(cp_tok)
    _execution_id.reset(e_tok)


class TestSingleExecution:
    """Spec test #4 — body executes exactly once per (execution_id, key)."""

    def test_repeated_calls_in_same_execution_run_body_once(self, store):
        calls = []

        @idempotent
        def charge(amount, customer_id):
            calls.append((amount, customer_id))
            return {"charge_id": f"ch_{len(calls)}", "amount": amount}

        tokens = _bind("exec-A", store)
        try:
            r1 = charge(100, "cust-1")
            r2 = charge(100, "cust-1")
            r3 = charge(100, "cust-1")
        finally:
            _unbind(tokens)

        assert len(calls) == 1, f"body ran {len(calls)} times, expected 1"
        # All three calls return the same cached value.
        assert r1 == r2 == r3
        assert r1["charge_id"] == "ch_1"

    def test_different_args_have_different_keys(self, store):
        calls = []

        @idempotent
        def fn(x):
            calls.append(x)
            return {"x": x}

        tokens = _bind("exec-X", store)
        try:
            assert fn(1) == {"x": 1}
            assert fn(2) == {"x": 2}
            assert fn(1) == {"x": 1}  # cache hit on the (x=1) call
        finally:
            _unbind(tokens)

        assert calls == [1, 2]

    def test_kwargs_are_part_of_the_key(self, store):
        calls = []

        @idempotent
        def fn(**kw):
            calls.append(kw)
            return dict(kw)

        tokens = _bind("exec-K", store)
        try:
            fn(a=1, b=2)
            fn(b=2, a=1)  # same kwargs, different order — same key
            fn(a=1, b=3)  # different value, new key
        finally:
            _unbind(tokens)

        assert len(calls) == 2  # only two unique kwarg shapes


class TestCrossExecution:
    """Spec test #5 — cache is scoped per execution_id."""

    def test_different_execution_ids_each_run_body_once(self, store):
        calls = []

        @idempotent
        def fn(x):
            calls.append(x)
            return {"x": x}

        tokens = _bind("exec-1", store)
        try:
            fn(7)
            fn(7)  # cache hit
        finally:
            _unbind(tokens)

        tokens = _bind("exec-2", store)
        try:
            fn(7)  # cache miss — different execution
            fn(7)  # cache hit within exec-2
        finally:
            _unbind(tokens)

        assert calls == [7, 7], f"body should run once per execution_id, ran for: {calls}"


class TestOutsideChainRun:
    """No execution_id in scope → pass-through, no caching."""

    def test_calls_run_every_time_outside_a_chain(self, store):
        calls = []

        @idempotent
        def fn(x):
            calls.append(x)
            return x

        # No bindings set — both calls run the body.
        assert fn(1) == 1
        assert fn(1) == 1
        assert calls == [1, 1]

    def test_calls_run_every_time_when_no_checkpointer_is_set(self):
        calls = []

        @idempotent
        def fn(x):
            calls.append(x)
            return x

        # execution_id is set but no checkpointer — pass-through.
        tok = _execution_id.set("exec-no-cp")
        try:
            fn(1)
            fn(1)
        finally:
            _execution_id.reset(tok)

        assert calls == [1, 1]


class TestKeyFn:
    """Custom key function lets non-JSON args participate in caching."""

    def test_key_fn_overrides_default_hash(self, store):
        calls = []

        class User:
            def __init__(self, uid: str):
                self.id = uid

        @idempotent(key_fn=lambda user, req_id: f"{user.id}:{req_id}")
        def process(user, req_id):
            calls.append((user.id, req_id))
            return {"ok": True, "user": user.id, "req": req_id}

        tokens = _bind("exec-Q", store)
        try:
            u = User("u-1")
            process(u, "r-1")
            process(u, "r-1")  # cache hit via custom key
            process(u, "r-2")  # different key — runs body
        finally:
            _unbind(tokens)

        assert calls == [("u-1", "r-1"), ("u-1", "r-2")]


class TestNonSerializableResult:
    def test_returning_non_jsonable_value_raises_idempotency_error(self, store):
        @idempotent
        def fn():
            # A bare object() is not JSON-serializable and not a Pydantic /
            # dataclass instance — to_jsonable_python rejects it.
            return object()

        tokens = _bind("exec-bad", store)
        try:
            with pytest.raises(IdempotencyError, match="non-JSON-serializable"):
                fn()
        finally:
            _unbind(tokens)


class TestPydanticReturn:
    def test_pydantic_model_round_trips_as_dict(self, store):
        class Result(BaseModel):
            charge_id: str
            amount: int

        calls = []

        @idempotent
        def charge(amount):
            calls.append(amount)
            return Result(charge_id="ch_42", amount=amount)

        tokens = _bind("exec-P", store)
        try:
            first = charge(500)
            second = charge(500)
        finally:
            _unbind(tokens)

        assert len(calls) == 1
        # First call: returns the original Pydantic model.
        assert isinstance(first, Result)
        assert first.charge_id == "ch_42"
        # Second call: cached — comes back as the JSON-deserialized dict.
        # The decorator documents this in its docstring.
        assert second == {"charge_id": "ch_42", "amount": 500}
