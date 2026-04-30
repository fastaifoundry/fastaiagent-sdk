"""End-to-end quality gate — side-effect protection via ``@idempotent``.

Covers spec test #9 (the footgun) and #10 (the mitigation).

Scenario: a chain has an ``approval`` node that does two things in order —
(1) charge the customer's card, (2) call ``interrupt()`` for human review.
On the FIRST execution the chain charges the card and suspends. On
``chain.resume(...)`` the executor re-runs the same node from the top so
``interrupt()`` can return the resume value. Without protection, the charge
runs again — counter goes from 1 to 2 (the footgun). Wrapping the charge
function in ``@idempotent`` caches the first result, the second invocation
is a cache hit, and the counter stays at 1 (the mitigation).

Per the no-mocking memory rule, the "payment processor" is a real local
HTTP server (stdlib ``http.server`` on an ephemeral port) — no mocks.
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import httpx
import pytest

from tests.e2e.conftest import require_env

pytestmark = pytest.mark.e2e


# ---------- Real local HTTP counter server (no mocks) -------------------


class _CounterServer:
    """A tiny stdlib HTTP server that increments a counter on POST /charge.

    Runs on an ephemeral port in a background thread. ``count`` and ``url``
    are read from the test side; ``shutdown()`` stops it cleanly.
    """

    def __init__(self) -> None:
        self.count = 0
        self._lock = threading.Lock()

        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                if self.path != "/charge":
                    self.send_response(404)
                    self.end_headers()
                    return
                with outer._lock:
                    outer.count += 1
                    n = outer.count
                body = f'{{"count": {n}, "charge_id": "ch_{n}"}}'.encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_args: Any, **_kwargs: Any) -> None:
                # Silence per-request console logs during tests.
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="counter-server",
            daemon=True,
        )
        self._thread.start()

    @property
    def url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def shutdown(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


@pytest.fixture
def counter_server() -> Iterator[_CounterServer]:
    srv = _CounterServer()
    try:
        yield srv
    finally:
        srv.shutdown()


# ---------- Chain helper used by both footgun + mitigation tests --------


def _build_chain(checkpoint_db_path: str, charge_url: str, *, idempotent_charge: bool):
    """Build the same chain twice — once with @idempotent, once without.

    The chain is: ``approval`` (charges card, then interrupts).
    ``finalize`` runs after resume.
    """
    from fastaiagent import Chain, FunctionTool, SQLiteCheckpointer, idempotent
    from fastaiagent.chain.node import NodeType

    def _raw_charge(amount: int) -> dict[str, Any]:
        # Real HTTP call to the local counter server.
        r = httpx.post(f"{charge_url}/charge", json={"amount": amount}, timeout=5.0)
        r.raise_for_status()
        return r.json()

    if idempotent_charge:
        charge_fn = idempotent(_raw_charge)
    else:
        charge_fn = _raw_charge

    def _approval_fn(amount: str) -> dict[str, Any]:
        from fastaiagent import interrupt

        # Side effect FIRST — the classic footgun. On resume the executor
        # re-runs this whole node, so the charge runs again unless cached.
        receipt = charge_fn(int(amount))
        decision = interrupt(
            reason="manager_approval",
            context={"amount": int(amount), "receipt": receipt},
        )
        return {
            "charge_id": receipt["charge_id"],
            "approved": decision.approved,
            "step_approval_done": True,
        }

    def _finalize_fn(approved: str, charge_id: str) -> dict[str, Any]:
        is_approved = str(approved).lower() in ("true", "1", "yes")
        return {
            "final_approved": is_approved,
            "final_charge": charge_id,
            "step_finalize_done": True,
        }

    store = SQLiteCheckpointer(db_path=checkpoint_db_path)
    chain = Chain(
        "side-effects-gate",
        checkpoint_enabled=True,
        checkpointer=store,
    )
    chain.add_node(
        "approval",
        tool=FunctionTool(name="approval_tool", fn=_approval_fn),
        type=NodeType.tool,
        input_mapping={"amount": "{{state.amount}}"},
    )
    chain.add_node(
        "finalize",
        tool=FunctionTool(name="finalize_tool", fn=_finalize_fn),
        type=NodeType.tool,
        input_mapping={
            "approved": "{{state.output.approved}}",
            "charge_id": "{{state.output.charge_id}}",
        },
    )
    chain.connect("approval", "finalize")
    return chain, store


# ---------- The footgun and its mitigation ------------------------------


class TestSideEffectsGate:
    def test_09_footgun_double_charges_without_idempotent(
        self, tmp_path: Path, counter_server: _CounterServer
    ) -> None:
        """Spec test #9 — the footgun.

        Without protection, suspend-then-resume re-runs the side effect. We
        sanity-check that with a direct HTTP call: counter increments twice.
        """
        require_env()

        ckpt_db = str(tmp_path / "ckpt-footgun.db")
        chain, _ = _build_chain(ckpt_db, counter_server.url, idempotent_charge=False)
        execution_id = f"footgun-{uuid.uuid4().hex[:8]}"

        # First execution: charge happens (counter → 1), then interrupts.
        paused = chain.execute(
            {"amount": 50_000},
            execution_id=execution_id,
        )
        assert paused.status == "paused"
        assert counter_server.count == 1, (
            f"first run should have charged exactly once, counter={counter_server.count}"
        )

        # Resume re-runs the suspended approval node from the top — that
        # means another POST /charge. Without @idempotent, counter → 2.
        from fastaiagent import Resume
        from fastaiagent._internal.async_utils import run_sync

        result = run_sync(
            chain.resume(
                execution_id,
                resume_value=Resume(approved=True, metadata={"approver": "alice"}),
            )
        )
        assert result.status == "completed"
        assert counter_server.count == 2, (
            f"footgun: without @idempotent, resume re-charged the card; "
            f"expected counter=2 got counter={counter_server.count}"
        )

    def test_10_idempotent_caches_charge_across_resume(
        self, tmp_path: Path, counter_server: _CounterServer
    ) -> None:
        """Spec test #10 — the mitigation.

        Wrapping the charge function with ``@idempotent`` caches the first
        result for the execution. On resume the cached receipt is returned
        without hitting the payment processor; counter stays at 1.
        """
        require_env()

        ckpt_db = str(tmp_path / "ckpt-mitigation.db")
        chain, _ = _build_chain(ckpt_db, counter_server.url, idempotent_charge=True)
        execution_id = f"idemp-{uuid.uuid4().hex[:8]}"

        paused = chain.execute(
            {"amount": 50_000},
            execution_id=execution_id,
        )
        assert paused.status == "paused"
        assert counter_server.count == 1

        from fastaiagent import Resume
        from fastaiagent._internal.async_utils import run_sync

        result = run_sync(
            chain.resume(
                execution_id,
                resume_value=Resume(approved=True, metadata={"approver": "alice"}),
            )
        )
        assert result.status == "completed"
        assert counter_server.count == 1, (
            f"@idempotent should have served the cached receipt on resume; "
            f"expected counter=1 got counter={counter_server.count}"
        )

        # And the chain finalized using the cached charge_id.
        last_output = result.final_state.get("output")
        assert isinstance(last_output, dict)
        assert last_output.get("step_finalize_done") is True
        assert last_output.get("final_charge") == "ch_1"
        assert last_output.get("final_approved") is True
