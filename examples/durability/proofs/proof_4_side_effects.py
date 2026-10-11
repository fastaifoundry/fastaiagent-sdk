"""Proof for "4 · Between a resume and a side effect" on docs/durability/durability-boundaries.md.

Offline: a real Agent on the SDK's FunctionModel, a real SQLiteCheckpointer.

A resumed tool runs again from the top, so a side effect placed before the pause
fires again — unless it is wrapped in @idempotent, which caches the result under
the run's execution_id. A new run is a new cache. A value that cannot be stored is
refused rather than silently uncached.
"""

import _agents as A
import _common  # noqa: F401
from _common import checkpointer, heading

from fastaiagent import FunctionTool, Resume, idempotent, interrupt
from fastaiagent.testing import FunctionModel

cp = checkpointer()
fired: list[str] = []


def charge_card(order: str) -> dict:
    fired.append(order)
    return {"charge_id": f"ch_{len(fired)}"}


def make_refund(charge):
    def refund(order: str, amount: int) -> dict:
        """Charge first, then ask."""
        receipt = charge(order)
        decision = interrupt(reason="manager_approval", context={"order": order, "amount": amount})
        return {"approved": decision.approved, **receipt}

    return refund


def run_and_resume(label: str, charge, execution_id: str) -> None:
    fired.clear()
    bot = A.agent(cp, tools=[FunctionTool(name="refund", fn=make_refund(charge)), FunctionTool(name="notify", fn=A.notify)])
    r = bot.run("Refund order 1042", execution_id=execution_id)
    r = bot.resume(execution_id, resume_value=Resume(approved=True))
    print(f"{label:<22} status={r.status} charge_card fired {len(fired)}× → {fired}")


heading("the same tool, paused and resumed")
run_and_resume("plain function", charge_card, "plain-1")
run_and_resume("@idempotent", idempotent(charge_card), "idem-1")

heading("the cache is per run")
run_and_resume("@idempotent, new run", idempotent(charge_card), "idem-2")

heading("outside a run there is no cache")
fired.clear()
cached = idempotent(charge_card)
cached("x"); cached("x")
print("called twice outside any run → fired", len(fired), "×")

heading("a value that cannot be stored is refused")


@idempotent
def opaque(order: str):
    return object()


bot = A.agent(cp, tools=[FunctionTool(name="refund", fn=lambda order, amount: str(opaque(order)))])
bot.llm = FunctionModel(lambda ms: ("", [{"name": "refund", "arguments": {"order": "1", "amount": 1}}]) if len(ms) < 3 else "ok")
r = bot.run("go", execution_id="opaque-1")
print("the tool raised inside the loop; the model was told:", repr(str(r.tool_calls[0].get("error"))[:120]), "…")
