"""Proof for "3 · Between a crash and a rerun" on docs/durability/durability-boundaries.md.

Offline, two processes: a child is killed with os._exit inside a tool, then inside the
model call; this process resumes each run. A tool-boundary crash re-invokes the tool
with the saved arguments and no model call; a turn-boundary crash re-issues the model.
When two tools were asked for in one turn and the first paused, the second is not run
on resume: the model is told so.
"""

import os
import subprocess
import sys
from pathlib import Path

import _agents as A
import _common
from _common import checkpointer, heading, rows

from fastaiagent import Resume

HERE = Path(__file__).resolve().parent
CHILD = """
import os, sys; sys.path.insert(0, {here!r})
os.environ["PROOF_CKPT_DB"] = {db!r}
import _common, _agents as A
from _common import checkpointer
A.agent(checkpointer(), amount=20, two_tools={two}).run("Refund order 1042", execution_id=sys.argv[1])
print("child finished normally")
"""


def crash_child_amount(execution_id: str) -> int:
    """A large refund whose tool is killed before it reaches interrupt()."""
    env = {**os.environ, "PROOF_CRASH": "tool"}
    child = CHILD.format(here=str(HERE), db=str(_common.CKPT_DB), two=False).replace("amount=20", "amount=50000")
    return subprocess.run([sys.executable, "-c", child, execution_id], capture_output=True, text=True,
                          cwd=str(HERE), env=env).returncode


def crash_child(execution_id: str, where: str, *, two_tools: bool = False) -> int:
    env = {**os.environ, "PROOF_CRASH": where}
    run = subprocess.run([sys.executable, "-c", CHILD.format(here=str(HERE), db=str(_common.CKPT_DB), two=two_tools),
                          execution_id], capture_output=True, text=True, cwd=str(HERE), env=env)
    return run.returncode


cp = checkpointer()
os.environ.pop("PROOF_CRASH", None)

heading("killed inside the tool (exit code 2)")
print("child exit code:", crash_child("crash-tool", "tool"))
for line in rows(cp, "crash-tool"):
    print(" ", line)
A.model_calls.clear(); A.charges.clear()
r = A.agent(cp, amount=20).resume("crash-tool")
print(f"resume → status={r.status} model calls={len(A.model_calls)} tool ran with saved args → charges={A.charges}")
for line in rows(cp, "crash-tool")[2:]:
    print(" ", line)

heading("killed waiting on the model (exit code 3)")
print("child exit code:", crash_child("crash-model", "model"))
for line in rows(cp, "crash-model"):
    print(" ", line)
A.model_calls.clear(); A.charges.clear()
r = A.agent(cp, amount=20).resume("crash-model")
print(f"resume → status={r.status} model calls={len(A.model_calls)} charges={A.charges} (the tool was not re-run)")
for line in rows(cp, "crash-model")[3:]:
    print(" ", line)

heading("two tools in one turn; the first pauses")
A.model_calls.clear(); A.charges.clear(); A.notified.clear()
r = A.agent(cp, two_tools=True).run("Refund order 1042 and tell the customer", execution_id="siblings")
print("status:", r.status, "· notify ran during the first pass:", A.notified)
r = A.agent(cp, two_tools=True).resume("siblings", resume_value=Resume(approved=True))
print("resume → status:", r.status, "· notify ran on resume:", A.notified)
snapshot = [c for c in cp.list("siblings") if c.node_id == "turn:1" and c.step_type == "llm_call"]
tool_msgs = [m for m in snapshot[-1].state_snapshot["messages"] if m.get("role") == "tool"] if snapshot else []
print("tool results the model was shown at the next turn:", [str(m.get("content"))[:64] for m in tool_msgs])

heading("the edge: a tool re-invoked after a crash pauses")
print("child exit code:", crash_child("crash-then-pause", "tool", two_tools=False) if False else crash_child_amount("crash-then-pause"))
try:
    r = A.agent(cp).resume("crash-then-pause")
    print("resume → status:", r.status)
except Exception as e:
    print(f"resume → {type(e).__name__} escaped resume(): {e} — not a paused result; nothing checkpointed (1.87.0)")
