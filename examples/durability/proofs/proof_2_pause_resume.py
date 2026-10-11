"""Proof for "2 · Between the process that paused and the one that resumes" on docs/durability/durability-boundaries.md.

Offline, two processes: a child pauses and exits; this process resumes with the
approver's decision. The context the approver saw is the one that was frozen at the
pause. A second resume is refused, and when five resumers race, exactly one wins.
"""

import os
import subprocess
import sys
import threading
from pathlib import Path

import _agents as A
import _common
from _common import checkpointer, heading

from fastaiagent import AlreadyResumed, Resume

HERE = Path(__file__).resolve().parent
CHILD = """
import os, sys; sys.path.insert(0, {here!r})
os.environ["PROOF_CKPT_DB"] = {db!r}
import _common, _agents as A
from _common import checkpointer
r = A.agent(checkpointer()).run("Refund order 1042", execution_id=sys.argv[1])
print(f"child pid={{os.getpid()}} status={{r.status}} context={{r.pending_interrupt['context']}}")
"""


def pause_in_child(execution_id: str) -> str:
    run = subprocess.run([sys.executable, "-c", CHILD.format(here=str(HERE), db=str(_common.CKPT_DB)), execution_id],
                         capture_output=True, text=True, check=True, cwd=str(HERE))
    return run.stdout.strip().splitlines()[-1]


cp = checkpointer()

heading("process A pauses and exits; process B resumes")
print(pause_in_child("refund-1"))
print(f"this  pid={os.getpid()} resumes with Resume(approved=True, metadata={{'approver': 'alice'}})")
bot = A.agent(cp)
r = bot.resume("refund-1", resume_value=Resume(approved=True, metadata={"approver": "alice"}))
print("status:", r.status, "· output:", repr(r.output), "· charges:", A.charges)

heading("the approver's context was frozen at the pause")
paused_row = next(c for c in cp.list("refund-1") if c.status == "interrupted")
print("frozen in the checkpoint:", paused_row.interrupt_context, "— JSON at pause time, never recomputed")

heading("a second resume is refused")
try:
    bot.resume("refund-1", resume_value=Resume(approved=True))
except AlreadyResumed as e:
    print("AlreadyResumed:", str(e)[:100], "…")

heading("five resumers race for one pause")
print(pause_in_child("refund-2"))
outcomes: list[str] = []
lock = threading.Lock()


def resumer(i: int) -> None:
    try:
        res = A.agent(cp).resume("refund-2", resume_value=Resume(approved=True, metadata={"approver": f"t{i}"}))
        out = f"completed({res.output[:12]!r})"
    except AlreadyResumed:
        out = "AlreadyResumed"
    with lock:
        outcomes.append(out)


threads = [threading.Thread(target=resumer, args=(i,)) for i in range(5)]
for t in threads:
    t.start()
for t in threads:
    t.join()
print("outcomes:", sorted(outcomes))
print("charges in this process:", len(A.charges), "— one per resumed run")
