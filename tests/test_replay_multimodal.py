"""Phase 5 tests — Replay accepts multimodal ``modify_input``.

The structural test here verifies that ``ForkedReplay.modify_input`` does
not mangle ``Image``/``PDF``/list inputs on the way to ``Agent.arun``. Real
LLM-driven Replay forks are exercised by the e2e gates.
"""

from __future__ import annotations

from pathlib import Path

from fastaiagent import PDF, Image
from fastaiagent.trace.replay import ForkedReplay
from fastaiagent.trace.storage import SpanData, TraceData

FIXTURES = Path(__file__).parent / "fixtures" / "multimodal"


def _empty_fork() -> ForkedReplay:
    trace = TraceData(
        trace_id="t-mm",
        spans=[SpanData(span_id="s0", trace_id="t-mm", name="agent.test")],
    )
    return ForkedReplay(original_trace=trace, fork_point=0, steps=[])


def test_modify_input_accepts_string() -> None:
    fork = _empty_fork()
    fork.modify_input("plain string")
    assert fork._modifications["input"] == "plain string"


def test_modify_input_accepts_single_image() -> None:
    fork = _empty_fork()
    img = Image.from_file(FIXTURES / "cat.jpg")
    fork.modify_input(img)
    assert fork._modifications["input"] is img


def test_modify_input_accepts_single_pdf() -> None:
    fork = _empty_fork()
    pdf = PDF.from_file(FIXTURES / "contract.pdf")
    fork.modify_input(pdf)
    assert fork._modifications["input"] is pdf


def test_modify_input_accepts_mixed_list_preserves_order() -> None:
    fork = _empty_fork()
    img = Image.from_file(FIXTURES / "cat.jpg")
    pdf = PDF.from_file(FIXTURES / "contract.pdf")
    fork.modify_input(["intro", img, "between", pdf])
    stored = fork._modifications["input"]
    assert isinstance(stored, list)
    assert stored[0] == "intro"
    assert stored[1] is img
    assert stored[2] == "between"
    assert stored[3] is pdf


def test_modify_input_chainable_returns_self() -> None:
    fork = _empty_fork()
    img = Image.from_file(FIXTURES / "cat.jpg")
    out = fork.modify_input(["q", img]).modify_prompt("revised system")
    assert out is fork
    assert fork._modifications["input"][1] is img
    assert fork._modifications["prompt"] == "revised system"
