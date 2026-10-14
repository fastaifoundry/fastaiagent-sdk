"""One tracer provider per process, however many threads ask at once (1.89.0).

``get_tracer_provider()`` built the provider without a lock, so threads that
asked at the same moment each built their own (eight threads, eight providers).
Each one stored spans locally, but an exporter (the plane's included) attaches
to one provider only, so the other threads' spans were never exported, and OTel
logged "Overriding of current TracerProvider is not allowed". ``evaluate()``
with ``concurrency>1`` hit it on its first call.

Real threads, the real provider and OTel's in-memory exporter; no mocks.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pytest

from fastaiagent.trace import otel

N = 16


@pytest.fixture
def fresh(isolated_local_db: Path) -> Any:
    otel.reset()
    yield
    otel.reset()


def _race(fn: Any) -> list[Any]:
    barrier = threading.Barrier(N)
    out: list[Any] = [None] * N

    def run(i: int) -> None:
        barrier.wait()
        out[i] = fn()

    threads = [threading.Thread(target=run, args=(i,)) for i in range(N)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return out


@pytest.mark.parametrize("attempt", range(5))
def test_threads_asking_at_once_get_one_provider(fresh: Any, attempt: int) -> None:
    providers = _race(otel.get_tracer_provider)
    assert len({id(p) for p in providers}) == 1


def test_an_exporter_added_after_the_race_sees_every_threads_span(fresh: Any) -> None:
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    providers = _race(otel.get_tracer_provider)
    exporter = InMemorySpanExporter()
    assert otel.add_span_processor(SimpleSpanProcessor(exporter))
    for i, provider in enumerate(providers):
        provider.get_tracer("race").start_span(f"thread-{i}").end()
    names = {s.name for s in exporter.get_finished_spans()}
    assert names == {f"thread-{i}" for i in range(N)}, f"lost: {N - len(names)}"


_EVALUATE = """
import logging, sys
logging.basicConfig(level=logging.WARNING)
from fastaiagent import Agent
from fastaiagent.eval import evaluate
from fastaiagent.llm.client import LLMResponse
from fastaiagent.testing import FunctionModel

agent = Agent(name="race", llm=FunctionModel(lambda m: LLMResponse(content="hi")))
results = evaluate(agent.run, sys.argv[1], scorers=["contains"], concurrency=4)
print("passed", results.case_counts["case_pass_count"])
"""


def test_concurrent_evaluate_does_not_override_the_provider(tmp_path: Path) -> None:
    """A fresh process: OTel's global slot is first-wins, once per process."""
    import os
    import subprocess
    import sys

    data = tmp_path / "d.jsonl"
    data.write_text("".join(f'{{"input": "q{i}", "expected_output": "hi"}}\n' for i in range(8)))
    env = {**os.environ, "FASTAIAGENT_LOCAL_DB": str(tmp_path / "local.db")}
    env.pop("FASTAIAGENT_API_KEY", None)
    done = subprocess.run(
        [sys.executable, "-c", _EVALUATE, str(data)],
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
        timeout=120,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    assert "passed 8" in done.stdout
    assert "Overriding of current TracerProvider" not in done.stderr
