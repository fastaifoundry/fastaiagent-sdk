"""Example 04: Agent Replay — fork-and-rerun debugging.

Shows how to load a trace, step through it, fork at a failing step,
modify the prompt, and rerun.
"""

from fastaiagent.trace.replay import Replay
from fastaiagent.trace.storage import SpanData, TraceData

# Create a sample trace (in production, you'd load from storage)
sample_trace = TraceData(
    trace_id="demo_trace_001",
    name="agent.support-bot",
    start_time="2025-01-15T10:00:00Z",
    end_time="2025-01-15T10:00:05Z",
    spans=[
        SpanData(
            span_id="s1",
            trace_id="demo_trace_001",
            name="agent.run",
            start_time="2025-01-15T10:00:00Z",
            end_time="2025-01-15T10:00:05Z",
        ),
        SpanData(
            span_id="s2",
            trace_id="demo_trace_001",
            name="llm.completion",
            start_time="2025-01-15T10:00:00.5Z",
            end_time="2025-01-15T10:00:02Z",
            parent_span_id="s1",
        ),
        SpanData(
            span_id="s3",
            trace_id="demo_trace_001",
            name="tool.search",
            start_time="2025-01-15T10:00:02.1Z",
            end_time="2025-01-15T10:00:03Z",
            parent_span_id="s1",
        ),
        SpanData(
            span_id="s4",
            trace_id="demo_trace_001",
            name="llm.completion",
            start_time="2025-01-15T10:00:03.1Z",
            end_time="2025-01-15T10:00:04.5Z",
            parent_span_id="s1",
            attributes={"note": "hallucinated refund policy"},
        ),
    ],
)

if __name__ == "__main__":
    replay = Replay(sample_trace)

    # View summary
    print(replay.summary())
    print()

    # Step through
    for step in replay.step_through():
        print(f"  Step {step.step}: {step.span_name}")
        if step.attributes:
            print(f"    Attributes: {step.attributes}")

    # Fork at the problematic step
    print("\nForking at step 3 (hallucinated response)...")
    forked = replay.fork_at(step=3)
    forked.modify_prompt("Always cite the exact policy section number.")

    result = forked.rerun()
    print(f"Rerun result: {result}")
