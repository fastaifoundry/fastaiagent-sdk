"""Capture a third-party OpenInference span in the FastAIAgent Local UI.

Runs a *plain* OpenAI call (no FastAIAgent agent) with the OpenInference OpenAI
instrumentor active and ``enable_otel_capture()`` turned on, then prints the
canonical attributes that landed in ``.fastaiagent/local.db``. Open the Local UI
with ``fastaiagent ui`` afterwards to see the span render richly — model,
tokens, cost, and Input/Output content.

Requires:
    pip install "fastaiagent[openai]" openinference-instrumentation-openai
    export OPENAI_API_KEY=...

Run:
    python examples/otel-openinference/capture.py
"""

from __future__ import annotations

import os

import fastaiagent as fa


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY to run this example.")

    import openai
    from openinference.instrumentation.openai import OpenAIInstrumentor

    # 1. Enable a third-party instrumentor on a non-FastAIAgent call path.
    OpenAIInstrumentor().instrument()

    # 2. Opt in to capture + rich rendering. Default behavior is unchanged
    #    until this is called.
    fa.enable_otel_capture()

    # 3. A normal OpenAI call — no FastAIAgent agent involved.
    client = openai.OpenAI()
    client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Summarize the quarterly report."}],
        max_tokens=64,
    )

    # 4. The span is now in the local store, normalized. Print proof.
    from fastaiagent.trace import TraceStore

    store = TraceStore()
    try:
        traces = store.list_traces()
        if not traces:
            print("No trace captured — is the instrumentor installed?")
            return
        for summary in traces:
            trace = store.get_trace(summary.trace_id)
            for span in trace.spans:
                attrs = span.attributes
                if attrs.get("gen_ai.request.model"):
                    print(f"captured span: {span.name}")
                    print(f"  framework   : {attrs.get('fastaiagent.framework')}")
                    print(f"  runner.type : {attrs.get('fastaiagent.runner.type')}")
                    print(f"  model       : {attrs.get('gen_ai.request.model')}")
                    print(f"  input_tokens: {attrs.get('gen_ai.usage.input_tokens')}")
                    print(f"  output_toks : {attrs.get('gen_ai.usage.output_tokens')}")
                    print("\nView it in the Local UI:  fastaiagent ui")
                    return
    finally:
        store.close()
        fa.disable_otel_capture()
        OpenAIInstrumentor().uninstrument()


if __name__ == "__main__":
    main()
