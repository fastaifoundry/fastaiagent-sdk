"""Example 57 — Run ``fa.evaluate()`` against a LangGraph agent.

``as_evaluable()`` adapts a compiled LangGraph (or any LangChain
``Runnable``) into the callable shape ``fa.evaluate()`` expects. The
adapter opens an outer ``eval.case`` span per case so ``trace_id`` is
captured and persisted on each ``EvalCaseRecord`` — the Local UI's
"Traces using this eval" panel can then deep-link to the exact trace
the case produced.

Run:
    pip install "fastaiagent[langchain,ui]" langchain langchain-openai langgraph
    OPENAI_API_KEY=sk-... python examples/57_eval_langchain.py
"""

from __future__ import annotations

import os
import sys


def main() -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set — skipping example.")
        return 0

    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        print(
            "LangChain is not installed. Install with: "
            'pip install "fastaiagent[langchain]" langchain langchain-openai'
        )
        return 0
    try:
        from langchain.agents import create_agent  # LangChain 1.x
    except ImportError:  # LangChain 0.x
        from langgraph.prebuilt import create_react_agent as create_agent

    import fastaiagent as fa
    from fastaiagent.integrations import langchain as lc

    lc.enable()
    graph = create_agent(ChatOpenAI(model="gpt-4o-mini", temperature=0), tools=[])

    dataset = [
        {"input": "Capital of France? One word.", "expected": "Paris"},
        {"input": "Capital of Japan? One word.", "expected": "Tokyo"},
        {"input": "Capital of Italy? One word.", "expected": "Rome"},
    ]
    evaluable = lc.as_evaluable(graph)

    results = fa.evaluate(
        evaluable,
        dataset=dataset,
        # ``contains``, not ``exact_match``: the model answers "Paris." and an
        # exact match on "Paris" scored a correct agent 0%.
        scorers=["contains"],
        persist=False,
        run_name="example-57-langchain-capitals",
    )
    print(results.summary())

    print("\nPer-case trace_ids:")
    for case in results.cases:
        print(
            f"  expected={case.expected_output:8s}  got={case.actual_output[:30]:30s}  "
            f"trace_id={case.trace_id}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
