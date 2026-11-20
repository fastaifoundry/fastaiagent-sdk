"""Example 02: Chain with a retry loop (cyclic workflow).

draft -> check -> publish, where ``check`` can send the work back to ``draft``:

* ``draft``   — an agent writes a summary
* ``check``   — a plain function judges it (here: a word limit) and answers PASS or FAIL
* ``publish`` — a transformer emits the draft that passed; it runs once, after the last pass

The edge that goes back carries the two loop settings:

* ``exit_condition`` — leave the loop when the check answers ``PASS``
* ``max_iterations`` — at most 3 passes; the chain raises ``ChainCycleError`` if the draft
  still has not passed by then

How data moves, which decides what each node can be:

* An **agent** node's input is the chain's input: the state at the start of the pass. It does not
  receive the previous node's output. On a second pass that state holds the check's feedback,
  which is how ``draft`` learns what to fix.
* A **tool** or **transformer** node can read any earlier node's result with a template
  (``{{node_results.draft.output}}``). That is why the judge here is a tool node.
* Conditions are templates too. A bare word (``quality >= 0.8``) is compared as text and is
  never true.

Usage:
    export OPENAI_API_KEY=sk-...
    python examples/02_chain_with_cycles.py
"""

from fastaiagent import Agent, Chain, FunctionTool, LLMClient
from fastaiagent._internal.errors import ChainCycleError
from fastaiagent.chain import NodeType

MAX_PASSES = 3
WORD_LIMIT = 15
checks: list[str] = []  # every verdict, so the run can show how many passes it took


def check_length(draft: str) -> str:
    """PASS when the draft is within the word limit, else FAIL with what to fix."""
    words = len(draft.split())
    if words <= WORD_LIMIT:
        verdict = f"PASS ({words} words)"
    else:
        verdict = f"FAIL: the draft is {words} words; rewrite it in at most {WORD_LIMIT} words."
    checks.append(verdict)
    return verdict


chain = Chain("research-pipeline", checkpoint_enabled=False)
chain.add_node(
    "draft",
    agent=Agent(
        name="drafter",
        system_prompt=(
            "Write a summary of the topic in the input's 'message'. If the input has an 'output' "
            "that starts with FAIL, it is feedback on your previous draft: follow it exactly. "
            "Reply with the summary only."
        ),
        llm=LLMClient(provider="openai", model="gpt-5-mini"),
    ),
)
chain.add_node(
    "check",
    tool=FunctionTool(name="check_length", fn=check_length),
    input_mapping={"draft": "{{node_results.draft.output}}"},
)
chain.add_node("publish", type=NodeType.transformer, template="{{node_results.draft.output}}")
chain.connect("draft", "check")
chain.connect("check", "publish")
chain.connect(
    "check",
    "draft",  # the edge that goes back
    max_iterations=MAX_PASSES,
    exit_condition="{{node_results.check.output}} startswith PASS",
)

if __name__ == "__main__":
    import os

    problems = chain.validate()
    assert problems == [], problems

    if not os.environ.get("OPENAI_API_KEY"):
        print("Skipping: OPENAI_API_KEY not set")
        print("Run: export OPENAI_API_KEY=sk-... && python examples/02_chain_with_cycles.py")
    else:
        print(f"Running draft -> check -> publish (at most {MAX_PASSES} passes)...")
        try:
            result = chain.execute(
                {"message": "What an AI agent framework is, and why teams use it"}
            )
        except ChainCycleError as err:
            print(f"The draft never passed the check: {err}")
        else:
            print(f"\nPasses: {len(checks)}")
            for n, verdict in enumerate(checks, 1):
                print(f"  check {n}: {verdict}")
            print(f"Published: {result.node_results['publish']['output']}")
            print(f"Execution ID: {result.execution_id}")
            print(f"Nodes executed: {list(result.node_results.keys())}")
            print(
                "\nTo render the topology view in the Local UI, register "
                "this chain with build_app:\n"
                "    from fastaiagent.ui.server import build_app\n"
                "    app = build_app(runners=[chain])\n"
                "Then visit http://127.0.0.1:7843/workflows/chain/research-pipeline"
            )
