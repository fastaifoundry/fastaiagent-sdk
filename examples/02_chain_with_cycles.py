"""Example 02: Chain with retry loop (cyclic workflow).

Demonstrates a research → evaluate → retry pattern where
the evaluator can send work back to the researcher up to 3 times.
"""

from fastaiagent import Agent, Chain, LLMClient


def make_agent(name: str, prompt: str) -> Agent:
    return Agent(
        name=name,
        system_prompt=prompt,
        llm=LLMClient(provider="openai", model="gpt-4o-mini"),
    )


chain = Chain("research-pipeline")
chain.add_node("research", agent=make_agent("researcher", "Research the given topic thoroughly."))
chain.add_node(
    "evaluate",
    agent=make_agent(
        "evaluator",
        "Evaluate if the research is sufficient. Set quality score.",
    ),
)
chain.add_node(
    "respond",
    agent=make_agent(
        "responder",
        "Write a final response using the research.",
    ),
)

chain.connect("research", "evaluate")
chain.connect("evaluate", "research", max_iterations=3, exit_condition="quality >= 0.8")
chain.connect("evaluate", "respond", condition="quality >= 0.8")

if __name__ == "__main__":
    import os

    if not os.environ.get("OPENAI_API_KEY"):
        print("Skipping: OPENAI_API_KEY not set")
        print("Run with: export OPENAI_API_KEY=sk-... && python examples/02_chain_with_cycles.py")
    else:
        result = chain.execute({"message": "What are the latest trends in AI agents?"})
        print(f"Output: {result.output}")
        print(f"Execution ID: {result.execution_id}")
