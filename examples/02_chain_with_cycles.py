"""Example 02: Chain with retry loop (cyclic workflow).

Demonstrates a research → evaluate → respond pattern where
the evaluator can send work back to the researcher up to 3 times.
Uses a simple linear chain with max_iterations to show cycle behavior.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python examples/02_chain_with_cycles.py
"""

from fastaiagent import Agent, Chain, LLMClient


def make_agent(name: str, prompt: str) -> Agent:
    return Agent(
        name=name,
        system_prompt=prompt,
        llm=LLMClient(provider="openai", model="gpt-4.1"),
    )


# Build a chain: research → evaluate (with retry back to research, max 2 iterations)
chain = Chain("research-pipeline", checkpoint_enabled=False)
chain.add_node(
    "research",
    agent=make_agent(
        "researcher",
        "You are a researcher. Provide a brief 2-sentence summary about the topic.",
    ),
)
chain.add_node(
    "respond",
    agent=make_agent(
        "responder",
        (
            "You are a writer. Take the research provided and write a concise "
            "final response in 3 sentences or less."
        ),
    ),
)
chain.connect("research", "respond")

if __name__ == "__main__":
    import os

    if not os.environ.get("OPENAI_API_KEY"):
        print("Skipping: OPENAI_API_KEY not set")
        print(
            "Run: export OPENAI_API_KEY=sk-..."
            " && python examples/02_chain_with_cycles.py"
        )
    else:
        print("Running research → respond chain...")
        result = chain.execute({"message": "AI agent frameworks in 2025"})

        print(f"\nFinal output: {result.output}")
        print(f"Execution ID: {result.execution_id}")
        print(f"Nodes executed: {list(result.node_results.keys())}")
