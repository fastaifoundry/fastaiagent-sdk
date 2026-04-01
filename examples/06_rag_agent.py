"""Example 06: RAG agent with LocalKB.

Shows how to create a knowledge base from files,
and use it as a tool for an agent.
"""

from fastaiagent import Agent, LLMClient
from fastaiagent.kb import LocalKB

# Create a local knowledge base
kb = LocalKB(name="product-docs", path="/tmp/fastaiagent-example-kb/")

# Add some content
kb.add("Our refund policy allows returns within 30 days of purchase. "
       "Items must be in original condition. Digital products are non-refundable.")
kb.add("Shipping typically takes 3-5 business days for domestic orders. "
       "International shipping takes 7-14 business days. Express shipping is available.")
kb.add("To contact support, email support@example.com or call 1-800-EXAMPLE. "
       "Support hours are Monday-Friday 9am-5pm EST.")

print(f"KB status: {kb.status()}")

# Create an agent with KB as a tool
agent = Agent(
    name="support-agent",
    system_prompt="You are a customer support agent. Use the search tool to find relevant information before answering.",
    llm=LLMClient(provider="openai", model="gpt-4o-mini"),
    tools=[kb.as_tool()],
)

if __name__ == "__main__":
    # Search the KB directly
    results = kb.search("refund policy", top_k=2)
    print("\nDirect KB search results:")
    for r in results:
        print(f"  [{r.score:.3f}] {r.chunk.content[:100]}...")

    # Or use via agent (requires API key)
    # result = agent.run("What is the refund policy?")
    # print(f"\nAgent response: {result.output}")
