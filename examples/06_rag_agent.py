"""Example 06: RAG agent with LocalKB.

Shows how to create a persistent knowledge base with hybrid search
(FAISS + BM25), add content from text and files, and use it as a tool
for an agent.
"""

from fastaiagent import Agent, LLMClient
from fastaiagent.kb import LocalKB

# Create a persistent local knowledge base (hybrid search by default)
kb = LocalKB(name="product-docs", path="/tmp/fastaiagent-example-kb/")

# Add some content — persisted to SQLite, no re-embedding on restart
kb.add(
    "Our refund policy allows returns within 30 days of purchase. "
    "Items must be in original condition. Digital products are non-refundable."
)
kb.add(
    "Shipping typically takes 3-5 business days for domestic orders. "
    "International shipping takes 7-14 business days. Express shipping is available."
)
kb.add(
    "To contact support, email support@example.com or call 1-800-EXAMPLE. "
    "Support hours are Monday-Friday 9am-5pm EST."
)
kb.add(
    "Error code ERR-4012 occurs when the payment gateway times out. "
    "Retry after 30 seconds or contact support with your transaction ID."
)

print(f"KB status: {kb.status()}")

# Create an agent with KB as a tool
agent = Agent(
    name="support-agent",
    system_prompt=(
        "You are a customer support agent. "
        "Use the search tool to find relevant information before answering."
    ),
    llm=LLMClient(provider="openai", model="gpt-4o-mini"),
    tools=[kb.as_tool()],
)

if __name__ == "__main__":
    # --- Hybrid search: BM25 catches exact terms, FAISS catches semantics ---
    print("\n--- Hybrid search (default) ---")
    results = kb.search("ERR-4012 payment failed", top_k=2)
    for r in results:
        print(f"  [{r.score:.3f}] {r.chunk.content[:100]}...")

    # --- Keyword-only search: great for error codes, IDs ---
    kb_keyword = LocalKB(name="logs", path="/tmp/fastaiagent-example-kb/", search_type="keyword", persist=False)
    kb_keyword.add("Error code ERR-4012 payment gateway timeout.")
    kb_keyword.add("Error code ERR-5001 authentication failure.")
    print("\n--- Keyword search ---")
    results = kb_keyword.search("ERR-4012", top_k=1)
    for r in results:
        print(f"  [{r.score:.3f}] {r.chunk.content[:100]}...")

    # --- Persistence: restart-safe ---
    print(f"\n--- Persistence ---")
    print(f"  Chunks in KB after restart: {kb.status()['chunk_count']}")

    # --- CRUD: delete by source ---
    # kb.delete_by_source("docs/old-faq.md")

    # --- Agent usage (requires API key) ---
    # result = agent.run("What is error ERR-4012?")
    # print(f"\nAgent response: {result.output}")

    kb.close()
