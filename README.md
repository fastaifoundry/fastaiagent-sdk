# FastAIAgent SDK

**Build, debug, evaluate, and operate AI agents.**
The only SDK with **Agent Replay** — fork-and-rerun debugging — and a
**zero-ceremony Local UI** that ships inside the Python wheel.

Works standalone or connected to the [FastAIAgent Platform](https://fastaiagent.net) for visual editing, production monitoring, and team collaboration.

[![PyPI](https://img.shields.io/pypi/v/fastaiagent?v=0.9.2)](https://pypi.org/project/fastaiagent/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Tests](https://github.com/fastaifoundry/fastaiagent-sdk/actions/workflows/ci.yml/badge.svg)](https://github.com/fastaifoundry/fastaiagent-sdk/actions)
[![Python](https://img.shields.io/pypi/pyversions/fastaiagent)](https://pypi.org/project/fastaiagent/)

---

## Quickstart

```python
from fastaiagent import Agent, LLMClient

# Create an LLM client
llm = LLMClient(provider="openai", model="gpt-4o")

# Create an agent
agent = Agent(
    name="my-agent",
    system_prompt="You are a helpful assistant.",
    llm=llm,
)

# Run it
result = agent.run("What is the capital of France?")
print(result.output)
print(result.trace_id)  # every run is traced — use this ID for replay/debugging
```

## Debug a failing agent in 30 seconds

```python
from fastaiagent.trace import Replay

# Load a trace from a production failure
replay = Replay.load("trace_abc123")

# Step through to find the problem
replay.step_through()
# Step 3: LLM hallucinated the refund policy ← found it

# Fork at the failing step, fix, rerun
forked = replay.fork_at(step=3)
forked.modify_prompt("Always cite the exact policy section...")
result = forked.rerun()
```

**No other SDK can do this.**

## See every trace, eval, and prompt in your browser — no Docker, no signup

```bash
pip install 'fastaiagent[ui]'
fastaiagent ui
```

Opens a polished web UI at `http://127.0.0.1:7842`. Every agent run you
execute lands here — span tree with Gantt-style timing, JSON-viewer
inspector, Agent Replay fork-and-rerun in the browser, eval runs with
pass-rate trend charts, prompt editor with version lineage, guardrail
events, agent scorecards, and a **read-only browser + search playground
for every `LocalKB`** you've built. Everything stored in one SQLite file at
`./.fastaiagent/local.db`. Bcrypt-hashed local auth. Nothing phones home.

![FastAIAgent Local UI — trace detail](https://raw.githubusercontent.com/fastaifoundry/fastaiagent-sdk/main/docs/ui/screenshots/03-trace-detail.png)

See [docs/ui/](https://github.com/fastaifoundry/fastaiagent-sdk/blob/main/docs/ui/index.md) for the full tour; the KB browser is documented at [docs/ui/kb.md](https://github.com/fastaifoundry/fastaiagent-sdk/blob/main/docs/ui/kb.md).

## Evaluate agents systematically

```python
from fastaiagent.eval import evaluate

results = evaluate(
    agent_fn=my_agent.run,
    dataset="test_cases.jsonl",
    scorers=["correctness", "relevance"]
)
print(results.summary())
# correctness: 92% | relevance: 88%
```

## Trace any agent — yours or LangChain/CrewAI

```python
import fastaiagent
fastaiagent.integrations.langchain.enable()

# Your existing LangChain agent, now with full tracing
result = langchain_agent.invoke({"input": "..."})
# → Traces stored locally or pushed to FastAIAgent Platform
```

## Build agents with guardrails and cyclic workflows

```python
from fastaiagent import Agent, Chain, LLMClient, Guardrail
from fastaiagent.guardrail import no_pii, json_valid

agent = Agent(
    name="support-bot",
    system_prompt="You are a helpful support agent...",
    llm=LLMClient(provider="openai", model="gpt-4o"),
    tools=[search_tool, refund_tool],
    guardrails=[no_pii(), json_valid()]
)

# Chains with loops (retry until quality is good enough)
chain = Chain("support-pipeline", state_schema=SupportState)
chain.add_node("research", agent=researcher)
chain.add_node("evaluate", agent=evaluator)
chain.add_node("respond", agent=responder)
chain.connect("research", "evaluate")
chain.connect("evaluate", "research", max_iterations=3, exit_condition="quality >= 0.8")
chain.connect("evaluate", "respond", condition="quality >= 0.8")

result = chain.execute({"message": "My order is late"}, trace=True)
```

## Deploying

A fastaiagent agent is a plain Python object — wrap it in any web framework and ship it anywhere Python runs. [docs/deployment](docs/deployment/index.md) has copy-paste recipes for:

- **[FastAPI + Uvicorn](docs/deployment/fastapi.md)** — the baseline. Works on a laptop or any VM / container.
- **[Docker → Cloud Run / Fly / Render / Railway](docs/deployment/docker.md)** — one Dockerfile, four managed container platforms.
- **[Modal](docs/deployment/modal.md)** — serverless Python with no container work.
- **[Replicate (Cog)](docs/deployment/replicate.md)** — public inference endpoint.

Every recipe exposes the same `POST /run` + `POST /run/stream` contract so callers don't care where the agent lives. See the runnable starter: [examples/33_deploy_fastapi.py](examples/33_deploy_fastapi.py).

## Expose agents as MCP servers (Claude Desktop / Cursor / Continue / Zed)

Any `Agent` or `Chain` becomes an MCP server with one line:

```python
from fastaiagent import Agent, LLMClient

agent = Agent(name="research_assistant", llm=LLMClient(provider="openai", model="gpt-4o"))

if __name__ == "__main__":
    import asyncio
    asyncio.run(agent.as_mcp_server(transport="stdio").run())
```

Register it in `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "research-assistant": {
      "command": "python",
      "args": ["/absolute/path/to/my_agent.py"]
    }
  }
}
```

Claude Desktop now treats your fastaiagent as a callable tool. Same config shape for Cursor / Continue / Zed. Or use the CLI: `fastaiagent mcp serve my_agent.py:agent`. See [docs/tools/mcp-server.md](docs/tools/mcp-server.md).

Install: `pip install 'fastaiagent[mcp-server]'`.

## Peer-to-peer swarms with handoffs

Beyond the central-coordinator Supervisor/Worker pattern, agents can hand off to each other directly:

```python
from fastaiagent import Agent, LLMClient, Swarm

llm = LLMClient(provider="openai", model="gpt-4o-mini")

triage = Agent(name="triage", llm=llm, system_prompt="Hand off to the right specialist.")
coder = Agent(name="coder", llm=llm, system_prompt="Answer Python questions.")
writer = Agent(name="writer", llm=llm, system_prompt="Help with prose.")

swarm = Swarm(
    name="help_desk",
    agents=[triage, coder, writer],
    entrypoint="triage",
    handoffs={"triage": ["coder", "writer"], "coder": [], "writer": []},
)
result = swarm.run("How do I reverse a list in Python?")
```

The currently active agent decides when to transfer control — no central LLM. See [docs/agents/swarm.md](docs/agents/swarm.md) for the full guide, and [Swarm vs Supervisor](docs/agents/swarm.md#swarm-vs-supervisor--when-to-use-which) for when to pick which.

## Long-term memory with composable blocks

Beyond a sliding window, layer static facts, a rolling summary, semantic recall, and fact extraction into one memory object:

```python
from fastaiagent import Agent, LLMClient, ComposableMemory, AgentMemory
from fastaiagent import StaticBlock, SummaryBlock, VectorBlock, FactExtractionBlock
from fastaiagent.kb.backends.faiss import FaissVectorStore

llm = LLMClient(provider="openai", model="gpt-4o-mini")

agent = Agent(
    name="assistant",
    llm=llm,
    memory=ComposableMemory(
        blocks=[
            StaticBlock("User is Upendra. Prefers terse answers."),
            SummaryBlock(llm=llm, keep_last=10, summarize_every=5),
            VectorBlock(store=FaissVectorStore(dimension=384)),
            FactExtractionBlock(llm=llm, max_facts=100),
        ],
        primary=AgentMemory(max_messages=20),
    ),
)
```

`VectorBlock` works with any `VectorStore` (Qdrant / Chroma / custom). Write your own block by subclassing `MemoryBlock` with two methods. See [docs/agents/memory.md](docs/agents/memory.md).

## Swap the KB storage layer

Default `LocalKB` ships with FAISS + BM25 + SQLite — zero setup. Point at Qdrant, Chroma, or your own backend with one kwarg:

```python
from fastaiagent.kb import LocalKB
from fastaiagent.kb.backends.qdrant import QdrantVectorStore

kb = LocalKB(
    name="product-docs",
    search_type="vector",
    vector_store=QdrantVectorStore(
        url="http://localhost:6333",
        collection="product-docs",
        dimension=1536,
    ),
)
kb.add("docs/")
results = kb.search("refund policy", top_k=5)
```

Adapters shipped: **FAISS**, **BM25**, **SQLite** (defaults), **Qdrant** (`fastaiagent[qdrant]`), **Chroma** (`fastaiagent[chroma]`). Write your own against the `VectorStore` / `KeywordStore` / `MetadataStore` protocols — see [docs/knowledge-base/backends.md](docs/knowledge-base/backends.md).

**Platform-hosted KBs.** For KBs uploaded and managed on the FastAIAgent platform, use `fa.PlatformKB(kb_id=...)` — same `.search()` / `.as_tool()` surface, retrieval (hybrid + rerank + relevance gate) runs on the platform. See [docs/knowledge-base/platform-kb.md](docs/knowledge-base/platform-kb.md).

## Shape agent behavior with middleware

Compose pre/post model hooks and tool wrappers without subclassing `Agent`:

```python
from fastaiagent import Agent, LLMClient, TrimLongMessages, RedactPII, ToolBudget

agent = Agent(
    name="controlled",
    llm=LLMClient(provider="openai", model="gpt-4o"),
    tools=[search_tool],
    middleware=[
        TrimLongMessages(keep_last=30),   # cap history size
        RedactPII(),                      # scrub emails/phones/SSNs both directions
        ToolBudget(max_calls=5),          # cooperatively stop after 5 tool calls
    ],
)
```

Write your own by subclassing `AgentMiddleware` and overriding `before_model`, `after_model`, or `wrap_tool`. See [docs/agents/middleware.md](docs/agents/middleware.md) for ordering, hook reference, and custom patterns.

## Multi-agent teams with context

```python
from fastaiagent import Agent, LLMClient, RunContext, Supervisor, Worker, tool

@tool(name="get_tickets")
def get_tickets(ctx: RunContext[AppState], status: str) -> str:
    """Get support tickets for the current user."""
    return ctx.state.db.query("tickets", user_id=ctx.state.user_id, status=status)

support = Agent(name="support", llm=llm, tools=[get_tickets], system_prompt="Handle tickets.")
billing = Agent(name="billing", llm=llm, tools=[get_billing], system_prompt="Handle billing.")

supervisor = Supervisor(
    name="customer-service",
    llm=LLMClient(provider="openai", model="gpt-4o"),
    workers=[
        Worker(agent=support, role="support", description="Manages tickets"),
        Worker(agent=billing, role="billing", description="Handles billing"),
    ],
    system_prompt=lambda ctx: f"You lead support for {ctx.state.company}. Be helpful.",
)

# Context flows to all workers and their tools
ctx = RunContext(state=AppState(db=db, user_id="u-1", company="Acme"))
result = supervisor.run("Show my open tickets and billing", context=ctx)

# Stream the supervisor's response
async for event in supervisor.astream("Help me", context=ctx):
    if isinstance(event, TextDelta):
        print(event.text, end="")
```

## Connect to FastAIAgent Platform (optional)

```python
import fastaiagent as fa

fa.connect(api_key="fa-...", project="my-project")

# Traces automatically sent to platform dashboard
result = agent.run("Help me")

# Pull versioned prompts from platform
prompt = PromptRegistry().get("support-prompt")

# Publish eval results to platform
results = evaluate(agent, dataset=dataset)
results.publish()
```

**SDK works standalone. Platform adds: production observability, prompt management,
evaluation dashboards, team collaboration, HITL approval workflows.**

[Free tier available →](https://app.fastaiagent.net)

---

## Install

```bash
pip install fastaiagent
```

With optional integrations:
```bash
pip install "fastaiagent[openai]"       # OpenAI auto-tracing
pip install "fastaiagent[langchain]"    # LangChain auto-tracing
pip install "fastaiagent[kb]"           # Local knowledge base
pip install "fastaiagent[all]"          # Everything
```

## Documentation

- [Getting Started](https://github.com/fastaifoundry/fastaiagent-sdk/tree/main/docs/getting-started)
- [Agent Replay Guide](https://github.com/fastaifoundry/fastaiagent-sdk/tree/main/docs/replay)
- [Building Chains with Cycles](https://github.com/fastaifoundry/fastaiagent-sdk/tree/main/docs/chains)
- [Guardrails](https://github.com/fastaifoundry/fastaiagent-sdk/tree/main/docs/guardrails)
- [Evaluation](https://github.com/fastaifoundry/fastaiagent-sdk/tree/main/docs/evaluation)
- [API Reference](https://github.com/fastaifoundry/fastaiagent-sdk/tree/main/docs/api-reference)

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Apache 2.0 — see [LICENSE](LICENSE).
