# Customer Support Agent

A production-shaped customer support agent built with [FastAIAgent SDK](https://github.com/fastaifoundry/fastaiagent-sdk) v1.6.0. Answers questions from a knowledge base, creates support tickets with human-in-the-loop approval, enforces PII guardrails, streams responses, accepts screenshots, and ships with an evaluation suite covering both LLM-judge and RAG metrics.

**Capabilities demonstrated**

- `Agent` with tools, `LocalKB`, guardrails, and `RunContext[Deps]` dependency injection
- **Multi-turn memory** via `AgentMemory` so the REPL remembers prior turns
- **Middleware** — `ToolBudget` + `TrimLongMessages` for cost and context-window control
- **PromptRegistry**-backed system prompt (editable from the Local UI Playground)
- **HITL approval** via `interrupt()` + `SQLiteCheckpointer` + `agent.aresume()` on high-impact tickets
- **Idempotent** ticket-id allocation that survives resume / `Replay.fork_at` reruns
- **Streaming** via `agent.astream()` (token-by-token output)
- **Multimodal input** via `fa.Image` (customer sends a screenshot)
- **Eval suite**: `LLMJudge` + `Faithfulness` + `AnswerRelevancy`
- **Replay**: fork-and-rerun debugging
- Optional `fa.connect()` to export traces to the platform

---

## Quick Start

```bash
# from the SDK root, install the local SDK + this example's deps
pip install -e .
cd examples/customer-support-agent
cp .env.example .env        # add OPENAI_API_KEY
pip install -r requirements.txt

python agent.py             # interactive REPL with memory + HITL
```

First run takes ~10 seconds to ingest the sample knowledge base. Subsequent runs use the cached index in `.fastaiagent-kb/`.

---

## Files

```
customer-support-agent/
├── README.md               # You are here
├── .env.example            # Environment variable template
├── requirements.txt        # fastaiagent>=1.6.0
├── agent.py                # Main agent + REPL (memory, HITL loop, middleware)
├── tools.py                # Tool functions (interrupt + @idempotent on create_ticket)
├── context.py              # RunContext deps (mock CRM/Ticket/Order clients)
├── guardrails.py           # PII output filter + toxicity input filter
├── eval_suite.py           # LLMJudge + RAG scorers
├── streaming_demo.py       # agent.astream() token-by-token
├── replay_demo.py          # fa.Replay.fork_at(...).modify_input(...).rerun()
├── multimodal_demo.py      # fa.Image + agent.arun([text, image])
└── knowledge/
    ├── faq.md
    └── policies.md
```

---

## How it's wired

### Agent construction ([agent.py](agent.py))

```python
import fastaiagent as fa
from fastaiagent.agent.memory import AgentMemory
from fastaiagent.agent.middleware import ToolBudget, TrimLongMessages

agent = fa.Agent(
    name="customer-support",
    llm=fa.LLMClient(provider="openai", model="gpt-4o"),
    system_prompt=SYSTEM_PROMPT,                        # loaded from PromptRegistry
    tools=[search_kb, create_ticket, lookup_account, check_order_status],
    guardrails=[pii_filter, toxicity_check],
    memory=AgentMemory(max_messages=20),                # multi-turn memory
    middleware=[
        ToolBudget(max_calls=10),                       # cap tool invocations
        TrimLongMessages(keep_last=20),                 # trim history
    ],
    checkpointer=fa.SQLiteCheckpointer(),               # enables HITL + idempotency
)
```

### `RunContext` — dependency injection ([context.py](context.py))

```python
@dataclass
class Deps:
    ticket_client: TicketClient
    crm_client: CRMClient
    order_client: OrderClient
    user_email: str

deps = await create_deps(user_email="alice@acme.com")
ctx = fa.RunContext(state=deps)
result = await agent.arun("I can't log in", context=ctx)
```

Tools receive `ctx: fa.RunContext[Deps]` as their last parameter.

### Knowledge base ([tools.py](tools.py))

```python
kb = fa.LocalKB(
    name="support-kb",
    path="./.fastaiagent-kb",
    chunk_size=512,
    chunk_overlap=50,
)
if kb.status()["chunk_count"] == 0:
    kb.add("./knowledge")          # bulk ingest on first run

@fa.tool()
async def search_kb(query: str, ctx: fa.RunContext[Deps]) -> str:
    """Search the support knowledge base."""
    results = kb.search(query, top_k=3)
    return "\n\n---\n\n".join(
        f"**{r.chunk.metadata.get('source', 'KB')}** (relevance: {r.score:.2f})\n{r.chunk.content}"
        for r in results
    )
```

### Guardrails ([guardrails.py](guardrails.py))

```python
import fastaiagent as fa
from fastaiagent.guardrail import GuardrailPosition

pii_filter = fa.no_pii(position=GuardrailPosition.output)
toxicity_check = fa.toxicity_check(position=GuardrailPosition.input)
```

### HITL approval on high-impact tickets ([tools.py](tools.py))

```python
@fa.tool()
async def create_ticket(subject, description, priority, ctx: fa.RunContext[Deps]) -> str:
    if priority in ("high", "urgent") or "billing" in subject.lower():
        decision = fa.interrupt(
            reason="ticket_approval_required",
            context={"subject": subject, "priority": priority, "user_email": ctx.state.user_email},
        )
        if not decision.approved:
            return "Ticket creation declined by reviewer."
    allocation = _allocate_ticket_id(
        user_email=ctx.state.user_email, subject=subject, priority=priority
    )
    ticket = await ctx.state.ticket_client.create(...)
    return f"Ticket {allocation['ticket_id']} created."
```

The Agent's `SQLiteCheckpointer` persists the suspension. The REPL handles it:

```python
result = await agent.arun(query, context=ctx)
while result.status == "paused":
    info = result.pending_interrupt
    approved = input(f"Approve {info['reason']}? [y/N] ").lower() == "y"
    result = await agent.aresume(
        result.execution_id,
        resume_value=fa.Resume(approved=approved, metadata={"approver": "cli"}),
        context=ctx,
    )
```

### Idempotent ticket-id allocation ([tools.py](tools.py))

```python
from fastaiagent.chain.idempotent import idempotent

@idempotent(key_fn=_ticket_idem_key)
def _allocate_ticket_id(*, user_email, subject, priority) -> dict:
    return {"ticket_id": f"TKT-{int(time.time() * 1000)}", ...}
```

After a HITL `interrupt()` + `aresume()`, the agent loop replays the tool call. Without `@idempotent` we'd mint a fresh ticket id; with it the resume reuses the cached allocation, so the user sees the same `TKT-xxxx` regardless of how many times the workflow is replayed.

### Streaming ([streaming_demo.py](streaming_demo.py))

```python
async for event in agent.astream("How do I upgrade my plan?", context=ctx):
    if isinstance(event, fa.TextDelta):
        print(event.text, end="", flush=True)
```

`astream()` reaches parity with `arun()` for middleware, guardrails, tool calls, and checkpoint writes. Suspensions during streaming propagate as exceptions rather than returning a paused result — the main REPL uses `arun` for that reason.

### Multimodal — screenshots ([multimodal_demo.py](multimodal_demo.py))

```python
image = fa.Image.from_file("error.png")
result = await agent.arun(["What does this error mean?", image], context=ctx)
```

Pass a list of parts (text + `Image` + `PDF`) as the agent input. `LLMClient` handles provider-specific wire formatting (OpenAI vision parts, Anthropic image blocks, etc.).

### Evaluation ([eval_suite.py](eval_suite.py))

```python
from fastaiagent.eval.llm_judge import LLMJudge
from fastaiagent.eval.rag import Faithfulness, AnswerRelevancy

scorers = [
    LLMJudge(criteria="correctness", prompt_template=..., scale="binary"),
    LLMJudge(criteria="helpfulness", prompt_template=..., scale="0-1"),
    LLMJudge(criteria="safety",      prompt_template=..., scale="binary"),
    Faithfulness(),       # claims supported by KB context
    AnswerRelevancy(),    # response addresses the question
]

results = fa.evaluate(
    agent_fn=lambda q: agent.run(q, context=ctx),
    dataset=fa.Dataset.from_list(EVAL_CASES),
    scorers=scorers,
    context=KB_CORPUS,    # forwarded to every scorer; Faithfulness uses it
)
print(results.summary())
```

### Replay — fork and rerun ([replay_demo.py](replay_demo.py))

```python
result = await agent.arun(query, context=ctx)
replay = fa.Replay.load(result.trace_id)
print(replay.summary())

# Pick a step, modify the recorded input, rerun from there.
forked = replay.fork_at(1).modify_input({"email": "bob@startup.io"})
forked_result = forked.rerun()
diff = forked.compare(forked_result)
print(f"Diverged at step {diff.diverged_at}")
```

---

## Local UI

Run the Local UI in a second terminal to inspect traces, costs, agent dependencies, and (when the agent suspends) approvals:

```bash
fastaiagent ui start            # serves http://127.0.0.1:7842 and opens your browser
fastaiagent ui start --no-auth  # skip the local auth prompt for throwaway use
```

Highlights:

- `/traces` — span tree, costs, FTS5 search across runs
- `/playground` — edit the `support-system-prompt` registered by `agent.py` and replay
- `/agents` — dependency graph (tools, KBs, prompts, guardrails)
- `/evals` — `eval_suite.py` runs are persisted here

---

## Running each entry point

```bash
# Interactive REPL with memory + HITL approvals
python agent.py

# One-shot query
python agent.py --query "What is your refund policy?"

# Token-by-token streaming
python streaming_demo.py --query "How do I upgrade my plan?"

# Replay debugging
python replay_demo.py

# Image + text
python multimodal_demo.py path/to/screenshot.png "What does this error mean?"

# Eval suite (LLMJudge + RAG)
python eval_suite.py
python eval_suite.py --publish        # also publish to platform
```

To test HITL interactively, ask: *"I was charged twice this month — please file a billing ticket."* The agent will pause for approval; type `y` to file, anything else to decline.

---

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key for GPT-4o |
| `LLM_MODEL` | No | Override the default `gpt-4o` |
| `FASTAIAGENT_API_KEY` | No | Platform API key for `fa.connect()` |
| `FASTAIAGENT_PROJECT` | No | Platform project name (default: `support-bot`) |
| `FASTAIAGENT_TARGET` | No | Platform URL (default: `https://app.fastaiagent.net`) |

---

## Customising

**Swap the LLM provider**:

```python
fa.LLMClient(provider="anthropic", model="claude-sonnet-4-6")
fa.LLMClient(provider="ollama",    model="llama3:8b")
```

**Replace the knowledge base**: drop your own markdown / PDF / TXT files into `knowledge/` and delete `.fastaiagent-kb/` to force re-ingest.

**Add a tool**:

```python
@fa.tool()
async def escalate_to_human(reason: str, ctx: fa.RunContext[Deps]) -> str:
    """Escalate to a human agent."""
    # Your escalation logic here
    return f"Escalated: {reason}."
```

**Connect to the platform**:

```bash
export FASTAIAGENT_API_KEY="fa_k_your_key"
export FASTAIAGENT_PROJECT="my-project"
python agent.py --connect
```

---

## What this example does NOT demonstrate

- **Universal harness** (v1.6.0) — this agent is native `fastaiagent`, not LangChain / CrewAI / PydanticAI. See `examples/55_trace_crewai.py`, `56_trace_pydanticai.py`, `57_eval_langchain.py` for those.
- **Swarm / Supervisor** topologies — see `examples/18_supervisor_worker.py`, `31_swarm_research_team.py`.
- **MCP server** (expose this agent as MCP tools) — see `examples/32_mcp_expose_agent.py`.

---

## License

Apache 2.0 — same as the SDK.
