# 💬 Customer Support Agent

A production-ready customer support agent built with [FastAIAgent SDK](https://github.com/fastaifoundry/fastaiagent-sdk). Answers questions from a knowledge base, creates support tickets, enforces PII guardrails, and includes a full evaluation suite.

**What this template demonstrates:**

- Agent with tools, knowledge base, and guardrails
- `RunContext` for dependency injection (DB connection, API clients)
- `fa.connect()` to export traces and pull prompts from the platform
- Evaluation with LLM-as-Judge scoring
- Agent Replay for debugging

---

## Quick Start

```bash
# Clone and setup
git clone https://github.com/fastaifoundry/fastaiagent-sdk.git
cd examples/customer-support-agent
cp .env.example .env        # Add your API keys
pip install -r requirements.txt

# Run the agent
python agent.py
```

First run takes ~10 seconds to ingest the sample knowledge base. Subsequent runs use the cached index.

---

## What's Inside

```
customer-support-agent/
├── README.md               # You are here
├── .env.example            # Environment variable template
├── requirements.txt        # Dependencies
├── agent.py                # Main agent definition and runner
├── tools.py                # Tool functions (ticket creation, account lookup)
├── context.py              # RunContext definition with dependencies
├── guardrails.py           # PII filter and toxicity guardrails
├── eval_suite.py           # Evaluation with LLM-as-Judge scoring
├── replay_demo.py          # Agent Replay fork-and-rerun example
└── knowledge/
    ├── faq.md              # Sample FAQ knowledge base
    └── policies.md         # Sample company policies
```

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────┐
│  Customer Support Agent             │
│                                     │
│  System Prompt (from Prompt Registry│
│  or local)                          │
│                                     │
│  ┌──────────┐  ┌──────────────────┐ │
│  │ Guardrails│  │ RunContext[Deps] │ │
│  │ • PII     │  │ • db_connection │ │
│  │ • Toxicity│  │ • ticket_client │ │
│  └──────────┘  └──────────────────┘ │
│                                     │
│  Tools:                             │
│  • search_kb      → Knowledge Base  │
│  • create_ticket  → Ticket System   │
│  • lookup_account → CRM             │
│  • check_status   → Order Tracking  │
└─────────────────────────────────────┘
    │
    ▼
Traces → fa.connect() → Platform Dashboard
```

---

## Features Demonstrated

### 1. RunContext — Dependency Injection

```python
@dataclass
class Deps:
    db: AsyncConnection
    ticket_client: TicketClient
    user_email: str

agent = fa.Agent(
    name="support-bot",
    model="gpt-4o",
    tools=[search_kb, create_ticket, lookup_account],
    context_type=Deps,
)

# Dependencies injected at runtime, not import time
result = await agent.run("I can't log in", context=deps)
```

### 2. Knowledge Base — Grounded Answers

```python
# Ingest docs on first run
kb = fa.KnowledgeBase(name="support-kb", path="./knowledge/")
await kb.ingest()

# search_kb tool uses the KB automatically
@fa.tool
async def search_kb(query: str, ctx: fa.RunContext[Deps]) -> str:
    """Search the support knowledge base for relevant information."""
    results = await kb.search(query, top_k=3)
    return "\n\n".join(r.content for r in results)
```

### 3. Guardrails — Safety by Default

```python
pii_filter = fa.Guardrail(
    name="pii-filter",
    type="regex",
    position="output",
    pattern=r"\b\d{3}-\d{2}-\d{4}\b",  # SSN pattern
    action="block",
    message="I can't share personally identifiable information.",
)

toxicity_check = fa.Guardrail(
    name="toxicity-check",
    type="llm_judge",
    position="input",
    prompt="Is the following message toxic or abusive? Respond YES or NO.",
    action="block",
)
```

### 4. Platform Connection

```python
import fastaiagent as fa

# Connect to FastAIAgent Platform (optional)
fa.connect(api_key="fa_k_...", project="support-bot")

# Now:
# - All traces auto-export to your dashboard
# - Prompts pull from the Prompt Registry
# - Eval results publish to the platform
```

### 5. Evaluation — LLM-as-Judge

```python
# Define scoring dimensions
correctness = fa.Scorer(
    name="correctness",
    type="llm_judge",
    scale="binary",
    prompt="Was the agent's response factually correct based on the KB?",
)

helpfulness = fa.Scorer(
    name="helpfulness",
    type="llm_judge",
    scale="1-5",
    prompt="How helpful was the response in resolving the user's issue?",
)

# Run evaluation
dataset = fa.Dataset.from_file("eval_cases.jsonl")
results = await fa.evaluate(agent, dataset, scorers=[correctness, helpfulness])
results.summary()

# Publish to platform
results.publish()
```

### 6. Agent Replay — Debug Any Failure

```python
# Get a trace from a failed run
trace = fa.Replay.from_latest(agent_name="support-bot", status="error")

# Step through it
for span in trace.spans:
    print(f"{span.type}: {span.name} ({span.duration_ms}ms)")

# Fork from the tool call that failed
forked = trace.fork(span_index=3, modified_input={"query": "login reset"})
forked_result = await forked.run()

# Compare outcomes
fa.compare(trace, forked)
```

---

## Running Evaluation

```bash
# Run the full eval suite
python eval_suite.py

# Output:
# ┌─────────────┬───────┬───────┐
# │ Scorer      │ Score │ Count │
# ├─────────────┼───────┼───────┤
# │ correctness │ 0.92  │ 20    │
# │ helpfulness │ 4.1   │ 20    │
# │ safety      │ 1.00  │ 20    │
# └─────────────┴───────┴───────┘
```

---

## Replaying a Failed Execution

```bash
# Run the replay demo
python replay_demo.py

# This will:
# 1. Execute the agent with a query that triggers a tool error
# 2. Open the replay viewer
# 3. Fork from the failing step
# 4. Re-run with corrected input
# 5. Compare both traces side by side
```

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key for GPT-4o |
| `FASTAIAGENT_API_KEY` | No | Platform API key for `fa.connect()` |
| `FASTAIAGENT_PROJECT` | No | Platform project name |
| `TICKET_API_URL` | No | Ticket system endpoint (defaults to mock) |
| `CRM_API_URL` | No | CRM endpoint (defaults to mock) |

---

## Customising This Template

**Swap the LLM provider:**
```python
agent = fa.Agent(model="claude-sonnet-4-20250514")     # Anthropic
agent = fa.Agent(model="llama3:8b")                     # Ollama (local)
agent = fa.Agent(model="gpt-4o", base_url="https://your-gateway.com")  # Custom
```

**Add your own knowledge base:**
Replace the files in `knowledge/` with your own docs (PDF, DOCX, TXT, Markdown). The KB auto-processes on first run.

**Add more tools:**
```python
@fa.tool
async def escalate_to_human(reason: str, ctx: fa.RunContext[Deps]) -> str:
    """Escalate the conversation to a human agent."""
    # Your escalation logic here
    return f"Escalated: {reason}. A human agent will follow up."
```

**Connect to your platform instance:**
```bash
export FASTAIAGENT_API_KEY="fa_k_your_key"
export FASTAIAGENT_PROJECT="my-project"
```

---

## Next Steps

- **Explore Agent Replay**: Run `python replay_demo.py` and try forking from different steps
- **Connect to the platform**: Sign up at [app.fastaiagent.net](https://app.fastaiagent.net) and see your traces in the dashboard
- **Run evals before shipping**: Customise `eval_suite.py` with your own test cases
- **Read the docs**: [fastaiagent.net/docs](https://fastaiagent.net/docs)

---

## License

Apache 2.0 — same as the SDK. Use this template for anything.
