# Templates

Production-ready, end-to-end examples shipped with the SDK. Each template lives under [`examples/`](https://github.com/fastaifoundry/fastaiagent-sdk/tree/main/examples) and includes its own README, eval suite, and tests.

| Template | Pattern | What it shows |
|---|---|---|
| [Deep Research Agent](deep-research-agent.md) | Scope → parallel Research → Write | Long-horizon research with structured trace spans, real web search/fetch, parallel sub-researchers via `asyncio.gather`. Pairs with the trace-learning loop for self-improving agents. |
| Customer Support Agent | Single Agent + HITL | Multi-turn support with checkpointer, memory, tool budget, KB retrieval, eval suite. |
| Research Agent | Supervisor + 3 workers | Researcher → Writer → Verifier with revision loop and citation auditing. |
| Sales SDR Agent | Multi-step workflow | Lead qualification chain with structured outputs. |
| Meeting Notes Agent | Structured output | Transcript → schema-validated meeting notes. |
| Personal Assistant | Long-context memory | Multi-tool orchestration with composable memory blocks. |

Templates are meant to be **read, copied, and adapted** — not used as black-box dependencies. Each one stays under ~500 lines so a developer can hold it in their head.
