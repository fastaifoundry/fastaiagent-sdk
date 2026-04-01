# Phase 39-40: SDK Extraction + Bidirectional Sync

**Project:** Brahmastra | **Duration:** 10 weeks (5+5) | **Team:** 20 devs  
**Prerequisite:** Phase 37-38 complete (all 6 gap features live in platform)  
**Outputs to:** Phase 41 (Polish + Launch)

---

## Architecture: Separate Repositories

The SDK is a **separate open-source repository**. It has zero dependency on the platform codebase. They communicate exclusively via the platform's existing Public API (43+ endpoints, Phase 19).

```
GitHub Organization: github.com/fastaifoundry
│
├── fastaiagent-sdk              ← NEW: Open source, Apache 2.0
│   ├── fastaiagent/             ← Python package (pip install fastaiagent)
│   ├── tests/
│   ├── docs/
│   ├── benchmarks/
│   ├── examples/
│   ├── fixtures/                ← Canonical format JSON examples (contract tests)
│   ├── pyproject.toml
│   ├── LICENSE                  ← Apache 2.0
│   ├── README.md
│   └── CONTRIBUTING.md
│
└── [platform repo]              ← EXISTING: Private, proprietary
    ├── backend/
    ├── frontend/
    ├── fixtures/                ← Same canonical format JSON (mirror of SDK fixtures)
    └── ...
```

**Communication:**
```
┌─────────────────────────────┐       HTTPS        ┌────────────────────────────┐
│  fastaiagent-sdk            │ ──────────────────→ │  FastAIAgent Platform      │
│  (github.com/fastaifoundry │                     │  (private repo)            │
│   /fastaiagent-sdk)         │  Public API         │                            │
│                             │  POST /api/v1/agents│  API routes (71 files)     │
│  pip install fastaiagent    │  GET /api/v1/traces │  Services (178 files)      │
│                             │  POST /api/v1/chains│  Frontend (94 pages)       │
│  Apache 2.0                 │  ...43+ endpoints   │  Proprietary               │
└─────────────────────────────┘                     └────────────────────────────┘
```

**Rules:**
1. SDK repo NEVER imports platform code. No shared Python packages.
2. Platform repo NEVER imports SDK code. The platform has its own services.
3. All communication is via HTTP (Public API). Same API that external developers use.
4. Canonical JSON format (for push/pull) is the contract. Tested in both repos via shared fixture files.
5. SDK can run 100% standalone with ZERO network calls. Platform connection is optional.

---

## Canonical Format Contract

The SDK and platform must agree on JSON structure for agents, chains, tools, prompts, traces. This is managed via:

1. **Fixture files** — identical JSON examples in both repos (`fixtures/` directory)
2. **Contract tests** — SDK tests verify `to_dict()` output matches fixtures. Platform tests verify API accepts fixture JSON.
3. **Versioned contract** — `fixtures/VERSION` file tracks contract version. Both repos must match.

```
fixtures/
├── VERSION                      # "1.0" — both repos must match
├── agent_simple.json            # Single agent with tools
├── agent_with_guardrails.json   # Agent with guardrails attached
├── chain_simple.json            # Linear chain (A → B → C)
├── chain_cyclic.json            # Chain with cycles and max_iterations
├── chain_typed_state.json       # Chain with Pydantic state schema
├── chain_with_hitl.json         # Chain with approval nodes
├── chain_with_parallel.json     # Chain with parallel execution
├── chain_full.json              # Chain with cycles + guardrails + typed state + HITL
├── tool_function.json           # FunctionTool definition
├── tool_rest.json               # RESTTool definition
├── tool_mcp.json                # MCPTool definition
├── prompt_simple.json           # Prompt with variables
├── prompt_with_fragments.json   # Prompt with fragment composition
├── guardrail_code.json          # Code guardrail
├── guardrail_llm_judge.json     # LLM judge guardrail
├── guardrail_regex.json         # Regex guardrail
├── trace_agent.json             # Agent execution trace with spans
├── trace_chain.json             # Chain execution trace with checkpoints
└── eval_dataset_multiturn.json  # Multi-turn eval dataset
```

**Contract test in SDK repo:**
```python
# tests/test_canonical_format.py

import json
from pathlib import Path
from fastaiagent import Agent, Chain, Tool, Guardrail

FIXTURES = Path(__file__).parent.parent / "fixtures"

def test_agent_serialization_matches_fixture():
    """SDK Agent.to_dict() produces JSON matching the canonical fixture."""
    expected = json.loads((FIXTURES / "agent_simple.json").read_text())
    
    agent = Agent(
        name=expected["name"],
        system_prompt=expected["system_prompt"],
        llm=LLMClient(**expected["llm_endpoint"]),
        tools=[Tool.from_dict(t) for t in expected["tools"]]
    )
    
    actual = agent.to_dict()
    assert actual["name"] == expected["name"]
    assert actual["system_prompt"] == expected["system_prompt"]
    assert actual["tools"] == expected["tools"]
    # ... verify all fields

def test_chain_serialization_matches_fixture():
    """SDK Chain.to_dict() produces JSON matching platform ReactFlow format."""
    expected = json.loads((FIXTURES / "chain_cyclic.json").read_text())
    
    chain = Chain.from_dict(expected)
    actual = chain.to_dict()
    
    assert actual["nodes"] == expected["nodes"]
    assert actual["edges"] == expected["edges"]
    assert actual["state_schema"] == expected.get("state_schema")

def test_chain_roundtrip():
    """Chain → to_dict() → from_dict() → to_dict() produces identical JSON."""
    original = json.loads((FIXTURES / "chain_full.json").read_text())
    chain = Chain.from_dict(original)
    roundtripped = chain.to_dict()
    assert roundtripped == original
```

**Contract test in platform repo:**
```python
# tests/test_sdk_contract.py

import json
from pathlib import Path

FIXTURES = Path(__file__).parent.parent / "fixtures"

async def test_platform_accepts_sdk_chain_format(client, auth_headers):
    """Platform API accepts chain JSON in canonical format."""
    chain_json = json.loads((FIXTURES / "chain_full.json").read_text())
    response = await client.post(
        "/api/v1/chains",
        json=chain_json,
        headers=auth_headers
    )
    assert response.status_code == 201
    
    # Verify it renders (would appear in visual editor)
    chain_id = response.json()["id"]
    get_response = await client.get(f"/api/v1/chains/{chain_id}", headers=auth_headers)
    assert get_response.status_code == 200
    assert get_response.json()["nodes"] == chain_json["nodes"]

async def test_platform_returns_sdk_compatible_format(client, auth_headers):
    """Platform API returns JSON that SDK can deserialize."""
    # Create a chain via platform
    response = await client.post("/api/v1/chains", json={...}, headers=auth_headers)
    chain_data = response.json()
    
    # Verify it has all fields the SDK expects
    assert "nodes" in chain_data
    assert "edges" in chain_data
    for edge in chain_data["edges"]:
        assert "source" in edge
        assert "target" in edge
        assert "is_cyclic" in edge  # Phase 37 field
```

When the canonical format changes (new fields, renamed fields), update:
1. Fixture files in both repos
2. Contract tests in both repos
3. `fixtures/VERSION` bumped in both repos
4. SDK release notes document the format change

---

## Phase 39: SDK Foundation + Complete Extraction

**Duration:** 5 weeks | **Goal:** Working standalone SDK on PyPI

### Team Structure

| Sub-team | Devs | Focus |
|----------|------|-------|
| **SDK Core** | 6 | Repo scaffold, Agent/Chain/Tool/LLM/Guardrail extraction |
| **Tracing + Replay** | 4 | OTel tracing, local storage, Agent Replay, CLI |
| **Prompts + KB + Eval** | 6 | Prompt registry, LocalKB, eval framework, scorers |
| **Integrations + CLI + QA** | 4 | Framework auto-tracing, CLI, packaging, compatibility |

### Week-by-Week

```
Week 1:  Repo scaffold + CI/CD + package structure
         Begin Agent/Chain/Tool/LLM extraction
         Begin OTel tracing extraction

Week 2:  Complete Agent/Tool/LLM extraction with tests
         Chain extraction (with cycles, typed state, checkpointing)
         Guardrail extraction
         Local trace storage (SQLite)

Week 3:  Agent Replay extraction (fork-and-rerun)
         Prompt registry (local mode with fragments)
         LocalKB (file-based with FastEmbed)
         Begin eval framework extraction

Week 4:  Complete eval extraction (multi-turn, trajectory)
         Framework auto-tracing (OpenAI, LangChain, CrewAI, Anthropic)
         CLI commands
         Schema drift detection extraction

Week 5:  Full test suite (200+ tests)
         Standalone verification (100% offline)
         Compatibility testing (Python 3.10-3.13, Linux/macOS/Windows)
         Performance benchmarks
         PyPI v0.1.0-alpha publish
```

---

### 39.1 — Repository Scaffold

**Duration:** 3 days | **Devs:** 4

**Repository:** `github.com/fastaifoundry/fastaiagent-sdk`

**Step 1: Create repository on GitHub**
- Organization: `fastaifoundry`
- Repository name: `fastaiagent-sdk`
- Visibility: Public
- License: Apache 2.0
- Description: "Build, debug, evaluate, and operate AI agents. The only SDK with fork-and-rerun Agent Replay."
- Topics: `ai-agents`, `agent-debugging`, `agent-replay`, `llm-evaluation`, `agent-observability`, `opentelemetry`, `python`

**Step 2: pyproject.toml**

```toml
[project]
name = "fastaiagent"
version = "0.1.0a1"
description = "Build, debug, evaluate, and operate AI agents. The only SDK with fork-and-rerun Agent Replay."
readme = "README.md"
license = {text = "Apache-2.0"}
requires-python = ">=3.10"
authors = [{name = "FastAIFoundry", email = "support@fastaifoundry.com"}]

keywords = [
    "ai-agents", "agent-debugging", "agent-replay", 
    "llm-evaluation", "agent-observability", "opentelemetry"
]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: Apache Software License",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]

dependencies = [
    "pydantic>=2.0",
    "httpx>=0.25",
    "opentelemetry-api>=1.20",
    "opentelemetry-sdk>=1.20",
    "typer>=0.9",
    "rich>=13.0",
]

[project.optional-dependencies]
openai = ["openai>=1.0"]
anthropic = ["anthropic>=0.20"]
langchain = ["langchain-core>=0.2"]
crewai = ["crewai>=1.0"]
kb = ["fastembed>=0.3", "pymupdf>=1.23"]
otel-export = [
    "opentelemetry-exporter-otlp>=1.20",
]
all = ["fastaiagent[openai,anthropic,langchain,crewai,kb,otel-export]"]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "pytest-cov>=5.0",
    "ruff>=0.3",
    "mypy>=1.8",
]

[project.scripts]
fastaiagent = "fastaiagent.cli.main:app"

[project.urls]
Homepage = "https://fastaiagent.net"
Documentation = "https://docs.fastaiagent.net"
Repository = "https://github.com/fastaifoundry/fastaiagent-sdk"
Issues = "https://github.com/fastaifoundry/fastaiagent-sdk/issues"
Changelog = "https://github.com/fastaifoundry/fastaiagent-sdk/blob/main/CHANGELOG.md"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["fastaiagent"]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"

[tool.ruff]
target-version = "py310"
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP"]

[tool.mypy]
python_version = "3.10"
strict = true
warn_unused_ignores = true
```

**Step 3: Complete directory structure**

```
fastaiagent-sdk/
├── .github/
│   ├── workflows/
│   │   ├── ci.yml                    # Test on every PR
│   │   ├── publish.yml               # Publish to PyPI on tag
│   │   └── compatibility.yml         # Weekly full matrix test
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.yml
│   │   ├── feature_request.yml
│   │   └── integration_request.yml
│   ├── PULL_REQUEST_TEMPLATE.md
│   └── FUNDING.yml
│
├── fastaiagent/
│   ├── __init__.py
│   ├── _version.py                   # __version__ = "0.1.0a1"
│   ├── client.py                     # FastAI client class (Phase 40)
│   │
│   ├── agent/
│   │   ├── __init__.py               # exports: Agent, AgentConfig, AgentResult
│   │   ├── agent.py
│   │   ├── team.py                   # Supervisor, Worker
│   │   ├── executor.py               # tool-calling loop
│   │   └── memory.py                 # conversation memory
│   │
│   ├── chain/
│   │   ├── __init__.py               # exports: Chain, ChainResult, ChainState
│   │   ├── chain.py
│   │   ├── node.py                   # NodeType enum, NodeConfig
│   │   ├── state.py                  # ChainState with Pydantic schema
│   │   ├── executor.py               # state-machine executor (cycles, parallel)
│   │   ├── checkpoint.py             # local SQLite checkpointing
│   │   └── validator.py              # cycle detection, schema validation
│   │
│   ├── tool/
│   │   ├── __init__.py               # exports: Tool, FunctionTool, RESTTool, MCPTool
│   │   ├── base.py                   # Tool base, ToolResult
│   │   ├── function.py
│   │   ├── rest.py
│   │   ├── mcp.py                    # MCP client
│   │   └── schema.py                 # schema validation + drift detection
│   │
│   ├── llm/
│   │   ├── __init__.py               # exports: LLMClient, Message
│   │   ├── client.py                 # LLMClient abstraction
│   │   ├── message.py                # SystemMessage, UserMessage, etc.
│   │   └── providers/
│   │       ├── __init__.py
│   │       ├── openai.py
│   │       ├── anthropic.py
│   │       ├── ollama.py
│   │       ├── azure.py
│   │       ├── bedrock.py
│   │       └── custom.py
│   │
│   ├── guardrail/
│   │   ├── __init__.py               # exports: Guardrail, GuardrailResult
│   │   ├── guardrail.py              # Guardrail class
│   │   ├── executor.py               # blocking/parallel execution
│   │   ├── implementations.py        # code, llm_judge, regex, schema runners
│   │   └── builtins.py               # no_pii(), json_valid(), toxicity_check(), etc.
│   │
│   ├── prompt/
│   │   ├── __init__.py               # exports: PromptRegistry, Prompt, Fragment
│   │   ├── registry.py               # local file-based registry
│   │   ├── prompt.py                 # Prompt with {{variable}} substitution
│   │   ├── fragment.py               # Fragment + {{@fragment}} composition
│   │   └── storage.py                # YAML file storage
│   │
│   ├── kb/
│   │   ├── __init__.py               # exports: LocalKB
│   │   ├── local.py                  # LocalKB class
│   │   ├── document.py               # file ingestion (PDF, MD, TXT, DOCX)
│   │   ├── chunking.py               # recursive text chunking
│   │   ├── embedding.py              # FastEmbed, OpenAI embeddings
│   │   └── search.py                 # cosine similarity search
│   │
│   ├── eval/
│   │   ├── __init__.py               # exports: evaluate, Scorer, Dataset, EvalResults
│   │   ├── evaluate.py               # evaluate() main function
│   │   ├── dataset.py                # Dataset (JSONL, CSV, list, dict)
│   │   ├── scorer.py                 # Scorer base + @Scorer.code decorator
│   │   ├── builtins.py               # ExactMatch, Contains, JSONValid, Latency, etc.
│   │   ├── llm_judge.py              # LLMJudge scorer
│   │   ├── session.py                # multi-turn session scorers
│   │   ├── trajectory.py             # trajectory/path scorers
│   │   └── results.py                # EvalResults with summary + export
│   │
│   ├── trace/
│   │   ├── __init__.py               # exports: trace (context manager), TraceStore, Replay
│   │   ├── tracer.py                 # tracing context manager
│   │   ├── span.py                   # span helpers + GenAI semantic conventions
│   │   ├── storage.py                # local SQLite trace storage
│   │   ├── replay.py                 # Agent Replay (step-through, fork, rerun)
│   │   ├── export.py                 # OTel exporter helpers
│   │   └── otel.py                   # OTel SDK setup, custom processors
│   │
│   ├── integrations/
│   │   ├── __init__.py
│   │   ├── openai.py                 # auto-trace for OpenAI SDK (<200 lines)
│   │   ├── langchain.py              # auto-trace for LangChain (<200 lines)
│   │   ├── crewai.py                 # auto-trace for CrewAI (<200 lines)
│   │   └── anthropic.py              # auto-trace for Anthropic SDK (<200 lines)
│   │
│   ├── deploy/                       # Phase 40
│   │   ├── __init__.py               # exports: push, pull
│   │   ├── push.py
│   │   └── pull.py
│   │
│   ├── cli/
│   │   ├── __init__.py
│   │   ├── main.py                   # typer app entry point
│   │   ├── replay.py                 # fastaiagent replay <trace_id>
│   │   ├── eval.py                   # fastaiagent eval run / compare
│   │   ├── traces.py                 # fastaiagent traces list / export
│   │   ├── prompts.py                # fastaiagent prompts list / diff
│   │   └── kb.py                     # fastaiagent kb status / add
│   │
│   ├── _platform/                    # Phase 40 — platform API client
│   │   ├── __init__.py
│   │   ├── api.py                    # HTTP client to Public API
│   │   ├── prompts.py                # PlatformPromptRegistry
│   │   ├── kb.py                     # PlatformKB
│   │   ├── traces.py                 # PlatformTraceStore
│   │   ├── eval.py                   # PlatformEval
│   │   └── cache.py                  # OfflineCache
│   │
│   └── _internal/
│       ├── __init__.py
│       ├── config.py                 # SDK configuration (env vars, defaults)
│       ├── errors.py                 # all custom exception classes
│       ├── serialization.py          # to_dict / from_dict helpers
│       └── storage.py                # SQLite helpers
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                   # shared fixtures (mocked LLM, temp dirs)
│   ├── test_agent.py                 # 25+ tests
│   ├── test_chain.py                 # 40+ tests (cycles, state, checkpoints, resume)
│   ├── test_tool.py                  # 20+ tests
│   ├── test_llm.py                   # 15+ tests (mocked providers)
│   ├── test_guardrail.py             # 25+ tests
│   ├── test_trace.py                 # 20+ tests
│   ├── test_replay.py                # 20+ tests
│   ├── test_prompt.py                # 20+ tests
│   ├── test_kb.py                    # 15+ tests
│   ├── test_eval.py                  # 30+ tests
│   ├── test_cli.py                   # 15+ tests
│   ├── test_integrations.py          # 12+ tests
│   ├── test_canonical_format.py      # contract tests against fixtures
│   └── test_platform_client.py       # 25+ tests (mocked HTTP, Phase 40)
│
├── fixtures/                         # canonical format JSON (shared with platform)
│   ├── VERSION                       # "1.0"
│   ├── agent_simple.json
│   ├── agent_with_guardrails.json
│   ├── chain_simple.json
│   ├── chain_cyclic.json
│   ├── chain_typed_state.json
│   ├── chain_with_hitl.json
│   ├── chain_with_parallel.json
│   ├── chain_full.json
│   ├── tool_function.json
│   ├── tool_rest.json
│   ├── tool_mcp.json
│   ├── prompt_simple.json
│   ├── prompt_with_fragments.json
│   ├── guardrail_code.json
│   ├── guardrail_llm_judge.json
│   ├── guardrail_regex.json
│   ├── trace_agent.json
│   ├── trace_chain.json
│   └── eval_dataset_multiturn.json
│
├── examples/
│   ├── 01_simple_agent.py            # Minimal agent with one tool
│   ├── 02_chain_with_cycles.py       # Retry loop pattern
│   ├── 03_guardrails.py              # Input/output/tool guardrails
│   ├── 04_agent_replay.py            # Fork-and-rerun debugging
│   ├── 05_prompt_fragments.py        # Modular prompt composition
│   ├── 06_rag_agent.py               # Agent with LocalKB
│   ├── 07_eval_pipeline.py           # Evaluate agent with scorers
│   ├── 08_trace_langchain.py         # Trace a LangChain agent
│   ├── 09_otel_export.py             # Export traces to Jaeger
│   └── 10_platform_sync.py           # Push/pull with platform (Phase 40)
│
├── benchmarks/
│   ├── bench_trace_overhead.py       # Target: <5% overhead
│   ├── bench_checkpoint_overhead.py  # Target: <50ms per node
│   ├── bench_cycle_performance.py    # Target: 100 iterations <10s
│   ├── bench_replay_load.py          # Target: 1000-span trace <2s
│   ├── bench_local_kb.py             # Target: search <500ms
│   └── bench_eval.py                 # Target: 100 cases <60s
│
├── docs/                             # MkDocs source (Phase 41)
│   └── ...
│
├── .gitignore
├── .python-version                   # "3.10"
├── LICENSE                           # Apache 2.0
├── README.md
├── CHANGELOG.md
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
└── py.typed                          # PEP 561 marker for type checkers
```

**Step 4: CI/CD pipelines**

```yaml
# .github/workflows/ci.yml
name: CI
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        python: ["3.10", "3.11", "3.12", "3.13"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python }}
      - name: Install dependencies
        run: pip install -e ".[dev,all]"
      - name: Run tests
        run: pytest tests/ -v --tb=short --cov=fastaiagent --cov-report=xml
      - name: Upload coverage
        if: matrix.os == 'ubuntu-latest' && matrix.python == '3.12'
        uses: codecov/codecov-action@v4

  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install ruff mypy
      - run: ruff check .
      - run: ruff format --check .

  type-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install -e ".[dev,all]"
      - run: mypy fastaiagent/
```

```yaml
# .github/workflows/publish.yml
name: Publish to PyPI
on:
  push:
    tags: ["v*"]

jobs:
  publish:
    runs-on: ubuntu-latest
    environment: pypi
    permissions:
      id-token: write  # for trusted publishing
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Install build tools
        run: pip install build
      - name: Build package
        run: python -m build
      - name: Publish to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1
```

**Step 5: README.md (positioning-first, NOT primitives-first)**

```markdown
# FastAIAgent SDK

**Build, debug, evaluate, and operate AI agents.**  
The only SDK with **Agent Replay** — fork-and-rerun debugging for AI agents.

Works standalone or connected to the [FastAIAgent Platform](https://fastaiagent.net) for visual editing, production monitoring, and team collaboration.

[![PyPI](https://img.shields.io/pypi/v/fastaiagent)](https://pypi.org/project/fastaiagent/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Tests](https://github.com/fastaifoundry/fastaiagent-sdk/actions/workflows/ci.yml/badge.svg)](https://github.com/fastaifoundry/fastaiagent-sdk/actions)
[![Python](https://img.shields.io/pypi/pyversions/fastaiagent)](https://pypi.org/project/fastaiagent/)

---

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
# ✅ Fixed.
```

**No other SDK can do this.**

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

## Connect to FastAIAgent Platform (optional)

```python
from fastaiagent import FastAI

fa = FastAI(api_key="sk-...", project="customer-support")

# Push your agent to the platform — see it in the visual editor
fa.push(chain)

# Traces appear in the platform dashboard
# Prompts sync between code and platform
# Eval results visible in the platform
```

**SDK works standalone. Platform adds: visual chain editor, production monitoring,
advanced KB intelligence, prompt optimization, team collaboration, HITL approval workflows.**

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

- [Getting Started](https://docs.fastaiagent.net/getting-started/)
- [Agent Replay Guide](https://docs.fastaiagent.net/replay/)
- [Building Chains with Cycles](https://docs.fastaiagent.net/chains/cyclic-workflows/)
- [Guardrails](https://docs.fastaiagent.net/guardrails/)
- [Evaluation](https://docs.fastaiagent.net/evaluation/)
- [API Reference](https://docs.fastaiagent.net/api-reference/)

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Apache 2.0 — see [LICENSE](LICENSE).
```

**Tasks for 39.1:**
- [ ] Create repository on GitHub under `fastaifoundry` organization
- [ ] Push all config files: pyproject.toml, LICENSE, README, CONTRIBUTING, CODE_OF_CONDUCT
- [ ] Create complete directory structure with all `__init__.py` files
- [ ] Set up CI workflow (runs on PR, tests on 4 Python versions × 3 OS)
- [ ] Set up publish workflow (tag → PyPI)
- [ ] Set up branch protection on `main` (require PR + CI)
- [ ] Create fixture files directory with VERSION file
- [ ] Create examples directory with placeholder files
- [ ] Verify `pip install -e .` works in clean virtualenv
- [ ] Verify `fastaiagent --help` CLI works
- [ ] Verify CI pipeline passes (with minimal placeholder test)

---

### 39.2 — Core Extraction: Agent + Chain + Tool + LLM + Guardrails

**Duration:** 3 weeks | **Devs:** 6

**Extraction methodology — apply to every component:**

1. **Identify** the service classes in the platform codebase
2. **Copy** the business logic into the SDK module
3. **Strip all platform dependencies:**
   - SQLAlchemy models → Pydantic models or dataclasses
   - FastAPI deps (Depends, Request, Response) → pure Python
   - Database queries → local storage (SQLite, JSON, in-memory)
   - Async-only methods → provide both sync and async interfaces
   - Platform config → SDK config (env vars, explicit params)
4. **Add serialization:** `to_dict()` and `from_dict()` using canonical format
5. **Verify** against fixture files (contract tests)
6. **Write SDK-specific tests** (mocked LLM providers, no real API calls in CI)

**Agent extraction source files (platform):**
```
backend/app/models/agent.py           → agent model fields
backend/app/agents/agent_service.py   → agent execution logic
backend/app/agents/executor.py        → tool-calling loop
backend/app/agents/memory_service.py  → conversation memory
backend/app/schemas/agent_schemas.py  → request/response models
```

**SDK Agent target:**
```python
# fastaiagent/agent/agent.py

from pydantic import BaseModel, Field
from typing import Optional
from fastaiagent.llm import LLMClient
from fastaiagent.tool import Tool
from fastaiagent.guardrail import Guardrail
from fastaiagent.trace import get_tracer

class AgentConfig(BaseModel):
    max_iterations: int = Field(default=10, ge=1, le=100)
    tool_choice: str = "auto"  # "auto", "required", "none"
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None

class AgentResult(BaseModel):
    output: str
    tool_calls: list[dict] = []
    tokens_used: int = 0
    cost: float = 0.0
    latency_ms: int = 0
    trace: Optional["Trace"] = None

class Agent:
    """An AI agent with tools, guardrails, and full tracing."""
    
    def __init__(
        self,
        name: str,
        system_prompt: str = "",
        llm: LLMClient | None = None,
        tools: list[Tool] | None = None,
        guardrails: list[Guardrail] | None = None,
        memory: "AgentMemory | None" = None,
        config: AgentConfig | None = None,
    ):
        self.name = name
        self.system_prompt = system_prompt
        self.llm = llm or LLMClient()
        self.tools = tools or []
        self.guardrails = guardrails or []
        self.memory = memory
        self.config = config or AgentConfig()
    
    def run(self, input: str, *, trace: bool = True, **kwargs) -> AgentResult:
        """Synchronous execution."""
        import asyncio
        return asyncio.run(self.arun(input, trace=trace, **kwargs))
    
    async def arun(self, input: str, *, trace: bool = True, **kwargs) -> AgentResult:
        """Async execution with tool-calling loop."""
        tracer = get_tracer() if trace else None
        
        # Execute input guardrails (blocking)
        if self.guardrails:
            await self._execute_guardrails("input", input)
        
        # Tool-calling loop (extracted from platform executor)
        messages = self._build_messages(input)
        result = await self._execute_loop(messages, tracer)
        
        # Execute output guardrails
        if self.guardrails:
            await self._execute_guardrails("output", result.output)
        
        return result
    
    async def stream(self, input: str, **kwargs):
        """Streaming execution (token-by-token + tool calls)."""
        # ... SSE-compatible streaming
        pass
    
    def to_dict(self) -> dict:
        """Serialize to canonical format for platform push."""
        return {
            "name": self.name,
            "agent_type": "single",
            "system_prompt": self.system_prompt,
            "llm_endpoint": self.llm.to_dict(),
            "tool_ids": [t.to_dict() for t in self.tools],
            "config": self.config.model_dump(),
            # guardrails serialized separately
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Agent":
        """Deserialize from canonical format (platform pull)."""
        return cls(
            name=data["name"],
            system_prompt=data.get("system_prompt", ""),
            llm=LLMClient.from_dict(data.get("llm_endpoint", {})),
            tools=[Tool.from_dict(t) for t in data.get("tool_ids", [])],
            config=AgentConfig(**data.get("config", {})),
        )
```

**Chain extraction source files (platform):**
```
backend/app/models/chain.py                    → chain model, edge model
backend/app/models/chain_checkpoint.py         → Phase 37 checkpoint model
backend/app/chains/executor.py                 → state-machine executor (Phase 37)
backend/app/chains/validator.py                → cycle detection, validation (Phase 37)
backend/app/chains/checkpoint_service.py       → Phase 37 checkpoint service
backend/app/chains/state_service.py            → Phase 38 typed state
```

SDK Chain must include ALL Phase 37-38 features:
- Directed graph with cycles (not DAG)
- Typed state via Pydantic schema
- Local SQLite checkpointing
- Resume-from-failure with state modification
- Guardrail integration at chain level
- HITL approval nodes (local: CLI prompt)
- Parallel execution nodes
- Conditional branching

**Guardrail extraction source files (platform):**
```
backend/app/models/guardrail.py               → Phase 38 guardrail model
backend/app/agents/guardrail_service.py       → Phase 38 execution service
```

SDK Guardrail includes all 5 implementation types:
- Code (sandboxed Python function)
- LLM Judge (prompt-based validation)
- Regex (pattern matching)
- Schema (JSON Schema validation)
- Classifier (toxicity, PII, sentiment)

Plus built-in factories: `no_pii()`, `json_valid()`, `toxicity_check()`, `cost_limit()`, `allowed_domains()`

**LLM Client extraction:**
```
backend/app/shared/llm_client.py              → provider abstraction
```

6 providers: OpenAI, Anthropic, Ollama, Azure, Bedrock, Custom endpoint.

**Tool extraction:**
```
backend/app/models/tool.py                    → tool model
backend/app/shared/tool_executor.py           → REST/function/MCP execution
backend/app/shared/schema_drift.py            → schema drift detection
backend/app/mcp_hosting/                      → MCP client
```

**Tasks for 39.2:**
- [ ] Extract Agent class with tool-calling loop, sync/async/stream — 25+ tests
- [ ] Extract Chain class with cycles, typed state, checkpointing, resume — 40+ tests
- [ ] Extract Tool (FunctionTool, RESTTool, MCPTool) — 20+ tests
- [ ] Extract LLMClient (6 providers, mocked in tests) — 15+ tests
- [ ] Extract Guardrail (5 types + built-in factories) — 25+ tests
- [ ] Extract Schema Drift Detection — 5+ tests
- [ ] Extract Supervisor/Worker teams — 5+ tests
- [ ] All `to_dict()` / `from_dict()` verified against fixture files
- [ ] Contract tests passing against all fixture JSONs

---

### 39.3 — Tracing + Replay Extraction

**Duration:** 2 weeks | **Devs:** 4

**Source files (platform):**
```
backend/app/shared/tracing/otel_bridge.py      → Phase 38 OTel bridge
backend/app/shared/tracing/processors.py       → Phase 38 span processors
backend/app/models/trace.py                    → trace/span models
backend/app/agents/replay_service.py           → Agent Replay service
```

**SDK tracing architecture (OTel-native from day one):**

```python
# fastaiagent/trace/otel.py

from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter

_provider: TracerProvider | None = None

def get_tracer_provider() -> TracerProvider:
    global _provider
    if _provider is None:
        _provider = TracerProvider()
        # Always add local storage by default
        _provider.add_span_processor(
            LocalStorageProcessor()  # writes to .fastaiagent/traces.db
        )
        otel_trace.set_tracer_provider(_provider)
    return _provider

def get_tracer(name: str = "fastaiagent") -> otel_trace.Tracer:
    return get_tracer_provider().get_tracer(name, version=__version__)

def add_exporter(exporter: SpanExporter):
    """Add any OTel-compatible exporter (Datadog, Jaeger, etc.)."""
    get_tracer_provider().add_span_processor(
        BatchSpanProcessor(exporter)
    )
```

**Local storage processor:**
```python
# fastaiagent/trace/storage.py

class LocalStorageProcessor(SpanProcessor):
    """Write OTel spans to local SQLite database."""
    
    def __init__(self, db_path: str = ".fastaiagent/traces.db"):
        self.db_path = db_path
        self._init_db()
    
    def on_end(self, span: ReadableSpan):
        """Called when a span completes — write to SQLite."""
        self._write_span(span)

class TraceStore:
    """Query interface for local trace storage."""
    
    def __init__(self, db_path: str = ".fastaiagent/traces.db"):
        self.db_path = db_path
    
    def get_trace(self, trace_id: str) -> TraceData: ...
    def list_traces(self, last_hours: int = 24, **filters) -> list[TraceSummary]: ...
    def search(self, query: str = None, **filters) -> list[TraceSummary]: ...
    def export(self, trace_id: str, format: str = "json") -> str: ...
```

**GenAI semantic conventions mapping:**
```python
# fastaiagent/trace/span.py

# Maps FastAIAgent concepts to OTel GenAI semantic conventions
GENAI_ATTRIBUTES = {
    "gen_ai.system": str,            # "openai", "anthropic", "ollama"
    "gen_ai.request.model": str,     # "gpt-4o", "claude-sonnet-4-20250514"
    "gen_ai.request.temperature": float,
    "gen_ai.request.max_tokens": int,
    "gen_ai.usage.input_tokens": int,
    "gen_ai.usage.output_tokens": int,
    "gen_ai.response.finish_reasons": list,
}

# FastAIAgent custom attributes (namespaced)
FASTAI_ATTRIBUTES = {
    "fastai.agent.name": str,
    "fastai.chain.name": str,
    "fastai.chain.node_id": str,
    "fastai.chain.iteration": int,
    "fastai.tool.name": str,
    "fastai.checkpoint.id": str,
    "fastai.guardrail.name": str,
    "fastai.guardrail.passed": bool,
    "fastai.prompt.name": str,
    "fastai.prompt.version": int,
    "fastai.cost.total_usd": float,
}
```

**Agent Replay extraction:**
```python
# fastaiagent/trace/replay.py

class Replay:
    """Replay and debug agent/chain executions from traces."""
    
    @classmethod
    def load(cls, trace_id: str, store: TraceStore | None = None) -> "Replay":
        store = store or TraceStore.default()
        trace_data = store.get_trace(trace_id)
        return cls(trace_data)
    
    def summary(self) -> str: ...
    def steps(self) -> list[ReplayStep]: ...
    def inspect(self, step: int) -> ReplayStep: ...
    
    def fork_at(self, step: int) -> "ForkedReplay":
        """Fork the execution at a specific step for re-execution."""
        return ForkedReplay(
            original_trace=self._trace,
            fork_point=step,
            state_at_fork=self._get_state_at_step(step)
        )

class ForkedReplay:
    """A forked execution that can be modified and rerun."""
    
    def modify_input(self, new_input: dict): ...
    def modify_prompt(self, new_prompt: str): ...
    def modify_config(self, **kwargs): ...
    def modify_state(self, new_state: dict): ...
    
    async def rerun(self) -> "ReplayResult":
        """Rerun from fork point with modifications."""
        # Reconstruct agent/chain from trace metadata
        # Apply modifications
        # Execute from fork point
        # Return comparison-ready result
        ...
    
    def compare(self, original: Replay) -> "ComparisonResult": ...
```

**Tasks for 39.3:**
- [ ] OTel tracer provider with local SQLite processor — 8+ tests
- [ ] TraceStore with query, list, search, export — 8+ tests
- [ ] GenAI semantic convention attribute mapping — 4+ tests
- [ ] Agent Replay: load, summary, steps, inspect — 8+ tests
- [ ] ForkedReplay: modify_input, modify_prompt, modify_state, rerun, compare — 12+ tests
- [ ] CLI: `fastaiagent traces list`, `fastaiagent replay` with interactive mode — 5+ tests
- [ ] OTel export helper: `add_exporter()` works with OTLP — 3+ tests
- [ ] Context manager: `with fastaiagent.trace("name"):` — 3+ tests

---

### 39.4 — Prompts + KB + Eval Extraction

**Duration:** 2 weeks | **Devs:** 6

**Prompt Registry (local mode):**

Source: `backend/app/prompts/` domain package

SDK: File-based YAML storage in `.prompts/` directory. Fragment composition with `{{@fragment}}`. Versioning, aliases, diff. See Phase 38 plan for full API design.

**LocalKB:**

Source: `backend/app/kb/` domain package (basic parts only, NOT the advanced intelligence from Phases 33-36)

SDK: File ingestion (PDF via PyMuPDF, MD, TXT, DOCX), recursive chunking, FastEmbed local embeddings, cosine similarity search, `kb.as_tool()` for agent integration.

**Eval Framework:**

Source: `backend/app/evals/` domain package

SDK: `evaluate()` function, Dataset class, built-in scorers, LLMJudge, multi-turn session scorers (Phase 38), trajectory scorers (Phase 38), custom code scorers, results with summary and export.

**Tasks for 39.4:**
- [ ] PromptRegistry: local YAML storage, fragments, versioning, aliases, diff — 20+ tests
- [ ] LocalKB: file ingestion, chunking, embedding, search, as_tool() — 15+ tests
- [ ] Eval: evaluate(), Dataset, built-in scorers, LLMJudge — 15+ tests
- [ ] Eval: session scorers, trajectory scorers — 10+ tests
- [ ] CLI: `fastaiagent prompts list/diff`, `fastaiagent kb status/add`, `fastaiagent eval run/compare` — 10+ tests

---

### 39.5 — Framework Auto-Tracing Integrations

**Duration:** 1 week | **Devs:** 4

Four thin wrappers. Each < 200 lines, single file. Uses monkey-patching or framework callbacks to capture OTel spans automatically.

**OpenAI** (`integrations/openai.py`): Patch `openai.resources.chat.completions.Completions.create` and `.acreate`.

**Anthropic** (`integrations/anthropic.py`): Patch `anthropic.Anthropic.messages.create`.

**LangChain** (`integrations/langchain.py`): Register a `BaseCallbackHandler` that emits OTel spans.

**CrewAI** (`integrations/crewai.py`): Register CrewAI callback handler.

**Manual fallback:** `with fastaiagent.trace("name"):` context manager always works, regardless of framework.

**`fastaiagent.init()` shorthand:**
```python
# fastaiagent/__init__.py

def init(api_key: str = None, target: str = "https://app.fastaiagent.net", project: str = None):
    """Quick setup: connect to platform + enable auto-tracing."""
    if api_key:
        from fastaiagent.client import FastAI
        _default_client = FastAI(api_key=api_key, target=target, project=project)
    # Does NOT auto-enable framework integrations
    # User must explicitly call fastaiagent.integrations.openai.enable()
```

**Tasks for 39.5:**
- [ ] OpenAI auto-trace wrapper — 3+ tests
- [ ] Anthropic auto-trace wrapper — 3+ tests
- [ ] LangChain callback tracer — 3+ tests
- [ ] CrewAI callback tracer — 3+ tests
- [ ] `fastaiagent.init()` shorthand — 2+ tests
- [ ] Documentation for each integration (in README + dedicated docs page)

---

### 39.6 — Standalone Verification + PyPI Alpha Publish

**Duration:** 1 week | **All 20 devs**

**Verification checklist (EVERY item must pass before publishing):**

Standalone mode (ZERO network calls):
- [ ] Agent execution with tools (mocked LLM in CI, real LLM in manual test)
- [ ] Chain execution with cycles, checkpointing, resume
- [ ] Guardrails: all 5 implementation types
- [ ] Typed state validation at each chain node
- [ ] Tracing produces OTel spans in local SQLite
- [ ] Agent Replay: load, step-through, fork, rerun from CLI
- [ ] Prompts: register, version, fragment composition, aliases, diff
- [ ] LocalKB: add files, search, as_tool()
- [ ] Eval: evaluate() with built-in + custom + LLM judge scorers
- [ ] Eval: multi-turn session evaluation
- [ ] Eval: trajectory evaluation
- [ ] CLI: all commands return expected output
- [ ] Schema drift detection works on tool responses
- [ ] Framework auto-tracing works (with each framework pip-installed)
- [ ] OTel export: `add_exporter()` sends spans to OTLP endpoint

Performance benchmarks:
- [ ] Trace overhead < 5% of execution time
- [ ] Checkpoint overhead < 50ms per node
- [ ] 100-iteration cycle completes in < 10s (excluding LLM calls)
- [ ] LocalKB search over 1,000 chunks < 500ms
- [ ] Eval of 100 test cases < 60s (excluding LLM calls)
- [ ] Replay load of 1,000-span trace < 2s

Compatibility:
- [ ] Python 3.10, 3.11, 3.12, 3.13 — all tests pass
- [ ] Linux (Ubuntu 22.04+), macOS (13+), Windows (10+) — all tests pass
- [ ] Core only (no optional deps): all core features work, optional imports fail gracefully
- [ ] Each optional extra works when installed independently

Contract:
- [ ] All fixture files validated against `to_dict()` output
- [ ] Canonical format roundtrip: `from_dict(to_dict(x)) == x` for all types
- [ ] Contract VERSION matches across SDK and platform repos

**Publish:**
```bash
# Tag the release
git tag v0.1.0a1
git push origin v0.1.0a1
# → CI publishes to PyPI automatically
```

Verify: `pip install fastaiagent==0.1.0a1` works in a clean environment.

**Phase 39 Definition of Done:**
- [ ] Repository live at `github.com/fastaifoundry/fastaiagent-sdk`
- [ ] 200+ tests passing in CI across all platforms
- [ ] `pip install fastaiagent` works from PyPI (v0.1.0-alpha)
- [ ] All features working standalone (zero network dependency)
- [ ] All Phase 37-38 features (cycles, checkpoints, guardrails, typed state, OTel, multi-turn eval) present in SDK
- [ ] All pre-existing features (agents, chains, tools, prompts, KB, eval, replay) present in SDK
- [ ] CLI working: replay, eval, traces, prompts, kb
- [ ] 4 framework integrations working (OpenAI, LangChain, CrewAI, Anthropic)
- [ ] OTel export to external collectors verified
- [ ] Performance benchmarks met
- [ ] Contract tests passing against fixture files
- [ ] All 10 example scripts run successfully

---

## Phase 40: Platform Connection + Bidirectional Sync

**Duration:** 5 weeks | **Goal:** SDK ↔ platform seamless bridge

### Team Structure

| Sub-team | Devs | Focus |
|----------|------|-------|
| **Platform Client** | 5 | FastAI client, API integration, offline cache |
| **Push** | 5 | SDK → Platform sync (agents, chains, tools, prompts) |
| **Pull + Export** | 5 | Platform → SDK sync, Export as Python, trace pull |
| **Migration + Integration Test** | 5 | Migration wizard, full round-trip testing |

### Week-by-Week

```
Week 1:  FastAI client class
         Platform prompt registry via SDK
         Platform trace push (async)

Week 2:  Platform KB via SDK (proxy to full intelligence)
         Platform eval via SDK (run, pull datasets, push results)
         Offline cache implementation

Week 3:  Push: agents, chains (with all Phase 37-38 features), tools → platform
         Verify chains render correctly in visual editor after push

Week 4:  Pull: platform → functional SDK objects
         Pull traces for local replay
         Export as Python (platform-side code generator + API endpoint + UI button)

Week 5:  Migration wizard (trace analysis → native chain scaffolding)
         Full round-trip integration testing (20 devs, all hands)
```

---

### 40.1 — FastAI Client

**File:** `fastaiagent/client.py`

```python
from fastaiagent._platform.api import PlatformAPI
from fastaiagent._platform.cache import OfflineCache
from fastaiagent._platform.prompts import PlatformPromptRegistry
from fastaiagent._platform.kb import PlatformKB
from fastaiagent._platform.traces import PlatformTraceStore
from fastaiagent._platform.eval import PlatformEval
from fastaiagent.trace.otel import get_tracer_provider

class FastAI:
    """Connect the SDK to the FastAIAgent platform.
    
    Works with any tier: Free, Pro, or Enterprise.
    Also works with self-hosted instances (change target URL).
    
    Example:
        fa = FastAI(api_key="sk-...", project="customer-support")
        
        # Load prompts from platform
        prompt = fa.prompts.load("support-classifier", alias="production")
        
        # Search platform KB (full intelligence: hybrid search, reranking, etc.)
        results = fa.kb("product-docs").search("refund policy")
        
        # Push an agent to the platform (appears in visual editor)
        fa.push(my_agent)
        
        # Pull a production trace for local debugging
        trace = fa.traces.pull("trace_abc123")
    """
    
    def __init__(
        self,
        api_key: str,
        target: str = "https://app.fastaiagent.net",
        project: str | None = None,
        offline_cache: bool = True,
        timeout: int = 30,
    ):
        self._api = PlatformAPI(
            api_key=api_key,
            base_url=target,
            timeout=timeout,
        )
        self.project = project
        self._cache = OfflineCache() if offline_cache else None
        self._target = target
        
        # Add platform as a trace export destination
        from fastaiagent.trace.otel import get_tracer_provider
        from fastaiagent._platform.traces import PlatformSpanProcessor
        provider = get_tracer_provider()
        provider.add_span_processor(
            PlatformSpanProcessor(self._api, project)
        )
    
    # --- Properties for platform-backed features ---
    
    @property
    def prompts(self) -> PlatformPromptRegistry:
        """Access the platform prompt registry."""
        return PlatformPromptRegistry(self._api, self.project, self._cache)
    
    def kb(self, name: str) -> PlatformKB:
        """Access a platform knowledge base (full intelligence)."""
        return PlatformKB(self._api, self.project, name)
    
    @property
    def eval(self) -> PlatformEval:
        """Access platform evaluation features."""
        return PlatformEval(self._api, self.project)
    
    @property
    def traces(self) -> PlatformTraceStore:
        """Access platform trace storage."""
        return PlatformTraceStore(self._api, self.project)
    
    # --- Push/Pull ---
    
    def push(self, resource, **kwargs) -> "PushResult":
        """Push an Agent, Chain, or Tool to the platform.
        
        The resource appears in the platform UI (visual editor for chains,
        agent list for agents, tool library for tools).
        
        Dependencies are resolved automatically:
        - Pushing a chain auto-pushes its agents
        - Pushing an agent auto-pushes its tools and guardrails
        - Pushing a prompt auto-pushes its fragments
        """
        from fastaiagent.deploy.push import push_resource
        return push_resource(self._api, self.project, resource, **kwargs)
    
    def push_all(self, resources: list = None) -> list["PushResult"]:
        """Push multiple resources to the platform."""
        from fastaiagent.deploy.push import push_all
        return push_all(self._api, self.project, resources)
    
    def pull_agent(self, name: str) -> "Agent":
        """Pull an agent from the platform as a functional SDK object."""
        from fastaiagent.deploy.pull import pull_agent
        return pull_agent(self._api, self.project, name)
    
    def pull_chain(self, name: str) -> "Chain":
        """Pull a chain from the platform as a functional SDK object."""
        from fastaiagent.deploy.pull import pull_chain
        return pull_chain(self._api, self.project, name)
    
    def pull_tool(self, name: str) -> "Tool":
        """Pull a tool definition from the platform."""
        from fastaiagent.deploy.pull import pull_tool
        return pull_tool(self._api, self.project, name)
    
    # --- OTel export ---
    
    def add_trace_exporter(self, exporter):
        """Add an OTel exporter for traces (Datadog, Jaeger, etc.)."""
        from fastaiagent.trace.otel import get_tracer_provider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        get_tracer_provider().add_span_processor(
            BatchSpanProcessor(exporter)
        )
    
    # --- Context manager for traces ---
    
    def trace(self, name: str):
        """Context manager that creates a trace sent to the platform."""
        from fastaiagent.trace.tracer import trace_context
        return trace_context(name)
```

**Platform API Client:**

```python
# fastaiagent/_platform/api.py

import httpx
from fastaiagent._internal.errors import (
    PlatformAuthError, PlatformTierLimitError, 
    PlatformNotFoundError, PlatformConnectionError
)

class PlatformAPI:
    """HTTP client to FastAIAgent platform Public API."""
    
    def __init__(self, api_key: str, base_url: str, timeout: int = 30):
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "User-Agent": f"fastaiagent-sdk/{__version__}",
                "Content-Type": "application/json",
            },
            timeout=timeout,
        )
        self._async_client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "User-Agent": f"fastaiagent-sdk/{__version__}",
                "Content-Type": "application/json",
            },
            timeout=timeout,
        )
    
    def _handle_response(self, response: httpx.Response) -> dict:
        """Handle API response, raise appropriate SDK errors."""
        if response.status_code == 401:
            raise PlatformAuthError(
                "Invalid API key. Check your key at https://app.fastaiagent.net/settings/api-keys"
            )
        elif response.status_code == 403:
            body = response.json()
            if "tier" in body.get("detail", "").lower():
                raise PlatformTierLimitError(
                    f"Tier limit reached: {body.get('detail', 'Unknown')}. "
                    f"Upgrade at https://app.fastaiagent.net/billing"
                )
            raise PlatformAuthError(f"Forbidden: {body.get('detail', 'Unknown')}")
        elif response.status_code == 404:
            raise PlatformNotFoundError(f"Resource not found: {response.url}")
        elif response.status_code == 429:
            raise PlatformRateLimitError(
                "Rate limit exceeded. Slow down or upgrade your tier."
            )
        elif response.status_code >= 500:
            raise PlatformConnectionError(
                f"Platform server error ({response.status_code}). "
                f"Check status at https://status.fastaiagent.net"
            )
        
        response.raise_for_status()
        return response.json()
    
    # Sync methods
    def get(self, path: str, **params) -> dict:
        try:
            response = self._client.get(path, params=params)
            return self._handle_response(response)
        except httpx.ConnectError:
            raise PlatformConnectionError(
                "Cannot connect to platform. Check your internet connection "
                "and verify the target URL is correct."
            )
    
    def post(self, path: str, data: dict) -> dict:
        try:
            response = self._client.post(path, json=data)
            return self._handle_response(response)
        except httpx.ConnectError:
            raise PlatformConnectionError("Cannot connect to platform.")
    
    def put(self, path: str, data: dict) -> dict:
        response = self._client.put(path, json=data)
        return self._handle_response(response)
    
    def delete(self, path: str) -> dict:
        response = self._client.delete(path)
        return self._handle_response(response)
    
    # Async methods
    async def aget(self, path: str, **params) -> dict: ...
    async def apost(self, path: str, data: dict) -> dict: ...
    async def aput(self, path: str, data: dict) -> dict: ...
    async def adelete(self, path: str) -> dict: ...
    
    # --- Convenience methods for specific resources ---
    
    # Agents
    def list_agents(self, project: str) -> list[dict]:
        return self.get(f"/api/v1/projects/{project}/agents")
    
    def create_agent(self, project: str, data: dict) -> dict:
        return self.post(f"/api/v1/projects/{project}/agents", data)
    
    def get_agent(self, project: str, name: str) -> dict:
        agents = self.list_agents(project)
        match = [a for a in agents if a["name"] == name]
        if not match:
            raise PlatformNotFoundError(f"Agent '{name}' not found in project '{project}'")
        return match[0]
    
    # Chains
    def create_chain(self, project: str, data: dict) -> dict:
        return self.post(f"/api/v1/projects/{project}/chains", data)
    
    def get_chain(self, project: str, name: str) -> dict:
        chains = self.get(f"/api/v1/projects/{project}/chains")
        match = [c for c in chains if c["name"] == name]
        if not match:
            raise PlatformNotFoundError(f"Chain '{name}' not found in project '{project}'")
        return match[0]
    
    # Traces
    def push_trace(self, project: str, trace_data: dict) -> dict:
        return self.post(f"/api/v1/projects/{project}/traces", trace_data)
    
    def get_trace(self, trace_id: str) -> dict:
        return self.get(f"/api/v1/traces/{trace_id}")
    
    # Prompts
    def load_prompt(self, project: str, name: str, version: int = None, alias: str = None) -> dict:
        params = {}
        if version: params["version"] = version
        if alias: params["alias"] = alias
        return self.get(f"/api/v1/projects/{project}/prompts/{name}", **params)
    
    def register_prompt(self, project: str, data: dict) -> dict:
        return self.post(f"/api/v1/projects/{project}/prompts", data)
    
    # KB
    def kb_search(self, project: str, kb_name: str, query: str, top_k: int = 5) -> dict:
        return self.post(f"/api/v1/projects/{project}/kb/{kb_name}/search", {
            "query": query, "top_k": top_k
        })
    
    # Eval
    def run_eval(self, project: str, data: dict) -> dict:
        return self.post(f"/api/v1/projects/{project}/evals/runs", data)
    
    def pull_dataset(self, project: str, name: str) -> dict:
        return self.get(f"/api/v1/projects/{project}/evals/datasets/{name}")
```

**Offline Cache:**

```python
# fastaiagent/_platform/cache.py

class OfflineCache:
    """Local cache for platform data. Used when platform is unreachable."""
    
    def __init__(self, cache_dir: str = ".fastaiagent/cache/"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get(self, key: str) -> dict | None:
        """Get cached data. Returns None if expired or missing."""
        path = self.cache_dir / f"{key}.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        if data.get("expires_at") and datetime.fromisoformat(data["expires_at"]) < datetime.utcnow():
            return None  # expired
        return data.get("value")
    
    def set(self, key: str, value: dict, ttl_seconds: int = 3600):
        """Cache data with TTL."""
        path = self.cache_dir / f"{key}.json"
        expires_at = (datetime.utcnow() + timedelta(seconds=ttl_seconds)).isoformat()
        path.write_text(json.dumps({"value": value, "expires_at": expires_at}))
    
    def buffer_trace(self, trace_data: dict):
        """Buffer a trace for later push when platform is reachable."""
        buffer_dir = self.cache_dir / "trace_buffer"
        buffer_dir.mkdir(exist_ok=True)
        path = buffer_dir / f"{trace_data['id']}.json"
        path.write_text(json.dumps(trace_data))
    
    def flush_buffered_traces(self, api: PlatformAPI, project: str) -> int:
        """Push buffered traces to platform. Returns count of flushed traces."""
        buffer_dir = self.cache_dir / "trace_buffer"
        if not buffer_dir.exists():
            return 0
        count = 0
        for path in buffer_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text())
                api.push_trace(project, data)
                path.unlink()
                count += 1
            except Exception:
                pass  # will retry next time
        return count
```

**Tasks for 40.1:**
- [ ] FastAI client class with all properties and methods — 5+ tests
- [ ] PlatformAPI HTTP client with error handling — 10+ tests (mocked)
- [ ] PlatformPromptRegistry: load, register, push, pull — 5+ tests
- [ ] PlatformKB: search (proxy), add_documents — 3+ tests
- [ ] PlatformEval: run, pull_dataset, push_results — 3+ tests
- [ ] PlatformTraceStore: push (async), pull, search — 5+ tests
- [ ] PlatformSpanProcessor: sends OTel spans to platform — 3+ tests
- [ ] OfflineCache: get/set with TTL, trace buffering, flush — 5+ tests
- [ ] Tier-aware error messages for all limit errors — 3+ tests
- [ ] Verify: platform unreachable → graceful degradation → offline cache used

---

### 40.2 — Push: SDK → Platform

**File:** `fastaiagent/deploy/push.py`

```python
from fastaiagent._platform.api import PlatformAPI
from fastaiagent.agent import Agent
from fastaiagent.chain import Chain
from fastaiagent.tool import Tool
from fastaiagent.guardrail import Guardrail

class PushResult:
    resource_type: str  # "agent", "chain", "tool"
    name: str
    platform_id: str
    url: str  # platform URL to view the resource
    dependencies_pushed: list[str]  # names of auto-pushed dependencies
    created: bool  # True if new, False if updated

def push_resource(api: PlatformAPI, project: str, resource, **kwargs) -> PushResult:
    """Push any resource to the platform."""
    if isinstance(resource, Chain):
        return _push_chain(api, project, resource, **kwargs)
    elif isinstance(resource, Agent):
        return _push_agent(api, project, resource, **kwargs)
    elif isinstance(resource, Tool):
        return _push_tool(api, project, resource, **kwargs)
    else:
        raise TypeError(
            f"Cannot push {type(resource).__name__}. "
            f"Pushable types: Agent, Chain, Tool."
        )

def _push_chain(api: PlatformAPI, project: str, chain: Chain, **kwargs) -> PushResult:
    """Push a chain and all its dependencies."""
    deps_pushed = []
    
    # 1. Push all agents referenced by chain nodes
    for node in chain.nodes:
        if node.agent:
            agent_result = _push_agent(api, project, node.agent)
            deps_pushed.append(f"agent:{agent_result.name}")
    
    # 2. Push the chain itself
    chain_data = chain.to_dict()
    # to_dict() produces canonical format matching platform ReactFlow schema
    
    try:
        # Try to update existing
        existing = api.get_chain(project, chain.name)
        result = api.put(f"/api/v1/chains/{existing['id']}", chain_data)
        created = False
    except PlatformNotFoundError:
        # Create new
        result = api.create_chain(project, chain_data)
        created = True
    
    return PushResult(
        resource_type="chain",
        name=chain.name,
        platform_id=result["id"],
        url=f"{api._client.base_url}/projects/{project}/chains/{result['id']}",
        dependencies_pushed=deps_pushed,
        created=created,
    )

def _push_agent(api: PlatformAPI, project: str, agent: Agent, **kwargs) -> PushResult:
    """Push an agent and its tool/guardrail dependencies."""
    deps_pushed = []
    
    # Push tools
    for tool in agent.tools:
        tool_result = _push_tool(api, project, tool)
        deps_pushed.append(f"tool:{tool_result.name}")
    
    # Push guardrails
    for guardrail in agent.guardrails:
        gr_data = guardrail.to_dict()
        # ... push guardrail
        deps_pushed.append(f"guardrail:{guardrail.name}")
    
    # Push agent
    agent_data = agent.to_dict()
    try:
        existing = api.get_agent(project, agent.name)
        result = api.put(f"/api/v1/agents/{existing['id']}", agent_data)
        created = False
    except PlatformNotFoundError:
        result = api.create_agent(project, agent_data)
        created = True
    
    return PushResult(
        resource_type="agent",
        name=agent.name,
        platform_id=result["id"],
        url=f"{api._client.base_url}/projects/{project}/agents/{result['id']}",
        dependencies_pushed=deps_pushed,
        created=created,
    )
```

**Tasks for 40.2:**
- [ ] `push_resource()` dispatcher for Agent, Chain, Tool
- [ ] `_push_chain()` with dependency resolution (auto-push agents → tools → guardrails)
- [ ] `_push_agent()` with dependency resolution
- [ ] `_push_tool()` 
- [ ] `fa.push_all()` for multiple resources
- [ ] Conflict resolution: update existing if name matches
- [ ] PushResult with platform URL and dependency list
- [ ] Verify: pushed chain renders correctly in visual editor (manual test against staging)
- [ ] 15+ tests (mocked API)

---

### 40.3 — Pull: Platform → SDK

**File:** `fastaiagent/deploy/pull.py`

```python
def pull_chain(api: PlatformAPI, project: str, name: str) -> Chain:
    """Pull a chain from the platform as a functional SDK Chain."""
    chain_data = api.get_chain(project, name)
    chain = Chain.from_dict(chain_data)
    
    # Pull agents referenced by nodes
    for node in chain.nodes:
        if node.agent_name:
            agent_data = api.get_agent(project, node.agent_name)
            node.agent = Agent.from_dict(agent_data)
    
    return chain  # fully functional — can execute locally

def pull_agent(api: PlatformAPI, project: str, name: str) -> Agent:
    """Pull an agent from the platform as a functional SDK Agent."""
    agent_data = api.get_agent(project, name)
    return Agent.from_dict(agent_data)

def pull_tool(api: PlatformAPI, project: str, name: str) -> Tool:
    """Pull a tool definition from the platform."""
    tool_data = api.get_tool(project, name)
    return Tool.from_dict(tool_data)
```

**Trace pull for local replay:**
```python
# In PlatformTraceStore
def pull(self, trace_id: str) -> TraceData:
    """Pull a production trace for local Agent Replay."""
    trace_data = self._api.get_trace(trace_id)
    # Convert to local TraceData format
    return TraceData.from_platform(trace_data)
    # Developer can now: Replay(trace_data).step_through().fork_at(3).rerun()
```

**Tasks for 40.3:**
- [ ] `pull_chain()` — returns functional Chain with agents resolved
- [ ] `pull_agent()` — returns functional Agent
- [ ] `pull_tool()` — returns functional Tool
- [ ] `traces.pull()` — returns trace data loadable by Replay
- [ ] All pulled objects fully executable locally
- [ ] 10+ tests (mocked API)

---

### 40.4 — Export as Python

**Platform-side addition** (added to platform repo, not SDK repo):

New API endpoint: `GET /api/v1/chains/{id}/export?format=python`

New service in platform backend: `PythonCodeGenerator` that reads chain/agent/tool from DB and generates valid, runnable `fastaiagent` SDK code.

The generated code imports from `fastaiagent` (the SDK package name on PyPI).

**Platform UI addition:** "Export" dropdown button on chain detail page and agent detail page. Options: Python, JSON, YAML.

**Tasks for 40.4 (platform repo):**
- [ ] PythonCodeGenerator service
- [ ] API endpoint for export (chain, agent)
- [ ] UI: Export dropdown with format options
- [ ] Generated Python verified: export → run locally → same behavior
- [ ] 10+ tests in platform repo

---

### 40.5 — Migration Wizard

**Platform-side addition** (platform repo):

When external framework traces (from SDK's LangChain/CrewAI auto-tracing) appear in the platform, the platform can analyze the trace and offer to scaffold a native chain.

**Tasks for 40.5 (platform repo):**
- [ ] TraceAnalyzer service: detect step types from OTel spans
- [ ] Chain scaffolding: generate canonical format JSON from analysis
- [ ] Platform UI: "Convert to Native Chain" banner on external traces
- [ ] Generated chain editable in visual editor
- [ ] 10+ tests in platform repo

---

### 40.6 — Full Round-Trip Integration Testing

**Duration:** 1 week | **All 20 devs**

These tests run against a real platform staging instance (not mocked). They verify the complete SDK ↔ platform contract.

```python
# tests/integration/test_roundtrip.py
# NOTE: These run against staging platform with real API key
# Set FASTAIAGENT_TEST_API_KEY and FASTAIAGENT_TEST_TARGET env vars

@pytest.fixture
def fa():
    return FastAI(
        api_key=os.environ["FASTAIAGENT_TEST_API_KEY"],
        target=os.environ.get("FASTAIAGENT_TEST_TARGET", "https://staging.fastaiagent.net"),
        project="sdk-integration-tests"
    )

def test_push_chain_appears_in_platform(fa):
    """Define chain in SDK → push → verify via API it exists on platform."""
    chain = Chain("test-roundtrip-chain")
    chain.add_node("a", agent=Agent(name="test-agent-a", system_prompt="test", llm=LLMClient(...)))
    chain.add_node("b", agent=Agent(name="test-agent-b", system_prompt="test", llm=LLMClient(...)))
    chain.connect("a", "b")
    
    result = fa.push(chain)
    assert result.created or not result.created  # exists
    assert result.url  # has platform URL
    
    # Verify via pull
    pulled = fa.pull_chain("test-roundtrip-chain")
    assert len(pulled.nodes) == 2
    assert len(pulled.edges) == 1

def test_push_cyclic_chain_with_guardrails(fa):
    """Push chain with cycles + guardrails + typed state → verify renders on platform."""
    chain = Chain("test-cyclic-guarded", state_schema={"type": "object", "properties": {...}})
    chain.add_node("research", agent=Agent(name="r", guardrails=[no_pii()], ...))
    chain.add_node("evaluate", agent=Agent(name="e", ...))
    chain.connect("research", "evaluate")
    chain.connect("evaluate", "research", max_iterations=3, exit_condition="quality >= 0.8")
    
    result = fa.push(chain)
    pulled = fa.pull_chain("test-cyclic-guarded")
    
    # Verify cycle preserved
    cyclic_edges = [e for e in pulled.edges if e.get("is_cyclic")]
    assert len(cyclic_edges) == 1
    assert cyclic_edges[0]["cycle_config"]["max_iterations"] == 3

def test_trace_push_pull_replay(fa):
    """Execute agent → push trace → pull trace → replay locally."""
    agent = Agent(name="test-traced", system_prompt="Say hello", llm=LLMClient(...))
    
    with fa.trace("test-trace"):
        result = agent.run("Hello")
    
    # Wait for async trace push
    import time; time.sleep(2)
    
    # Search for the trace on platform
    traces = fa.traces.search(last_minutes=5)
    assert len(traces) > 0
    
    # Pull and replay
    trace = fa.traces.pull(traces[0].id)
    replay = Replay(trace)
    assert len(replay.steps()) > 0

def test_prompt_push_pull_with_fragments(fa):
    """Create prompt with fragments locally → push → pull → verify fragments resolved."""
    from fastaiagent.prompt import PromptRegistry
    
    local_reg = PromptRegistry(store="local", path="/tmp/test-prompts/")
    local_reg.register_fragment(name="tone", content="Be professional.")
    local_reg.register(name="test-prompt", template="Hello {{name}}. {{@tone}}", fragments=["tone"])
    
    # Push to platform
    fa.prompts.push_from_local(local_reg, "test-prompt")
    
    # Load from platform
    prompt = fa.prompts.load("test-prompt")
    formatted = prompt.format(name="World")
    assert "Be professional" in formatted

def test_canonical_format_roundtrip(fa):
    """SDK to_dict() → push → platform stores → pull → from_dict() → identical."""
    original_agent = Agent(
        name="format-test",
        system_prompt="Test agent",
        llm=LLMClient(provider="openai", model="gpt-4o-mini"),
        tools=[FunctionTool(name="greet", fn=lambda name: f"Hello {name}")],
        guardrails=[no_pii()],
    )
    
    original_dict = original_agent.to_dict()
    fa.push(original_agent)
    
    pulled = fa.pull_agent("format-test")
    pulled_dict = pulled.to_dict()
    
    # Core fields should match
    assert pulled_dict["name"] == original_dict["name"]
    assert pulled_dict["system_prompt"] == original_dict["system_prompt"]
```

**Cleanup:** Integration tests create resources on staging. Add cleanup fixture that deletes test resources after each test run.

**Tasks for 40.6:**
- [ ] Set up staging platform for integration testing
- [ ] Write 15+ round-trip integration tests
- [ ] Test push/pull for all resource types (agent, chain, tool, prompt, trace)
- [ ] Test cyclic chains with guardrails and typed state round-trip
- [ ] Test trace push → pull → replay workflow
- [ ] Test prompt with fragments round-trip
- [ ] Test canonical format preservation across push/pull
- [ ] Test offline mode: disconnect network → verify cache → reconnect → verify sync
- [ ] Test tier limits: verify clear error messages on Free tier limits
- [ ] Cleanup fixture to remove test resources

---

## Phase 40 — Complete Definition of Done

- [ ] FastAI client connects to platform with all features
- [ ] Push works: agents, chains (with cycles, guardrails, typed state), tools, prompts
- [ ] Pull works: all resource types return functional SDK objects
- [ ] Pushed chains render correctly in visual editor (verified manually)
- [ ] Trace push (async) works — traces visible in platform UI
- [ ] Trace pull works — production traces replayable locally
- [ ] Export as Python generates valid, runnable code (platform feature)
- [ ] Migration wizard scaffolds native chains from external traces (platform feature)
- [ ] Offline cache works with graceful degradation
- [ ] Tier-aware error messages work for all limits
- [ ] Round-trip integration tests passing against staging platform
- [ ] 65+ new tests in Phase 40
- [ ] **Cumulative: 265+ tests across Phases 39-40**
- [ ] No regressions in standalone mode (offline still works perfectly)
- [ ] Contract tests passing: fixture files validated in both repos
- [ ] `pip install fastaiagent` still works (no dependency on platform for standalone features)

---

## What Exists After Phase 39-40

**On PyPI (`pip install fastaiagent`):**
- Complete agent framework: agents, chains (with cycles), tools, guardrails, typed state
- OTel-native tracing with local storage + any exporter
- Agent Replay with fork-and-rerun (unique — nobody else has this)
- Prompt registry with fragment composition (unique)
- Local KB with basic search
- Eval framework with multi-turn + trajectory scoring
- Schema drift detection (unique)
- Auto-tracing for OpenAI, LangChain, CrewAI, Anthropic
- Full CLI
- Platform client for SaaS connection (push/pull/sync)
- 265+ tests, Python 3.10-3.13, Linux/macOS/Windows
- Apache 2.0 license

**On the platform (existing customers):**
- 6 new features from Phase 37-38 (cycles, checkpoints, guardrails, typed state, OTel, multi-turn eval)
- Export as Python button
- Migration wizard for external framework users
- SDK contract fixtures for API compatibility

**Repository:** `github.com/fastaifoundry/fastaiagent-sdk`

Ready for Phase 41: polish, documentation, and launch.
