# Installation

## Requirements

- Python 3.10 or higher
- pip (or your preferred package manager)

## Install

```bash
pip install fastaiagent
```

## Optional Integrations

FastAIAgent uses optional dependencies to keep the core package lightweight.

| Extra | What It Adds | Install Command |
|-------|-------------|-----------------|
| `openai` | OpenAI SDK auto-tracing | `pip install "fastaiagent[openai]"` |
| `anthropic` | Anthropic SDK auto-tracing | `pip install "fastaiagent[anthropic]"` |
| `langchain` | LangChain auto-tracing | `pip install "fastaiagent[langchain]"` |
| `crewai` | CrewAI auto-tracing | `pip install "fastaiagent[crewai]"` |
| `kb` | Local knowledge base (FastEmbed + PyMuPDF) | `pip install "fastaiagent[kb]"` |
| `otel-export` | OpenTelemetry OTLP exporter | `pip install "fastaiagent[otel-export]"` |
| `all` | All of the above | `pip install "fastaiagent[all]"` |

## Development Setup

```bash
git clone https://github.com/fastaifoundry/fastaiagent-sdk.git
cd fastaiagent-sdk
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev,all]"
```

## Verify Installation

```python
import fastaiagent
print(fastaiagent.__version__)  # 0.1.0a1
```

## Environment Variables

| Variable | Purpose | Required |
|----------|---------|----------|
| `OPENAI_API_KEY` | OpenAI API calls | If using OpenAI provider |
| `ANTHROPIC_API_KEY` | Anthropic API calls | If using Anthropic provider |
| `FASTAIAGENT_API_KEY` | Platform sync | If connecting to platform |
| `FASTAIAGENT_PROJECT` | Platform project name | If connecting to platform |

## Next Steps

- [Build Your First Agent](first-agent.md)
- [Explore the Documentation](../index.md)
