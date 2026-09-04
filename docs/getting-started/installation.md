# Installation

## Requirements

- Python 3.10 or higher
- pip (or your preferred package manager)

## Install

```bash
pip install fastaiagent
```

The core install is deliberately small and **permissively licensed throughout** —
no copyleft anywhere in the resolved tree. A `clean-core` CI job installs the
package with no extras on every pull request and fails the build if an AGPL or
GPL package appears, or if the library-usage surface (`run_guardrail`,
`plane_guardrails_for_agent`, guardrail evaluation) stops working without extras.
Dropping the SDK into an existing LangChain or CrewAI project should not drag a
licence review along with it.

## Optional Integrations

FastAIAgent uses optional dependencies to keep the core package lightweight.

| Extra | What It Adds | Install Command |
|-------|-------------|-----------------|
| `openai` | OpenAI SDK auto-tracing | `pip install "fastaiagent[openai]"` |
| `anthropic` | Anthropic SDK auto-tracing | `pip install "fastaiagent[anthropic]"` |
| `langchain` | LangChain auto-tracing | `pip install "fastaiagent[langchain]"` |
| `crewai` | CrewAI auto-tracing | `pip install "fastaiagent[crewai]"` |
| `pdf` | Local PDF decoding — text, page count, page rendering (pypdfium2) | `pip install "fastaiagent[pdf]"` |
| `kb` | Local knowledge base (FastEmbed + faiss + `pdf`) | `pip install "fastaiagent[kb]"` |
| `otel-export` | OpenTelemetry OTLP exporter | `pip install "fastaiagent[otel-export]"` |
| `postgres` | Postgres durability backend (psycopg3) | `pip install "fastaiagent[postgres]"` |
| `all` | All of the above | `pip install "fastaiagent[all]"` |

!!! note "Local PDF decoding is optional — three ways to avoid it"

    `PDF.extract_text()`, `PDF.page_count()`, `PDF.to_page_images()`,
    `LocalKB.add("file.pdf")` and Local UI PDF thumbnails decode locally, which
    the core install does not ship an engine for. You have three options:

    1. **Do nothing.** `pdf_mode="auto"` (the default) selects `native` for
       Claude 3.5/3.7/4.x, GPT-4o/4.1/5/o-series, Azure, Bedrock Claude and
       Gemini — the raw PDF goes to the provider and nothing is parsed locally.
    2. **Bring your own parser** —
       `PDF.from_file(p, text=my_parser.extract(p))`. No engine involved.
    3. **Install the extra** — `pip install "fastaiagent[pdf]"`.

    See [PDFs](../multimodal/pdfs.md).

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
print(fastaiagent.__version__)  # 1.0.0
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
