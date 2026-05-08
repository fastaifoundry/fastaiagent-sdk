# Deep Research over a Financial Document (10-K)

A worked example of pointing the [Deep Research Agent template](deep-research-agent.md) at a *local* financial filing instead of the open web. Same Scope → parallel Research → Write pattern, same SDK primitives — only the search tool changes.

**Use case:** structured extraction of the canonical "snapshot" attributes from a public-company 10-K. We bundled Apple's most recent annual report (filed 2025-10-31, fiscal year ending September 27, 2025) and asked the pipeline to capture 12 attributes with citations.

## Why deep research fits

A 10-K's table of contents is *literally* a subtopic decomposition. The Scope agent's job — break the topic into independent research tracks — maps onto the filing's natural sections:

| Subtopic the Scope agent picks | Section of the 10-K | Attributes covered |
|---|---|---|
| Income Statement Metrics | Consolidated Statements of Operations | revenue, gross margin, operating income, net income, diluted EPS, R&D |
| Balance Sheet Metrics | Consolidated Balance Sheets | cash, total assets, total liabilities |
| Cash Flow Statement Details | Consolidated Statements of Cash Flows | operating cash flow |
| Human-Capital and Operational Data | Item 1 / Human Capital | employee count |

Researchers can run in parallel because the sections are independent. The writer composes one coherent report.

## How it's wired (no new SDK code)

The complete usage code lives in [`examples/deep-research-agent/financial_demo.py`](https://github.com/fastaifoundry/fastaiagent-sdk/blob/main/examples/deep-research-agent/financial_demo.py) (~190 lines, all composition).

The single piece that changes versus the headline `python agent.py` flow: a tiny `search_filing` tool that greps a bundled file instead of calling Tavily. Researchers see a hermetic, deterministic corpus.

```python
from pathlib import Path
import fastaiagent as fa

_FIXTURE = Path(__file__).parent / "fixtures" / "apple_10k_fy2025_excerpts.txt"
_DOC = _FIXTURE.read_text()

@fa.tool()
def search_filing(query: str) -> str:
    """Drop-in replacement for web_search — returns paragraphs from the 10-K
    whose tokens overlap with the query."""
    q = {t.lower().strip(",.;:()$") for t in query.split() if len(t) > 3}
    scored = [(len(q & {t.lower().strip(",.;:()$") for t in p.split()}), p)
              for p in _DOC.split("\n\n") if p.strip()]
    scored.sort(key=lambda x: -x[0])
    return "\n\n--- next match ---\n\n".join(p for ov, p in scored[:5] if ov > 0)
```

Researchers reuse the prompt and structured output type from the existing `topology.py`:

```python
from topology import researcher_prompt, ResearchFindings

agent = fa.Agent(
    name=f"researcher:{subtopic[:40]}",
    system_prompt=researcher_prompt(subtopic, rationale),
    llm=fa.LLMClient(provider="openai", model="gpt-4o-mini"),
    tools=[search_filing],                          # ← only line that changes
    middleware=[ToolBudget(max_calls=15, ...)],
    config=fa.AgentConfig(max_iterations=20),
    output_type=ResearchFindings,
)
```

Scope and Writer agents are imported as-is from `topology.py`. The session span is stamped with `set_template_kind(span, "financial-extraction")` so this trace shows up under a different template kind from the headline `deep-research` runs:

```python
from fastaiagent.trace.span import set_template_kind

with trace_context("deep_research.session") as session_span:
    set_template_kind(session_span, "financial-extraction")
    ...
```

## Setting up the fixture

The 10-K excerpts were downloaded from SEC EDGAR ahead of time:

```sh
curl -A "your-name your-email@example.com" \
  "https://www.sec.gov/Archives/edgar/data/320193/000032019325000079/aapl-20250927.htm" \
  > raw_10k.htm
```

A short Python script then strips the XBRL HTML to plain text and slices the four canonical sections (income statement, balance sheet, cash flow, human capital) into `fixtures/apple_10k_fy2025_excerpts.txt` (~12 KB). The full extraction script is reproducible from the [demo file](https://github.com/fastaifoundry/fastaiagent-sdk/blob/main/examples/deep-research-agent/financial_demo.py).

## Running the demo

```sh
cd examples/deep-research-agent
pip install -r requirements.txt
python financial_demo.py
```

Cost: ~$0.30 per run on `gpt-4o` (writer + scope) plus `gpt-4o-mini` (4 parallel researchers).

## Captured output (real run, real LLM)

Below is the actual report the pipeline produced on a clean run against the bundled fixture. It captures **10 of the 12 target attributes**.

```markdown
# Extract Key Financial Attributes from Apple Inc.'s FY2025 10-K Filing

## Summary
Apple Inc.'s fiscal year 2025 10-K filing reveals significant financial
metrics, including a notable revenue of $416.161 billion and a net income
of $112.01 billion, reflecting robust operational performance. The company
has improved its balance sheet position with increased cash and cash
equivalents and reduced liabilities, reinforcing its financial stability.

## Findings

### Income Statement Metrics
The income statement metrics provide a comprehensive view of Apple's
financial performance for FY2025. Apple generated total net sales of
$416.161 billion, with a gross margin of $195.201 billion [1]. Operating
income was reported at $133.05 billion, while net income totaled $112.01
billion [1]. Additionally, diluted earnings per share stood at $7.46 [1].

### Balance Sheet Metrics
The balance sheet for FY2025 reveals total assets of $359.241 billion [2].
Total liabilities decreased to $285.508 billion [2]. The company's cash and
cash equivalents rose to $35.934 billion [2]. Shareholders' equity was
$73.733 billion [2].

### Human-Capital and Operational Data
In 2025, Apple reported approximately 166,000 full-time equivalent
employees [3]. Apple's expenditure on research and development reached
$34,550 million, reflecting its strategic focus on innovation [4].

## Sources
[1] Apple Inc. 10-K Report 2025: Consolidated Statements of Operations
[2] Apple Inc — Form 10-K (Fiscal Year 2025): Consolidated Balance Sheets
[3] Apple Inc. 2025 Form 10-K - Human Capital
[4] Apple Inc. 2025 Form 10-K - R&D expenses
```

### Score against the 12 canonical FY2025 attributes

| # | Attribute | Canonical (from 10-K) | Captured | ✓ |
|---|---|---|---|---|
| 1 | Total net sales (revenue) | $416,161 M | $416.161 B | ✓ |
| 2 | Gross margin | $195,201 M | $195.201 B | ✓ |
| 3 | Operating income | $133,050 M | $133.05 B | ✓ |
| 4 | Net income | $112,010 M | $112.01 B | ✓ |
| 5 | Diluted EPS | $7.46 | $7.46 | ✓ |
| 6 | Cash & cash equivalents | $35,934 M | $35.934 B | ✓ |
| 7 | Total assets | $359,241 M | $359.241 B | ✓ |
| 8 | Total liabilities | $285,508 M | $285.508 B | ✓ |
| 9 | Operating cash flow | $111,482 M | (writer dropped section) | ✗ |
| 10 | Full-time employees | ~166,000 | ~166,000 | ✓ |
| 11 | Fiscal year end | Sep 27, 2025 | Inferred ("FY2025") only | ✗ |
| 12 | R&D expense | $34,550 M | $34,550 M | ✓ |

**10 of 12 captured exactly.** Every value the writer included is correct to the dollar.

### Honest note on LLM variance

Across multiple runs of the same demo we saw 9-11 of 12 attributes captured. The consistent failure modes:

- **Cash flow section gets dropped.** The cash flow statement has the most cross-referenced line items per attribute, so the researcher hits the tool budget faster. Even when the researcher *did* extract `$111,482 M` cleanly, the writer occasionally synthesized a 3-section report instead of 4.
- **Fiscal year end date.** The writer often paraphrases as "FY2025" instead of reproducing "September 27, 2025" verbatim, even though the date is in the income statement excerpt.

Mitigations that work on the existing primitives (no SDK changes):

1. **Use `gpt-4o` (not mini) for the researcher** — fewer dropped findings, ~3× the cost.
2. **Bump `RESEARCH_TOOL_BUDGET`** — 15 is enough for income/balance/HC, sometimes too tight for cash flow. 20+ for the cash-flow branch in particular.
3. **Add a verifier subtopic** — append a 5th researcher whose only job is to re-read the writer draft against the brief and flag missing attributes. Pattern is identical to `examples/research-agent`'s verifier.
4. **Switch the writer's `output_type`** to a Pydantic `FinancialSnapshot` with required fields. Pydantic's validation will error if the writer produces a partial result, which surfaces missing-attribute drops at runtime.

## Trace shape (live, queryable)

The session span carries `fastaiagent.template.kind = "financial-extraction"` so the local UI distinguishes these from generic deep-research traces:

```sql
SELECT trace_id, json_extract(attributes, '$.fastaiagent.research.topic') AS topic
FROM spans
WHERE json_extract(attributes, '$.fastaiagent.template.kind') = 'financial-extraction';
```

Or via the REST API while `fastaiagent ui` is running:

```sh
curl -s 'http://127.0.0.1:7843/api/traces?last_hours=24' \
  | jq '.rows[] | select(.name == "deep_research.session")'
```

The trace below was captured from the run that produced the report above:

```
deep_research.session             template.kind="financial-extraction"
                                  topic="Apple Inc FY2025 financial snapshot"
  ├── deep_research.scope         (ResearchBrief: 4 subtopics)
  ├── deep_research.research × 4  Income Statement Metrics
  │                               Balance Sheet Metrics
  │                               Cash Flow Statement Details
  │                               Human-Capital and Operational Data
  └── deep_research.write         (2,569-char Markdown report)
```

## What this demo does NOT add to the SDK

This page is a usage walkthrough — **no new SDK code, no new Pydantic schemas, no new database tables.** The demo is two example files:

- `examples/deep-research-agent/financial_demo.py` — a script that composes the existing `build_scope_agent`, `build_writer_agent`, and `researcher_prompt` from `topology.py`.
- `examples/deep-research-agent/fixtures/apple_10k_fy2025_excerpts.txt` — the bundled 10-K text downloaded ahead of time from SEC.gov.

The pattern is portable: swap the fixture for any other filing or document, adjust the topic prompt, and you're extracting from a different domain. The SDK primitives stay unchanged.
