# Meeting Notes Agent (Chain with parallel analysis)

A Granola / Otter / Fireflies-style meeting-notes generator built with [FastAIAgent SDK](https://github.com/fastaifoundry/fastaiagent-sdk) v1.6.1+. A `Chain` DAG loads a transcript, runs three single-purpose analyzer agents **in parallel**, validates the merged output against a Pydantic `MeetingNotes` schema, and optionally drafts personalized followup emails per attendee.

```
input: {"path": "fixtures/sample_transcript.md"}
                │
                ▼
        ┌───────────────┐
        │   load        │ tool — reads .md / .txt / .pdf, extracts
        │               │        title + date heuristically
        └───────┬───────┘
                ▼
        ┌───────────────┐
        │   analyze     │ tool — fans out to 3 LLM agents IN PARALLEL
        │               │   (asyncio.gather inside one tool):
        │               │     • summarizer
        │               │     • action_extractor
        │               │     • decision_extractor
        └───────┬───────┘
                ▼
        ┌───────────────┐
        │   merge       │ tool — Pydantic-validates each analyzer's JSON
        │               │        and produces a typed MeetingNotes model
        └───────┬───────┘
                ▼
   (end of chain — agent.py optionally fans out to per-attendee
    followup emails via draft_followup_email)
```

**What this example demonstrates** (vs. prior templates):

- **Parallel LLM fan-out** via `asyncio.gather` inside a single tool node — three analyzer agents run concurrently against the same transcript
- **Pydantic schema enforcement** at the chain merge step — `MeetingNotes` / `ActionItem` / `Decision` models are the contract every downstream tool reads, so a malformed analyzer output surfaces as a Pydantic validation error rather than cascading
- **Multimodal input** via `fa.PDF` — `load_transcript` accepts `.pdf` as a first-class input alongside `.md` / `.txt`
- **Per-recipient personalization** — `MeetingNotes.for_attendee(name)` slices the notes down to one person's action items + the decisions affecting them, fed to a drafter agent for personalized followup
- **Custom Python eval scorers** for completeness (`action_item_recall`, `decision_recall`) and quality (`owner_attribution` — catches "the team" attributions)

---

## Quick Start

```bash
# from the SDK root
pip install -e .
cd examples/meeting-notes-agent
cp .env.example .env       # only OPENAI_API_KEY is required
pip install -r requirements.txt

python agent.py                                          # default sample transcript
python agent.py --transcript fixtures/sample_transcript.md
python agent.py --transcript meeting.pdf                 # PDF works too
python agent.py --transcript ... --notify                # also draft followup emails
```

The default fixture is a roadmap-sync transcript with 4 attendees, 4 action items, 4 decisions. First run takes ~5–8 seconds (three concurrent LLM calls during `analyze`).

---

## Files

```
meeting-notes-agent/
├── README.md
├── .env.example
├── requirements.txt
├── tools.py             # load_transcript / analyze_meeting / merge_into_notes / draft_followup_email
├── workflow.py          # Chain DAG + the 3 analyzer system prompts + followup prompt
├── schema.py            # Pydantic MeetingNotes / ActionItem / Decision
├── agent.py             # CLI entry point
├── streaming_demo.py    # tail the trace store as the chain runs
├── replay_demo.py       # fork the chain at any node, rerun
├── eval_suite.py        # 3 custom scorers + golden fixture
├── fixtures/
│   └── sample_transcript.md   # Q3 roadmap sync — 4 attendees, 4 actions, 4 decisions
└── tests/
    └── test_smoke.py    # 9 offline regression tests
```

---

## How it's wired

### Chain construction ([workflow.py](workflow.py))

```python
chain = fa.Chain("meeting-notes", checkpoint_enabled=True)

chain.add_node("load",    type=NodeType.tool, tool=load_transcript,
               input_mapping={"path": "{{state.path}}"})
chain.add_node("analyze", type=NodeType.tool, tool=analyze_meeting,
               input_mapping={"transcript": "{{state.output.text}}"})
chain.add_node("merge",   type=NodeType.tool, tool=merge_into_notes,
               input_mapping={
                   "transcript_meta": "{{node_results.load.output}}",
                   "analysis":        "{{node_results.analyze.output}}",
               })

chain.connect("load", "analyze")
chain.connect("analyze", "merge")
```

### Parallel LLM fan-out ([tools.py](tools.py))

The interesting parallelism lives inside `analyze_meeting`. Three concurrent `agent.arun(transcript)` calls via `asyncio.gather`:

```python
@fa.tool()
async def analyze_meeting(transcript: str, ctx: fa.RunContext[MeetingDeps]) -> dict:
    summarizer, action_extractor, decision_extractor = _build_agents(llm)
    summary_r, actions_r, decisions_r = await asyncio.gather(
        summarizer.arun(transcript, context=ctx),
        action_extractor.arun(transcript, context=ctx),
        decision_extractor.arun(transcript, context=ctx),
        return_exceptions=True,
    )
    return {
        "summary_raw":    _safe_output(summary_r),
        "actions_raw":    _safe_output(actions_r),
        "decisions_raw":  _safe_output(decisions_r),
    }
```

> **Why not `NodeType.parallel`?**
> The chain executor's parallel-node contract passes `context["input"]` (the chain's *initial state*, stringified if it's a dict) to every child agent. For our use case each analyzer needs the raw transcript text, which is cleaner expressed as `asyncio.gather` over three explicit `arun(transcript)` calls inside one tool. The chain DAG stays linear; the parallelism is honest and visible in source. `NodeType.parallel` is the right primitive when each child agent runs against the full initial-state dict.

### Schema-enforced merge ([schema.py](schema.py), [tools.py](tools.py))

```python
class ActionItem(BaseModel):
    text: str
    owner: str
    due: str | None = None

class Decision(BaseModel):
    text: str
    rationale: str | None = None

class MeetingNotes(BaseModel):
    title: str = ""
    date: str | None = None
    attendees: list[str] = []
    summary: str
    action_items: list[ActionItem] = []
    decisions: list[Decision] = []
```

The merge tool runs Pydantic validation on each analyzer's JSON. **Per-field parse failures fall through to a partial model** rather than raising — keeps the chain useful when one analyzer returns malformed JSON; the eval suite then surfaces *which* analyzer dropped the ball.

### Per-attendee personalization ([schema.py](schema.py))

```python
class MeetingNotes(BaseModel):
    ...
    def for_attendee(self, name: str) -> dict:
        their_actions = [a for a in self.action_items if a.owner.lower() == name.lower()]
        return {"name": name, "action_items": [...], "decisions": [...], "summary": self.summary}
```

`agent.py --notify` calls `draft_followup_email` once per attendee. Skips attendees with zero action items.

### Multimodal: PDF transcripts

`load_transcript` reads `.pdf` files via `fa.PDF.extract_text()`:

```python
from fastaiagent.multimodal.pdf import PDF
text = PDF.from_file(p).extract_text()
```

For LLM-native PDF reading (vision model), pass an `fa.PDF` instance directly to a vision-capable agent: `agent.arun([prompt, fa.PDF.from_file(path)])`. The chain in this template uses extracted text so the analyzers can stay on a cheaper model.

---

## Running each entry point

```bash
# Single transcript
python agent.py
python agent.py --transcript meeting.pdf
python agent.py --notify              # also draft per-attendee followups

# Stream the chain trace as it executes — see the 3 parallel LLM calls land
python streaming_demo.py

# Replay debugging — fork at a node boundary, swap input, rerun
python replay_demo.py

# Eval suite — 3 custom scorers
python eval_suite.py

# Smoke tests — 9 offline tests
python -m pytest tests/
```

---

## Local UI

```bash
fastaiagent ui start             # http://127.0.0.1:7842
```

What this example populates:

- **`/traces`** — `chain.meeting-notes` root span; inside it: `tool.load_transcript`, then `tool.analyze_meeting` containing **three concurrent `agent.<role>` sub-trees** with their own `llm.openai.gpt-4o` spans, then `tool.merge_into_notes`.
- **`/agents`** — `meeting-summarizer`, `action-extractor`, `decision-extractor`, plus `followup-drafter` if you ran with `--notify`.
- **`/evals`** — `meeting-notes eval` against dataset `meeting-fixtures-golden`.

The streaming demo is the most fun thing to watch in the UI: the three LLM spans appear within ~10ms of each other under the `analyze` parent.

---

## Customising

**Replace the sample transcript** — drop your own meeting transcripts (any `.md` / `.txt` / `.pdf`) into `fixtures/`. The first heading line `# Title — YYYY-MM-DD` and an optional `date:` line in the first 6 lines are heuristically extracted.

**Add a fourth analyzer** (e.g., a "risks" extractor):

```python
RISKS_PROMPT = """You are the risks extractor. Return JSON:
  {"risks": [{"text": "...", "severity": "low|med|high"}]}"""

# in workflow.py: add to _build_agents tuple
# in tools.py: add the 4th gather() task and merge_into_notes branch
# in schema.py: add Risk model and MeetingNotes.risks field
```

**Tighten output_type** — give the analyzer agents a Pydantic `output_type` to enforce JSON shape at the LLM client layer (uses provider-side structured-output mode):

```python
class SummaryOutput(BaseModel):
    summary: str
    attendees: list[str]

summarizer = fa.Agent(name="...", output_type=SummaryOutput, ...)
```

The merge tool then sees `result.parsed: SummaryOutput | None` populated.

**Wire real SendGrid** — uncomment `httpx` in `requirements.txt`, set `SENDGRID_API_KEY` + `SENDGRID_FROM`, flip `EMAIL_BACKEND=sendgrid` in `.env`. No code change.

**HITL on followup-send** — add `fa.interrupt()` inside `draft_followup_email` before the actual send, mirroring the sales-sdr template's pattern. Useful when org policy requires manager approval for any outbound meeting-followup.

---

## What this example does NOT demonstrate

- **Single-agent + memory** — see `examples/customer-support-agent/`.
- **Supervisor / Worker multi-agent** — see `examples/research-agent/`.
- **Conditional branching + HITL gate** — see `examples/sales-sdr-agent/`.
- **Live audio transcription** — assumes the transcript already exists; bring your own Otter / AssemblyAI / Whisper step upstream.
- **Calendar integration** — out of scope; add a `lookup_calendar(date)` tool if you want followup emails to also book a synch-up.

---

## License

Apache 2.0 — same as the SDK.
