# AutoLLM Recipes

[AutoLLM](optimization.md) turns an eval set into a better prompt. These are the
situations where that pays off most, each with a runnable example. Every number
quoted below comes from a real run of that example.

| Your situation | What you need | Scorer | Levers | Example |
|---|---|---|---|---|
| A live agent breaks rules only your team's labels know | its traces + labels | deterministic | instructions + few-shot | [`autollm-loop/`](#a-live-agent-and-your-labels) |
| Move to a cheaper, local or new model | the same labelled set | the same | per model | [`switch_models.py`](#switch-models) |
| Your LLM judge disagrees with your reviewers | outputs labelled pass/fail + a note | agreement with reviewers | instructions | [`calibrate_judge.py`](#calibrate-a-judge) |
| No reference answers (replies, summaries) | inputs only | a calibrated judge + a different audit judge | instructions | [`calibrate_judge.py --tune-agent`](#no-reference-answers) |
| Extraction with conventions (units, signs, formats) | documents + expected values | exact / numeric | instructions | [`autollm/financials.py`](#extraction-conventions), `jaarrekening.py` |
| The format is wrong, not the reasoning | a small labelled set | `exact_match` | instructions | `autollm/agent.py` |
| Which learned facts to inject | facts from `fastaiagent learn` | your task scorer | memory | `101_optimize_memory_agent.py` |
| Keep a prompt tuned as data grows | the curated set, in the repo | a built-in scorer | instructions | [scheduled re-tune](#re-tune-on-a-schedule) |

## A live agent and your labels

The flagship: [AutoLLM Closed Loop](../flagships/autollm-closed-loop.md). A
ticket-triage agent's prompt lives in the registry; its traffic becomes a
labelled dataset; AutoLLM recovers the house rules (a EUR 500 threshold, plan-based
outage priority, a 5-business-day cutoff) that live only in the labels; the winner
becomes a registry version, passes a pytest gate and ships by moving one alias.
Our run: 56% → 96% on 120 tickets, holdout 0.633 → 0.967, ~$0.30.

**Why it works:** the proposer sees each failing case's expected output and the
scorer's reason (`priority P2, expected P1`). A scorer whose reason names the miss
is half the recipe.

## Switch models

A prompt is tuned to its model. To move models, re-tune rather than copy:
[`switch_models.py`](https://github.com/fastaifoundry/fastaiagent-sdk/blob/main/examples/autollm-loop/switch_models.py)
runs AutoLLM per model on the same dataset and seed — so every model faces the same
holdout — and measures cost from the traces. On the triage agent, `gpt-4.1-nano`
ran the shipped prompt at 0.867 on the holdout and its own re-tuned prompt at
0.933, at $0.065 per thousand tickets — 1/17 of `gpt-4.1`'s cost. A forced
migration is the same run with the old and new model.

## Calibrate a judge

An LLM judge is a prompt, and it can be wrong in the same way an agent can.
[`calibrate_judge.py`](https://github.com/fastaifoundry/fastaiagent-sdk/blob/main/examples/autollm/calibrate_judge.py)
treats the judge as an `Agent` whose answer is a verdict, the dataset as replies
your reviewers labelled pass/fail with a one-line note, and the scorer as agreement
with them. AutoLLM rewrites the judge until it agrees; the tuned prompt then becomes
an `LLMJudge(prompt_template=…)` and is re-checked on labelled replies the
calibration never saw: 15/20 for the naive judge, 18/20 calibrated. The full
walkthrough: [Calibrate Your LLM Judge](../flagships/judge-calibration.md).

**Put the reviewers' note in the expected value.** "Fail" alone leaves the
proposer guessing; "fail: promises a refund" names the rule.

## No reference answers

For outputs with no single right answer, the judge is the only signal — so it
selects, and a *different* judge audits:

```python
fa.OptimizeConfig(
    selection_judge=LLMJudge(prompt_template=calibrated, scale="0-1", name="policy_judge"),
    audit_judge=GEval(name="policy_audit", evaluation_steps=policy, llm=fa.LLMClient(model="gpt-4.1")),
)
```

Selecting and reporting with the same judge rewards whatever that judge likes,
including its blind spots. See [the two-judge guard](optimization.md#the-two-judge-guard).

## Extraction conventions

When the model reads the document fine but writes the answer in the wrong
convention — thousands not units, a dropped sign, a Dutch decimal comma — the
convention is in your expected values, not your prompt.
`autollm/financials.py` takes financial-table extraction from 0% to 86–100% on
dev; `autollm/jaarrekening.py` does the same across three real Dutch annual
reports, from 0.30–0.50 to 0.92–1.00 on dev and 0.93–1.00 on the holdout. Use
documents from more than one source, or the optimizer can memorise one source's
answers.

## Re-tune on a schedule

Labels accumulate. A weekly job can re-run AutoLLM on the curated set and open a
pull request when the prompt changes — a reviewer reads the diff, CI gates it:

```yaml
name: autollm-retune
on:
  schedule: [{cron: "0 6 * * 1"}]
  workflow_dispatch: {}
jobs:
  retune:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: "3.12"}
      - run: pip install "fastaiagent>=1.87.0"
      - env: {OPENAI_API_KEY: "${{ secrets.OPENAI_API_KEY }}"}
        run: |
          fastaiagent optimize --agent app/triage.py:agent --dataset evals/triage.jsonl \
            --scorers exact_match --levers instructions --max-iterations 6 \
            --out prompts/triage.txt --no-persist
      - uses: peter-evans/create-pull-request@v6
        with: {branch: autollm/triage, title: "AutoLLM: re-tuned triage prompt"}
```

- `--out` writes the winning **system prompt** — so keep this recipe to the
  `instructions` lever; few-shot examples are not in that file.
- When nothing beats the current prompt, `--out` rewrites the same text, the
  checkout has no diff, and no pull request is opened.
- `--scorers` takes built-in scorer names; for a custom scorer, call
  `optimize()` from a script instead.

## When not to

- **Fewer than ~15 cases.** There's no meaningful train/dev/holdout split; run
  [`harden()`](agent-hardening.md) once and read its suggestions.
- **The tools are wrong, not the prompt.** Fix the tools first.
- **A scorer you haven't checked.** Debug the scorer against real outputs before
  optimizing against it — a scorer bug looks exactly like an agent failure, and
  AutoLLM will happily "fix" it.
