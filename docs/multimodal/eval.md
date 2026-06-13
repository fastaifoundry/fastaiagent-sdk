# Multimodal Eval

`Dataset.from_jsonl(...)` recognises typed parts in an item's `input`
field and resolves file paths into real `Image` / `PDF` objects at load
time.

## JSONL syntax

Each line is a single test case; `input` may be a string (legacy /
text-only) or a list of typed parts:

```jsonl
{"input": "What is 2 + 2?", "expected": "4"}
{"input": [{"type": "text", "text": "Letters in this image?"}, {"type": "image", "path": "fixtures/cat.jpg"}], "expected": "CAT"}
{"input": [{"type": "text", "text": "Summarise this contract."}, {"type": "pdf", "path": "fixtures/contract.pdf"}], "expected": "Two-year service agreement"}
{"input": [{"type": "text", "text": "Describe."}, {"type": "image", "url": "https://example.com/x.png"}], "expected": "..."}
```

Supported part types:

| Type    | Required keys             | Optional keys |
|---------|---------------------------|---------------|
| `text`  | `text`                    | —             |
| `image` | `path` *or* `url`         | `detail`      |
| `pdf`   | `path` *or* `url`         | —             |

Paths are resolved relative to the JSONL file's directory — moving the
dataset moves its referenced media along with it.

## Running an eval

```python
from fastaiagent import Agent, LLMClient, evaluate, Dataset

agent = Agent(name="vision-eval", llm=LLMClient(provider="openai", model="gpt-4o"))

ds = Dataset.from_jsonl("eval/multimodal_cases.jsonl")
results = evaluate(
    agent_fn=lambda mm_input: agent.run(mm_input).output,
    dataset=ds,
    scorers=["exact_match", "contains"],
)
print(results.summary())
```

`evaluate()` calls `agent_fn` with each item's `input` **value** (not the whole
item dict) — already a list of `str | Image | PDF` once the dataset is loaded —
so `agent.run(mm_input)` works without any transformation. You can also pass
`agent_fn=agent.run` directly. See `examples/78_multimodal_eval.py` for a
runnable script.

## Vision-quality scoring

The built-in scorers (`exact_match`, `contains`, `similarity`) operate on
the agent's text output and are unchanged by multimodal input. For
vision-quality scoring (e.g. "did the agent identify the right object?")
write a custom LLM-as-Judge scorer using the existing
[scorer framework](../evaluation/llm-judge.md).
