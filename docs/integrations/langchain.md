# LangChain / LangGraph

The harness wraps any LangChain `Runnable` (LCEL chain, LangGraph
compiled graph, prebuilt agent). Auto-tracing covers the full LangChain
callback ABI — chains, LLM / chat-model calls, tool calls, retrievers
— and threads parent-context properly so the trace tree mirrors the
graph topology.

```bash
pip install "fastaiagent[langchain]"  # bumps to langchain-core>=0.3, langgraph>=0.2
```

## 1. Auto-tracing

```python
import fastaiagent as fa
from fastaiagent.integrations import langchain as lc

lc.enable()  # idempotent — safe to call multiple times

from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

@tool
def echo(text: str) -> str:
    """Echo input back."""
    return f"echo: {text}"

graph = create_react_agent(ChatOpenAI(model="gpt-4o-mini"), tools=[echo])
result = graph.invoke(
    {"messages": [HumanMessage(content="...")]},
    config={"callbacks": [lc.get_callback_handler()]},
)
```

After `enable()`, our `FastAIAgentCallbackHandler` is registered via
`langchain_core.tracers.context.register_configure_hook`, so it auto-
attaches to `RunnableConfig`s that don't carry their own callbacks. If
you compose your own `RunnableConfig`, pass
`get_callback_handler()` explicitly inside the `callbacks` list — that's
the most reliable way to guarantee the handler fires, especially in
test fixtures.

**What lands in each span:**

| Span | Captures |
|---|---|
| `langgraph.{name}` / `langchain.{name}` (root) | inputs, outputs, `fastaiagent.framework=langchain`, `framework.version` |
| `node.{node_name}` | state in / state out |
| `llm.{provider}.{model}` | full messages JSON, `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, computed `fastaiagent.cost.total_usd`, response content |
| `tool.{tool_name}` | tool args, tool output (truncated at 10 KB), latency |
| `retrieval.{retriever_name}` | query, document count, first 200 chars of each doc, `top_k` |

LangGraph node spans arrive as standard child `on_chain_start` events
from the compiled `Pregel`, so we get them for free. Conditional-edge
evaluation has no callback — that's a documented LangGraph limitation,
not a harness one.

## 2. Eval

```python
import fastaiagent as fa
from fastaiagent.integrations import langchain as lc

evaluable = lc.as_evaluable(graph)
results = fa.evaluate(
    evaluable,
    dataset=[{"input": "Capital of France?", "expected": "Paris"}],
    scorers=["exact_match"],
)
```

The adapter opens an outer `eval.case` span per case so `trace_id`
is captured *while it's still active* and lands on each
`EvalCaseRecord`. Click an eval case in the Local UI → deep-link to
the trace.

Custom mappers:

```python
evaluable = lc.as_evaluable(
    graph,
    input_mapper=lambda text: {"messages": [HumanMessage(content=text)], "extra": "..."},
    output_mapper=lambda result: result["final_answer"],
)
```

Defaults: `input_mapper` wraps a string in a single `HumanMessage`
inside `MessagesState` shape; `output_mapper` reads the last message's
content, falling back to common keys (`response`, `output`, `answer`,
`result`, `text`, `content`) for custom states.

## 3. Guardrails

```python
from fastaiagent.guardrail.builtins import no_pii, toxicity_check
from fastaiagent.integrations.langchain import with_guardrails

guarded = lc.with_guardrails(
    graph,
    name="support-bot",
    input_guardrails=[no_pii(position="input")],
    output_guardrails=[toxicity_check()],
)
```

The wrapper is a `_GuardedRunnable` that proxies every other attribute
to the wrapped object, so consumer code keeps working unchanged.
Overridden methods: `invoke`, `ainvoke`, `stream`, `astream`, `batch`.

**Block-only semantics**: a failing blocking guardrail logs a
`guardrail_events` row tagged `framework=langchain` *and* raises
`GuardrailBlocked`.
There is no redaction — see the [Limitations section](overview.md#limitations).

**Streaming**: input guardrails run before the stream opens; output
guardrails buffer the entire stream and check the final chunk. If you
want zero-latency streaming, only pass `input_guardrails`.

## 4. Prompt registry

```python
from fastaiagent.integrations.langchain import prompt_from_registry

template = lc.prompt_from_registry("support-system", agent="support-bot")
chain = template | ChatOpenAI(model="gpt-4o-mini")
chain.invoke({"name": "world"})
```

Returns a `ChatPromptTemplate` with `template_format="mustache"` so the
registry's `{{var}}` syntax works natively. When the template is rendered
inside a traced LangGraph run, the next `on_chat_model_start` /
`on_llm_start` event tags the LLM span with
`fastaiagent.prompt.slug` and `fastaiagent.prompt.version`, which the
Prompt detail page's "Traces using this prompt" panel reads.

## 5. Knowledge base as a retriever

```python
from fastaiagent.integrations.langchain import kb_as_retriever
from langchain_core.runnables import RunnablePassthrough

retriever = lc.kb_as_retriever("support-kb", top_k=5, agent="support-bot")
rag = (
    {"context": retriever, "question": RunnablePassthrough()}
    | template
    | ChatOpenAI(model="gpt-4o-mini")
)
```

`kb_as_retriever` returns a real `BaseRetriever` subclass that delegates
to `LocalKB.search()`. Documents come back as
`Document(page_content=..., metadata={..., "score": float})` — the
`score` lives in metadata so it doesn't collide with retrievers that
populate other metadata keys.

## 6. Register the agent

```python
from fastaiagent.integrations.langchain import register_agent

register_agent(graph, name="support-bot")
```

Walks `compiled.get_graph()` and writes:
- nodes (classified as llm / tool / retriever / function),
- edges (with `conditional` flag),
- model name (from the first LLM node)

into the `external_agents` SQLite table. The Local UI's
`/agents/support-bot` page picks this up and renders the dependency
graph + workflow visualization. If you skip `register_agent` and
just use `with_guardrails(name=...)` / `prompt_from_registry(agent=...)` /
`kb_as_retriever(agent=...)`, a stub agent row is created lazily —
the dependency graph then shows just the harness layers without
native nodes.

## Migration path

When a workflow needs **Replay** (fork-and-rerun for debugging),
**durability** (checkpoint-resumable runs across process crashes),
or **suspending HITL** (human approval mid-run), build it natively in
FastAIAgent. The harness can't deliver those because they need
execution control of the framework's state machine.

A pragmatic migration order:
1. Keep the LangGraph agent. Enable auto-tracing + register it. Use
   the Local UI to understand its behaviour.
2. Add evals with `as_evaluable()` against your test set.
3. Add guardrails with `with_guardrails()`.
4. Plug in the prompt registry and KB.
5. When you next need to build a new workflow with Replay /
   durability / HITL, build that one natively.

There is no automatic LangGraph → FastAIAgent code conversion (too
fragile, too version-dependent across LangGraph releases).

## Examples

- `examples/08_trace_langchain.py`
- `examples/57_eval_langchain.py`
- `examples/59_register_external_agent.py`
