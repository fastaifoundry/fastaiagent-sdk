# Streaming

Streaming delivers LLM tokens to your application as they are generated, rather than waiting for the full response. This enables real-time chat interfaces, lower perceived latency, and progressive output rendering.

## Quick Start

```python
import asyncio
from fastaiagent import Agent, LLMClient
from fastaiagent.llm.stream import TextDelta

agent = Agent(
    name="assistant",
    system_prompt="You are a helpful assistant.",
    llm=LLMClient(provider="openai", model="gpt-4.1"),
)

async def main():
    async for event in agent.astream("Explain quantum computing in 3 sentences."):
        if isinstance(event, TextDelta):
            print(event.text, end="", flush=True)
    print()

asyncio.run(main())
```

## StreamEvent Types

Every streaming method yields `StreamEvent` objects. There are five event types:

| Event | Fields | When Emitted |
|-------|--------|-------------|
| `TextDelta` | `text: str` | Each token/chunk of text from the LLM |
| `ToolCallStart` | `call_id: str`, `tool_name: str` | LLM initiates a tool call |
| `ToolCallEnd` | `call_id: str`, `tool_name: str`, `arguments: dict` | Tool call arguments fully parsed |
| `Usage` | `prompt_tokens: int`, `completion_tokens: int` | Token counts (typically at end of response) |
| `StreamDone` | *(none)* | End-of-stream marker |

```python
from fastaiagent.llm.stream import (
    StreamEvent, TextDelta, ToolCallStart, ToolCallEnd, Usage, StreamDone
)
```

These types align with the FastAIAgent Platform's streaming protocol for seamless compatibility.

## Streaming Layers

Streaming is available at three layers, from low-level to high-level:

### LLMClient.astream()

Stream directly from the LLM provider. No tool execution — just raw tokens.

```python
from fastaiagent import LLMClient
from fastaiagent.llm.message import UserMessage
from fastaiagent.llm.stream import TextDelta, Usage

llm = LLMClient(provider="openai", model="gpt-4.1")

async for event in llm.astream([UserMessage("Hello!")]):
    if isinstance(event, TextDelta):
        print(event.text, end="", flush=True)
    elif isinstance(event, Usage):
        print(f"\nTokens: {event.prompt_tokens} in, {event.completion_tokens} out")
```

**Supported streaming providers:**

| Provider | Streaming | Protocol |
|----------|:-:|---------|
| OpenAI | Yes | SSE (`data: {json}` lines) |
| Anthropic | Yes | SSE (Anthropic event format) |
| Ollama | Yes | Newline-delimited JSON |
| Azure | Yes | SSE (OpenAI-compatible) |
| Custom | Yes | SSE (OpenAI-compatible) |
| Bedrock | No | Use `acomplete()` instead |

### stream_tool_loop()

Streaming with tool execution. The loop streams LLM tokens, detects tool calls, executes tools, and continues until the LLM produces a final text response.

```python
from fastaiagent.agent.executor import stream_tool_loop
from fastaiagent.llm.message import SystemMessage, UserMessage
from fastaiagent.llm.stream import TextDelta, ToolCallStart, ToolCallEnd

messages = [SystemMessage("You are helpful."), UserMessage("What's the weather?")]

async for event in stream_tool_loop(llm=llm, messages=messages, tools=[weather_tool]):
    if isinstance(event, TextDelta):
        print(event.text, end="", flush=True)
    elif isinstance(event, ToolCallStart):
        print(f"\n[Calling {event.tool_name}...]")
    elif isinstance(event, ToolCallEnd):
        print(f"[{event.tool_name} done]")
```

### Agent.astream()

Full agent streaming with guardrails, memory, and tool execution.

```python
from fastaiagent import Agent
from fastaiagent.guardrail import no_pii
from fastaiagent.llm.stream import TextDelta, ToolCallStart

agent = Agent(
    name="assistant",
    system_prompt="You are helpful.",
    llm=llm,
    tools=[search_tool],
    guardrails=[no_pii()],
)

async for event in agent.astream("Find me a restaurant"):
    if isinstance(event, TextDelta):
        print(event.text, end="", flush=True)
    elif isinstance(event, ToolCallStart):
        print(f"\n  -> Using {event.tool_name}...")
```

**Execution order:**
1. Input guardrails run **before** streaming begins
2. Stream events are yielded during the tool-calling loop
3. Output guardrails run **after** streaming completes
4. Memory is updated at the end

## Handling Tool Calls

When an agent uses tools, the stream emits events in this order for each tool-calling iteration:

```
TextDelta         (optional — LLM may emit text before tool calls)
ToolCallStart     (call_id, tool_name)
ToolCallEnd       (call_id, tool_name, arguments)
Usage             (token counts for this iteration)
--- tool executes, result appended to messages ---
TextDelta ...     (next iteration's tokens)
```

Using Python 3.10+ pattern matching:

```python
async for event in agent.astream("Search for Python tutorials"):
    match event:
        case TextDelta(text=text):
            print(text, end="", flush=True)
        case ToolCallStart(tool_name=name):
            print(f"\n[Tool: {name}]", end="")
        case ToolCallEnd(tool_name=name):
            print(f" [done]")
        case Usage(prompt_tokens=p, completion_tokens=c):
            print(f"\n({p}+{c} tokens)", end="")
        case _:
            pass
```

## Sync vs Async

Every streaming method has both sync and async versions:

```python
# Async — yields events in real time
async for event in agent.astream("Hello"):
    ...

# Sync — collects all events into a single result
result = agent.stream("Hello")   # returns AgentResult
print(result.output)
```

```python
# Async — yields events from LLM
async for event in llm.astream(messages):
    ...

# Sync — collects into LLMResponse
response = llm.stream(messages)  # returns LLMResponse
print(response.content)
```

The sync `stream()` safely handles being called from within an async context (e.g., Jupyter notebooks, async frameworks).

## Building a Chat UI

Streaming is ideal for chat interfaces. Here is a pattern for a streaming chat loop:

```python
import asyncio
from fastaiagent import Agent, LLMClient
from fastaiagent.agent import AgentMemory
from fastaiagent.llm.stream import TextDelta

agent = Agent(
    name="chatbot",
    system_prompt="You are a friendly assistant.",
    llm=LLMClient(provider="openai", model="gpt-4.1"),
    memory=AgentMemory(),
)

async def chat():
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ("quit", "exit"):
            break
        print("Assistant: ", end="", flush=True)
        async for event in agent.astream(user_input):
            if isinstance(event, TextDelta):
                print(event.text, end="", flush=True)
        print()

asyncio.run(chat())
```

## Error Handling

Streaming errors are raised as exceptions, same as non-streaming:

```python
from fastaiagent._internal.errors import (
    LLMProviderError,        # LLM API error (auth, rate limit, etc.)
    MaxIterationsError,      # Tool loop exceeded max_iterations
    GuardrailBlockedError,   # Guardrail rejected input/output
    LLMError,                # Streaming not supported for provider
)

try:
    async for event in agent.astream("Do something"):
        if isinstance(event, TextDelta):
            print(event.text, end="")
except GuardrailBlockedError as e:
    print(f"\nBlocked by guardrail: {e}")
except MaxIterationsError:
    print("\nAgent hit iteration limit")
except LLMProviderError as e:
    print(f"\nLLM error: {e}")
```

> **Note:** Input guardrails raise `GuardrailBlockedError` before any streaming begins. Output guardrails raise after streaming completes. In both cases, your `except` block handles it normally.

## Platform Compatibility

The SDK's `StreamEvent` types (`TextDelta`, `ToolCallStart`, `ToolCallEnd`, `Usage`, `StreamDone`) align with the FastAIAgent Platform's streaming protocol. The same event types work in both local SDK streaming and platform streaming, making it straightforward to build clients that work with both.

## Middleware, Durability, and HITL Parity (1.5.0+)

As of 1.5.0, `Agent.astream()` is at full feature parity with `Agent.run()` / `arun()`:

- **Middleware hooks fire during streaming.** `before_model`, `after_model`, and `wrap_tool` are invoked at the same logical points they're invoked during a non-streaming run. A configured `ToolBudget`, `TrimLongMessages`, or custom `AgentMiddleware` works identically for both modes.
- **Checkpoints are written during streaming.** When the agent has a `Checkpointer` configured, turn-boundary and pre-tool checkpoints are persisted as the loop runs — so a process crash mid-stream can resume from the last checkpoint with `chain.aresume(...)`.
- **`InterruptSignal` works inside streamed tool calls.** Calling `interrupt(...)` from within a tool that runs during `astream()` pauses the run identically to `arun()` — the streaming generator surfaces the interrupt and the execution resumes via the standard `aresume` flow.

Before 1.5.0, all three were silently bypassed during streaming — middleware was ignored, no checkpoints were written, and `interrupt()` raised an unhandled exception. If you upgrade and your existing streaming code starts seeing middleware applied for the first time, that is the intended behavior.

---

## Next Steps

- [Agents](../agents/index.md) — Agent construction, tools, guardrails, and configuration
- [Tools](../tools/index.md) — Deep dive into FunctionTool, RESTTool, and MCPTool
- [Tracing](../tracing/index.md) — Trace streaming executions with OTel spans
- [Guardrails](../guardrails/index.md) — Input/output validation during streaming
- [Replay](../replay/index.md) — Debug streaming executions with fork-and-rerun
