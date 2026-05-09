"""Test utilities for fastaiagent users.

Two LLM stand-ins that swap into ``Agent(llm=...)`` without any HTTP:

- :class:`TestModel` — returns canned responses (string or list, optional
  tool calls). Round-robins through a list of responses.
- :class:`FunctionModel` — wraps a callable that receives the message list
  and returns either ``str`` or ``(str, list[ToolCall])``.

Both honour the full :class:`fastaiagent.llm.LLMClient` API surface
(``complete`` / ``acomplete`` / ``stream`` / ``astream``) and emit
the same ``StreamEvent`` types as real providers, so tracing, replay,
and the local UI work end-to-end against fake runs.

Example:

    from fastaiagent.testing import TestModel
    from fastaiagent.agent import Agent

    agent = Agent(name="hello", llm=TestModel(response="hi"))
    assert agent.run("anything").output == "hi"
"""

from fastaiagent.testing.models import FunctionModel, TestModel

__all__ = ["TestModel", "FunctionModel"]
