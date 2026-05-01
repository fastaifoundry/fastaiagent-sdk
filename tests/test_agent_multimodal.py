"""Phase 3 unit tests for ``Agent`` multimodal acceptance.

Mock-free: every assertion is on the result of pure SDK code paths
(``_build_messages``, ``_coerce_tool_output_to_message_content``,
``_input_summary_text``) operating on real ``Image``/``PDF`` instances
loaded from the committed fixtures. No LLM calls and no HTTP — those live
in ``tests/e2e/test_gate_multimodal_*.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fastaiagent import PDF, Agent, AgentConfig, Image, LLMClient
from fastaiagent.agent.agent import _input_summary_text
from fastaiagent.agent.executor import _coerce_tool_output_to_message_content
from fastaiagent.llm.message import MessageRole

FIXTURES = Path(__file__).parent / "fixtures" / "multimodal"


def _make_agent(model: str = "gpt-4o") -> Agent:
    return Agent(
        name="test",
        system_prompt="you are a test agent",
        llm=LLMClient(provider="openai", model=model),
        config=AgentConfig(max_iterations=1),
    )


# --- _build_messages: input shape coverage ---


def test_build_messages_string_input_unchanged_legacy_shape() -> None:
    agent = _make_agent()
    msgs = agent._build_messages("hello there")
    assert msgs[-1].role == MessageRole.user
    assert msgs[-1].content == "hello there"
    assert msgs[-1].has_multimodal_content() is False


def test_build_messages_single_image_input() -> None:
    agent = _make_agent()
    img = Image.from_file(FIXTURES / "cat.jpg")
    msgs = agent._build_messages(img)
    assert msgs[-1].role == MessageRole.user
    assert msgs[-1].has_multimodal_content() is True
    assert msgs[-1].content == [img]


def test_build_messages_single_pdf_input() -> None:
    agent = _make_agent()
    pdf = PDF.from_file(FIXTURES / "contract.pdf")
    msgs = agent._build_messages(pdf)
    assert msgs[-1].has_multimodal_content() is True
    assert msgs[-1].content == [pdf]


def test_build_messages_mixed_list_preserves_order() -> None:
    agent = _make_agent()
    img = Image.from_file(FIXTURES / "cat.jpg")
    pdf = PDF.from_file(FIXTURES / "contract.pdf")
    msgs = agent._build_messages(["caption", img, "between", pdf])
    user_msg = msgs[-1]
    assert user_msg.has_multimodal_content() is True
    assert user_msg.content == ["caption", img, "between", pdf]


def test_build_messages_singleton_string_list_collapses_to_string() -> None:
    """A list containing only a single string degrades to the legacy text shape.

    Keeps the wire payload byte-identical to the pre-multimodal era when
    callers happen to wrap text in a one-element list.
    """
    agent = _make_agent()
    msgs = agent._build_messages(["just text"])
    assert msgs[-1].content == "just text"
    assert msgs[-1].has_multimodal_content() is False


def test_build_messages_includes_system_prompt() -> None:
    agent = _make_agent()
    img = Image.from_file(FIXTURES / "cat.jpg")
    msgs = agent._build_messages(["caption", img])
    assert msgs[0].role == MessageRole.system
    assert msgs[0].content == "you are a test agent"


# --- _input_summary_text ---


def test_input_summary_text_interleaves_text_and_markers() -> None:
    img = Image.from_file(FIXTURES / "cat.jpg")
    pdf = PDF.from_file(FIXTURES / "contract.pdf")
    summary = _input_summary_text(["alpha", img, "beta", pdf, "gamma"])
    # Text portions appear, in order. Media parts become readable markers
    # rather than the original bytes.
    a_idx = summary.index("alpha")
    b_idx = summary.index("beta")
    g_idx = summary.index("gamma")
    assert a_idx < b_idx < g_idx
    assert "image:image/jpeg" in summary
    assert "pdf:" in summary


def test_input_summary_text_string_only() -> None:
    assert _input_summary_text(["hello", "world"]) == "hello world"


# --- Tool result coercion (Test #6 spec) ---


def test_tool_returns_image_produces_multimodal_content() -> None:
    img = Image.from_file(FIXTURES / "cat.jpg")
    content, summary = _coerce_tool_output_to_message_content(img)
    assert isinstance(content, list)
    assert len(content) == 2
    assert isinstance(content[0], str)
    assert content[1] is img
    assert "image" in summary
    assert str(img.size_bytes()) in summary


def test_tool_returns_pdf_produces_multimodal_content() -> None:
    pdf = PDF.from_file(FIXTURES / "contract.pdf")
    content, summary = _coerce_tool_output_to_message_content(pdf)
    assert isinstance(content, list)
    assert content[1] is pdf
    assert "pdf" in summary
    assert "pages=2" in summary


def test_tool_returns_string_passes_through() -> None:
    content, summary = _coerce_tool_output_to_message_content("plain result")
    assert content == "plain result"
    assert summary == "plain result"


def test_tool_returns_dict_serializes_to_json() -> None:
    content, summary = _coerce_tool_output_to_message_content({"a": 1, "b": [2, 3]})
    assert isinstance(content, str)
    assert summary == content
    assert '"a"' in content


def test_tool_returns_list_with_image_preserves_list() -> None:
    img = Image.from_file(FIXTURES / "cat.jpg")
    content, summary = _coerce_tool_output_to_message_content(["caption", img])
    assert isinstance(content, list)
    assert content == ["caption", img]
    assert "multimodal" in summary


# --- Type acceptance: rejection paths ---


def test_build_messages_rejects_bare_bytes_input() -> None:
    agent = _make_agent()
    bad: object = b"raw bytes"
    with pytest.raises(TypeError):
        agent._build_messages(bad)  # type: ignore[arg-type]


def test_build_messages_rejects_dict_input() -> None:
    agent = _make_agent()
    bad: object = {"text": "hi"}
    with pytest.raises(TypeError):
        agent._build_messages(bad)  # type: ignore[arg-type]


# --- Non-vision model surfaces error path before LLM call ---


def test_non_vision_model_raises_when_image_passed_through_to_provider_dict() -> None:
    """The check fires when ``LLMClient`` serializes the multimodal user
    message — verified by exercising the same ``to_provider_dict`` path that
    every ``_call_<provider>`` method drives."""
    from fastaiagent._internal.errors import NonVisionModelError

    agent = _make_agent(model="gpt-3.5-turbo")
    img = Image.from_file(FIXTURES / "cat.jpg")
    msgs = agent._build_messages(["describe", img])
    user_msg = msgs[-1]
    with pytest.raises(NonVisionModelError):
        user_msg.to_provider_dict("openai", **agent.llm._provider_dict_kwargs())
