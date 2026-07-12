"""Gemini native multimodal conversion — images and PDFs become inlineData.

Pure-function tests on the Gemini wire's message converter (no network). Uses
real Pillow/pymupdf-constructed parts, the same objects production passes.
"""

from __future__ import annotations

from pathlib import Path

from fastaiagent import PDF, Image
from fastaiagent.llm.message import UserMessage
from fastaiagent.llm.providers.gemini import _convert_messages, _user_parts

FIXTURES = Path(__file__).parent / "fixtures" / "multimodal"


def test_user_parts_pdf_becomes_inline_data() -> None:
    pdf = PDF.from_file(FIXTURES / "contract.pdf")
    parts = _user_parts(["Summarize this.", pdf])
    assert parts[0] == {"text": "Summarize this."}
    blob = parts[1]["inlineData"]
    assert blob["mimeType"] == "application/pdf"
    assert blob["data"] == pdf.to_base64()


def test_user_parts_image_becomes_inline_data() -> None:
    img = Image.from_file(FIXTURES / "cat.jpg")
    parts = _user_parts([img])
    blob = parts[0]["inlineData"]
    assert blob["mimeType"] == img.media_type
    assert blob["data"] == img.to_base64()


def test_user_parts_plain_string_passthrough() -> None:
    assert _user_parts("hello") == [{"text": "hello"}]


def test_convert_messages_carries_pdf_into_contents() -> None:
    pdf = PDF.from_file(FIXTURES / "contract.pdf")
    _system, contents = _convert_messages([UserMessage(["read it", pdf])])
    assert contents[0]["role"] == "user"
    kinds = [("inlineData" if "inlineData" in p else "text") for p in contents[0]["parts"]]
    assert kinds == ["text", "inlineData"]
    assert contents[0]["parts"][1]["inlineData"]["mimeType"] == "application/pdf"
