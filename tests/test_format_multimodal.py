"""Pure-function tests for ``format_multimodal_message`` across providers.

These are unit tests on a pure function — no HTTP, no mock LLM. They verify
the wire-format dict shape against the spec for each provider, and they use
real Pillow / pymupdf to construct the input parts (the same code paths
production calls hit).

Spec test mappings:

* Test #2 — PDF text mode → :py:func:`test_openai_pdf_text_mode_emits_text_block`
* Test #3 — PDF vision mode → :py:func:`test_openai_pdf_vision_mode_emits_image_blocks`
* Test #4 — multi-provider wire format → ``test_<provider>_image_block_shape`` x5
* Test #5 — mixed input order preserved → :py:func:`test_mixed_text_image_pdf_blocks_in_order`
* Test #14 — non-vision model error → :py:func:`test_non_vision_model_raises`
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fastaiagent import PDF, Image
from fastaiagent._internal.errors import NonVisionModelError
from fastaiagent.multimodal.format import format_multimodal_message

FIXTURES = Path(__file__).parent / "fixtures" / "multimodal"


# --- OpenAI / Azure / Custom ---


def test_openai_text_only_collapses_to_string() -> None:
    """Text-only payloads collapse back to a plain string so legacy
    text-only models (e.g. gpt-3.5-turbo) accept the request — the array
    content shape is reserved for true multimodal calls."""
    out = format_multimodal_message(["hello"], "openai")
    assert out == {"content": "hello"}


def test_openai_image_block_shape() -> None:
    img = Image.from_file(FIXTURES / "cat.jpg")
    out = format_multimodal_message(["caption", img], "openai")
    blocks = out["content"]
    assert blocks[0] == {"type": "text", "text": "caption"}
    assert blocks[1]["type"] == "image_url"
    url = blocks[1]["image_url"]["url"]
    assert url.startswith("data:image/jpeg;base64,")
    assert url.endswith(img.to_base64())
    assert blocks[1]["image_url"]["detail"] == "auto"


def test_azure_uses_openai_shape() -> None:
    img = Image.from_file(FIXTURES / "cat.jpg")
    out_openai = format_multimodal_message([img], "openai")
    out_azure = format_multimodal_message([img], "azure")
    assert out_openai == out_azure


def test_openai_pdf_text_mode_emits_text_block() -> None:
    pdf = PDF.from_file(FIXTURES / "contract.pdf")
    out = format_multimodal_message([pdf], "openai", pdf_mode="text")
    # Text-only payload collapses to a plain string for OpenAI.
    assert isinstance(out["content"], str)
    assert "Service Agreement" in out["content"]
    assert "two years" in out["content"]


def test_openai_pdf_vision_mode_emits_image_blocks() -> None:
    pdf = PDF.from_file(FIXTURES / "contract.pdf")
    out = format_multimodal_message([pdf], "openai", pdf_mode="vision")
    blocks = out["content"]
    assert len(blocks) == 2  # contract.pdf has 2 pages
    for block in blocks:
        assert block["type"] == "image_url"
        assert block["image_url"]["url"].startswith("data:image/png;base64,")


# --- Anthropic ---


def test_anthropic_image_block_shape() -> None:
    img = Image.from_file(FIXTURES / "cat.jpg")
    out = format_multimodal_message(["caption", img], "anthropic")
    blocks = out["content"]
    assert blocks[0] == {"type": "text", "text": "caption"}
    assert blocks[1] == {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/jpeg",
            "data": img.to_base64(),
        },
    }


def test_anthropic_pdf_native_block_when_model_supports() -> None:
    pdf = PDF.from_file(FIXTURES / "contract.pdf")
    out = format_multimodal_message(
        [pdf],
        "anthropic",
        model="claude-sonnet-4-6",
        pdf_mode="auto",
    )
    block = out["content"][0]
    assert block["type"] == "document"
    assert block["source"]["media_type"] == "application/pdf"
    assert block["source"]["data"] == pdf.to_base64()


def test_anthropic_pdf_text_mode_extracts() -> None:
    pdf = PDF.from_file(FIXTURES / "contract.pdf")
    out = format_multimodal_message([pdf], "anthropic", model="claude-sonnet-4-6", pdf_mode="text")
    assert out["content"][0]["type"] == "text"
    assert "Service Agreement" in out["content"][0]["text"]


# --- Ollama ---


def test_ollama_returns_content_string_and_top_level_images() -> None:
    img = Image.from_file(FIXTURES / "cat.jpg")
    out = format_multimodal_message(["caption", img], "ollama")
    assert out["content"] == "caption"
    assert isinstance(out["images"], list)
    assert len(out["images"]) == 1
    assert out["images"][0] == img.to_base64()


def test_ollama_concatenates_text_chunks() -> None:
    img = Image.from_file(FIXTURES / "cat.jpg")
    out = format_multimodal_message(["a", img, "b"], "ollama")
    assert out["content"] == "a\n\nb"
    assert len(out["images"]) == 1


# --- Bedrock ---


def test_bedrock_image_block_shape_uses_raw_bytes() -> None:
    img = Image.from_file(FIXTURES / "cat.jpg")
    out = format_multimodal_message(["caption", img], "bedrock")
    blocks = out["content"]
    assert blocks[0] == {"text": "caption"}
    assert blocks[1] == {"image": {"format": "jpeg", "source": {"bytes": img.data}}}


def test_bedrock_png_format_label() -> None:
    img = Image.from_file(FIXTURES / "receipt.png")
    out = format_multimodal_message([img], "bedrock")
    assert out["content"][0]["image"]["format"] == "png"


def test_bedrock_pdf_native_document_block() -> None:
    pdf = PDF.from_file(FIXTURES / "contract.pdf")
    out = format_multimodal_message(
        [pdf], "bedrock", model="anthropic.claude-sonnet-4", pdf_mode="native"
    )
    doc = out["content"][0]["document"]
    assert doc["format"] == "pdf"
    assert doc["name"] == "contract"
    assert doc["source"]["bytes"] == pdf.data  # raw bytes, not base64


def test_bedrock_pdf_auto_uses_native_for_claude() -> None:
    # Bedrock-hosted Claude is native-PDF capable; auto must not render locally.
    pdf = PDF.from_file(FIXTURES / "contract.pdf")
    out = format_multimodal_message([pdf], "bedrock", model="anthropic.claude-sonnet-4")
    assert "document" in out["content"][0]


# --- Mixed / order ---


def test_mixed_text_image_pdf_blocks_in_order() -> None:
    img = Image.from_file(FIXTURES / "cat.jpg")
    pdf = PDF.from_file(FIXTURES / "contract.pdf")
    out = format_multimodal_message(["intro", img, "between", pdf], "openai", pdf_mode="text")
    blocks = out["content"]
    assert blocks[0]["text"] == "intro"
    assert blocks[1]["type"] == "image_url"
    assert blocks[2]["text"] == "between"
    assert blocks[3]["type"] == "text"  # pdf in text mode
    assert "Service Agreement" in blocks[3]["text"]


# --- Capability gates ---


def test_non_vision_model_raises() -> None:
    img = Image.from_file(FIXTURES / "cat.jpg")
    with pytest.raises(NonVisionModelError):
        format_multimodal_message(
            ["caption", img],
            "openai",
            model="gpt-3.5-turbo",
            is_vision_capable=False,
        )


def test_text_only_does_not_raise_on_non_vision_model() -> None:
    out = format_multimodal_message(
        ["plain text"], "openai", model="gpt-3.5-turbo", is_vision_capable=False
    )
    # Non-vision models must receive the legacy string shape, not an array.
    assert out["content"] == "plain text"


def test_unknown_provider_raises() -> None:
    img = Image.from_file(FIXTURES / "cat.jpg")
    with pytest.raises(Exception):
        format_multimodal_message([img], "vertex")


# --- Detail param passthrough (OpenAI) ---


def test_openai_image_detail_high_passthrough() -> None:
    img = Image.from_file(FIXTURES / "cat.jpg", detail="high")
    out = format_multimodal_message([img], "openai")
    assert out["content"][0]["image_url"]["detail"] == "high"


# --- pdf_mode=auto resolution ---


def test_auto_mode_falls_back_to_text_for_non_vision() -> None:
    pdf = PDF.from_file(FIXTURES / "contract.pdf")
    out = format_multimodal_message([pdf], "openai", pdf_mode="auto", is_vision_capable=False)
    # Text-only payload collapses to a string for OpenAI compatibility.
    assert isinstance(out["content"], str)
    assert "Service Agreement" in out["content"]


def test_auto_mode_uses_native_for_anthropic_vision_model() -> None:
    pdf = PDF.from_file(FIXTURES / "contract.pdf")
    out = format_multimodal_message([pdf], "anthropic", model="claude-sonnet-4-6", pdf_mode="auto")
    assert out["content"][0]["type"] == "document"


def test_auto_mode_uses_native_for_openai_gpt4o() -> None:
    # gpt-4o accepts native PDF file input, so auto forwards the raw PDF
    # instead of rendering pages locally with PyMuPDF (matches raw OpenAI SDK).
    pdf = PDF.from_file(FIXTURES / "contract.pdf")
    out = format_multimodal_message([pdf], "openai", model="gpt-4o", pdf_mode="auto")
    assert out["content"][0]["type"] == "file"


# --- OpenAI native PDF ---


def test_openai_pdf_native_block_shape() -> None:
    pdf = PDF.from_file(FIXTURES / "contract.pdf")
    out = format_multimodal_message([pdf], "openai", model="gpt-4o", pdf_mode="native")
    block = out["content"][0]
    assert block["type"] == "file"
    assert block["file"]["filename"] == "contract.pdf"
    file_data = block["file"]["file_data"]
    assert file_data.startswith("data:application/pdf;base64,")
    assert file_data.endswith(pdf.to_base64())


def test_openai_pdf_native_filename_defaults_to_document_pdf() -> None:
    # PDF.from_bytes carries no source_path/source_url → generic filename.
    pdf = PDF.from_bytes((FIXTURES / "contract.pdf").read_bytes())
    out = format_multimodal_message([pdf], "openai", model="gpt-4o", pdf_mode="native")
    assert out["content"][0]["file"]["filename"] == "document.pdf"


def test_azure_pdf_native_matches_openai() -> None:
    pdf = PDF.from_file(FIXTURES / "contract.pdf")
    out_openai = format_multimodal_message([pdf], "openai", model="gpt-4o", pdf_mode="native")
    out_azure = format_multimodal_message([pdf], "azure", model="gpt-4o", pdf_mode="native")
    assert out_openai == out_azure


def test_openai_pdf_native_coexists_with_text_and_image() -> None:
    img = Image.from_file(FIXTURES / "cat.jpg")
    pdf = PDF.from_file(FIXTURES / "contract.pdf")
    out = format_multimodal_message(
        ["question", pdf, img], "openai", model="gpt-4o", pdf_mode="native"
    )
    blocks = out["content"]
    assert blocks[0] == {"type": "text", "text": "question"}
    assert blocks[1]["type"] == "file"
    assert blocks[2]["type"] == "image_url"


def test_custom_pdf_auto_stays_vision() -> None:
    # Custom endpoints aren't in the native registry; auto must keep rendering
    # pages so arbitrary OpenAI-compatible servers keep working.
    pdf = PDF.from_file(FIXTURES / "contract.pdf")
    out = format_multimodal_message([pdf], "custom", model="my-model", pdf_mode="auto")
    assert out["content"][0]["type"] == "image_url"


def test_custom_pdf_explicit_native_opt_in() -> None:
    # Explicit native is honored for custom (integrator vouches for the endpoint).
    pdf = PDF.from_file(FIXTURES / "contract.pdf")
    out = format_multimodal_message([pdf], "custom", model="my-model", pdf_mode="native")
    assert out["content"][0]["type"] == "file"


# --- max_pdf_pages truncation ---


def test_pdf_vision_truncated_at_max_pages() -> None:
    pdf = PDF.from_file(FIXTURES / "contract.pdf")
    out = format_multimodal_message([pdf], "openai", pdf_mode="vision", max_pdf_pages=1)
    assert len(out["content"]) == 1
