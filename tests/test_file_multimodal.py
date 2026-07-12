"""Generic ``File`` input — mime sniffing, auto-detect, per-provider routing.

Pure-function tests (no network). Real bytes for PDF/image fixtures; small
synthetic byte strings for audio/csv where a fixture would add no value.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fastaiagent import File
from fastaiagent._internal.errors import MultimodalError
from fastaiagent.multimodal.file import sniff_mime
from fastaiagent.multimodal.format import format_multimodal_message
from fastaiagent.multimodal.types import normalize_input

FIXTURES = Path(__file__).parent / "fixtures" / "multimodal"

PDF_BYTES = (FIXTURES / "contract.pdf").read_bytes()
JPG_BYTES = (FIXTURES / "cat.jpg").read_bytes()
PNG_BYTES = (FIXTURES / "receipt.png").read_bytes()
WAV_BYTES = b"RIFF\x24\x00\x00\x00WAVEfmt "
CSV_BYTES = b"name,amount\nacme,1200\n"


# --- sniffing + classification ---


def test_sniff_mime_from_magic_bytes() -> None:
    assert sniff_mime(PDF_BYTES) == "application/pdf"
    assert sniff_mime(PNG_BYTES) == "image/png"
    assert sniff_mime(JPG_BYTES) == "image/jpeg"
    assert sniff_mime(WAV_BYTES) == "audio/wav"


def test_sniff_mime_falls_back_to_extension() -> None:
    assert sniff_mime(CSV_BYTES, "data.csv") == "text/csv"
    assert sniff_mime(b"\x00\x01", "notes.md") == "text/markdown"


def test_category_classification() -> None:
    assert File.from_bytes(PDF_BYTES).category == "pdf"
    assert File.from_bytes(JPG_BYTES).category == "image"
    assert File.from_bytes(WAV_BYTES).category == "audio"
    assert File.from_bytes(CSV_BYTES, filename="d.csv").category == "document"


def test_from_path_sets_filename_and_mime(tmp_path: Path) -> None:
    p = tmp_path / "report.pdf"
    p.write_bytes(PDF_BYTES)
    f = File.from_path(p)
    assert f.filename == "report.pdf"
    assert f.mime_type == "application/pdf"
    assert f.source_path == str(p)


def test_from_file_id_carries_no_bytes() -> None:
    f = File.from_file_id("file-abc123", mime_type="application/pdf")
    assert f.file_id == "file-abc123"
    assert f.data == b""


def test_empty_bytes_without_file_id_raises() -> None:
    with pytest.raises(MultimodalError):
        File.from_bytes(b"")


# --- auto-detect in normalize_input ---


def test_normalize_bytes_becomes_file() -> None:
    parts = normalize_input(PDF_BYTES)
    assert isinstance(parts[0], File)
    assert parts[0].category == "pdf"


def test_normalize_path_becomes_file(tmp_path: Path) -> None:
    p = tmp_path / "x.png"
    p.write_bytes(PNG_BYTES)
    parts = normalize_input(p)
    assert isinstance(parts[0], File)
    assert parts[0].category == "image"


def test_normalize_mixed_list() -> None:
    parts = normalize_input(["question", PDF_BYTES])
    assert parts[0] == "question"
    assert isinstance(parts[1], File)


# --- OpenAI routing ---


def test_openai_pdf_file_becomes_file_part() -> None:
    out = format_multimodal_message(
        [File.from_bytes(PDF_BYTES, filename="c.pdf")], "openai", model="gpt-4o"
    )
    block = out["content"][0]
    assert block["type"] == "file"
    assert block["file"]["filename"] == "c.pdf"
    assert block["file"]["file_data"].startswith("data:application/pdf;base64,")


def test_openai_image_file_becomes_image_url() -> None:
    out = format_multimodal_message([File.from_bytes(JPG_BYTES)], "openai", model="gpt-4o")
    assert out["content"][0]["type"] == "image_url"


def test_openai_audio_file_becomes_input_audio() -> None:
    out = format_multimodal_message([File.from_bytes(WAV_BYTES)], "openai", model="gpt-4o")
    block = out["content"][0]
    assert block["type"] == "input_audio"
    assert block["input_audio"]["format"] == "wav"


def test_openai_file_id_passthrough() -> None:
    out = format_multimodal_message(
        [File.from_file_id("file-xyz")], "openai", model="gpt-4o"
    )
    assert out["content"][0]["file"] == {"file_id": "file-xyz"}


def test_openai_csv_document_raises_with_files_api_hint() -> None:
    with pytest.raises(MultimodalError, match="Files API"):
        format_multimodal_message(
            [File.from_bytes(CSV_BYTES, filename="d.csv")], "openai", model="gpt-4o"
        )


# --- Anthropic routing ---


def test_anthropic_pdf_file_becomes_document() -> None:
    out = format_multimodal_message(
        [File.from_bytes(PDF_BYTES)], "anthropic", model="claude-sonnet-4-6"
    )
    block = out["content"][0]
    assert block["type"] == "document"
    assert block["source"]["media_type"] == "application/pdf"


def test_anthropic_text_document_inline() -> None:
    out = format_multimodal_message(
        [File.from_bytes(b"hello world", filename="n.txt")],
        "anthropic",
        model="claude-sonnet-4-6",
    )
    src = out["content"][0]["source"]
    assert src["type"] == "text"
    assert src["data"] == "hello world"


# --- Bedrock routing ---


def test_bedrock_csv_file_becomes_document() -> None:
    out = format_multimodal_message(
        [File.from_bytes(CSV_BYTES, filename="data.csv")],
        "bedrock",
        model="anthropic.claude-sonnet-4",
    )
    doc = out["content"][0]["document"]
    assert doc["format"] == "csv"
    assert doc["name"] == "data"
    assert doc["source"]["bytes"] == CSV_BYTES


# --- Ollama routing ---


def test_ollama_document_file_raises() -> None:
    with pytest.raises(MultimodalError):
        format_multimodal_message(
            [File.from_bytes(CSV_BYTES, filename="d.csv")], "ollama", model="llava"
        )


# --- round-trip ---


def test_to_dict_from_dict_round_trip() -> None:
    f = File.from_bytes(PDF_BYTES, filename="c.pdf")
    f2 = File.from_dict(f.to_dict())
    assert f2.data == f.data
    assert f2.mime_type == f.mime_type
    assert f2.filename == f.filename
