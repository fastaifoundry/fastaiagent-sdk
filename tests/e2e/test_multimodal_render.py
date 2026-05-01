"""E2E tests for the multimodal trace rendering surface.

Covers two paths:

1. The ``extract_content_parts`` helper detects images, PDFs, and text
   in the message shapes the SDK and providers actually emit.
2. The ``GET /api/traces/{tid}/spans/{sid}/attachments`` endpoint streams
   thumbnail bytes, and ``?full=1`` returns the full payload when stored.

No mocking — uses a real SQLite DB seeded by ``seed_ui_sprint1.py`` and
the real FastAPI app via ``TestClient``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.e2e

pytest.importorskip("fastapi")
pytest.importorskip("bcrypt")
pytest.importorskip("itsdangerous")

from fastapi.testclient import TestClient  # noqa: E402

from fastaiagent.ui.attrs import (  # noqa: E402
    extract_content_parts,
    has_multimodal_part,
)
from fastaiagent.ui.db import init_local_db  # noqa: E402
from fastaiagent.ui.server import build_app  # noqa: E402


# ---------------------------------------------------------------------------
# Pure helper
# ---------------------------------------------------------------------------


def test_extract_content_parts_handles_strings() -> None:
    parts = extract_content_parts("hello world")
    assert parts == [{"type": "text", "text": "hello world"}]
    assert not has_multimodal_part("hello world")


def test_extract_content_parts_walks_message_lists() -> None:
    msg = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": "https://example.com/cat.jpg",
                    "media_type": "image/jpeg",
                },
            ],
        }
    ]
    parts = extract_content_parts(msg)
    # text + image part, in original order
    assert len(parts) == 2
    assert parts[0]["type"] == "text"
    assert parts[1]["type"] == "image_url"
    assert has_multimodal_part(msg)


def test_extract_content_parts_detects_pdf_via_type_or_media_type() -> None:
    by_type = [{"role": "user", "content": [{"type": "input_pdf", "data": "..."}]}]
    by_media = [
        {
            "role": "user",
            "content": [{"type": "document", "media_type": "application/pdf"}],
        }
    ]
    assert has_multimodal_part(by_type)
    assert has_multimodal_part(by_media)


def test_extract_content_parts_handles_plain_text_messages() -> None:
    msg = [{"role": "user", "content": "just text"}]
    parts = extract_content_parts(msg)
    assert parts == [{"type": "text", "text": "just text"}]
    assert not has_multimodal_part(msg)


# ---------------------------------------------------------------------------
# Attachments endpoint
# ---------------------------------------------------------------------------


@pytest.fixture
def seeded_sprint1_db(tmp_path: Path) -> Path:
    """Use the Sprint 1 seeder to land a multimodal trace + attachment row."""
    from scripts.seed_ui_snapshot import seed as seed_base
    from scripts.seed_ui_sprint1 import seed as seed_sprint1

    db_path = tmp_path / "local.db"
    init_local_db(db_path).close()
    seed_base(db_path)
    seed_sprint1(db_path)
    return db_path


@pytest.fixture
def no_auth_client(seeded_sprint1_db: Path) -> TestClient:
    app = build_app(db_path=str(seeded_sprint1_db), no_auth=True)
    return TestClient(app)


# Constants must match scripts/seed_ui_sprint1.py.
TRACE_ID = "mm00000000000000000000000000mm01"
SPAN_LLM = "mmllmspan0000000000000000000mm01"


def test_attachments_list_returns_metadata(no_auth_client: TestClient) -> None:
    r = no_auth_client.get(f"/api/traces/{TRACE_ID}/spans/{SPAN_LLM}/attachments")
    assert r.status_code == 200, r.text
    body = r.json()
    items = body.get("attachments", [])
    assert len(items) == 1
    item = items[0]
    assert item["media_type"] == "image/jpeg"
    assert item["size_bytes"] > 0
    assert item.get("metadata", {}).get("width") == 320
    assert item["has_full_data"] is True


def test_attachments_thumbnail_stream(no_auth_client: TestClient) -> None:
    list_r = no_auth_client.get(
        f"/api/traces/{TRACE_ID}/spans/{SPAN_LLM}/attachments"
    )
    assert list_r.status_code == 200
    aid = list_r.json()["attachments"][0]["attachment_id"]
    r = no_auth_client.get(
        f"/api/traces/{TRACE_ID}/spans/{SPAN_LLM}/attachments/{aid}"
    )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("image/")
    assert len(r.content) > 0


def test_attachments_full_data_stream(no_auth_client: TestClient) -> None:
    list_r = no_auth_client.get(
        f"/api/traces/{TRACE_ID}/spans/{SPAN_LLM}/attachments"
    )
    aid = list_r.json()["attachments"][0]["attachment_id"]
    r = no_auth_client.get(
        f"/api/traces/{TRACE_ID}/spans/{SPAN_LLM}/attachments/{aid}?full=1"
    )
    # Sprint 1 seed stores full_data, so ?full=1 succeeds.
    assert r.status_code == 200
    assert len(r.content) > 0


def test_traces_input_attribute_carries_image_content_part(
    no_auth_client: TestClient,
) -> None:
    """The seeded trace's LLM span has a content-parts message in
    ``gen_ai.request.messages`` — the frontend renders this inline."""
    r = no_auth_client.get(f"/api/traces/{TRACE_ID}/spans")
    assert r.status_code == 200, r.text
    body = r.json()
    # Find the LLM span by name.
    spans = _flatten(body.get("tree"))
    llm = next(s for s in spans if s["span_id"] == SPAN_LLM)
    import json as _json

    msg_attr = llm["attributes"].get("gen_ai.request.messages")
    assert msg_attr is not None
    # Spans store attribute values as JSON-stringified blobs; the frontend
    # parses them back before rendering.
    if isinstance(msg_attr, str):
        msg_attr = _json.loads(msg_attr)
    parts = extract_content_parts(msg_attr)
    assert any(p.get("type") == "image_url" for p in parts), parts


def _flatten(tree: dict) -> list[dict]:
    """Walk a span tree node and return all spans as a flat list."""
    out: list[dict] = []
    if not tree:
        return out
    nodes = [tree]
    while nodes:
        n = nodes.pop()
        if "span" in n:
            out.append(n["span"])
        nodes.extend(n.get("children", []))
    return out
