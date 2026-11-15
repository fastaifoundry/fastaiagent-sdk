"""``PlatformAPI`` verbs added in 1.102.0: patch, delete, post_multipart, download.

Each runs against ``CaptureServer`` — a real localhost HTTP server with real ``httpx`` —
so what is asserted is what went over a socket, not what a stand-in was handed.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from fastaiagent._internal.errors import (
    PlatformAuthError,
    PlatformConnectionError,
    PlatformNotFoundError,
)
from fastaiagent._platform.api import PlatformAPI


def _api(server) -> PlatformAPI:
    return PlatformAPI(api_key="fa_k_test", base_url=server.url)


def _headers(request: dict) -> dict[str, str]:
    return {k.lower(): v for k, v in request["headers"].items()}


def test_patch_sends_json_with_the_key_and_wire_headers(capture_server) -> None:
    capture_server.set_response_sequence([(200, {"id": "d1", "name": "renamed"})])
    out = _api(capture_server).patch(
        "/x/drafts/d1", {"name": "renamed"}, params={"project_id": "p"}
    )
    assert out == {"id": "d1", "name": "renamed"}
    (request,) = capture_server.requests
    assert request["method"] == "PATCH"
    assert request["path"] == "/x/drafts/d1?project_id=p"
    assert request["body"] == {"name": "renamed"}
    headers = _headers(request)
    assert headers["x-api-key"] == "fa_k_test"
    assert headers["x-faa-protocol"] == "v1.12"
    assert headers["content-type"] == "application/json"


def test_delete_answers_empty_for_a_204(capture_server) -> None:
    capture_server.set_response_sequence([(204, b"")])
    assert _api(capture_server).delete("/x/drafts/d1") == {}
    (request,) = capture_server.requests
    assert request["method"] == "DELETE"
    assert request.raw == b""


def test_a_list_body_comes_back_as_a_list(capture_server) -> None:
    capture_server.set_response_sequence([(200, [{"id": "a"}, {"id": "b"}])])
    assert _api(capture_server).get("/x/drafts") == [{"id": "a"}, {"id": "b"}]


def test_post_with_no_body_and_a_query(capture_server) -> None:
    capture_server.set_response_sequence([(201, {"status": "requested"})])
    out = _api(capture_server).post("/x/drafts/d1/sandbox", None, params={"project_id": "p"})
    assert out == {"status": "requested"}
    (request,) = capture_server.requests
    assert request["path"] == "/x/drafts/d1/sandbox?project_id=p"
    assert request.raw == b""


def test_post_multipart_sends_each_file_under_one_field(capture_server) -> None:
    capture_server.set_response_sequence([(201, [{"name": "orders"}, {"name": "notes"}])])
    out = _api(capture_server).post_multipart(
        "/x/drafts/d1/fixtures",
        [("orders.csv", b"order_id,status\n1003,shipped\n"), ("notes.txt", b"hello")],
    )
    assert [f["name"] for f in out] == ["orders", "notes"]
    (request,) = capture_server.requests
    content_type = _headers(request)["content-type"]
    assert content_type.startswith("multipart/form-data; boundary=")
    raw = request.raw
    assert raw.count(b'name="files"') == 2
    assert b'filename="orders.csv"' in raw and b"Content-Type: text/csv" in raw
    assert b"1003,shipped" in raw and b'filename="notes.txt"' in raw
    # The JSON content type must not leak onto an upload.
    assert "application/json" not in content_type


def test_download_streams_the_body_to_a_file(capture_server, tmp_path: Path) -> None:
    payload = bytes(range(256)) * 4096  # 1 MiB of non-text bytes
    capture_server.set_response_sequence([(200, payload, "application/zip")])
    dest = tmp_path / "project.zip"
    result = _api(capture_server).download("/x/drafts/d1/export", dest, params={"version_id": "v"})
    assert result == dest
    assert dest.read_bytes() == payload
    (request,) = capture_server.requests
    assert request["path"] == "/x/drafts/d1/export?version_id=v"


def test_download_writes_nothing_when_the_plane_refuses(capture_server, tmp_path: Path) -> None:
    capture_server.set_response_sequence([(404, {"detail": {"error": "not_found"}})])
    dest = tmp_path / "project.zip"
    with pytest.raises(PlatformNotFoundError):
        _api(capture_server).download("/x/drafts/nope/export", dest)
    assert not dest.exists()


def test_the_new_verbs_share_the_existing_error_mapping(capture_server) -> None:
    capture_server.set_response_sequence([(401, {"detail": "no"})])
    with pytest.raises(PlatformAuthError):
        _api(capture_server).delete("/x/drafts/d1")


def test_a_dead_server_is_a_connection_error_for_every_verb(capture_server) -> None:
    api = _api(capture_server)
    capture_server.stop()
    for call in (
        lambda: api.get("/x"),
        lambda: api.patch("/x", {}),
        lambda: api.delete("/x"),
        lambda: api.post_multipart("/x", [("a.txt", b"a")]),
    ):
        with pytest.raises(PlatformConnectionError, match="Cannot connect"):
            call()


def test_async_verbs_match_the_sync_ones(capture_server, tmp_path: Path) -> None:
    api = _api(capture_server)

    async def run() -> None:
        capture_server.set_response_sequence(
            [
                (200, {"ok": 1}),
                (204, b""),
                (201, [{"name": "orders"}]),
                (200, b"zipbytes", "application/zip"),
            ]
        )
        assert await api.apatch("/x/a", {"k": 1}) == {"ok": 1}
        assert await api.adelete("/x/a") == {}
        assert await api.apost_multipart("/x/f", [("orders.csv", b"a,b\n1,2\n")]) == [
            {"name": "orders"}
        ]
        dest = tmp_path / "out.zip"
        assert await api.adownload("/x/export", dest) == dest
        assert dest.read_bytes() == b"zipbytes"

    asyncio.run(run())
    assert [r["method"] for r in capture_server.requests] == ["PATCH", "DELETE", "POST", "GET"]
