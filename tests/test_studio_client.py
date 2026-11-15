"""``StudioClient`` against a real local HTTP server (``CaptureServer``) — no mocks.

What is asserted is what went over a socket: method, path, query, headers and body for
every route, the plane's error codes mapped to errors, and the polling rules (a
monotonic deadline, ``retry_after_seconds``, one dropped connection tolerated).
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import pytest

from fastaiagent._internal.errors import (
    PlatformAuthError,
    PlatformConnectionError,
    PlatformRateLimitError,
    StudioError,
)
from fastaiagent.studio import StudioClient
from fastaiagent.studio.models import Draft, Trial

BASE = "/public/v1/agent-studio"
SPEC = {"spec_version": 1, "name": "order-desk", "architecture": "single"}
VERSION_WITH_STUDIO = {
    "current": "v1.12",
    "endpoints": ["GET /public/v1/agents", f"GET {BASE}/context"],
}


def _client(server, **kwargs: Any) -> StudioClient:
    return StudioClient(api_key="fa_k_test", target=server.url, **kwargs)


def _serve(server, *responses: tuple[Any, ...]) -> None:
    """Queue the plane's answers, in the order the client will ask."""
    server.set_response_sequence(list(responses))


def _calls(server) -> list[dict[str, Any]]:
    """The requests the client made."""
    return list(server.requests)


def _draft(**extra: Any) -> dict[str, Any]:
    return {"id": "d1", "name": "Order desk", "status": "drafting", **extra}


def _trial(status: str = "completed", retry: float = 0, **extra: Any) -> dict[str, Any]:
    return {"id": "t1", "status": status, "retry_after_seconds": retry, **extra}


def _error(code: str, message: str = "no", **extra: Any) -> dict[str, Any]:
    return {"detail": {"error": code, "message": message, **extra}}


# -- construction ----------------------------------------------------------------------


def test_a_missing_key_or_target_is_said_plainly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FASTAIAGENT_API_KEY", raising=False)
    monkeypatch.delenv("FASTAIAGENT_TARGET", raising=False)
    with pytest.raises(ValueError, match="No API key"):
        StudioClient(target="http://localhost:1")
    with pytest.raises(ValueError, match="No target"):
        StudioClient(api_key="fa_k_x")


def test_key_and_target_fall_back_to_the_environment(
    monkeypatch: pytest.MonkeyPatch, capture_server
) -> None:
    monkeypatch.setenv("FASTAIAGENT_API_KEY", "fa_k_env")
    monkeypatch.setenv("FASTAIAGENT_TARGET", capture_server.url + "/")
    _serve(capture_server, (200, {"project": {"id": "p"}}))
    StudioClient().context()
    assert capture_server.requests[-1]["headers"]["X-API-Key"] == "fa_k_env"


def test_the_repr_never_carries_the_key(capture_server) -> None:
    assert "fa_k_test" not in repr(_client(capture_server))


# -- does this plane have Studio? -----------------------------------------------------


def test_a_plane_without_the_studio_routes_is_named(capture_server) -> None:
    # The route is not there (a bare 404); the plane's own listing then says why.
    _serve(
        capture_server,
        (404, {"detail": "Not Found"}),
        (200, {"endpoints": ["GET /public/v1/agents"]}),
    )
    with pytest.raises(StudioError) as err:
        _client(capture_server).context()
    assert err.value.code == "studio_not_available"
    assert "does not have headless Agent Studio" in err.value.message
    assert "0.20.0" in err.value.message
    assert [r["path"] for r in _calls(capture_server)] == [
        f"{BASE}/context",
        "/public/v1/version",
    ]


def test_a_plane_with_no_version_route_either_is_named_too(capture_server) -> None:
    _serve(capture_server, (404, {"detail": "Not Found"}), (404, {"detail": "Not Found"}))
    with pytest.raises(StudioError) as err:
        _client(capture_server).list_drafts()
    assert err.value.code == "studio_not_available"


def test_the_happy_path_pays_no_version_request(capture_server) -> None:
    # The key allows 60 calls a minute and a command is a handful of them.
    _serve(capture_server, (200, []), (200, []), (200, []))
    studio = _client(capture_server)
    studio.list_drafts()
    studio.list_drafts()
    studio.list_drafts()
    assert "/public/v1/version" not in [r["path"] for r in _calls(capture_server)]


def test_a_404_from_a_plane_that_has_studio_is_not_blamed_on_the_version(capture_server) -> None:
    from fastaiagent._internal.errors import PlatformNotFoundError

    _serve(capture_server, (404, {"detail": "Not Found"}), (200, VERSION_WITH_STUDIO))
    with pytest.raises(PlatformNotFoundError):
        _client(capture_server).get_trial("t9")


def test_a_plane_that_stops_listing_its_routes_is_not_blamed(capture_server) -> None:
    from fastaiagent._internal.errors import PlatformNotFoundError

    _serve(capture_server, (404, {"detail": "Not Found"}), (200, {"current": "v1.12"}))
    with pytest.raises(PlatformNotFoundError):
        _client(capture_server).get_trial("t9")


def test_if_the_version_cannot_be_read_the_original_answer_stands(capture_server) -> None:
    from fastaiagent._internal.errors import PlatformNotFoundError

    _serve(capture_server, (404, {"detail": "Not Found"}), (500, {"detail": "down"}))
    with pytest.raises(PlatformNotFoundError):
        _client(capture_server).get_trial("t9")


def test_the_planes_own_not_found_is_a_studio_error_and_reads_no_version(capture_server) -> None:
    _serve(capture_server, (404, _error("not_found", "No such draft.")))
    with pytest.raises(StudioError) as err:
        _client(capture_server).get_draft("d9")
    assert err.value.code == "not_found"
    assert len(_calls(capture_server)) == 1


# -- every route -----------------------------------------------------------------------

ROUTES: list[tuple[str, tuple[Any, ...], dict[str, Any], tuple[int, Any], str, str, Any]] = [
    # method name, args, kwargs, the plane's answer, HTTP method, path+query, JSON body sent
    ("context", (), {}, (200, {"project": {"id": "p"}}), "GET", f"{BASE}/context", None),
    ("spec_schema", (), {}, (200, {"schema": {}}), "GET", f"{BASE}/spec-schema", None),
    (
        "list_drafts",
        (),
        {"status": "ready", "q": "order", "limit": 5, "skip": 10},
        (200, [_draft()]),
        "GET",
        f"{BASE}/drafts?status=ready&q=order&limit=5&skip=10",
        None,
    ),
    (
        "create_draft",
        ("Order desk", "Answer order questions.", "ep1"),
        {"use_harness": True},
        (201, _draft()),
        "POST",
        f"{BASE}/drafts",
        {
            "name": "Order desk",
            "requirements_text": "Answer order questions.",
            "llm_endpoint_id": "ep1",
            "use_harness": True,
        },
    ),
    (
        "get_draft",
        ("d1",),
        {"messages_after": "m9"},
        (200, _draft()),
        "GET",
        f"{BASE}/drafts/d1?messages_after=m9",
        None,
    ),
    (
        "update_draft",
        ("d1",),
        {"name": "New name"},
        (200, _draft()),
        "PATCH",
        f"{BASE}/drafts/d1",
        {"name": "New name"},
    ),
    ("delete_draft", ("d1",), {}, (204, b""), "DELETE", f"{BASE}/drafts/d1", None),
    (
        "add_text_fixture",
        ("d1", "scan", "text read from a photo"),
        {"source_filename": "receipt.jpg"},
        (201, {"id": "f1", "name": "scan", "kind": "doc"}),
        "POST",
        f"{BASE}/drafts/d1/fixtures/text",
        {"name": "scan", "text": "text read from a photo", "source_filename": "receipt.jpg"},
    ),
    (
        "get_fixture",
        ("f1",),
        {},
        (200, {"id": "f1"}),
        "GET",
        f"{BASE}/fixtures/f1",
        None,
    ),
    ("delete_fixture", ("f1",), {}, (204, b""), "DELETE", f"{BASE}/fixtures/f1", None),
    (
        "validate_spec",
        ("d1", SPEC),
        {},
        (200, {"ok": True}),
        "POST",
        f"{BASE}/drafts/d1/spec/validate",
        {"spec": SPEC},
    ),
    (
        "create_version",
        ("d1", SPEC),
        {"note": "first design"},
        (202, {"version_id": "v1", "version": 1}),
        "POST",
        f"{BASE}/drafts/d1/versions",
        {"spec": SPEC, "note": "first design"},
    ),
    (
        "list_versions",
        ("d1",),
        {},
        (200, [{"id": "v1", "version": 1}]),
        "GET",
        f"{BASE}/drafts/d1/versions",
        None,
    ),
    (
        "get_version",
        ("d1", "v1"),
        {"include_files": True},
        (200, {"id": "v1", "version": 1}),
        "GET",
        f"{BASE}/drafts/d1/versions/v1?include=files",
        None,
    ),
    (
        "diff_versions",
        ("d1", "v2", "v1"),
        {},
        (200, {"from": 1, "to": 2, "spec_changes": []}),
        "GET",
        f"{BASE}/drafts/d1/versions/v2/diff?against=v1",
        None,
    ),
    (
        "start_sandbox",
        ("d1",),
        {},
        (201, {"status": "requested"}),
        "POST",
        f"{BASE}/drafts/d1/sandbox",
        None,
    ),
    (
        "get_sandbox",
        ("d1",),
        {},
        (200, {"status": "ready"}),
        "GET",
        f"{BASE}/drafts/d1/sandbox",
        None,
    ),
    (
        "start_trial",
        ("d1", "Where is order 1003?"),
        {"worker": "orders"},
        (201, _trial("queued", 2)),
        "POST",
        f"{BASE}/drafts/d1/trials",
        {"input": "Where is order 1003?", "worker": "orders"},
    ),
    ("get_trial", ("t1",), {}, (200, _trial()), "GET", f"{BASE}/trials/t1", None),
    (
        "list_trials",
        ("d1",),
        {"kind": "eval_preview"},
        (200, [_trial()]),
        "GET",
        f"{BASE}/drafts/d1/trials?kind=eval_preview",
        None,
    ),
    (
        "start_eval_run",
        ("d1",),
        {"worker": "orders", "case": "found"},
        (201, {"runs": [_trial("queued", 2)]}),
        "POST",
        f"{BASE}/drafts/d1/evals/run",
        {"worker": "orders", "case": "found"},
    ),
    (
        "get_evals",
        ("d1",),
        {"version_id": "v1"},
        (200, {"version": 1, "totals": {"passed": 2}}),
        "GET",
        f"{BASE}/drafts/d1/evals?version_id=v1",
        None,
    ),
    (
        "link_eval_dataset",
        ("d1",),
        {},
        (201, {"dataset_id": "ds1"}),
        "POST",
        f"{BASE}/drafts/d1/eval-dataset",
        {},
    ),
    (
        "import_eval_dataset",
        ("d1",),
        {},
        (202, {"version_id": "v2", "version": 2}),
        "POST",
        f"{BASE}/drafts/d1/eval-dataset/import",
        None,
    ),
    (
        "promote",
        ("d1", "agent-9"),
        {},
        (201, {"suite_id": "s1", "dataset_id": "ds1", "cases": 3}),
        "POST",
        f"{BASE}/drafts/d1/promote",
        {"agent_id": "agent-9"},
    ),
]


@pytest.mark.parametrize("route", ROUTES, ids=[r[0] for r in ROUTES])
def test_each_route_sends_what_the_contract_says(capture_server, route) -> None:
    name, args, kwargs, answer, method, path, body = route
    _serve(capture_server, answer)
    getattr(_client(capture_server), name)(*args, **kwargs)
    (request,) = _calls(capture_server)
    assert request["method"] == method
    assert request["path"] == path
    assert request["body"] == body
    headers = {k.lower(): v for k, v in request["headers"].items()}
    assert headers["x-api-key"] == "fa_k_test"
    assert headers["x-faa-protocol"] == "v1.12"


def test_every_route_in_the_contract_has_a_method() -> None:
    """The plane lists its routes at ``/version``; each one is a method here."""
    listed = {
        ("GET", "/context"),
        ("GET", "/spec-schema"),
        ("GET", "/drafts"),
        ("POST", "/drafts"),
        ("GET", "/drafts/{id}"),
        ("PATCH", "/drafts/{id}"),
        ("DELETE", "/drafts/{id}"),
        ("POST", "/drafts/{id}/fixtures"),
        ("POST", "/drafts/{id}/fixtures/text"),
        ("GET", "/fixtures/{id}"),
        ("DELETE", "/fixtures/{id}"),
        ("POST", "/drafts/{id}/spec/validate"),
        ("POST", "/drafts/{id}/versions"),
        ("GET", "/drafts/{id}/versions"),
        ("GET", "/drafts/{id}/versions/{vid}"),
        ("GET", "/drafts/{id}/versions/{vid}/diff"),
        ("POST", "/drafts/{id}/sandbox"),
        ("GET", "/drafts/{id}/sandbox"),
        ("POST", "/drafts/{id}/trials"),
        ("GET", "/trials/{id}"),
        ("GET", "/drafts/{id}/trials"),
        ("POST", "/drafts/{id}/evals/run"),
        ("GET", "/drafts/{id}/evals"),
        ("POST", "/drafts/{id}/eval-dataset"),
        ("POST", "/drafts/{id}/eval-dataset/import"),
        ("GET", "/drafts/{id}/export"),
        ("POST", "/drafts/{id}/promote"),
    }
    covered = {(r[4], r[5].split("?")[0]) for r in ROUTES}
    # Upload and export are not JSON calls; their own tests cover them below.
    covered |= {("POST", f"{BASE}/drafts/d1/fixtures"), ("GET", f"{BASE}/drafts/d1/export")}
    normal = {
        (m, p.replace(BASE, "").replace("/d1", "/{id}").replace("/f1", "/{id}")) for m, p in covered
    }
    normal = {
        (m, p.replace("/t1", "/{id}").replace("/v2", "/{vid}").replace("/v1", "/{vid}"))
        for m, p in normal
    }
    missing = listed - normal
    assert not missing, f"routes with no method under test: {sorted(missing)}"


def test_upload_fixtures_is_a_multipart_form(capture_server) -> None:
    _serve(capture_server, (201, [{"id": "f1", "name": "orders", "kind": "csv", "rows": []}]))
    out = _client(capture_server).upload_fixtures(
        "d1", [("orders.csv", b"order_id,status\n1003,shipped\n")]
    )
    (request,) = _calls(capture_server)
    assert request["path"] == f"{BASE}/drafts/d1/fixtures"
    assert request["headers"]["Content-Type"].startswith("multipart/form-data")
    assert b'filename="orders.csv"' in request.raw and b"1003,shipped" in request.raw
    assert out[0].name == "orders" and out[0].kind == "csv"


def test_export_streams_the_zip_to_a_file(capture_server, tmp_path: Path) -> None:
    payload = b"PK\x03\x04" + bytes(range(256)) * 64
    _serve(capture_server, (200, payload, "application/zip"))
    dest = tmp_path / "project.zip"
    assert _client(capture_server).export("d1", dest, version_id="v1") == dest
    assert dest.read_bytes() == payload
    assert _calls(capture_server)[0]["path"] == f"{BASE}/drafts/d1/export?version_id=v1"


def test_a_domain_level_key_names_its_project_on_every_call(capture_server) -> None:
    _serve(capture_server, (200, []), (204, b""))
    studio = _client(capture_server, project="proj-1")
    studio.list_drafts(q="order")
    studio.delete_draft("d1")
    first, second = _calls(capture_server)
    assert first["path"] == f"{BASE}/drafts?q=order&project_id=proj-1"
    assert second["path"] == f"{BASE}/drafts/d1?project_id=proj-1"


def test_no_project_is_sent_when_none_is_named(capture_server) -> None:
    _serve(capture_server, (200, []))
    _client(capture_server).list_drafts()
    assert "project_id" not in _calls(capture_server)[0]["path"]


# -- the answers -----------------------------------------------------------------------


def test_the_models_keep_keys_they_do_not_know_and_read_null_as_absent(capture_server) -> None:
    _serve(
        capture_server,
        (200, [_draft(updated_at=None, brand_new_key={"a": 1}, steps_summary={"built": True})]),
    )
    (draft,) = _client(capture_server).list_drafts()
    assert isinstance(draft, Draft)
    assert draft.updated_at == ""  # null → the field's default
    assert draft.model_extra == {"brand_new_key": {"a": 1}}
    assert draft.steps_summary is not None and draft.steps_summary.built is True


def test_a_trial_reads_its_answer_whatever_shape_result_has() -> None:
    assert Trial(id="t", result={"output": "Order 1003 shipped"}).output == "Order 1003 shipped"
    assert Trial(id="t", result="plain").output == "plain"
    assert Trial(id="t").output is None


def test_the_schema_answer_keeps_the_json_schema_under_a_safe_name(capture_server) -> None:
    _serve(
        capture_server,
        (
            200,
            {"spec_version": 1, "schema": {"title": "AgentSpec"}, "guardrails": [{"factory": "x"}]},
        ),
    )
    schema = _client(capture_server).spec_schema()
    assert schema.json_schema == {"title": "AgentSpec"}
    assert schema.guardrails[0].factory == "x"


# -- errors ----------------------------------------------------------------------------


def test_a_refused_spec_carries_its_problems(capture_server) -> None:
    problems = [{"where": "tools[0]", "message": "fixture 'nope' is not uploaded"}]
    _serve(
        capture_server, (422, _error("spec_invalid", "The spec has problems.", problems=problems))
    )
    with pytest.raises(StudioError) as err:
        _client(capture_server).create_version("d1", SPEC)
    assert err.value.code == "spec_invalid"
    assert err.value.status == 422
    assert err.value.problems == problems
    assert "spec_invalid" in str(err.value)


@pytest.mark.parametrize(
    ("status", "code", "extra"),
    [
        (403, "insufficient_scope", {"required_scope": "studio:write"}),
        (403, "environment_rule", {}),
        (403, "key_has_no_owner", {}),
        (404, "not_found", {}),
        (409, "name_taken", {}),
        (409, "no_llm_endpoint", {}),
        (409, "build_in_progress", {}),
        (400, "worker_required", {"workers": ["orders", "billing"]}),
        (404, "unknown_case", {}),
        (409, "dataset_already_linked", {"dataset_id": "ds1"}),
        (422, "no_text_layer", {}),
    ],
)
def test_the_planes_codes_become_studio_errors(capture_server, status, code, extra) -> None:
    _serve(capture_server, (status, _error(code, "words", **extra)))
    with pytest.raises(StudioError) as err:
        _client(capture_server).start_trial("d1", "hi")
    assert (err.value.code, err.value.status, err.value.message) == (code, status, "words")
    assert err.value.details == extra


@pytest.mark.parametrize(("status", "code"), [(401, "invalid_api_key"), (400, "customer_required")])
def test_no_usable_key_is_an_auth_error_whichever_way_the_plane_says_it(
    capture_server, status, code
) -> None:
    # A SaaS server that holds several workspaces answers 400 before it can say 401.
    _serve(capture_server, (status, _error(code, "A key is required.")))
    with pytest.raises(PlatformAuthError, match="studio:read and studio:write"):
        _client(capture_server).context()


def _limited(seconds: str | None) -> tuple[Any, ...]:
    headers = {"Retry-After": seconds} if seconds is not None else {}
    body = _error("rate_limited", "slow down", retry_after=seconds)
    return (429, body, "application/json", headers)


def test_a_long_rate_limit_reaches_the_caller_with_the_wait_on_it(capture_server) -> None:
    _serve(capture_server, _limited("60"), (503, {"detail": "down"}))
    studio = _client(capture_server)
    started = time.monotonic()
    with pytest.raises(PlatformRateLimitError, match="60") as err:
        studio.context()
    assert err.value.retry_after == 60.0
    assert time.monotonic() - started < 5  # a minute is told, not silently waited out
    assert len(_calls(capture_server)) == 1
    with pytest.raises(PlatformConnectionError):
        studio.context()  # a server error keeps its own class


def test_a_short_rate_limit_is_waited_out_and_the_call_made_again(capture_server) -> None:
    _serve(capture_server, _limited("0.3"), (200, {"project": {"id": "p"}}))
    started = time.monotonic()
    assert _client(capture_server).context().project.id == "p"
    assert time.monotonic() - started >= 0.3
    assert len(_calls(capture_server)) == 2


def test_a_write_refused_by_the_rate_limit_is_asked_again_once_not_twice(capture_server) -> None:
    # The plane refuses before it does anything, so the retry cannot double a write.
    _serve(capture_server, _limited("0.05"), (201, _draft(status="drafting")))
    draft = _client(capture_server).create_draft("Order desk", "Answer order questions.", "ep1")
    assert draft.id == "d1"
    assert [r["method"] for r in _calls(capture_server)] == ["POST", "POST"]


def test_a_rate_limit_that_will_not_lift_gives_up_after_two_retries(capture_server) -> None:
    _serve(capture_server, *[_limited("0.05")] * 5)
    with pytest.raises(PlatformRateLimitError):
        _client(capture_server).context()
    assert len(_calls(capture_server)) == 3  # the call and two more


def test_a_rate_limit_with_no_retry_after_is_not_guessed_at(capture_server) -> None:
    _serve(capture_server, _limited(None))
    with pytest.raises(PlatformRateLimitError):
        _client(capture_server).context()
    assert len(_calls(capture_server)) == 1  # no number from the plane, so nothing to wait on


def test_a_wait_keeps_going_through_a_rate_limit_until_its_deadline(capture_server) -> None:
    done = {"id": "t1", "status": "completed", "retry_after_seconds": 0}
    # More 429s than the short retries absorb: the wait itself must carry on.
    _serve(capture_server, *[_limited("0.05")] * 4, (200, done))
    assert _client(capture_server).wait_for_trial("t1", timeout=20).status == "completed"


def test_the_rate_limit_reaches_a_wait_that_has_run_out_of_time(capture_server) -> None:
    _serve(capture_server, *[_limited("1")] * 50)
    with pytest.raises(StudioError) as err:
        _client(capture_server).wait_for_trial("t1", timeout=0.5)
    assert err.value.code == "wait_timeout"


def test_the_server_keeps_its_platform_error(capture_server) -> None:
    _serve(capture_server, (503, {"detail": "down"}))
    with pytest.raises(PlatformConnectionError):
        _client(capture_server).context()


def test_an_error_body_that_is_not_the_studio_shape_falls_back(capture_server) -> None:
    from fastaiagent._internal.errors import PlatformNotFoundError

    _serve(capture_server, (404, {"detail": "Not Found"}))
    with pytest.raises(PlatformNotFoundError):
        _client(capture_server).get_draft("d1")


# -- waiting ---------------------------------------------------------------------------


def test_wait_for_version_polls_until_the_plane_says_stop(capture_server) -> None:
    pending = {"id": "v1", "generation_status": "running", "retry_after_seconds": 0.01}
    done = {"id": "v1", "generation_status": "completed", "retry_after_seconds": 0}
    _serve(capture_server, (200, pending), (200, pending), (200, done))
    version = _client(capture_server).wait_for_version("d1", "v1")
    assert version.generation_status == "completed"
    assert len(_calls(capture_server)) == 3


def test_a_failed_version_is_final_too(capture_server) -> None:
    failed = {
        "id": "v1",
        "generation_status": "failed",
        "error_message": "bad",
        "retry_after_seconds": 0,
    }
    _serve(capture_server, (200, failed))
    version = _client(capture_server).wait_for_version("d1", "v1")
    assert (version.generation_status, version.error_message) == ("failed", "bad")


def test_the_pause_between_reads_is_the_planes_not_a_guess(capture_server) -> None:
    slow = {"id": "t1", "status": "running", "retry_after_seconds": 0.4}
    _serve(capture_server, (200, slow), (200, _trial()))
    started = time.monotonic()
    _client(capture_server).wait_for_trial("t1")
    assert time.monotonic() - started >= 0.4


def test_a_wait_stops_at_its_deadline_by_the_clock_not_by_counting(capture_server) -> None:
    capture_server.delay = 0.05  # each read takes time too; a sleep counter would overshoot
    forever = {"id": "t1", "status": "running", "retry_after_seconds": 0.1}
    capture_server.set_response_sequence([(200, forever)] * 200)
    started = time.monotonic()
    with pytest.raises(StudioError) as err:
        _client(capture_server).wait_for_trial("t1", timeout=0.6)
    elapsed = time.monotonic() - started
    assert err.value.code == "wait_timeout"
    assert 0.5 <= elapsed < 1.5


def test_a_plane_that_sends_no_pause_is_judged_by_status(capture_server) -> None:
    _serve(capture_server, (200, {"id": "s", "status": "ready"}))
    assert _client(capture_server).wait_for_sandbox("d1").status == "ready"


class _DropsOnce:
    """A real server that closes the first connection without answering, then behaves."""

    def __init__(self) -> None:
        self.hits = 0
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:
                pass

            def do_GET(self) -> None:  # noqa: N802
                outer.hits += 1
                if outer.hits == 1:
                    self.connection.close()  # no status line: a dropped connection
                    return
                body = json.dumps(
                    {"id": "t1", "status": "completed", "retry_after_seconds": 0}
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()


def test_one_dropped_connection_is_tolerated_while_waiting() -> None:
    server = _DropsOnce()
    try:
        trial = _client(server).wait_for_trial("t1", timeout=10)
    finally:
        server.stop()
    assert trial.status == "completed"
    assert server.hits == 2  # the dropped read, and the one retry


def test_a_dead_plane_is_not_retried_forever() -> None:
    server = _DropsOnce()
    studio = _client(server)
    server.stop()
    with pytest.raises(PlatformConnectionError):
        studio.wait_for_trial("t1", timeout=10)


# -- the sandbox trap ------------------------------------------------------------------

READY = (200, {"status": "ready", "retry_after_seconds": 0})
STARTING = (201, {"status": "requested", "retry_after_seconds": 3})


def test_run_trial_starts_the_sandbox_then_asks(capture_server) -> None:
    _serve(
        capture_server,
        STARTING,
        READY,
        (201, _trial("queued", 0.01)),
        (200, _trial("completed", 0, result={"output": "shipped"})),
    )
    trial = _client(capture_server).run_trial("d1", "Where is order 1003?")
    assert trial.output == "shipped"
    assert [(r["method"], r["path"]) for r in _calls(capture_server)] == [
        ("POST", f"{BASE}/drafts/d1/sandbox"),
        ("GET", f"{BASE}/drafts/d1/sandbox"),
        ("POST", f"{BASE}/drafts/d1/trials"),
        ("GET", f"{BASE}/trials/t1"),
    ]


def test_run_trial_re_arms_a_sandbox_that_was_ready_but_gone(capture_server) -> None:
    _serve(
        capture_server,
        STARTING,
        READY,
        (409, _error("no_active_sandbox", "not ready")),
        STARTING,
        READY,
        (201, _trial("queued", 0.01)),
        (200, _trial("completed", 0, result={"output": "ok"})),
    )
    trial = _client(capture_server).run_trial("d1", "hi")
    assert trial.status == "completed"
    posts = [r["path"] for r in _calls(capture_server) if r["method"] == "POST"]
    assert posts.count(f"{BASE}/drafts/d1/sandbox") == 2


def test_run_trial_re_arms_when_the_trial_ends_failed_on_a_dead_runner(capture_server) -> None:
    _serve(
        capture_server,
        STARTING,
        READY,
        (201, _trial("queued", 0.01)),
        (200, _trial("failed", 0, error="runner offline")),
        STARTING,
        READY,
        (201, _trial("queued", 0.01)),
        (200, _trial("completed", 0, result={"output": "ok"})),
    )
    assert _client(capture_server).run_trial("d1", "hi").status == "completed"


def test_run_trial_gives_up_after_three_attempts_and_says_why(capture_server) -> None:
    refusal = (409, _error("no_active_sandbox", "not ready"))
    _serve(capture_server, *([STARTING, READY, refusal] * 3))
    with pytest.raises(StudioError) as err:
        _client(capture_server).run_trial("d1", "hi")
    assert err.value.code == "no_active_sandbox"


def test_another_refusal_is_not_retried(capture_server) -> None:
    _serve(
        capture_server, STARTING, READY, (400, _error("worker_required", "name one", workers=["a"]))
    )
    with pytest.raises(StudioError) as err:
        _client(capture_server).run_trial("d1", "hi")
    assert err.value.code == "worker_required" and err.value.details["workers"] == ["a"]
    assert len(_calls(capture_server)) == 3


def test_run_trial_without_waiting_returns_the_started_trial(capture_server) -> None:
    _serve(capture_server, STARTING, READY, (201, _trial("queued", 2)))
    trial = _client(capture_server).run_trial("d1", "hi", wait=False)
    assert trial.status == "queued"
    assert len(_calls(capture_server)) == 3


def test_run_evals_waits_for_every_run_of_a_team(capture_server) -> None:
    _serve(
        capture_server,
        STARTING,
        READY,
        (
            201,
            {
                "runs": [
                    {**_trial("queued", 0.01), "id": "r1", "worker": "orders"},
                    {**_trial("queued", 0.01), "id": "r2", "worker": "billing"},
                ]
            },
        ),
        (200, {**_trial("completed"), "id": "r1", "worker": "orders"}),
        (200, {**_trial("completed"), "id": "r2", "worker": "billing"}),
    )
    runs = _client(capture_server).run_evals("team")
    assert [(r.worker, r.status) for r in runs] == [
        ("orders", "completed"),
        ("billing", "completed"),
    ]


def test_run_evals_recovers_from_a_runner_that_went_away(capture_server) -> None:
    gone = {**_trial("failed", 0, error="the runner went away"), "id": "r1"}
    ok = {**_trial("completed"), "id": "r1"}
    _serve(
        capture_server,
        STARTING,
        READY,
        (201, {"runs": [{**_trial("queued", 0.01), "id": "r1"}]}),
        (200, gone),
        STARTING,
        READY,
        (201, {"runs": [{**_trial("queued", 0.01), "id": "r1"}]}),
        (200, ok),
    )
    (run,) = _client(capture_server).run_evals("d1")
    assert run.status == "completed"


# -- finding a draft ---------------------------------------------------------------------


def test_a_draft_is_found_by_name_whatever_its_case(capture_server) -> None:
    _serve(
        capture_server,
        (200, [_draft(id="d9", name="Order Desk"), _draft(id="d8", name="Order desk 2")]),
    )
    found = _client(capture_server).find_draft("order desk")
    assert found.id == "d9"
    assert _calls(capture_server)[0]["path"].startswith(f"{BASE}/drafts?q=order+desk")


def test_a_uuid_is_read_directly_and_a_missing_name_is_said(capture_server) -> None:
    ident = "2d70c0de-1111-4222-8333-444455556666"
    _serve(capture_server, (200, _draft(id=ident)), (200, []))
    studio = _client(capture_server)
    assert studio.find_draft(ident).id == ident
    assert _calls(capture_server)[0]["path"] == f"{BASE}/drafts/{ident}"
    with pytest.raises(StudioError) as err:
        studio.find_draft("nothing here")
    assert err.value.code == "not_found"


# -- async -----------------------------------------------------------------------------


def test_the_async_twins_do_the_same(capture_server, tmp_path: Path) -> None:
    async def run() -> None:
        studio = _client(capture_server)
        _serve(
            capture_server,
            (200, {"project": {"id": "p"}}),
            (202, {"version_id": "v1", "version": 1}),
            (200, {"id": "v1", "generation_status": "running", "retry_after_seconds": 0.01}),
            (200, {"id": "v1", "generation_status": "completed", "retry_after_seconds": 0}),
            (201, {"status": "requested", "retry_after_seconds": 3}),
            (200, {"status": "ready", "retry_after_seconds": 0}),
            (201, _trial("queued", 0.01)),
            (200, _trial("completed", 0, result={"output": "async ok"})),
            (200, b"zip", "application/zip"),
        )
        assert (await studio.acontext()).project.id == "p"
        accepted = await studio.acreate_version("d1", SPEC, note="n")
        assert (
            await studio.await_for_version("d1", accepted.version_id)
        ).generation_status == "completed"
        assert (await studio.arun_trial("d1", "hi")).output == "async ok"
        dest = tmp_path / "p.zip"
        assert await studio.aexport("d1", dest) == dest and dest.read_bytes() == b"zip"

    asyncio.run(run())


def test_the_async_path_checks_the_plane_has_studio(capture_server) -> None:
    capture_server.set_response_sequence(
        [(404, {"detail": "Not Found"}), (200, {"endpoints": ["GET /public/v1/agents"]})]
    )
    with pytest.raises(StudioError) as err:
        asyncio.run(_client(capture_server).alist_drafts())
    assert err.value.code == "studio_not_available"
