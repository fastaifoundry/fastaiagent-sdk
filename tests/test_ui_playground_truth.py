"""The Playground tells the truth about a model call (1.89.0, WS5 P6–P8, P10).

* **Failures say why** (P8, D1): every failure read "LLM call failed.". Now the
  response carries ``kind`` (auth, not_found, bad_request, rate_limited,
  provider_error, connection, timeout, config, internal), the provider status,
  the provider's own message (scrubbed of keys, request and org ids) and a hint.
  Error codes, exception class names and the traceback never appear; the root
  span keeps the opaque message. ``FASTAIAGENT_UI_PROVIDER_ERRORS=redacted``
  restores the opaque response.
* **Sampling rules** (P7, D4): ``/models`` returns the table; it only hints.
* **Truncation** (P10f, D5): a streamed call recorded no finish reason, so a
  reply cut off at max_tokens looked complete.
* **Unfilled variables** (P10e, D10), **prompt attribution** (P10c, D9),
  **root span** (P10g), **endpoint required** (P10i), **save as eval case**
  with the run's image (P6), **aliases** and a **line diff** (P10a/b).

A real localhost HTTP server speaking the OpenAI / Anthropic wire format
(``tests/_sse_provider.py``), the real ``LLMClient``, real tracing and SQLite,
real FastAPI ``TestClient``; no mocks.
"""

from __future__ import annotations

import base64
import json
import socket
import sqlite3
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("itsdangerous")

from fastapi.testclient import TestClient  # noqa: E402

from tests._sse_provider import (  # noqa: E402
    ECHOED_KEY,
    Reply,
    SSEProvider,
    anthropic_stream,
    openai_chunk,
    openai_stream,
    openai_usage_chunk,
)

TOKEN = "sk-local-test-token-000000000000"


@pytest.fixture
def provider() -> Any:
    p = SSEProvider().start()
    yield p
    p.stop()


@pytest.fixture
def env(isolated_local_db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    from fastaiagent.trace import otel
    from fastaiagent.ui.server import build_app

    monkeypatch.delenv("FASTAIAGENT_UI_PROVIDER_ERRORS", raising=False)
    otel.reset()
    app = build_app(db_path=str(isolated_local_db), auth_path=tmp_path / "a.json", no_auth=True)
    with TestClient(app) as client:
        yield client, isolated_local_db
    otel.reset()


def _body(server: Any, **kw: Any) -> dict[str, Any]:
    base = {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "prompt_template": "Say hello to {{name}}",
        "variables": {"name": "Ada"},
        "base_url": f"{server.url}/v1",
        "api_key": TOKEN,
        "parameters": {"max_tokens": 64},
    }
    base.update(kw)
    return base


def _spans(db: Path) -> list[dict[str, Any]]:
    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT trace_id, name, parent_span_id, status, attributes, events FROM spans"
        ).fetchall()
    return [
        {**dict(r), "attributes": json.loads(r["attributes"] or "{}"), "events": r["events"]}
        for r in rows
    ]


def _sse_events(text: str) -> list[tuple[str, dict[str, Any]]]:
    out = []
    for block in text.strip().split("\n\n"):
        lines = block.splitlines()
        name = next((ln[7:] for ln in lines if ln.startswith("event: ")), "")
        data = next((ln[6:] for ln in lines if ln.startswith("data: ")), "{}")
        out.append((name, json.loads(data)))
    return out


# --------------------------------------------------------------------------- #
# P8: failures say why
# --------------------------------------------------------------------------- #

_LEAK = (ECHOED_KEY, "req_ABC123", "org-XYZ789", "invalid_api_key", "LLMProviderError")


def _error_body(message: str, code: str = "invalid_api_key") -> dict[str, Any]:
    return {"error": {"message": message, "type": "invalid_request_error", "code": code}}


@pytest.mark.parametrize("route", ["/api/playground/run", "/api/playground/stream"])
def test_an_auth_failure_says_why_without_leaking(env: Any, provider: Any, route: str) -> None:
    client, db = env
    provider.script(
        Reply(
            status=401,
            body=_error_body(
                f"Incorrect API key provided: {ECHOED_KEY}. request id req_ABC123 for org-XYZ789"
            ),
        )
    )
    r = client.post(route, json=_body(provider))
    if route.endswith("/run"):
        assert r.status_code == 502, r.text
        detail = r.json()["detail"]
    else:
        (name, detail) = [e for e in _sse_events(r.text) if e[0] == "error"][0]
        assert detail["message"] == "LLM call failed."
    assert detail["kind"] == "auth" and detail["provider_status"] == 401
    assert "Incorrect API key provided" in detail["provider_message"]
    assert detail["hint"] and "credential" in detail["hint"]
    assert detail["correlation_id"]
    for leak in _LEAK:
        assert leak not in r.text, leak

    root = next(s for s in _spans(db) if s["name"] == "playground.run")
    assert root["status"] == "ERROR"
    assert "Incorrect API key" not in json.dumps(root), "provider text reached the root span"


def test_a_claude_temperature_refusal_gets_the_rule_as_hint(env: Any, provider: Any) -> None:
    client, _ = env
    provider.script(
        Reply(
            status=400,
            body={
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": "`temperature` is deprecated for this model.",
                },
            },
        )
    )
    r = client.post(
        "/api/playground/run",
        json=_body(
            provider,
            provider="anthropic",
            model="claude-opus-4-8",
            base_url=f"{provider.url}/v1",
            parameters={"temperature": 0.7, "max_tokens": 64},
        ),
    )
    detail = r.json()["detail"]
    assert detail["kind"] == "bad_request"
    assert detail["provider_message"] == "`temperature` is deprecated for this model."
    assert "default temperature (1)" in detail["hint"]


def test_an_unknown_model_is_not_found(env: Any, provider: Any) -> None:
    client, _ = env
    provider.script(Reply(status=404, body=_error_body("The model `gpt-nope` does not exist", "m")))
    detail = client.post("/api/playground/run", json=_body(provider, model="gpt-nope")).json()[
        "detail"
    ]
    assert detail["kind"] == "not_found" and "gpt-nope" in detail["hint"]


def test_a_refused_connection_names_the_endpoint(env: Any) -> None:
    client, _ = env
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    body = {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "prompt_template": "hi",
        "base_url": f"http://127.0.0.1:{port}/v1",
        "api_key": TOKEN,
    }
    detail = client.post("/api/playground/run", json=body).json()["detail"]
    assert detail["kind"] == "connection"
    assert f"127.0.0.1:{port}" in detail["hint"]


def test_redacted_mode_restores_the_opaque_message(
    env: Any, provider: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, _ = env
    monkeypatch.setenv("FASTAIAGENT_UI_PROVIDER_ERRORS", "redacted")
    provider.script(Reply(status=401, body=_error_body("Incorrect API key provided")))
    detail = client.post("/api/playground/run", json=_body(provider)).json()["detail"]
    assert set(detail) == {"error", "correlation_id"}


@pytest.mark.parametrize(
    ("status", "kind"),
    [
        (401, "auth"),
        (403, "auth"),
        (404, "not_found"),
        (400, "bad_request"),
        (422, "bad_request"),
        (429, "rate_limited"),
        (500, "provider_error"),
        (503, "provider_error"),
    ],
)
def test_every_status_maps_to_a_kind(status: int, kind: str) -> None:
    from fastaiagent._internal.errors import LLMProviderError
    from fastaiagent.ui.llm_errors import describe_llm_failure

    exc = LLMProviderError("x", status_code=status, provider_message="why")
    failure = describe_llm_failure(exc, provider="openai", model="gpt-4o-mini")
    assert failure.kind == kind and failure.provider_status == status


def test_a_custom_provider_without_an_endpoint_is_refused_before_the_call(env: Any) -> None:
    client, _ = env
    rows = {p["provider"]: p for p in client.get("/api/playground/models").json()["providers"]}
    assert rows["custom"]["requires_endpoint"] is True
    assert rows["openai"]["requires_endpoint"] is False
    r = client.post(
        "/api/playground/run",
        json={"provider": "custom", "model": "m", "prompt_template": "hi", "api_key": TOKEN},
    )
    assert r.status_code == 400 and "needs an Endpoint" in r.json()["detail"]


# --------------------------------------------------------------------------- #
# P7: sampling rules (hints only)
# --------------------------------------------------------------------------- #


def test_models_carry_the_sampling_rules(env: Any) -> None:
    client, _ = env
    rules = client.get("/api/playground/models").json()["sampling_rules"]
    by = {(r["provider"], r["model_prefix"]): r for r in rules}
    assert by[("anthropic", "claude-opus-4-8")]["temperature"] == "default_only"
    assert by[("anthropic", "claude-")]["exclusive"] is True
    assert by[("openai", "gpt-5.1")]["temperature"] == "free"


@pytest.mark.parametrize(
    ("provider", "model", "temperature", "top_p", "exclusive"),
    [
        ("anthropic", "claude-haiku-4-5", "free", "free", True),
        ("anthropic", "claude-sonnet-5", "default_only", "unsupported", False),
        ("bedrock", "us.anthropic.claude-opus-5-v1", "default_only", "unsupported", False),
        ("openrouter", "anthropic/claude-fable-5", "default_only", "unsupported", False),
        ("openai", "gpt-5-mini", "default_only", "default_only", False),
        ("openai", "gpt-5.1", "free", "free", False),
        ("openai", "o3-mini", "default_only", "unsupported", False),
        ("azure", "gpt-5", "default_only", "default_only", False),
        ("ollama", "llama3.2", "free", "free", False),
    ],
)
def test_sampling_rule_resolution(
    provider: str, model: str, temperature: str, top_p: str, exclusive: bool
) -> None:
    from fastaiagent.llm.param_rules import sampling_rule

    rule = sampling_rule(provider, model)
    assert (rule.temperature, rule.top_p, rule.exclusive) == (temperature, top_p, exclusive)


def test_rules_never_block_a_call(env: Any, provider: Any) -> None:
    """D4: the server forwards what the user set; the provider is the authority."""
    client, _ = env
    provider.script(
        Reply(body={"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]})
    )
    r = client.post(
        "/api/playground/run",
        json=_body(provider, model="gpt-5-mini", parameters={"temperature": 0.3, "max_tokens": 8}),
    )
    assert r.status_code == 200, r.text
    assert provider.requests[-1]["body"]["temperature"] == 0.3


# --------------------------------------------------------------------------- #
# P10f: truncation
# --------------------------------------------------------------------------- #


def test_a_truncated_openai_stream_says_so(env: Any, provider: Any) -> None:
    client, db = env
    provider.script(
        Reply(
            events=[
                (None, openai_chunk("Hel")),
                (None, openai_chunk(finish="length")),
                (None, openai_usage_chunk()),
                (None, "[DONE]"),
            ]
        )
    )
    r = client.post("/api/playground/stream", json=_body(provider))
    (_, done) = [e for e in _sse_events(r.text) if e[0] == "done"][0]
    meta = done["metadata"]
    assert meta["finish_reason"] == "length" and meta["truncated"] is True
    assert meta["max_tokens"] == 64
    llm = next(s for s in _spans(db) if s["name"].startswith("llm."))
    assert llm["attributes"]["gen_ai.response.finish_reason"] == "length"
    root = next(s for s in _spans(db) if s["name"] == "playground.run")
    assert root["attributes"]["fastaiagent.playground.truncated"] is True


def test_a_truncated_anthropic_stream_says_so(env: Any, provider: Any) -> None:
    client, _ = env
    events = anthropic_stream("Hel")
    for i, (name, payload) in enumerate(events):
        if name == "message_delta":
            events[i] = (name, {**payload, "delta": {"stop_reason": "max_tokens"}})
    provider.script(Reply(events=events))
    r = client.post(
        "/api/playground/stream",
        json=_body(provider, provider="anthropic", model="claude-haiku-4-5"),
    )
    (_, done) = [e for e in _sse_events(r.text) if e[0] == "done"][0]
    assert done["metadata"]["finish_reason"] == "max_tokens"
    assert done["metadata"]["truncated"] is True


def test_a_complete_stream_is_not_truncated_and_sync_stream_reports_the_reason(
    env: Any, provider: Any
) -> None:
    from fastaiagent.llm import LLMClient, UserMessage

    client, _ = env
    provider.script(Reply(events=openai_stream("Hi", " there")))
    r = client.post("/api/playground/stream", json=_body(provider))
    (_, done) = [e for e in _sse_events(r.text) if e[0] == "done"][0]
    assert done["metadata"]["truncated"] is False and done["metadata"]["finish_reason"] == "stop"

    provider.script(
        Reply(
            events=[
                (None, openai_chunk("Hel")),
                (None, openai_chunk(finish="length")),
                (None, "[DONE]"),
            ]
        )
    )
    llm = LLMClient(
        provider="openai", model="gpt-4o-mini", base_url=f"{provider.url}/v1", api_key=TOKEN
    )
    assert llm.stream([UserMessage("hi")]).finish_reason == "length", "it was hard-coded to stop"


@pytest.mark.parametrize(
    ("raw", "kind"),
    [
        ("stop", "stop"),
        ("end_turn", "stop"),
        ("length", "length"),
        ("max_tokens", "length"),
        ("MAX_TOKENS", "length"),
        ("model_context_window_exceeded", "length"),
        ("tool_use", "tool_calls"),
        ("content_filter", "content_filter"),
        ("refusal", "refusal"),
        ("weird", "other"),
        ("", None),
    ],
)
def test_finish_reason_normalisation(raw: str, kind: str | None) -> None:
    from fastaiagent.llm.finish import normalize_finish_reason

    assert normalize_finish_reason(raw) == kind


# --------------------------------------------------------------------------- #
# P10e unfilled variables, P10g root span
# --------------------------------------------------------------------------- #


def test_unfilled_variables_are_reported_not_blocked(env: Any, provider: Any) -> None:
    client, db = env
    provider.script(
        Reply(body={"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]})
    )
    r = client.post(
        "/api/playground/run",
        json=_body(
            provider,
            prompt_template="Hi {{name}} from {{city}}, about {{topic}}",
            variables={"name": "Ada", "city": ""},
        ),
    )
    assert r.status_code == 200
    assert r.json()["unfilled_variables"] == ["city", "topic"]
    root = next(s for s in _spans(db) if s["name"] == "playground.run")
    assert root["attributes"]["fastaiagent.playground.unfilled_variables"] == "city,topic"


def test_the_root_span_is_a_playground_run_with_input_and_output(env: Any, provider: Any) -> None:
    from fastaiagent.trace.redaction import is_sensitive_attr

    client, db = env
    provider.script(
        Reply(body={"choices": [{"message": {"content": "Hello Ada"}, "finish_reason": "stop"}]})
    )
    trace_id = client.post("/api/playground/run", json=_body(provider)).json()["trace_id"]
    root = next(s for s in _spans(db) if s["name"] == "playground.run")
    a = root["attributes"]
    assert a["fastaiagent.runner.type"] == "playground"
    assert a["fastaiagent.playground.input"] == "Say hello to Ada"
    assert a["fastaiagent.playground.output"] == "Hello Ada"
    assert is_sensitive_attr("fastaiagent.playground.input")
    assert is_sensitive_attr("fastaiagent.playground.output")
    detail = client.get(f"/api/traces/{trace_id}").json()
    assert detail["runner_type"] == "playground"


# --------------------------------------------------------------------------- #
# P10c prompt attribution, P10a aliases, P10b diff
# --------------------------------------------------------------------------- #


def _register(db: Path, template: str, *, fragment: str | None = None) -> int:
    from fastaiagent.prompt.registry import PromptRegistry

    reg = PromptRegistry(path=str(db))
    if fragment:
        reg.register_fragment(name="tone", content=fragment)
    return reg.register(name="greeter", template=template).version


def test_an_unchanged_registry_prompt_is_attributed_to_its_version(env: Any, provider: Any) -> None:
    client, db = env
    v = _register(db, "Say hello to {{name}}. {{@tone}}", fragment="Be brief.")
    provider.script(
        Reply(body={"choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}]})
    )
    r = client.post(
        "/api/playground/run",
        json=_body(
            provider,
            prompt_template="Say hello to {{name}}. {{@tone}}",
            prompt_ref={"slug": "greeter", "version": v},
        ),
    )
    assert r.json()["prompt"] == {
        "slug": "greeter",
        "version": v,
        "attributed": True,
        "modified": False,
    }
    sent = provider.requests[-1]["body"]["messages"][-1]["content"]
    assert "Be brief." in json.dumps(sent), "the fragment reference was sent unresolved"
    llm = next(s for s in _spans(db) if s["name"].startswith("llm."))
    assert llm["attributes"]["fastaiagent.prompt.name"] == "greeter"
    assert llm["attributes"]["fastaiagent.prompt.version"] == v
    lineage = client.get("/api/prompts/greeter/lineage").json()
    assert r.json()["trace_id"] in json.dumps(lineage)


def test_an_edited_prompt_is_marked_modified_not_attributed(env: Any, provider: Any) -> None:
    client, db = env
    v = _register(db, "Say hello to {{name}}")
    provider.script(
        Reply(body={"choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}]})
    )
    r = client.post(
        "/api/playground/run",
        json=_body(
            provider,
            prompt_template="Say HELLO loudly to {{name}}",
            prompt_ref={"slug": "greeter", "version": v},
        ),
    )
    assert r.json()["prompt"]["attributed"] is False and r.json()["prompt"]["modified"] is True
    root = next(s for s in _spans(db) if s["name"] == "playground.run")
    assert root["attributes"]["fastaiagent.playground.prompt.base"] == f"greeter@v{v}"
    llm = next(s for s in _spans(db) if s["name"].startswith("llm."))
    assert "fastaiagent.prompt.name" not in llm["attributes"]


def test_aliases_and_a_line_diff(env: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from fastaiagent.prompt.registry import PromptRegistry

    client, db = env
    reg = PromptRegistry(path=str(db))
    reg.register(name="greeter", template="line one\nline two\nline three")
    reg.register(name="greeter", template="line one\nline TWO\nline three")
    external = client.put("/api/prompts/greeter/aliases/production", json={"version": 1})
    assert external.status_code == 403, "a registry outside the project must stay read-only"
    monkeypatch.chdir(db.parent)
    r = client.put("/api/prompts/greeter/aliases/production", json={"version": 1})
    assert r.status_code == 200 and r.json()["aliases"] == {"production": 1}
    assert client.put("/api/prompts/greeter/aliases/x", json={"version": 9}).status_code == 404
    assert (
        client.put("/api/prompts/greeter/aliases/bad%20name", json={"version": 1}).status_code
        == 422
    )
    assert client.get("/api/prompts/greeter").json()["aliases"] == {"production": 1}
    versions = client.get("/api/prompts/greeter/versions").json()["versions"]
    assert [v["aliases"] for v in versions] == [["production"], []]
    assert reg.load("greeter", alias="production").version == 1

    diff = client.get("/api/prompts/greeter/diff?a=1&b=2").json()
    lines = diff["diff"].splitlines()
    assert lines[:2] == ["--- greeter v1", "+++ greeter v2"]
    assert "-line two" in lines and "+line TWO" in lines
    assert not any(ln.startswith(("-line one", "+line one")) for ln in lines), (
        "unchanged line in diff"
    )
    assert diff["a"]["template"].startswith("line one") and diff["b"]["version"] == 2

    assert client.delete("/api/prompts/greeter/aliases/production").json()["aliases"] == {}
    assert client.delete("/api/prompts/greeter/aliases/production").status_code == 404


def test_set_alias_refuses_a_missing_version(isolated_local_db: Path) -> None:
    from fastaiagent._internal.errors import PromptNotFoundError
    from fastaiagent.prompt.registry import PromptRegistry

    reg = PromptRegistry(path=str(isolated_local_db))
    reg.register(name="p", template="t")
    with pytest.raises(PromptNotFoundError):
        reg.set_alias("p", 5, "prod")


# --------------------------------------------------------------------------- #
# P6: save the run, image included
# --------------------------------------------------------------------------- #


def test_save_as_eval_keeps_the_runs_image_and_settings(env: Any) -> None:
    from fastaiagent.eval.dataset import Dataset
    from fastaiagent.multimodal.image import Image
    from tests.test_eval_multimodal_persist import _png

    client, db = env
    png = _png()
    r = client.post(
        "/api/playground/save-as-eval",
        json={
            "dataset_name": "pics",
            "input": "What colour is this?",
            "expected_output": "red",
            "system_prompt": "Be brief.",
            "model": "gpt-4o-mini",
            "provider": "openai",
            "image_b64": base64.b64encode(png).decode(),
            "image_media_type": "image/png",
            "image_name": "red square.png",
            "parameters": {"temperature": 0.2, "max_tokens": 64, "api_key": "nope"},
            "prompt": {"slug": "greeter", "version": 3},
            "trace_id": "abc123",
        },
    )
    assert r.status_code == 200, r.text
    path = Path(r.json()["path"])
    row = json.loads(path.read_text().splitlines()[-1])
    assert row["input"][0] == {"type": "text", "text": "What colour is this?"}
    assert row["input"][1]["path"].startswith("images/pics/")
    meta = row["metadata"]
    assert meta["parameters"] == {"temperature": 0.2, "max_tokens": 64}, "only sampling knobs kept"
    assert meta["prompt"] == {"slug": "greeter", "version": 3} and meta["trace_id"] == "abc123"
    assert meta["system_prompt"] == "Be brief."
    case = list(Dataset.from_jsonl(path))[-1]
    assert isinstance(case["input"][1], Image) and case["input"][1].data == png

    bad = client.post(
        "/api/playground/save-as-eval",
        json={
            "dataset_name": "pics",
            "input": "x",
            "expected_output": "y",
            "image_b64": "aGk=",
            "image_media_type": "text/html",
        },
    )
    assert bad.status_code == 400
