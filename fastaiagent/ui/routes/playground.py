"""Prompt Playground endpoints.

The Playground is a UI feature that lets developers select a prompt from the
registry, fill in variables, pick a model, and run the LLM call interactively.
It's the "iterate quickly without writing a script" loop.

Three endpoints:

* ``GET /api/playground/models`` — list known providers + models with
  ``has_key`` set so the UI can disable options for providers without
  configured API keys. The list comes from
  :mod:`fastaiagent.ui.model_catalog`, which layers a user ``models.json``
  over the shipped defaults so a stale dropdown can be fixed without an
  SDK release.
* ``POST /api/playground/run`` — non-streaming JSON LLM call. Returns the
  full response with metadata (latency, tokens, cost, trace_id).
* ``POST /api/playground/stream`` — same body as ``/run`` but streams tokens
  via Server-Sent Events.
* ``POST /api/playground/save-as-eval`` — append the (input, expected_output)
  pair to a JSONL file under ``./.fastaiagent/datasets/{name}.jsonl``.
  Saved files load directly via :py:meth:`fastaiagent.eval.dataset.Dataset.from_jsonl`.

Every LLM call is tagged with ``fastaiagent.source = "playground"`` on the
span so playground traces are filterable in the Traces page.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from fastaiagent.ui.deps import get_context, require_session
from fastaiagent.ui.model_catalog import (
    build_catalog,
    env_key_is_set,
    env_var_for,
    has_api_key,
    load_overrides,
)
from fastaiagent.ui.throttle import client_throttle_ip, get_llm_rate_limiter

logger = logging.getLogger(__name__)


def _scrubbed_traceback(exc: BaseException, body: Any) -> str:
    """The traceback of a failed Playground call, with credentials removed.

    Replaces ``exc_info=True`` (1.89.0): the formatted traceback includes the
    exception message, and a provider may echo the submitted key in it. The
    key typed into the form and the provider keys in the environment are
    scrubbed, on top of the scrubbing ``LLMProviderError`` already does.
    """
    import traceback

    from fastaiagent._internal.scrub import known_credential_values, scrub_secrets

    text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return scrub_secrets(text, [getattr(body, "api_key", None), *known_credential_values()])


def _llm_rate_key(request: Request, user: str) -> str:
    """Per-(IP, user) bucket for the LLM rate limiter (M5).

    IP resolution goes through :func:`client_throttle_ip` so a spoofed
    ``X-Forwarded-For`` can't reset the budget (N12).
    """
    return f"{client_throttle_ip(request)}|{user}"


def _enforce_llm_rate_limit(request: Request, user: str) -> None:
    """Refuse the call with HTTP 429 if the user has burned through the
    minute-window budget. Same primitive as login throttling, but tracks
    *every* call rather than just failures (M5).
    """
    allowed, retry_after = get_llm_rate_limiter().try_acquire(
        _llm_rate_key(request, user)
    )
    if not allowed:
        retry = max(int(retry_after) + 1, 1)
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            f"LLM rate limit reached. Retry in {retry}s.",
            headers={"Retry-After": str(retry)},
        )

router = APIRouter(prefix="/api/playground", tags=["playground"])


_VARIABLE_RE = re.compile(r"\{\{(\w+)\}\}")


def _env_key_for_provider(provider: str) -> str | None:
    """Env-var name for a provider, or ``None`` if unknown.

    ``""`` means "no key required" (local providers such as ollama). Thin
    wrapper over :func:`~fastaiagent.ui.model_catalog.env_var_for` so callers
    here don't each have to load the override file.
    """
    return env_var_for(provider, load_overrides(_db_path(None)))


def _has_api_key(provider: str, request: Request | None = None) -> bool:
    db_path = _db_path(request)
    return has_api_key(provider, load_overrides(db_path))


def _db_path(request: Request | None) -> str | None:
    """Best-effort local.db path, used to locate the catalog override file."""
    if request is None:
        return None
    try:
        return get_context(request).db_path
    except Exception:  # noqa: BLE001 — override file is optional, never fatal
        return None


@router.get("/models")
def list_models(
    request: Request, _user: str = Depends(require_session)
) -> dict[str, Any]:
    """Return the provider/model catalog with ``has_key`` flags.

    Sources, in order of precedence: a ``models.json`` override file, then the
    shipped defaults in :mod:`fastaiagent.ui.model_catalog`, merged with any
    presets registered via
    :func:`fastaiagent.llm.providers.register_provider`. See that module for
    the override file's shape and location.
    """
    from fastaiagent.llm.param_rules import rules_table

    return {
        "providers": build_catalog(_db_path(request)),
        # Which sampling parameters each model family takes (1.89.0, P7): UI
        # hints only, the server never blocks on them (decision D4).
        "sampling_rules": rules_table(),
    }


# ---------------------------------------------------------------------------
# Run / Stream
# ---------------------------------------------------------------------------


class PlaygroundParameters(BaseModel):
    """Sampling knobs. ``None`` means "don't send it" — not "send the default".

    ``temperature`` and ``top_p`` default to ``None`` deliberately. Sending
    both is a hard 400 on Anthropic ("`temperature` and `top_p` cannot both be
    specified for this model"), and Claude 5 rejects ``top_p`` on its own
    ("`top_p` is deprecated for this model"). Defaulting them to 1.0 meant
    every Anthropic call from the Playground failed. Only forward what the
    user explicitly asked for; let each provider apply its own defaults.
    """

    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=1024, ge=1, le=200_000)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)

    def client_kwargs(self) -> dict[str, Any]:
        """Only the knobs that were actually set, for ``LLMClient(**kwargs)``."""
        return {
            k: v
            for k, v in (
                ("temperature", self.temperature),
                ("max_tokens", self.max_tokens),
                ("top_p", self.top_p),
            )
            if v is not None
        }


# ~25 MiB raw → ~33.4M base64 chars (4/3 expansion). Cap at 35M to leave a
# safety margin for whitespace/padding without letting an attacker post a
# multi-GB string and OOM the worker before we ever decode it.
_MAX_IMAGE_B64_CHARS: int = 35_000_000
_MAX_IMAGE_DECODED_BYTES: int = 25 * 1024 * 1024  # 25 MiB


class PromptRef(BaseModel):
    slug: str = Field(min_length=1, max_length=200)
    version: int | None = Field(default=None, ge=1)
    alias: str | None = Field(default=None, max_length=64)


class PlaygroundRunRequest(BaseModel):
    """One Playground run.

    ``base_url`` and ``api_key`` are the AI-gateway path. Large orgs usually
    front their models with an OpenAI-compatible gateway reached at a private
    URL with a bearer token, and that token is frequently short-lived
    (SSO/OAuth-issued), so an env var read at server start is the wrong shape —
    you'd restart the UI on every rotation.

    Both are **per request**: not persisted, not written to the DB, not logged,
    and not echoed back in the response. The key exists only for the lifetime
    of the call. Anything you want to keep belongs in an env var (see
    ``models.json``'s ``env_var`` override for pointing a provider at a
    different variable).
    """

    provider: str
    model: str
    prompt_template: str
    variables: dict[str, Any] = Field(default_factory=dict)
    system_prompt: str | None = None
    parameters: PlaygroundParameters = Field(default_factory=PlaygroundParameters)
    image_b64: str | None = Field(default=None, max_length=_MAX_IMAGE_B64_CHARS)
    image_media_type: str | None = None
    base_url: str | None = Field(
        default=None,
        max_length=2048,
        description="Override the provider's endpoint (AI gateway, proxy, "
        "self-hosted server). http/https only.",
    )
    api_key: str | None = Field(
        default=None,
        max_length=8192,
        description="Per-request credential. Never stored, logged, or returned.",
    )
    prompt_ref: PromptRef | None = None
    """The registry prompt this template was loaded from (1.89.0). When the
    template is unchanged the call is attributed to that version (the prompt's
    lineage then lists it); an edited template is marked ``modified`` and not
    attributed (decision D9)."""

    @field_validator("base_url")
    @classmethod
    def _validate_base_url(cls, v: str | None) -> str | None:
        """Only http/https, and nothing that isn't a real absolute URL.

        The server makes the outbound call, so this value decides where the
        prompt goes. Rejecting other schemes keeps `file://`, `gopher://` and
        friends out of the request path.
        """
        if v is None:
            return None
        v = v.strip()
        if not v:
            return None
        parsed = urlparse(v)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError(
                "base_url must be an absolute http:// or https:// URL "
                "(e.g. https://ai-gateway.internal/v1)"
            )
        return v

    @field_validator("api_key")
    @classmethod
    def _validate_api_key(cls, v: str | None) -> str | None:
        # Treat whitespace-only as absent so an accidental space doesn't
        # shadow a perfectly good env var.
        if v is None:
            return None
        v = v.strip()
        return v or None

    def connection_kwargs(self) -> dict[str, Any]:
        """Endpoint/credential overrides for ``LLMClient``, omitting unset ones."""
        out: dict[str, Any] = {}
        if self.base_url:
            out["base_url"] = self.base_url
        if self.api_key:
            out["api_key"] = self.api_key
        return out


def _resolve_template(template: str, variables: dict[str, Any]) -> str:
    """Substitute ``{{name}}`` placeholders. Unknown variables are left as-is.

    Mirrors :py:meth:`fastaiagent.prompt.prompt.Prompt.format` but doesn't
    require constructing a Prompt instance for ad-hoc templates.
    """
    out = template
    for key, value in variables.items():
        out = out.replace(f"{{{{{key}}}}}", str(value))
    return out


def _unfilled_variables(template: str, variables: dict[str, Any]) -> list[str]:
    """``{{vars}}`` the run will send unfilled (1.89.0, P10e): still present
    after substitution, or given as an empty string. A warning, not a block
    (decision D10): a literal ``{{x}}`` can be intended."""
    resolved = _resolve_template(template, variables)
    left = set(_VARIABLE_RE.findall(resolved))
    empty = {k for k, v in variables.items() if isinstance(v, str) and not v.strip()}
    return sorted(left | (empty & set(_VARIABLE_RE.findall(template))))


def _prompt_context(
    request: Request, req: PlaygroundRunRequest
) -> tuple[str, dict[str, Any] | None, dict[str, Any]]:
    """``(template, provenance, root_attrs)`` for a run loaded from the registry.

    With ``prompt_ref`` the ``{{@fragment}}`` references are resolved (the editor
    hands the raw template over since 1.89.0). An unchanged template is
    attributed to its version: ``set_prompt_provenance`` stamps the ``llm.*``
    span the way an agent's registry prompt is, so the prompt's lineage lists
    the run. An edited one is not (decision D9): the root says what it started
    from and that it was modified.
    """
    from fastaiagent.prompt.registry import PromptRegistry

    template = req.prompt_template
    ref = req.prompt_ref
    if ref is None:
        return template, None, {}
    try:
        reg = PromptRegistry(path=_db_path(request))
        stored = reg.load(ref.slug, version=ref.version, alias=ref.alias, resolve_fragments=False)
        resolved = reg.resolve_fragments(stored.template)
        template = reg.resolve_fragments(req.prompt_template)
    except Exception:  # unknown prompt or version: run the text as typed
        logger.debug("Playground prompt_ref did not resolve", exc_info=True)
        return (
            template,
            None,
            {"playground.prompt.base": ref.slug, "playground.prompt.modified": True},
        )
    version = stored.version
    unchanged = req.prompt_template in (stored.template, resolved)
    root: dict[str, Any] = {
        "playground.prompt.name": ref.slug,
        "playground.prompt.version": version,
    }
    if not unchanged:
        return template, None, {
            "playground.prompt.base": f"{ref.slug}@v{version}",
            "playground.prompt.version": version,
            "playground.prompt.modified": True,
        }
    return template, {"name": ref.slug, "version": version}, root


def _build_messages(req: PlaygroundRunRequest, template: str | None = None) -> list[Any]:
    """Construct the message list for the LLM call.

    System prompt → system message (if set). Resolved template → user message,
    optionally with an attached :class:`Image` for vision models.
    """
    from fastaiagent.llm import SystemMessage, UserMessage
    from fastaiagent.multimodal import Image

    resolved = _resolve_template(
        req.prompt_template if template is None else template, req.variables
    )
    messages: list[Any] = []
    if req.system_prompt:
        messages.append(SystemMessage(req.system_prompt))
    if req.image_b64:
        if not req.image_media_type:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                "image_media_type is required when image_b64 is provided",
            )
        try:
            data = base64.b64decode(req.image_b64)
        except Exception as e:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                f"Invalid base64 image data: {e}",
            ) from e
        if len(data) > _MAX_IMAGE_DECODED_BYTES:
            raise HTTPException(
                status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                f"Decoded image exceeds {_MAX_IMAGE_DECODED_BYTES} bytes.",
            )
        img = Image.from_bytes(data, req.image_media_type)
        messages.append(UserMessage([resolved, img]))
    else:
        messages.append(UserMessage(resolved))
    return messages


def _check_api_key_or_400(
    provider: str,
    request: Request | None = None,
    body: PlaygroundRunRequest | None = None,
) -> None:
    """Refuse early when we have no credential to call the provider with.

    A per-request ``api_key`` satisfies this on its own — that's the whole
    point of the gateway-token path, and requiring the env var *as well* would
    defeat it.
    """
    from fastaiagent.ui.model_catalog import requires_endpoint

    if requires_endpoint(provider) and not (body is not None and body.base_url):
        # 1.89.0 (P10i): listed with a key, then failed at call time with an
        # opaque transport error.
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"Provider '{provider}' needs an Endpoint (base URL): set it in the "
            "Connection panel.",
        )
    if body is not None and body.api_key:
        return

    overrides = load_overrides(_db_path(request))

    # Never let the server's own key follow a user-supplied endpoint.
    #
    # ``provider`` selects the wire format; ``base_url`` selects where the
    # request goes. LLMClient's normal behaviour is to fall back to the
    # provider's env var when no key is passed, which is right when base_url
    # is the provider's own URL and badly wrong once it isn't: point the
    # Playground at any host with the Token box empty and OPENAI_API_KEY goes
    # with it. A typo'd or hostile URL becomes key exfiltration.
    #
    # We can't just pass an empty key — LLMClient does
    # ``self.api_key or os.environ.get(...)``, and "" is falsy, so it falls
    # through to the environment anyway. So refuse the request instead, and
    # say why. If no env key is configured there is nothing to leak, which is
    # the local-server case (Ollama/vLLM on a custom port) — allow that.
    if body is not None and body.base_url and env_key_is_set(provider, overrides):
        env_var = env_var_for(provider, overrides) or "the provider key"
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            (
                f"A custom endpoint needs its own token. Without one, "
                f"{env_var} from this server's environment would be sent to "
                f"{body.base_url} — which is only correct if that endpoint is "
                f"'{provider}' itself. Enter the endpoint's token, or clear "
                f"the Endpoint field to call {provider} directly."
            ),
        )

    if not has_api_key(provider, overrides):
        env_var = env_var_for(provider, overrides) or "(provider key)"
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            (
                f"No API key found for provider '{provider}'. "
                f"Set {env_var} in your environment and restart the UI, "
                f"or supply a token for this run in the Playground's "
                f"Connection panel."
            ),
        )


@router.post("/run")
async def run(
    request: Request,
    body: PlaygroundRunRequest,
    _user: str = Depends(require_session),
) -> dict[str, Any]:
    """Non-streaming LLM call. Returns the full response + metadata."""
    from fastaiagent.llm import LLMClient
    from fastaiagent.llm.finish import is_truncated
    from fastaiagent.prompt.provenance import reset_prompt_provenance, set_prompt_provenance
    from fastaiagent.trace.otel import get_tracer
    from fastaiagent.trace.span import set_fastaiagent_attributes
    from fastaiagent.ui.pricing import compute_cost_usd

    _enforce_llm_rate_limit(request, _user)
    _check_api_key_or_400(body.provider, request, body)
    template, provenance, prompt_attrs = _prompt_context(request, body)
    messages = _build_messages(body, template)
    unfilled = _unfilled_variables(template, body.variables)

    client = LLMClient(
        provider=body.provider,
        model=body.model,
        **body.parameters.client_kwargs(),
        **body.connection_kwargs(),
    )

    tracer = get_tracer("fastaiagent.ui.playground")
    trace_id_hex: str | None = None
    failure: dict[str, Any] | None = None
    # A provenance scope, not just a root attribute (1.89.0): every span of the
    # call, the llm.* span included, is stamped "playground" and stays local.
    from fastaiagent.trace.provenance import provenance_scope

    with (
        provenance_scope("playground"),
        tracer.start_as_current_span("playground.run") as span,
    ):
        _stamp_root(span, body, template, unfilled, prompt_attrs)
        try:
            sc = span.get_span_context()
            trace_id_hex = format(sc.trace_id, "032x") if sc and sc.trace_id else None
        except Exception:
            trace_id_hex = None

        start = time.monotonic()
        token = set_prompt_provenance(provenance) if provenance else None
        try:
            resp = await client.acomplete(messages)
        except Exception as e:
            failure = _record_failure(span, e, body, "/run")
        finally:
            if token is not None:
                reset_prompt_provenance(token)
        if failure is None:
            latency_ms = int((time.monotonic() - start) * 1000)
            truncated = is_truncated(resp.finish_reason)
            set_fastaiagent_attributes(
                span,
                **{
                    "playground.output": resp.content or "",
                    "playground.truncated": truncated,
                },
            )
    if failure is not None:
        # Raised after the span closed: the root records only the opaque
        # message (the plane derives ``trace.error`` from it), while the
        # response says why (P8, decision D1).
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, failure)

    input_tokens = (
        resp.usage.get("prompt_tokens") or resp.usage.get("input_tokens") or 0
    )
    output_tokens = (
        resp.usage.get("completion_tokens") or resp.usage.get("output_tokens") or 0
    )
    cost_usd = compute_cost_usd(body.model, input_tokens, output_tokens)

    return {
        "response": resp.content or "",
        "model": body.model,
        "provider": body.provider,
        "latency_ms": latency_ms,
        "tokens": {"input": int(input_tokens), "output": int(output_tokens)},
        "cost_usd": cost_usd,
        "trace_id": trace_id_hex,
        "trace_in_ui_db": _trace_in_ui_db(request),
        "finish_reason": resp.finish_reason,
        "truncated": truncated,
        "max_tokens": body.parameters.max_tokens,
        "unfilled_variables": unfilled,
        "prompt": _prompt_summary(body, provenance, prompt_attrs),
    }


def _stamp_root(
    span: Any,
    body: PlaygroundRunRequest,
    template: str,
    unfilled: list[str],
    prompt_attrs: dict[str, Any],
) -> None:
    """The ``playground.run`` root's attributes (1.89.0, P10g).

    ``runner.type=playground`` gives the trace its own badge (it fell through
    to "agent"), and ``playground.input`` is the resolved prompt, so the trace
    page's Input tab shows what was sent. Input and output are registered in
    ``SENSITIVE_ATTR_KEYS``: payload, gated on export.
    """
    from fastaiagent.trace.span import set_fastaiagent_attributes

    attrs: dict[str, Any] = {
        "llm.provider": body.provider,
        "llm.model": body.model,
        "runner.type": "playground",
        "playground.input": _resolve_template(template, body.variables),
    }
    if unfilled:
        attrs["playground.unfilled_variables"] = ",".join(unfilled)
    set_fastaiagent_attributes(span, source="playground", **attrs)
    for key, value in prompt_attrs.items():
        span.set_attribute(f"fastaiagent.{key}", value)


def _record_failure(
    span: Any, exc: BaseException, body: PlaygroundRunRequest, route: str
) -> dict[str, Any]:
    """Log the failure, mark the root, and build the client-facing detail."""
    from fastaiagent.trace.span import mark_span_error
    from fastaiagent.ui.llm_errors import describe_llm_failure, errors_redacted

    correlation_id = uuid.uuid4().hex
    logger.warning(
        "Playground %s LLM call failed (correlation_id=%s, provider=%s, model=%s)\n%s",
        route,
        correlation_id,
        body.provider,
        body.model,
        _scrubbed_traceback(exc, body),
    )
    failure = describe_llm_failure(
        exc,
        provider=body.provider,
        model=body.model,
        base_url=body.base_url,
        params=body.parameters.client_kwargs(),
        secrets=[body.api_key],
    )
    opaque = {"error": "LLM call failed.", "correlation_id": correlation_id}
    # The root keeps the opaque message and the failure kind only.
    mark_span_error(
        span,
        HTTPException(status.HTTP_502_BAD_GATEWAY, {**opaque, "kind": failure.kind}),
    )
    if errors_redacted():
        return opaque
    return {**opaque, **failure.to_dict()}


def _prompt_summary(
    body: PlaygroundRunRequest, provenance: dict[str, Any] | None, prompt_attrs: dict[str, Any]
) -> dict[str, Any] | None:
    if body.prompt_ref is None:
        return None
    return {
        "slug": body.prompt_ref.slug,
        "version": (provenance or {}).get("version")
        or prompt_attrs.get("playground.prompt.version"),
        "attributed": provenance is not None,
        "modified": bool(prompt_attrs.get("playground.prompt.modified")),
    }


def _trace_in_ui_db(request: Request) -> bool:
    """Whether this process's traces land in the store this UI reads (1.89.0).

    False only for an embedder whose ``build_app(db_path=...)`` differs from the
    configured trace store; the UI then shows where the trace went instead of a
    link that 404s.
    """
    import os

    from fastaiagent._internal.config import get_config

    ui_db = _db_path(request)
    if not ui_db:
        return True
    # realpath, not abspath: `ui start --db` stores the resolved path, and a
    # symlinked one (macOS /tmp, /var) must still compare equal.
    return os.path.realpath(get_config().resolved_trace_db_path) == os.path.realpath(ui_db)


@router.post("/stream")
async def stream(
    request: Request,
    body: PlaygroundRunRequest,
    _user: str = Depends(require_session),
) -> StreamingResponse:
    """SSE token stream. Each event is a JSON line tagged with ``event:`` type.

    Event flow::

        event: token   → {"text": "..."}
        event: token   → {"text": "..."}
        ...
        event: done    → {"metadata": {...}}

    On error::

        event: error   → {"message": "..."}
    """
    from fastaiagent.llm import LLMClient, StreamDone, TextDelta, Usage
    from fastaiagent.llm.finish import is_truncated
    from fastaiagent.prompt.provenance import reset_prompt_provenance, set_prompt_provenance
    from fastaiagent.trace.otel import get_tracer
    from fastaiagent.trace.span import set_fastaiagent_attributes
    from fastaiagent.ui.pricing import compute_cost_usd

    _enforce_llm_rate_limit(request, _user)
    _check_api_key_or_400(body.provider, request, body)
    template, provenance, prompt_attrs = _prompt_context(request, body)
    messages = _build_messages(body, template)
    unfilled = _unfilled_variables(template, body.variables)

    client = LLMClient(
        provider=body.provider,
        model=body.model,
        **body.parameters.client_kwargs(),
        **body.connection_kwargs(),
    )
    trace_in_ui_db = _trace_in_ui_db(request)

    async def event_stream() -> Any:
        from fastaiagent.trace.provenance import provenance_scope

        tracer = get_tracer("fastaiagent.ui.playground")
        with (
            provenance_scope("playground"),
            tracer.start_as_current_span("playground.run") as span,
        ):
            _stamp_root(span, body, template, unfilled, prompt_attrs)
            try:
                sc = span.get_span_context()
                trace_id_hex = (
                    format(sc.trace_id, "032x") if sc and sc.trace_id else None
                )
            except Exception:
                trace_id_hex = None

            input_tokens = 0
            output_tokens = 0
            finish_reason = ""
            text_parts: list[str] = []
            start = time.monotonic()
            token = set_prompt_provenance(provenance) if provenance else None
            try:
                async for ev in client.astream(messages):
                    if isinstance(ev, TextDelta) and ev.text:
                        text_parts.append(ev.text)
                        payload = json.dumps({"text": ev.text})
                        yield f"event: token\ndata: {payload}\n\n"
                    elif isinstance(ev, Usage):
                        input_tokens = ev.prompt_tokens
                        output_tokens = ev.completion_tokens
                    elif isinstance(ev, StreamDone):
                        finish_reason = ev.finish_reason
            except asyncio.CancelledError:  # client disconnected
                raise
            except Exception as e:
                # The root keeps the opaque message (P5); the event says why
                # (P8, decision D1).
                failure = _record_failure(span, e, body, "/stream")
                err = json.dumps({"message": failure.pop("error"), **failure})
                yield f"event: error\ndata: {err}\n\n"
                return
            finally:
                if token is not None:
                    reset_prompt_provenance(token)

            truncated = is_truncated(finish_reason)
            set_fastaiagent_attributes(
                span,
                **{
                    "playground.output": "".join(text_parts),
                    "playground.truncated": truncated,
                },
            )
            latency_ms = int((time.monotonic() - start) * 1000)
            cost_usd = compute_cost_usd(body.model, input_tokens, output_tokens)
            done = json.dumps(
                {
                    "metadata": {
                        "model": body.model,
                        "provider": body.provider,
                        "latency_ms": latency_ms,
                        "tokens": {
                            "input": int(input_tokens),
                            "output": int(output_tokens),
                        },
                        "cost_usd": cost_usd,
                        "trace_id": trace_id_hex,
                        "trace_in_ui_db": trace_in_ui_db,
                        # 1.89.0: the stream's own reason (it had none), so a
                        # reply cut off at max_tokens says so.
                        "finish_reason": finish_reason or None,
                        "truncated": truncated,
                        "max_tokens": body.parameters.max_tokens,
                        "unfilled_variables": unfilled,
                        "prompt": _prompt_summary(body, provenance, prompt_attrs),
                    }
                }
            )
            yield f"event: done\ndata: {done}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Save as eval case — append a JSONL line so Dataset.from_jsonl() can load it
# ---------------------------------------------------------------------------

# Restrict dataset names to a safe filename character set so we can't be
# tricked into writing outside the datasets directory.
_DATASET_NAME_RE = re.compile(r"^[A-Za-z0-9_\-]+$")


_SAVE_IMAGE_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}


class SaveAsEvalRequest(BaseModel):
    """One Playground run as an eval case.

    Since 1.89.0 the UI sends the run's own snapshot (it sent the *current
    form*, so editing the prompt after a run paired the old output with a new
    input), including the attached image, the parameters, the registry prompt
    and the trace.
    """

    dataset_name: str
    input: Any
    expected_output: Any
    system_prompt: str | None = None
    model: str | None = None
    provider: str | None = None
    image_b64: str | None = Field(default=None, max_length=_MAX_IMAGE_B64_CHARS)
    image_media_type: str | None = None
    image_name: str | None = Field(default=None, max_length=200)
    parameters: dict[str, Any] | None = None
    prompt: dict[str, Any] | None = None
    """``{slug, version}`` of the registry prompt the run used."""
    trace_id: str | None = Field(default=None, max_length=64)


def _datasets_dir(db_path: str) -> Path:
    """The datasets directory: the one the Datasets page reads.

    Delegates to :func:`fastaiagent.ui.routes.datasets._datasets_dir` (it was
    a copy of it).
    """
    from fastaiagent.ui.routes.datasets import _datasets_dir as datasets_dir

    return datasets_dir(db_path)


def _save_case_image(db_path: str, dataset: str, body: SaveAsEvalRequest) -> str:
    """Write the run's image next to the dataset; the row's path to it."""
    from fastaiagent.ui.routes.datasets import _images_dir, _safe_image_name

    if body.image_media_type not in _SAVE_IMAGE_TYPES:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"image_media_type must be one of {', '.join(sorted(_SAVE_IMAGE_TYPES))}",
        )
    try:
        data = base64.b64decode(body.image_b64 or "", validate=True)
    except Exception as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"Invalid base64 image data: {e}") from e
    if len(data) > _MAX_IMAGE_DECODED_BYTES:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            f"Decoded image exceeds {_MAX_IMAGE_DECODED_BYTES} bytes.",
        )
    ext = body.image_media_type.split("/")[-1].replace("jpeg", "jpg")
    safe = _safe_image_name(body.image_name or f"playground-image.{ext}")
    out_dir = _images_dir(db_path, dataset)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / safe
    if target.exists():
        stem, _, suffix = safe.rpartition(".")
        if not stem:
            stem, suffix = safe, ""
        safe = f"{stem}-{uuid.uuid4().hex[:8]}" + (f".{suffix}" if suffix else "")
        target = out_dir / safe
    target.write_bytes(data)
    return f"images/{dataset}/{safe}"


@router.post("/save-as-eval")
def save_as_eval(
    request: Request,
    body: SaveAsEvalRequest,
    _user: str = Depends(require_session),
) -> dict[str, Any]:
    """Append a single eval case as a JSONL line.

    File path: ``{db_dir}/datasets/{dataset_name}.jsonl``. Created if missing.
    The line shape matches what :py:meth:`Dataset.from_jsonl` expects, so the
    dataset is immediately runnable.
    """
    if not _DATASET_NAME_RE.match(body.dataset_name):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "dataset_name must match [A-Za-z0-9_-]+",
        )
    ctx = get_context(request)
    out_dir = _datasets_dir(ctx.db_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{body.dataset_name}.jsonl"

    case_input: Any = body.input
    if body.image_b64:
        # The multimodal row shape ``Dataset.from_jsonl`` loads (1.89.0): the
        # image used to be dropped, so the saved case asked about a picture
        # the dataset did not have.
        path = _save_case_image(ctx.db_path, body.dataset_name, body)
        text = body.input if isinstance(body.input, str) else json.dumps(body.input)
        case_input = [{"type": "text", "text": text}, {"type": "image", "path": path}]

    record: dict[str, Any] = {
        "input": case_input,
        "expected_output": body.expected_output,
    }
    # The system prompt stays in metadata: the input is the user turn, and the
    # system prompt belongs to the agent under test.
    metadata: dict[str, Any] = {}
    if body.system_prompt:
        metadata["system_prompt"] = body.system_prompt
    if body.model:
        metadata["model"] = body.model
    if body.provider:
        metadata["provider"] = body.provider
    if body.parameters:
        metadata["parameters"] = {
            k: v for k, v in body.parameters.items() if k in ("temperature", "top_p", "max_tokens")
        }
    if body.prompt and body.prompt.get("slug"):
        metadata["prompt"] = {"slug": body.prompt["slug"], "version": body.prompt.get("version")}
    if body.trace_id:
        metadata["trace_id"] = body.trace_id
    metadata["source"] = "playground"
    record["metadata"] = metadata

    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    line_count = sum(1 for _ in out_path.open("r", encoding="utf-8"))
    return {
        "dataset_name": body.dataset_name,
        "path": str(out_path),
        "line_count": line_count,
    }


__all__ = ["router"]


def _detect_variables(template: str) -> list[str]:
    """Public-ish helper kept here for parity with Prompt._extract_variables.

    Currently unused by the router itself but exposed for tests so the
    template-variable contract is single-sourced.
    """
    return sorted(set(_VARIABLE_RE.findall(template)))
