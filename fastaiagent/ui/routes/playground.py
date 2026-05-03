"""Prompt Playground endpoints.

The Playground is a UI feature that lets developers select a prompt from the
registry, fill in variables, pick a model, and run the LLM call interactively.
It's the "iterate quickly without writing a script" loop.

Three endpoints:

* ``GET /api/playground/models`` — list known providers + models with
  ``has_key`` set so the UI can disable options for providers without
  configured API keys.
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
import os
import re
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from fastaiagent.ui.deps import get_context, require_session

router = APIRouter(prefix="/api/playground", tags=["playground"])


# ---------------------------------------------------------------------------
# Provider catalog — kept here intentionally so the UI doesn't need to know
# the SDK's internal model registry. Add new defaults here as we ship them.
# Listed providers with prefix-only entries get the prefix as the model name.
# ---------------------------------------------------------------------------
_PROVIDER_CATALOG: dict[str, list[str]] = {
    "openai": [
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-4.1-mini",
        "gpt-4.1",
        "o3-mini",
    ],
    "anthropic": [
        "claude-haiku-4-5",
        "claude-sonnet-4-5",
        "claude-opus-4-5",
        "claude-3-5-sonnet-latest",
    ],
    "ollama": [
        "llama3.2",
        "llama3.2-vision",
        "qwen2.5",
    ],
}

_PROVIDER_ENV_KEY: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "ollama": "",  # local — no key required
}


_VARIABLE_RE = re.compile(r"\{\{(\w+)\}\}")


def _has_api_key(provider: str) -> bool:
    env_var = _PROVIDER_ENV_KEY.get(provider)
    if env_var is None:
        return False
    if env_var == "":
        return True
    return bool(os.environ.get(env_var))


@router.get("/models")
def list_models(_user: str = Depends(require_session)) -> dict[str, Any]:
    """Return the provider/model catalog with ``has_key`` flags."""
    providers: list[dict[str, Any]] = []
    for provider, models in _PROVIDER_CATALOG.items():
        providers.append(
            {
                "provider": provider,
                "models": models,
                "has_key": _has_api_key(provider),
                "env_var": _PROVIDER_ENV_KEY.get(provider) or None,
            }
        )
    return {"providers": providers}


# ---------------------------------------------------------------------------
# Run / Stream
# ---------------------------------------------------------------------------


class PlaygroundParameters(BaseModel):
    temperature: float | None = Field(default=1.0, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=1024, ge=1, le=200_000)
    top_p: float | None = Field(default=1.0, ge=0.0, le=1.0)


class PlaygroundRunRequest(BaseModel):
    provider: str
    model: str
    prompt_template: str
    variables: dict[str, Any] = Field(default_factory=dict)
    system_prompt: str | None = None
    parameters: PlaygroundParameters = Field(default_factory=PlaygroundParameters)
    image_b64: str | None = None
    image_media_type: str | None = None


def _resolve_template(template: str, variables: dict[str, Any]) -> str:
    """Substitute ``{{name}}`` placeholders. Unknown variables are left as-is.

    Mirrors :py:meth:`fastaiagent.prompt.prompt.Prompt.format` but doesn't
    require constructing a Prompt instance for ad-hoc templates.
    """
    out = template
    for key, value in variables.items():
        out = out.replace(f"{{{{{key}}}}}", str(value))
    return out


def _build_messages(req: PlaygroundRunRequest) -> list[Any]:
    """Construct the message list for the LLM call.

    System prompt → system message (if set). Resolved template → user message,
    optionally with an attached :class:`Image` for vision models.
    """
    from fastaiagent.llm import SystemMessage, UserMessage
    from fastaiagent.multimodal import Image

    resolved = _resolve_template(req.prompt_template, req.variables)
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
        img = Image.from_bytes(data, req.image_media_type)
        messages.append(UserMessage([resolved, img]))
    else:
        messages.append(UserMessage(resolved))
    return messages


def _check_api_key_or_400(provider: str) -> None:
    if not _has_api_key(provider):
        env_var = _PROVIDER_ENV_KEY.get(provider) or "(provider key)"
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            (
                f"No API key found for provider '{provider}'. "
                f"Set {env_var} in your environment and restart the UI."
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
    from fastaiagent.trace.otel import get_tracer
    from fastaiagent.trace.span import set_fastaiagent_attributes
    from fastaiagent.ui.pricing import compute_cost_usd

    _check_api_key_or_400(body.provider)
    messages = _build_messages(body)

    client = LLMClient(
        provider=body.provider,
        model=body.model,
        temperature=body.parameters.temperature,
        max_tokens=body.parameters.max_tokens,
        top_p=body.parameters.top_p,
    )

    tracer = get_tracer("fastaiagent.ui.playground")
    trace_id_hex: str | None = None
    with tracer.start_as_current_span("playground.run") as span:
        set_fastaiagent_attributes(
            span,
            source="playground",
            **{"llm.provider": body.provider, "llm.model": body.model},
        )
        try:
            sc = span.get_span_context()
            trace_id_hex = format(sc.trace_id, "032x") if sc and sc.trace_id else None
        except Exception:
            trace_id_hex = None

        start = time.monotonic()
        try:
            resp = await client.acomplete(messages)
        except Exception as e:
            raise HTTPException(
                status.HTTP_502_BAD_GATEWAY,
                f"LLM call failed: {type(e).__name__}: {e}",
            ) from e
        latency_ms = int((time.monotonic() - start) * 1000)

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
        "finish_reason": resp.finish_reason,
    }


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
    from fastaiagent.llm import LLMClient, TextDelta, Usage
    from fastaiagent.trace.otel import get_tracer
    from fastaiagent.trace.span import set_fastaiagent_attributes
    from fastaiagent.ui.pricing import compute_cost_usd

    _check_api_key_or_400(body.provider)
    messages = _build_messages(body)

    client = LLMClient(
        provider=body.provider,
        model=body.model,
        temperature=body.parameters.temperature,
        max_tokens=body.parameters.max_tokens,
        top_p=body.parameters.top_p,
    )

    async def event_stream() -> Any:
        tracer = get_tracer("fastaiagent.ui.playground")
        with tracer.start_as_current_span("playground.run") as span:
            set_fastaiagent_attributes(
                span,
                source="playground",
                **{"llm.provider": body.provider, "llm.model": body.model},
            )
            try:
                sc = span.get_span_context()
                trace_id_hex = (
                    format(sc.trace_id, "032x") if sc and sc.trace_id else None
                )
            except Exception:
                trace_id_hex = None

            input_tokens = 0
            output_tokens = 0
            start = time.monotonic()
            try:
                async for ev in client.astream(messages):
                    if isinstance(ev, TextDelta) and ev.text:
                        payload = json.dumps({"text": ev.text})
                        yield f"event: token\ndata: {payload}\n\n"
                    elif isinstance(ev, Usage):
                        input_tokens = ev.prompt_tokens
                        output_tokens = ev.completion_tokens
            except asyncio.CancelledError:  # client disconnected
                raise
            except Exception as e:
                err = json.dumps(
                    {"message": f"{type(e).__name__}: {e}"}
                )
                yield f"event: error\ndata: {err}\n\n"
                return

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


class SaveAsEvalRequest(BaseModel):
    dataset_name: str
    input: Any
    expected_output: Any
    system_prompt: str | None = None
    model: str | None = None
    provider: str | None = None


def _datasets_dir(db_path: str) -> Path:
    """Resolve ./.fastaiagent/datasets relative to the configured local.db.

    Falls back to the current working directory when ``db_path`` doesn't sit
    under a ``.fastaiagent`` directory (e.g. tests using a tmpfs).
    """
    db = Path(db_path)
    if db.parent.name == ".fastaiagent":
        return db.parent / "datasets"
    return Path.cwd() / ".fastaiagent" / "datasets"


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

    record: dict[str, Any] = {
        "input": body.input,
        "expected_output": body.expected_output,
    }
    metadata: dict[str, Any] = {}
    if body.system_prompt:
        metadata["system_prompt"] = body.system_prompt
    if body.model:
        metadata["model"] = body.model
    if body.provider:
        metadata["provider"] = body.provider
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
