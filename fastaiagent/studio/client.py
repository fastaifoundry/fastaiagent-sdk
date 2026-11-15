"""``StudioClient`` — headless Agent Studio, one method per route.

A coding tool's own model designs an agent and writes the design as a *spec*. Studio
checks it, renders the code, runs trials and evals in its sandbox, keeps every version
and exports. This client never asks Studio's model to plan or build: there is no route
for that.

    from fastaiagent.studio import StudioClient

    studio = StudioClient(api_key="fa_k_…", target="https://<your program address>")
    context = studio.context()                      # endpoints, knowledge bases, limits
    draft = studio.create_draft(
        name="Order desk", requirements_text="Answer order-status questions.",
        llm_endpoint_id=context.llm_endpoints[0].id,
    )
    check = studio.validate_spec(draft.id, spec)    # problems stop a version; warnings do not
    accepted = studio.create_version(draft.id, spec, note="first design")
    studio.wait_for_version(draft.id, accepted.version_id)
    trial = studio.run_trial(draft.id, "Where is order 1003?")

Every route has a plain method (``create_version``) and an ``a``-prefixed async twin
(``acreate_version``). The waiting helpers (``wait_for_*``, ``run_trial``,
``run_evals``) poll with a monotonic deadline and honour the plane's
``retry_after_seconds``; they never count sleeps.

Errors are the SDK's platform errors, plus :class:`StudioError` carrying the plane's
code. See ``fastaiagent.studio.models`` for the answers.
"""

from __future__ import annotations

import asyncio
import os
import re
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generic, NoReturn, TypeVar

from pydantic import BaseModel

from fastaiagent._internal.errors import (
    PlatformConnectionError,
    PlatformError,
    PlatformNotFoundError,
    PlatformRateLimitError,
    StudioError,
)
from fastaiagent.studio._transport import StudioTransport
from fastaiagent.studio.models import (
    DatasetLink,
    Draft,
    DraftDetail,
    EvalRollup,
    EvalRuns,
    Fixture,
    Promoted,
    Sandbox,
    SpecCheck,
    SpecSchema,
    StudioContext,
    Trial,
    Version,
    VersionAccepted,
    VersionDiff,
    VersionSummary,
)

T = TypeVar("T")
M = TypeVar("M", bound=BaseModel)

_BASE = "/public/v1/agent-studio"
_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)

#: A version is being made until ``completed`` or ``failed``.
_VERSION_FINAL = ("completed", "failed")
#: A sandbox is done changing when it is ``ready`` (or ``stopped``).
_SANDBOX_FINAL = ("ready", "stopped")
#: A trial or an eval run; an eval run reads ``scoring`` until its scores are written.
_TRIAL_FINAL = ("completed", "failed")

#: How long to wait before reading again when the plane did not say.
_DEFAULT_RETRY = 2.0
#: A trial whose runner went away says so in its error; the sandbox must be re-armed.
_RUNNER_GONE = "runner"
#: Attempts at one trial or eval run when the sandbox's runner has just gone.
_SANDBOX_ATTEMPTS = 3
#: Times a call is asked again after a 429, and the longest ``Retry-After`` waited out.
_RATE_RETRIES = 2
_RATE_WAIT_CAP = 30.0


@dataclass(frozen=True)
class _Req(Generic[T]):
    """One request, described once and executed by the sync or the async path."""

    method: str
    path: str
    parse: Callable[[Any], T]
    params: dict[str, Any] = field(default_factory=dict)
    body: dict[str, Any] | None = None
    files: list[tuple[str, bytes]] | None = None


def _one(model: type[M]) -> Callable[[Any], M]:
    return lambda raw: model.model_validate(raw)


def _many(model: type[M]) -> Callable[[Any], list[M]]:
    return lambda raw: [model.model_validate(item) for item in raw]


def _nothing(_raw: Any) -> None:
    return None


def _clean(values: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in values.items() if v is not None}


class StudioClient:
    """Client for the plane's ``/public/v1/agent-studio`` API (plane 0.20.0+, wire v1.12).

    ``api_key`` needs ``studio:read`` for reads and ``studio:write`` for everything else.
    ``target`` is your program address; both fall back to ``FASTAIAGENT_API_KEY`` and
    ``FASTAIAGENT_TARGET``. ``project`` is a project id and is only needed for a key
    that is not tied to one project (a domain-level key); the plane answers
    ``project_required`` without it.
    """

    def __init__(
        self,
        api_key: str | None = None,
        target: str | None = None,
        project: str | None = None,
        *,
        timeout: int = 60,
    ):
        from fastaiagent.client import _normalize_target

        key = api_key or os.environ.get("FASTAIAGENT_API_KEY") or ""
        if not key:
            raise ValueError("No API key. Pass api_key= or set FASTAIAGENT_API_KEY.")
        base = _normalize_target(target or os.environ.get("FASTAIAGENT_TARGET") or "")
        if not base:
            raise ValueError(
                "No target. Pass target= or set FASTAIAGENT_TARGET to your program "
                "address, shown in the console when you create an API key."
            )
        self.target = base
        self.project = project
        self._api = StudioTransport(api_key=key, base_url=base, timeout=timeout)

    def __repr__(self) -> str:
        return f"StudioClient(target={self.target!r}, project={self.project!r})"

    # -- what goes wrong around a call ---------------------------------------------------
    #
    # Two things, handled in one place so every route (and the export) gets them:
    #
    # * A 404 that is NOT the plane's own ``not_found`` is a plane without the routes. Only
    #   then is ``GET /public/v1/version`` read, to say "needs 0.20.0 or later" instead of a
    #   bare 404. The happy path pays no extra request, which matters: the key's limit is
    #   60 a minute and every command is a handful of calls.
    # * 429. The plane refuses before it does anything, so asking again is safe. The wait is
    #   the plane's own ``Retry-After``, and only a short one is waited out here; a long one
    #   reaches the caller.

    def _studio_missing(self) -> StudioError:
        return StudioError(
            "studio_not_available",
            "this plane does not have headless Agent Studio (it needs 0.20.0 or later).",
        )

    @staticmethod
    def _lists_studio(version: Any) -> bool:
        endpoints = version.get("endpoints") if isinstance(version, dict) else None
        # A plane that stops listing its routes is not proof the routes are gone.
        if not isinstance(endpoints, list):
            return True
        return any(isinstance(e, str) and "/public/v1/agent-studio" in e for e in endpoints)

    def _explain_not_found(self, err: PlatformNotFoundError) -> NoReturn:
        try:
            listed = self._lists_studio(self._api.get("/public/v1/version"))
        except PlatformNotFoundError:
            listed = False
        except PlatformError:
            raise err from None  # cannot tell; the original answer is the honest one
        raise (err if listed else self._studio_missing()) from err

    async def _aexplain_not_found(self, err: PlatformNotFoundError) -> NoReturn:
        try:
            listed = self._lists_studio(await self._api.aget("/public/v1/version"))
        except PlatformNotFoundError:
            listed = False
        except PlatformError:
            raise err from None
        raise (err if listed else self._studio_missing()) from err

    def _call(self, send: Callable[[], T]) -> T:
        for attempt in range(_RATE_RETRIES + 1):
            try:
                return send()
            except PlatformNotFoundError as err:
                self._explain_not_found(err)
            except PlatformRateLimitError as err:
                pause = _rate_pause(err)
                if pause is None or attempt == _RATE_RETRIES:
                    raise
                time.sleep(pause)
        raise AssertionError("unreachable")  # pragma: no cover

    async def _acall(self, send: Callable[[], Awaitable[T]]) -> T:
        for attempt in range(_RATE_RETRIES + 1):
            try:
                return await send()
            except PlatformNotFoundError as err:
                await self._aexplain_not_found(err)
            except PlatformRateLimitError as err:
                pause = _rate_pause(err)
                if pause is None or attempt == _RATE_RETRIES:
                    raise
                await asyncio.sleep(pause)
        raise AssertionError("unreachable")  # pragma: no cover

    # -- execution ------------------------------------------------------------------------

    def _params(self, req: _Req[Any]) -> dict[str, Any]:
        params = _clean(req.params)
        if self.project:
            params["project_id"] = self.project
        return params

    def _run(self, req: _Req[T]) -> T:
        params, api = self._params(req), self._api

        def send() -> T:
            raw: Any
            if req.files is not None:
                raw = api.post_multipart(req.path, req.files, params=params)
            elif req.method == "GET":
                raw = api.get(req.path, params=params)
            elif req.method == "POST":
                raw = api.post(req.path, req.body, params=params)
            elif req.method == "PATCH":
                raw = api.patch(req.path, req.body or {}, params=params)
            else:
                raw = api.delete(req.path, params=params)
            return req.parse(raw)

        return self._call(send)

    async def _arun(self, req: _Req[T]) -> T:
        params, api = self._params(req), self._api

        async def send() -> T:
            raw: Any
            if req.files is not None:
                raw = await api.apost_multipart(req.path, req.files, params=params)
            elif req.method == "GET":
                raw = await api.aget(req.path, params=params)
            elif req.method == "POST":
                raw = await api.apost(req.path, req.body, params=params)
            elif req.method == "PATCH":
                raw = await api.apatch(req.path, req.body or {}, params=params)
            else:
                raw = await api.adelete(req.path, params=params)
            return req.parse(raw)

        return await self._acall(send)

    # -- the routes: request descriptions ---------------------------------------------------

    @staticmethod
    def _r_context() -> _Req[StudioContext]:
        return _Req("GET", f"{_BASE}/context", _one(StudioContext))

    @staticmethod
    def _r_spec_schema() -> _Req[SpecSchema]:
        return _Req("GET", f"{_BASE}/spec-schema", _one(SpecSchema))

    @staticmethod
    def _r_list_drafts(
        status: str | None, q: str | None, limit: int | None, skip: int | None
    ) -> _Req[list[Draft]]:
        params = {"status": status, "q": q, "limit": limit, "skip": skip}
        return _Req("GET", f"{_BASE}/drafts", _many(Draft), params=params)

    @staticmethod
    def _r_create_draft(
        name: str, requirements_text: str, llm_endpoint_id: str, use_harness: bool
    ) -> _Req[Draft]:
        body = {
            "name": name,
            "requirements_text": requirements_text,
            "llm_endpoint_id": llm_endpoint_id,
            "use_harness": use_harness,
        }
        return _Req("POST", f"{_BASE}/drafts", _one(Draft), body=body)

    @staticmethod
    def _r_get_draft(draft_id: str, messages_after: str | None) -> _Req[DraftDetail]:
        params = {"messages_after": messages_after}
        return _Req("GET", f"{_BASE}/drafts/{draft_id}", _one(DraftDetail), params=params)

    @staticmethod
    def _r_update_draft(draft_id: str, changes: dict[str, Any]) -> _Req[Draft]:
        return _Req("PATCH", f"{_BASE}/drafts/{draft_id}", _one(Draft), body=_clean(changes))

    @staticmethod
    def _r_delete_draft(draft_id: str) -> _Req[None]:
        return _Req("DELETE", f"{_BASE}/drafts/{draft_id}", _nothing)

    @staticmethod
    def _r_upload_fixtures(
        draft_id: str, files: Sequence[tuple[str, bytes]]
    ) -> _Req[list[Fixture]]:
        return _Req(
            "POST", f"{_BASE}/drafts/{draft_id}/fixtures", _many(Fixture), files=list(files)
        )

    @staticmethod
    def _r_add_text_fixture(
        draft_id: str, name: str, text: str, source_filename: str | None
    ) -> _Req[Fixture]:
        body = _clean({"name": name, "text": text, "source_filename": source_filename})
        return _Req("POST", f"{_BASE}/drafts/{draft_id}/fixtures/text", _one(Fixture), body=body)

    @staticmethod
    def _r_get_fixture(fixture_id: str) -> _Req[Fixture]:
        return _Req("GET", f"{_BASE}/fixtures/{fixture_id}", _one(Fixture))

    @staticmethod
    def _r_delete_fixture(fixture_id: str) -> _Req[None]:
        return _Req("DELETE", f"{_BASE}/fixtures/{fixture_id}", _nothing)

    @staticmethod
    def _r_validate_spec(draft_id: str, spec: dict[str, Any]) -> _Req[SpecCheck]:
        return _Req(
            "POST", f"{_BASE}/drafts/{draft_id}/spec/validate", _one(SpecCheck), body={"spec": spec}
        )

    @staticmethod
    def _r_create_version(
        draft_id: str, spec: dict[str, Any], note: str | None
    ) -> _Req[VersionAccepted]:
        body = _clean({"spec": spec, "note": note})
        return _Req("POST", f"{_BASE}/drafts/{draft_id}/versions", _one(VersionAccepted), body=body)

    @staticmethod
    def _r_list_versions(draft_id: str) -> _Req[list[VersionSummary]]:
        return _Req("GET", f"{_BASE}/drafts/{draft_id}/versions", _many(VersionSummary))

    @staticmethod
    def _r_get_version(draft_id: str, version_id: str, include_files: bool) -> _Req[Version]:
        params = {"include": "files" if include_files else None}
        return _Req(
            "GET", f"{_BASE}/drafts/{draft_id}/versions/{version_id}", _one(Version), params=params
        )

    @staticmethod
    def _r_diff(draft_id: str, version_id: str, against: str) -> _Req[VersionDiff]:
        return _Req(
            "GET",
            f"{_BASE}/drafts/{draft_id}/versions/{version_id}/diff",
            _one(VersionDiff),
            params={"against": against},
        )

    @staticmethod
    def _r_start_sandbox(draft_id: str) -> _Req[Sandbox]:
        return _Req("POST", f"{_BASE}/drafts/{draft_id}/sandbox", _one(Sandbox))

    @staticmethod
    def _r_get_sandbox(draft_id: str) -> _Req[Sandbox]:
        return _Req("GET", f"{_BASE}/drafts/{draft_id}/sandbox", _one(Sandbox))

    @staticmethod
    def _r_start_trial(
        draft_id: str, input: str, worker: str | None, version_id: str | None
    ) -> _Req[Trial]:
        body = _clean({"input": input, "worker": worker, "version_id": version_id})
        return _Req("POST", f"{_BASE}/drafts/{draft_id}/trials", _one(Trial), body=body)

    @staticmethod
    def _r_get_trial(trial_id: str) -> _Req[Trial]:
        return _Req("GET", f"{_BASE}/trials/{trial_id}", _one(Trial))

    @staticmethod
    def _r_list_trials(
        draft_id: str, kind: str | None, version_id: str | None
    ) -> _Req[list[Trial]]:
        params = {"kind": kind, "version_id": version_id}
        return _Req("GET", f"{_BASE}/drafts/{draft_id}/trials", _many(Trial), params=params)

    @staticmethod
    def _r_start_eval_run(draft_id: str, worker: str | None, case: str | None) -> _Req[EvalRuns]:
        body = _clean({"worker": worker, "case": case})
        return _Req("POST", f"{_BASE}/drafts/{draft_id}/evals/run", _one(EvalRuns), body=body)

    @staticmethod
    def _r_get_evals(draft_id: str, version_id: str | None) -> _Req[EvalRollup]:
        params = {"version_id": version_id}
        return _Req("GET", f"{_BASE}/drafts/{draft_id}/evals", _one(EvalRollup), params=params)

    @staticmethod
    def _r_link_eval_dataset(draft_id: str, dataset_id: str | None) -> _Req[DatasetLink]:
        body = _clean({"dataset_id": dataset_id})
        return _Req("POST", f"{_BASE}/drafts/{draft_id}/eval-dataset", _one(DatasetLink), body=body)

    @staticmethod
    def _r_import_eval_dataset(draft_id: str) -> _Req[VersionAccepted]:
        return _Req("POST", f"{_BASE}/drafts/{draft_id}/eval-dataset/import", _one(VersionAccepted))

    @staticmethod
    def _r_promote(draft_id: str, agent_id: str) -> _Req[Promoted]:
        return _Req(
            "POST",
            f"{_BASE}/drafts/{draft_id}/promote",
            _one(Promoted),
            body={"agent_id": agent_id},
        )

    # -- the routes: sync -------------------------------------------------------------------

    def context(self) -> StudioContext:
        """What the project has: LLM endpoints, knowledge bases, the SDK contract, limits."""
        return self._run(self._r_context())

    def spec_schema(self) -> SpecSchema:
        """The spec's JSON Schema, the guardrail names with their arguments, the tool types."""
        return self._run(self._r_spec_schema())

    def list_drafts(
        self,
        *,
        status: str | None = None,
        q: str | None = None,
        limit: int | None = None,
        skip: int | None = None,
    ) -> list[Draft]:
        return self._run(self._r_list_drafts(status, q, limit, skip))

    def create_draft(
        self,
        name: str,
        requirements_text: str,
        llm_endpoint_id: str,
        *,
        use_harness: bool = False,
    ) -> Draft:
        """Make a draft. ``requirements_text`` is the brief (at least 10 characters); Studio
        keeps it for the record. ``use_harness`` wires prompts and eval cases to the plane."""
        return self._run(
            self._r_create_draft(name, requirements_text, llm_endpoint_id, use_harness)
        )

    def get_draft(self, draft_id: str, *, messages_after: str | None = None) -> DraftDetail:
        return self._run(self._r_get_draft(draft_id, messages_after))

    def update_draft(
        self,
        draft_id: str,
        *,
        name: str | None = None,
        requirements_text: str | None = None,
        llm_endpoint_id: str | None = None,
        use_harness: bool | None = None,
    ) -> Draft:
        changes = {
            "name": name,
            "requirements_text": requirements_text,
            "llm_endpoint_id": llm_endpoint_id,
            "use_harness": use_harness,
        }
        return self._run(self._r_update_draft(draft_id, changes))

    def delete_draft(self, draft_id: str) -> None:
        self._run(self._r_delete_draft(draft_id))

    def upload_fixtures(self, draft_id: str, files: Sequence[tuple[str, bytes]]) -> list[Fixture]:
        """Upload ``(filename, content)`` pairs: ``.csv``/``.json``/``.jsonl`` become rows,
        ``.md``/``.txt``/``.pdf``/``.docx`` become text. A scan or image is refused with
        ``no_text_layer``: read it yourself and send the text with :meth:`add_text_fixture`."""
        return self._run(self._r_upload_fixtures(draft_id, files))

    def add_text_fixture(
        self, draft_id: str, name: str, text: str, *, source_filename: str | None = None
    ) -> Fixture:
        """Sample data you read yourself — a scan, a photo, an OCR result."""
        return self._run(self._r_add_text_fixture(draft_id, name, text, source_filename))

    def get_fixture(self, fixture_id: str) -> Fixture:
        return self._run(self._r_get_fixture(fixture_id))

    def delete_fixture(self, fixture_id: str) -> None:
        self._run(self._r_delete_fixture(fixture_id))

    def validate_spec(self, draft_id: str, spec: dict[str, Any]) -> SpecCheck:
        """Check a spec. Makes nothing. Problems stop a version being made; warnings do not."""
        return self._run(self._r_validate_spec(draft_id, spec))

    def create_version(
        self, draft_id: str, spec: dict[str, Any], *, note: str | None = None
    ) -> VersionAccepted:
        """Make the next version from a whole spec. Poll it with :meth:`wait_for_version`.

        A spec with problems raises :class:`StudioError` (``spec_invalid``, with
        ``.problems``) and makes nothing. No model is called.
        """
        return self._run(self._r_create_version(draft_id, spec, note))

    def list_versions(self, draft_id: str) -> list[VersionSummary]:
        return self._run(self._r_list_versions(draft_id))

    def get_version(
        self, draft_id: str, version_id: str, *, include_files: bool = False
    ) -> Version:
        return self._run(self._r_get_version(draft_id, version_id, include_files))

    def diff_versions(self, draft_id: str, version_id: str, against: str) -> VersionDiff:
        """What changed in ``version_id`` compared with ``against``, by place."""
        return self._run(self._r_diff(draft_id, version_id, against))

    def start_sandbox(self, draft_id: str) -> Sandbox:
        """Start the project's sandbox, or re-arm one whose runner has gone."""
        return self._run(self._r_start_sandbox(draft_id))

    def get_sandbox(self, draft_id: str) -> Sandbox:
        return self._run(self._r_get_sandbox(draft_id))

    def start_trial(
        self,
        draft_id: str,
        input: str,
        *,
        worker: str | None = None,
        version_id: str | None = None,
    ) -> Trial:
        """Ask one question. A team needs ``worker``. Most callers want :meth:`run_trial`."""
        return self._run(self._r_start_trial(draft_id, input, worker, version_id))

    def get_trial(self, trial_id: str) -> Trial:
        return self._run(self._r_get_trial(trial_id))

    def list_trials(
        self, draft_id: str, *, kind: str | None = None, version_id: str | None = None
    ) -> list[Trial]:
        return self._run(self._r_list_trials(draft_id, kind, version_id))

    def start_eval_run(
        self, draft_id: str, *, worker: str | None = None, case: str | None = None
    ) -> EvalRuns:
        """Start eval runs. Most callers want :meth:`run_evals`."""
        return self._run(self._r_start_eval_run(draft_id, worker, case))

    def get_evals(self, draft_id: str, *, version_id: str | None = None) -> EvalRollup:
        """Eval results for ONE version (the current one by default)."""
        return self._run(self._r_get_evals(draft_id, version_id))

    def link_eval_dataset(self, draft_id: str, *, dataset_id: str | None = None) -> DatasetLink:
        return self._run(self._r_link_eval_dataset(draft_id, dataset_id))

    def import_eval_dataset(self, draft_id: str) -> VersionAccepted:
        """Bring curated cases back from the linked dataset as a new version."""
        return self._run(self._r_import_eval_dataset(draft_id))

    def export(self, draft_id: str, dest: str | Path, *, version_id: str | None = None) -> Path:
        """Download the project as a zip to ``dest``. Needs only ``studio:read``.

        After it, Studio's part is done: the code is yours.
        """
        params = _clean({"version_id": version_id})
        if self.project:
            params["project_id"] = self.project
        path = f"{_BASE}/drafts/{draft_id}/export"
        return self._call(lambda: self._api.download(path, dest, params=params))

    def promote(self, draft_id: str, agent_id: str) -> Promoted:
        """Turn the eval cases into a real eval suite against an agent the exported code
        has registered. A team's cases cannot be promoted (``topology_promote_unsupported``)."""
        return self._run(self._r_promote(draft_id, agent_id))

    # -- the routes: async ------------------------------------------------------------------

    async def acontext(self) -> StudioContext:
        return await self._arun(self._r_context())

    async def aspec_schema(self) -> SpecSchema:
        return await self._arun(self._r_spec_schema())

    async def alist_drafts(
        self,
        *,
        status: str | None = None,
        q: str | None = None,
        limit: int | None = None,
        skip: int | None = None,
    ) -> list[Draft]:
        return await self._arun(self._r_list_drafts(status, q, limit, skip))

    async def acreate_draft(
        self,
        name: str,
        requirements_text: str,
        llm_endpoint_id: str,
        *,
        use_harness: bool = False,
    ) -> Draft:
        return await self._arun(
            self._r_create_draft(name, requirements_text, llm_endpoint_id, use_harness)
        )

    async def aget_draft(self, draft_id: str, *, messages_after: str | None = None) -> DraftDetail:
        return await self._arun(self._r_get_draft(draft_id, messages_after))

    async def aupdate_draft(
        self,
        draft_id: str,
        *,
        name: str | None = None,
        requirements_text: str | None = None,
        llm_endpoint_id: str | None = None,
        use_harness: bool | None = None,
    ) -> Draft:
        changes = {
            "name": name,
            "requirements_text": requirements_text,
            "llm_endpoint_id": llm_endpoint_id,
            "use_harness": use_harness,
        }
        return await self._arun(self._r_update_draft(draft_id, changes))

    async def adelete_draft(self, draft_id: str) -> None:
        await self._arun(self._r_delete_draft(draft_id))

    async def aupload_fixtures(
        self, draft_id: str, files: Sequence[tuple[str, bytes]]
    ) -> list[Fixture]:
        return await self._arun(self._r_upload_fixtures(draft_id, files))

    async def aadd_text_fixture(
        self, draft_id: str, name: str, text: str, *, source_filename: str | None = None
    ) -> Fixture:
        return await self._arun(self._r_add_text_fixture(draft_id, name, text, source_filename))

    async def aget_fixture(self, fixture_id: str) -> Fixture:
        return await self._arun(self._r_get_fixture(fixture_id))

    async def adelete_fixture(self, fixture_id: str) -> None:
        await self._arun(self._r_delete_fixture(fixture_id))

    async def avalidate_spec(self, draft_id: str, spec: dict[str, Any]) -> SpecCheck:
        return await self._arun(self._r_validate_spec(draft_id, spec))

    async def acreate_version(
        self, draft_id: str, spec: dict[str, Any], *, note: str | None = None
    ) -> VersionAccepted:
        return await self._arun(self._r_create_version(draft_id, spec, note))

    async def alist_versions(self, draft_id: str) -> list[VersionSummary]:
        return await self._arun(self._r_list_versions(draft_id))

    async def aget_version(
        self, draft_id: str, version_id: str, *, include_files: bool = False
    ) -> Version:
        return await self._arun(self._r_get_version(draft_id, version_id, include_files))

    async def adiff_versions(self, draft_id: str, version_id: str, against: str) -> VersionDiff:
        return await self._arun(self._r_diff(draft_id, version_id, against))

    async def astart_sandbox(self, draft_id: str) -> Sandbox:
        return await self._arun(self._r_start_sandbox(draft_id))

    async def aget_sandbox(self, draft_id: str) -> Sandbox:
        return await self._arun(self._r_get_sandbox(draft_id))

    async def astart_trial(
        self,
        draft_id: str,
        input: str,
        *,
        worker: str | None = None,
        version_id: str | None = None,
    ) -> Trial:
        return await self._arun(self._r_start_trial(draft_id, input, worker, version_id))

    async def aget_trial(self, trial_id: str) -> Trial:
        return await self._arun(self._r_get_trial(trial_id))

    async def alist_trials(
        self, draft_id: str, *, kind: str | None = None, version_id: str | None = None
    ) -> list[Trial]:
        return await self._arun(self._r_list_trials(draft_id, kind, version_id))

    async def astart_eval_run(
        self, draft_id: str, *, worker: str | None = None, case: str | None = None
    ) -> EvalRuns:
        return await self._arun(self._r_start_eval_run(draft_id, worker, case))

    async def aget_evals(self, draft_id: str, *, version_id: str | None = None) -> EvalRollup:
        return await self._arun(self._r_get_evals(draft_id, version_id))

    async def alink_eval_dataset(
        self, draft_id: str, *, dataset_id: str | None = None
    ) -> DatasetLink:
        return await self._arun(self._r_link_eval_dataset(draft_id, dataset_id))

    async def aimport_eval_dataset(self, draft_id: str) -> VersionAccepted:
        return await self._arun(self._r_import_eval_dataset(draft_id))

    async def aexport(
        self, draft_id: str, dest: str | Path, *, version_id: str | None = None
    ) -> Path:
        params = _clean({"version_id": version_id})
        if self.project:
            params["project_id"] = self.project
        path = f"{_BASE}/drafts/{draft_id}/export"
        return await self._acall(lambda: self._api.adownload(path, dest, params=params))

    async def apromote(self, draft_id: str, agent_id: str) -> Promoted:
        return await self._arun(self._r_promote(draft_id, agent_id))

    # -- finding a draft by what a person calls it -----------------------------------------

    def find_draft(self, ref: str) -> Draft:
        """A draft by id, or by name (the plane keeps names unique within a project)."""
        if _UUID.match(ref):
            return self.get_draft(ref)
        return _by_name(ref, self.list_drafts(q=ref, limit=200))

    async def afind_draft(self, ref: str) -> Draft:
        if _UUID.match(ref):
            return await self.aget_draft(ref)
        return _by_name(ref, await self.alist_drafts(q=ref, limit=200))

    # -- waiting ---------------------------------------------------------------------------

    def wait_for_version(
        self, draft_id: str, version_id: str, *, timeout: float = 120.0
    ) -> Version:
        """Poll a version until it is ``completed`` or ``failed``."""
        return self._settle(
            lambda: self.get_version(draft_id, version_id),
            lambda v: v.generation_status in _VERSION_FINAL,
            f"version {version_id}",
            timeout,
        )

    async def await_for_version(
        self, draft_id: str, version_id: str, *, timeout: float = 120.0
    ) -> Version:
        return await self._asettle(
            lambda: self.aget_version(draft_id, version_id),
            lambda v: v.generation_status in _VERSION_FINAL,
            f"version {version_id}",
            timeout,
        )

    def wait_for_sandbox(self, draft_id: str, *, timeout: float = 180.0) -> Sandbox:
        """Poll the sandbox until it is ``ready`` (or ``stopped``)."""
        return self._settle(
            lambda: self.get_sandbox(draft_id),
            lambda s: s.status in _SANDBOX_FINAL,
            "the sandbox",
            timeout,
        )

    async def await_for_sandbox(self, draft_id: str, *, timeout: float = 180.0) -> Sandbox:
        return await self._asettle(
            lambda: self.aget_sandbox(draft_id),
            lambda s: s.status in _SANDBOX_FINAL,
            "the sandbox",
            timeout,
        )

    def wait_for_trial(self, trial_id: str, *, timeout: float = 300.0) -> Trial:
        """Poll a trial or an eval run until it is ``completed`` or ``failed``."""
        return self._settle(
            lambda: self.get_trial(trial_id),
            lambda t: t.status in _TRIAL_FINAL,
            f"trial {trial_id}",
            timeout,
        )

    async def await_for_trial(self, trial_id: str, *, timeout: float = 300.0) -> Trial:
        return await self._asettle(
            lambda: self.aget_trial(trial_id),
            lambda t: t.status in _TRIAL_FINAL,
            f"trial {trial_id}",
            timeout,
        )

    def ensure_sandbox(self, draft_id: str, *, timeout: float = 180.0) -> Sandbox:
        """Start the sandbox (or re-arm one whose runner has gone) and wait until it is ready."""
        self.start_sandbox(draft_id)
        return self.wait_for_sandbox(draft_id, timeout=timeout)

    async def aensure_sandbox(self, draft_id: str, *, timeout: float = 180.0) -> Sandbox:
        await self.astart_sandbox(draft_id)
        return await self.await_for_sandbox(draft_id, timeout=timeout)

    def run_trial(
        self,
        draft_id: str,
        input: str,
        *,
        worker: str | None = None,
        version_id: str | None = None,
        wait: bool = True,
        timeout: float = 300.0,
    ) -> Trial:
        """Ask one question in the sandbox, starting it first if needed.

        A sandbox can read ``ready`` while its runner has just gone (a restart, a crash).
        The trial is then refused with ``no_active_sandbox`` or ends ``failed`` with an
        error naming the runner. Either way this re-arms the sandbox and asks again, up to
        three attempts in all. With ``wait=False`` it returns the started trial.
        """
        deadline = time.monotonic() + timeout
        for attempt in range(_SANDBOX_ATTEMPTS):
            last = attempt == _SANDBOX_ATTEMPTS - 1
            self.ensure_sandbox(draft_id, timeout=_left(deadline))
            try:
                trial = self.start_trial(draft_id, input, worker=worker, version_id=version_id)
            except StudioError as err:
                if err.code == "no_active_sandbox" and not last:
                    continue
                raise
            if not wait:
                return trial
            done = self.wait_for_trial(trial.id, timeout=_left(deadline))
            if _runner_gone(done) and not last:
                continue
            return done
        raise AssertionError("unreachable")  # pragma: no cover

    async def arun_trial(
        self,
        draft_id: str,
        input: str,
        *,
        worker: str | None = None,
        version_id: str | None = None,
        wait: bool = True,
        timeout: float = 300.0,
    ) -> Trial:
        deadline = time.monotonic() + timeout
        for attempt in range(_SANDBOX_ATTEMPTS):
            last = attempt == _SANDBOX_ATTEMPTS - 1
            await self.aensure_sandbox(draft_id, timeout=_left(deadline))
            try:
                trial = await self.astart_trial(
                    draft_id, input, worker=worker, version_id=version_id
                )
            except StudioError as err:
                if err.code == "no_active_sandbox" and not last:
                    continue
                raise
            if not wait:
                return trial
            done = await self.await_for_trial(trial.id, timeout=_left(deadline))
            if _runner_gone(done) and not last:
                continue
            return done
        raise AssertionError("unreachable")  # pragma: no cover

    def run_evals(
        self,
        draft_id: str,
        *,
        worker: str | None = None,
        case: str | None = None,
        wait: bool = True,
        timeout: float = 600.0,
    ) -> list[Trial]:
        """Run eval cases in the sandbox and return one run per scope.

        A single agent is one run. A team with ``worker`` runs that worker's cases; a team
        without one runs every worker that has cases. Recovers from a sandbox whose runner
        has gone exactly as :meth:`run_trial` does. With ``wait=False`` it returns the
        runs as started.
        """
        deadline = time.monotonic() + timeout
        for attempt in range(_SANDBOX_ATTEMPTS):
            last = attempt == _SANDBOX_ATTEMPTS - 1
            self.ensure_sandbox(draft_id, timeout=_left(deadline))
            try:
                runs = self.start_eval_run(draft_id, worker=worker, case=case).runs
            except StudioError as err:
                if err.code == "no_active_sandbox" and not last:
                    continue
                raise
            if not wait:
                return runs
            done = [self.wait_for_trial(r.id, timeout=_left(deadline)) for r in runs]
            if any(_runner_gone(t) for t in done) and not last:
                continue
            return done
        raise AssertionError("unreachable")  # pragma: no cover

    async def arun_evals(
        self,
        draft_id: str,
        *,
        worker: str | None = None,
        case: str | None = None,
        wait: bool = True,
        timeout: float = 600.0,
    ) -> list[Trial]:
        deadline = time.monotonic() + timeout
        for attempt in range(_SANDBOX_ATTEMPTS):
            last = attempt == _SANDBOX_ATTEMPTS - 1
            await self.aensure_sandbox(draft_id, timeout=_left(deadline))
            try:
                runs = (await self.astart_eval_run(draft_id, worker=worker, case=case)).runs
            except StudioError as err:
                if err.code == "no_active_sandbox" and not last:
                    continue
                raise
            if not wait:
                return runs
            done = [await self.await_for_trial(r.id, timeout=_left(deadline)) for r in runs]
            if any(_runner_gone(t) for t in done) and not last:
                continue
            return done
        raise AssertionError("unreachable")  # pragma: no cover

    # -- polling ---------------------------------------------------------------------------

    def _settle(
        self,
        read: Callable[[], T],
        is_final: Callable[[T], bool],
        what: str,
        timeout: float,
    ) -> T:
        deadline = time.monotonic() + timeout
        dropped = False
        while True:
            try:
                thing = read()
                dropped = False
            except PlatformConnectionError:
                # One dropped connection is tolerated (an older plane closed an idle
                # connection after 2 s; a proxy may have its own timer). Two in a row is not.
                if dropped:
                    raise
                dropped = True
                time.sleep(min(1.0, _left(deadline, what)))
                continue
            except PlatformRateLimitError as err:
                # Still limited after the short retries: keep waiting, as long as the
                # deadline allows, for as long as the plane says.
                time.sleep(min(err.retry_after or _DEFAULT_RETRY, _left(deadline, what)))
                continue
            pause = _pause(thing, is_final)
            if pause is None:
                return thing
            time.sleep(min(pause, _left(deadline, what)))

    async def _asettle(
        self,
        read: Callable[[], Awaitable[T]],
        is_final: Callable[[T], bool],
        what: str,
        timeout: float,
    ) -> T:
        deadline = time.monotonic() + timeout
        dropped = False
        while True:
            try:
                thing = await read()
                dropped = False
            except PlatformConnectionError:
                if dropped:
                    raise
                dropped = True
                await asyncio.sleep(min(1.0, _left(deadline, what)))
                continue
            except PlatformRateLimitError as err:
                await asyncio.sleep(min(err.retry_after or _DEFAULT_RETRY, _left(deadline, what)))
                continue
            pause = _pause(thing, is_final)
            if pause is None:
                return thing
            await asyncio.sleep(min(pause, _left(deadline, what)))


def _pause(thing: Any, is_final: Callable[[Any], bool]) -> float | None:
    """How long to wait before reading again, or ``None`` when there is nothing left to wait for.

    The plane says so itself: every read of something that takes time carries
    ``retry_after_seconds``, and ``0`` means stop polling. A plane that sends none is
    judged by the status alone.
    """
    retry = getattr(thing, "retry_after_seconds", None)
    if retry is None:
        return None if is_final(thing) else _DEFAULT_RETRY
    return None if retry <= 0 else float(retry)


def _rate_pause(err: PlatformRateLimitError) -> float | None:
    """How long to wait before asking again after a 429, or ``None`` to give up.

    Only a ``Retry-After`` the plane actually sent, and only a short one: a caller that
    is told to wait a minute is better served by being told than by a silent minute.
    """
    wait = err.retry_after
    return wait if wait is not None and 0 < wait <= _RATE_WAIT_CAP else None


def _left(deadline: float, what: str = "the request") -> float:
    """Seconds until ``deadline`` (monotonic); past it, the wait has failed."""
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise StudioError("wait_timeout", f"{what} did not settle in time.")
    return remaining


def _runner_gone(trial: Trial) -> bool:
    return trial.status == "failed" and _RUNNER_GONE in (trial.error or "").lower()


def _by_name(ref: str, drafts: list[Draft]) -> Draft:
    wanted = ref.strip().lower()
    found = [d for d in drafts if d.name.strip().lower() == wanted]
    if len(found) == 1:
        return found[0]
    if not found:
        raise StudioError("not_found", f"No draft named {ref!r} in this project.")
    raise StudioError(
        "ambiguous_draft",
        f"{len(found)} drafts are named {ref!r}; use an id: " + ", ".join(d.id for d in found),
    )
