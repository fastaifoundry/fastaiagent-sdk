"""Eval Dataset Editor endpoints (Sprint 3).

JSONL files in ``<db_dir>/datasets/`` are the source of truth — same
files :py:meth:`Dataset.from_jsonl` already loads. This router exposes
CRUD over those files plus image upload + import/export so the local UI
can drive curation without a script-edit-rerun loop.

Whole-file rewrites on every mutation. Datasets are typically
≤ 1000 cases; atomic ``write+os.replace`` keeps the file consistent
even if the process dies mid-edit.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import unicodedata
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from fastaiagent.ui.deps import get_context, require_session

router = APIRouter(prefix="/api/datasets", tags=["datasets"])


# ---------------------------------------------------------------------------
# Path resolution + validation
# ---------------------------------------------------------------------------

# Same regex as the Playground's ``save-as-eval`` so the two surfaces stay
# consistent. Bars path-traversal (``../`` can't match), bars whitespace,
# bars dots — keeping the filename to a known-safe shape.
_DATASET_NAME_RE = re.compile(r"^[A-Za-z0-9_\-]+$")


def _datasets_dir(db_path: str) -> Path:
    """Resolve the project's ``datasets/`` directory.

    Mirrors :func:`fastaiagent.ui.routes.playground._datasets_dir` so
    Playground "Save as eval case" writes and the editor read/write the
    same files. Falls back to ``./.fastaiagent/datasets`` when the
    configured DB doesn't sit under a ``.fastaiagent`` directory (test
    fixtures with ad-hoc paths).
    """
    db = Path(db_path)
    if db.parent.name == ".fastaiagent":
        return db.parent / "datasets"
    return Path.cwd() / ".fastaiagent" / "datasets"


def _validate_name(name: str) -> str:
    if not _DATASET_NAME_RE.match(name):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "dataset name must match [A-Za-z0-9_-]+ (no slashes, dots, spaces)",
        )
    return name


def _dataset_path(db_path: str, name: str) -> Path:
    return _datasets_dir(db_path) / f"{_validate_name(name)}.jsonl"


def _images_dir(db_path: str, name: str) -> Path:
    """Per-dataset image directory: ``<datasets>/images/<name>/``.

    Per-dataset rather than shared so deleting a dataset can also clean
    up its images without orphaning anyone else's. The JSONL stores
    image paths as ``images/<name>/<file>`` (relative to the JSONL's
    parent), which is what :func:`Dataset._resolve_multimodal_part`
    walks.
    """
    return _datasets_dir(db_path) / "images" / _validate_name(name)


def _safe_image_name(filename: str) -> str:
    """Strip an uploaded filename to a safe basename.

    NFKD-normalise, drop non-ASCII, restrict to ``[A-Za-z0-9._-]``, and
    fall back to a uuid if the result is empty. This blocks both path
    traversal (``..``) and weird unicode names.
    """
    base = Path(filename or "").name
    if not base:
        base = uuid.uuid4().hex
    base = unicodedata.normalize("NFKD", base).encode("ascii", "ignore").decode()
    base = re.sub(r"[^A-Za-z0-9._-]", "_", base)
    base = base.strip("._") or uuid.uuid4().hex
    return base


# ---------------------------------------------------------------------------
# JSONL read / write
# ---------------------------------------------------------------------------


class CaseRow(BaseModel):
    """One eval case as the UI sees it.

    ``index`` is the row's position in the JSONL — it changes on
    insert/delete. ``input`` can be a plain string or a list of typed
    parts (``[{"type": "image", "path": "..."}]``) for multimodal
    cases.
    """

    index: int
    input: Any
    expected_output: Any | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DatasetSummary(BaseModel):
    name: str
    case_count: int
    modified_at: str
    created_at: str
    has_multimodal: bool = False


class DatasetDetail(BaseModel):
    name: str
    cases: list[CaseRow]


class CreateDatasetBody(BaseModel):
    name: str


class CaseBody(BaseModel):
    """Body for ``POST /cases`` and ``PUT /cases/{index}``.

    ``expected`` is accepted as an alias for ``expected_output`` to keep
    files written by hand (the spec format) compatible with files
    written by the Playground (which uses ``expected_output``).
    """

    input: Any
    expected_output: Any | None = None
    expected: Any | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


def _normalise_case(raw: dict[str, Any], index: int) -> CaseRow:
    """Map a JSONL line dict onto the public ``CaseRow`` shape.

    Accepts both ``expected`` and ``expected_output`` keys for
    backwards compatibility with hand-edited files (per spec) and
    Playground-written files (existing convention).
    """
    expected = raw.get("expected_output", raw.get("expected"))
    tags = raw.get("tags") or []
    if not isinstance(tags, list):
        tags = []
    metadata = raw.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}
    return CaseRow(
        index=index,
        input=raw.get("input"),
        expected_output=expected,
        tags=[str(t) for t in tags],
        metadata=metadata,
    )


def _looks_multimodal(value: Any) -> bool:
    if isinstance(value, list):
        return any(
            isinstance(p, dict) and p.get("type") in {"image", "pdf"}
            for p in value
        )
    return False


def _read_dataset(path: Path) -> list[CaseRow]:
    """Parse a JSONL file into ``CaseRow``s. Empty/missing → ``[]``."""
    if not path.exists():
        return []
    rows: list[CaseRow] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise HTTPException(
                    status.HTTP_500_INTERNAL_SERVER_ERROR,
                    f"dataset {path.name} has malformed JSONL on line {i + 1}: {exc.msg}",
                ) from exc
            if not isinstance(raw, dict) or "input" not in raw:
                raise HTTPException(
                    status.HTTP_500_INTERNAL_SERVER_ERROR,
                    f"dataset {path.name} line {i + 1}: each line must be a JSON object with an 'input' field",
                )
            rows.append(_normalise_case(raw, len(rows)))
    return rows


def _serialise_case(case: CaseRow) -> dict[str, Any]:
    """Convert a ``CaseRow`` back to the on-disk shape.

    Always emits ``expected_output`` (matches Playground's existing
    write format), drops empty tags/metadata to keep files tidy.
    """
    record: dict[str, Any] = {"input": case.input}
    if case.expected_output is not None:
        record["expected_output"] = case.expected_output
    if case.tags:
        record["tags"] = case.tags
    if case.metadata:
        record["metadata"] = case.metadata
    return record


def _atomic_write(path: Path, cases: list[CaseRow]) -> None:
    """Rewrite the JSONL file atomically — tmp file + ``os.replace``.

    Keeps the file consistent even if the process dies between lines.
    Sufficient for our typical dataset size (≤ 1000 cases).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{uuid.uuid4().hex}.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            for case in cases:
                f.write(json.dumps(_serialise_case(case), ensure_ascii=False) + "\n")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _ensure_exists(path: Path) -> None:
    if not path.exists():
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, f"dataset '{path.stem}' not found"
        )


def _summary_for(path: Path) -> DatasetSummary:
    rows = _read_dataset(path)
    stat = path.stat()
    has_mm = any(_looks_multimodal(r.input) for r in rows)
    return DatasetSummary(
        name=path.stem,
        case_count=len(rows),
        modified_at=datetime.fromtimestamp(
            stat.st_mtime, tz=timezone.utc
        ).isoformat(),
        # SQLite-style portability: birth time isn't reliable across
        # filesystems, so fall back to mtime if ctime equals 0.
        created_at=datetime.fromtimestamp(
            stat.st_ctime or stat.st_mtime, tz=timezone.utc
        ).isoformat(),
        has_multimodal=has_mm,
    )


# ---------------------------------------------------------------------------
# Routes — list / create / delete / detail
# ---------------------------------------------------------------------------


@router.get("", response_model=list[DatasetSummary])
def list_datasets(
    request: Request, _user: str = Depends(require_session)
) -> list[DatasetSummary]:
    """List every ``*.jsonl`` in the project's datasets directory."""
    ctx = get_context(request)
    base = _datasets_dir(ctx.db_path)
    if not base.exists():
        return []
    out: list[DatasetSummary] = []
    for entry in sorted(base.iterdir()):
        if entry.is_file() and entry.suffix == ".jsonl":
            out.append(_summary_for(entry))
    return out


@router.post("", response_model=DatasetSummary, status_code=status.HTTP_201_CREATED)
def create_dataset(
    request: Request,
    body: CreateDatasetBody,
    _user: str = Depends(require_session),
) -> DatasetSummary:
    ctx = get_context(request)
    path = _dataset_path(ctx.db_path, body.name)
    if path.exists():
        raise HTTPException(
            status.HTTP_409_CONFLICT, f"dataset '{body.name}' already exists"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")
    return _summary_for(path)


@router.delete("/{name}", status_code=status.HTTP_204_NO_CONTENT)
def delete_dataset(
    request: Request,
    name: str,
    _user: str = Depends(require_session),
) -> Response:
    ctx = get_context(request)
    path = _dataset_path(ctx.db_path, name)
    _ensure_exists(path)
    path.unlink()
    # Sweep the per-dataset image folder too if it exists.
    img = _images_dir(ctx.db_path, name)
    if img.exists():
        shutil.rmtree(img, ignore_errors=True)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/{name}", response_model=DatasetDetail)
def get_dataset(
    request: Request,
    name: str,
    _user: str = Depends(require_session),
) -> DatasetDetail:
    ctx = get_context(request)
    path = _dataset_path(ctx.db_path, name)
    _ensure_exists(path)
    return DatasetDetail(name=name, cases=_read_dataset(path))


# ---------------------------------------------------------------------------
# Routes — case CRUD
# ---------------------------------------------------------------------------


def _case_from_body(body: CaseBody, index: int) -> CaseRow:
    expected = body.expected_output if body.expected_output is not None else body.expected
    return CaseRow(
        index=index,
        input=body.input,
        expected_output=expected,
        tags=body.tags,
        metadata=body.metadata,
    )


@router.post("/{name}/cases", response_model=CaseRow, status_code=status.HTTP_201_CREATED)
def add_case(
    request: Request,
    name: str,
    body: CaseBody,
    _user: str = Depends(require_session),
) -> CaseRow:
    ctx = get_context(request)
    path = _dataset_path(ctx.db_path, name)
    _ensure_exists(path)
    cases = _read_dataset(path)
    new_case = _case_from_body(body, len(cases))
    cases.append(new_case)
    _atomic_write(path, cases)
    return new_case


@router.put("/{name}/cases/{index}", response_model=CaseRow)
def update_case(
    request: Request,
    name: str,
    index: int,
    body: CaseBody,
    _user: str = Depends(require_session),
) -> CaseRow:
    ctx = get_context(request)
    path = _dataset_path(ctx.db_path, name)
    _ensure_exists(path)
    cases = _read_dataset(path)
    if index < 0 or index >= len(cases):
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"case index {index} out of range (have {len(cases)} cases)",
        )
    updated = _case_from_body(body, index)
    cases[index] = updated
    _atomic_write(path, cases)
    return updated


@router.delete("/{name}/cases/{index}", status_code=status.HTTP_204_NO_CONTENT)
def delete_case(
    request: Request,
    name: str,
    index: int,
    _user: str = Depends(require_session),
) -> Response:
    ctx = get_context(request)
    path = _dataset_path(ctx.db_path, name)
    _ensure_exists(path)
    cases = _read_dataset(path)
    if index < 0 or index >= len(cases):
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"case index {index} out of range (have {len(cases)} cases)",
        )
    del cases[index]
    # Reflow indices so subsequent updates target the right rows.
    for i, c in enumerate(cases):
        c.index = i
    _atomic_write(path, cases)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# ---------------------------------------------------------------------------
# Routes — import / export / images
# ---------------------------------------------------------------------------


class ImportResult(BaseModel):
    name: str
    imported: int
    total: int


@router.post("/{name}/import", response_model=ImportResult)
async def import_jsonl(
    request: Request,
    name: str,
    file: UploadFile = File(...),
    mode: str = Form("append"),
    _user: str = Depends(require_session),
) -> ImportResult:
    """Upload a JSONL file. ``mode`` is ``append`` (default) or ``replace``.

    Validates each line has an ``input`` field; rejects with a 400 +
    line number on the first bad line so the user knows where to look.
    """
    if mode not in {"append", "replace"}:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, "mode must be 'append' or 'replace'"
        )
    ctx = get_context(request)
    path = _dataset_path(ctx.db_path, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (await file.read()).decode("utf-8", errors="replace")
    new_cases: list[CaseRow] = []
    for i, line in enumerate(raw.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                f"line {i + 1}: invalid JSON — {exc.msg}",
            ) from exc
        if not isinstance(obj, dict) or "input" not in obj:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                f"line {i + 1}: each line must be a JSON object with 'input'",
            )
        new_cases.append(_normalise_case(obj, len(new_cases)))

    if mode == "append" and path.exists():
        existing = _read_dataset(path)
        merged = existing + new_cases
        for i, c in enumerate(merged):
            c.index = i
    else:
        merged = new_cases
    _atomic_write(path, merged)
    return ImportResult(name=name, imported=len(new_cases), total=len(merged))


@router.get("/{name}/export")
def export_jsonl(
    request: Request,
    name: str,
    _user: str = Depends(require_session),
) -> Response:
    ctx = get_context(request)
    path = _dataset_path(ctx.db_path, name)
    _ensure_exists(path)
    body = path.read_text(encoding="utf-8")
    return Response(
        content=body,
        media_type="application/x-ndjson",
        headers={
            "Content-Disposition": f'attachment; filename="{name}.jsonl"'
        },
    )


class ImageUploadResult(BaseModel):
    path: str  # relative path stored in the JSONL
    filename: str
    size_bytes: int


@router.post("/{name}/images", response_model=ImageUploadResult, status_code=status.HTTP_201_CREATED)
async def upload_image(
    request: Request,
    name: str,
    file: UploadFile = File(...),
    _user: str = Depends(require_session),
) -> ImageUploadResult:
    """Save an uploaded image under ``images/<name>/`` and return the
    relative path the JSONL should reference.

    The relative path matches what :func:`Dataset._resolve_multimodal_part`
    walks at load time, so cases referencing it work end-to-end.
    """
    ctx = get_context(request)
    _validate_name(name)  # rejects path-traversal in the dataset name
    out_dir = _images_dir(ctx.db_path, name)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = _safe_image_name(file.filename or "")
    target = out_dir / safe
    # Avoid clobbering an existing file with the same name — append a
    # short uuid suffix.
    if target.exists():
        stem, _, ext = safe.rpartition(".")
        if not stem:
            stem, ext = safe, ""
        safe = f"{stem}-{uuid.uuid4().hex[:8]}" + (f".{ext}" if ext else "")
        target = out_dir / safe
    contents = await file.read()
    target.write_bytes(contents)
    return ImageUploadResult(
        path=f"images/{name}/{safe}",
        filename=safe,
        size_bytes=len(contents),
    )


@router.get("/{name}/images/{filename}")
def get_image(
    request: Request,
    name: str,
    filename: str,
    _user: str = Depends(require_session),
) -> FileResponse:
    """Serve a previously-uploaded image. Filename is constrained to a
    safe basename to bar path traversal."""
    ctx = get_context(request)
    safe = _safe_image_name(filename)
    if safe != filename:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "invalid filename")
    target = _images_dir(ctx.db_path, name) / safe
    if not target.exists() or not target.is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "image not found")
    return FileResponse(target)


# ---------------------------------------------------------------------------
# Run-eval — kicks off the existing eval framework
# ---------------------------------------------------------------------------


class RunEvalBody(BaseModel):
    agent_name: str | None = None
    scorers: list[str] = Field(default_factory=lambda: ["exact_match"])
    run_name: str | None = None


class RunEvalResult(BaseModel):
    run_id: str
    pass_rate: float | None
    pass_count: int
    fail_count: int


@router.post("/{name}/run-eval", response_model=RunEvalResult)
def run_eval(
    request: Request,
    name: str,
    body: RunEvalBody,
    _user: str = Depends(require_session),
) -> RunEvalResult:
    """Run the eval framework against this dataset and return the run_id.

    The agent function is ``lambda x: x`` when no ``agent_name`` is
    provided — useful for sanity-checking the dataset itself (every
    case should "pass" against ``exact_match`` only when its
    ``expected_output`` equals its ``input``). Real agents are
    selected from the registered runners on ``app.state.context``.
    """
    from fastaiagent.eval import evaluate
    from fastaiagent.eval.dataset import Dataset

    ctx = get_context(request)
    path = _dataset_path(ctx.db_path, name)
    _ensure_exists(path)
    dataset = Dataset.from_jsonl(path)

    runner = None
    if body.agent_name:
        runner = ctx.runners.get(body.agent_name)
        if runner is None:
            raise HTTPException(
                status.HTTP_404_NOT_FOUND,
                f"agent '{body.agent_name}' is not registered with this server "
                "(pass it via build_app(runners=[...]))",
            )

    def echo(x: Any) -> Any:
        return x

    agent_fn = runner.run if runner is not None else echo
    # Run with persist=False so we can persist into the AppContext's DB
    # (test fixtures override db_path; the eval framework's default-
    # persist path uses the global config and would land in the wrong
    # file under TestClient).
    results = evaluate(
        agent_fn=agent_fn,
        dataset=list(dataset),
        scorers=body.scorers,
        run_name=body.run_name or f"editor-{name}",
        dataset_name=f"{name}.jsonl",
        agent_name=body.agent_name or "editor-echo",
        persist=False,
    )
    run_id = results.persist_local(
        db_path=ctx.db_path,
        run_name=body.run_name or f"editor-{name}",
        dataset_name=f"{name}.jsonl",
        agent_name=body.agent_name or "editor-echo",
    )
    pass_count = sum(1 for c in results.cases if all(s["passed"] for s in c.per_scorer.values()))
    fail_count = len(results.cases) - pass_count
    pass_rate = pass_count / len(results.cases) if results.cases else None
    return RunEvalResult(
        run_id=run_id,
        pass_rate=pass_rate,
        pass_count=pass_count,
        fail_count=fail_count,
    )


__all__ = ["router"]
