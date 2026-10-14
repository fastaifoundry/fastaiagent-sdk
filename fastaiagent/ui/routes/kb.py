"""Read-only browser + search playground for LocalKB collections.

Scans a directory of LocalKB stores (default ``./.fastaiagent/kb/``, override
with ``FASTAIAGENT_KB_DIR``) and exposes list / detail / documents / search /
lineage endpoints. No mutations — add / delete / re-index stay in code.
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Literal, cast

from fastapi import APIRouter, Body, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from fastaiagent.ui.deps import get_context, require_session

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/kb", tags=["kb"])

_DEFAULT_KB_DIR = ".fastaiagent/kb"
#: Keyed on ``(root, name, search_type)``: one KB opened two ways is two entries.
_LOCALKB_CACHE: dict[tuple[str, str, str], Any] = {}
# Collection names are user-visible directory names. Restrict to a portable
# slug so a path-traversal segment (``../``) or a NUL byte cannot reach
# ``_collection_db`` and walk outside the KB root. Real KB names in the
# SDK and docs are short alphanumeric slugs (``my-kb``, ``docs``, …) so
# this regex is non-breaking for documented usage.
_COLLECTION_NAME_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")


def kb_root() -> Path:
    """The KB collections root — the single resolver for the whole UI.

    ``FASTAIAGENT_KB_DIR`` is re-read live (it always has been, and a long-lived
    UI process should follow a change to it), falling back to
    ``fastaiagent.config.kb_dir`` — a real field as of 1.67.0, which is what
    ``routes/agents.py`` reads when it builds the agent-detail payload. Before
    that it asked ``getattr(get_config(), "kb_dir", None)`` against a field that
    did not exist, so the agent view always looked in ``./.fastaiagent/kb``
    however ``FASTAIAGENT_KB_DIR`` was set.
    """
    from fastaiagent._internal.config import get_config
    from fastaiagent._internal.env import env_path

    explicit = env_path("FASTAIAGENT_KB_DIR")
    if explicit:
        return Path(explicit)
    return Path(get_config().kb_dir or _DEFAULT_KB_DIR).expanduser()


#: Backwards-compatible private alias — several call sites in this module use it.
_kb_root = kb_root


def _validate_collection_name(name: str) -> None:
    """Reject any KB name that could escape the KB root via traversal."""
    if not _COLLECTION_NAME_RE.fullmatch(name):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "Invalid KB name: must match [A-Za-z0-9_-]{1,64}",
        )


def _collection_db(root: Path, name: str) -> Path:
    """Resolve ``<root>/<name>/kb.sqlite`` and refuse anything outside ``root``.

    Belt-and-braces: the name regex above already prevents traversal, but a
    second ``is_relative_to`` check after path resolution defends against
    symlink shenanigans (a ``<root>/foo`` symlink pointing to ``/etc``).
    """
    _validate_collection_name(name)
    candidate = (root / name / "kb.sqlite").resolve()
    root_resolved = root.resolve()
    if not candidate.is_relative_to(root_resolved):
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"KB '{name}' not found")
    return candidate


def _is_collection(entry: Path) -> bool:
    return entry.is_dir() and (entry / "kb.sqlite").is_file()


def _open_readonly(db_path: Path) -> sqlite3.Connection:
    """Open the KB sqlite file read-only — never mutate user data from the UI."""
    uri = f"file:{db_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _safe_int(conn: sqlite3.Connection, sql: str) -> int:
    try:
        row = conn.execute(sql).fetchone()
        return int(row[0]) if row and row[0] is not None else 0
    except sqlite3.Error:
        logger.debug("Failed to execute KB stat query: %s", sql, exc_info=True)
        return 0


def _collection_stats(db_path: Path) -> dict[str, Any]:
    """Count chunks / distinct sources without loading the full KB."""
    if not db_path.is_file():
        return {"chunk_count": 0, "doc_count": 0}
    try:
        with _open_readonly(db_path) as conn:
            chunks = _safe_int(conn, "SELECT COUNT(*) FROM chunks")
            docs = _safe_int(conn, "SELECT COUNT(*) FROM kb_documents")
            # Documents table may be empty if the user only called add() with
            # raw text; fall back to distinct metadata.source on chunks.
            if docs == 0:
                try:
                    rows = conn.execute(
                        "SELECT DISTINCT json_extract(metadata, '$.source') "
                        "FROM chunks WHERE metadata IS NOT NULL"
                    ).fetchall()
                    docs = sum(1 for r in rows if r[0])
                except sqlite3.Error:
                    logger.debug("Failed to count distinct sources in KB", exc_info=True)
        return {"chunk_count": chunks, "doc_count": docs}
    except sqlite3.Error:
        logger.warning("Failed to read KB collection stats from %s", db_path, exc_info=True)
        return {"chunk_count": 0, "doc_count": 0}


@router.get("")
async def list_collections(
    request: Request,
    _username: str = Depends(require_session),
) -> dict[str, Any]:
    """List every LocalKB collection under the configured KB root."""
    root = _kb_root()
    collections: list[dict[str, Any]] = []
    if root.exists():
        for entry in sorted(root.iterdir()):
            if not _is_collection(entry):
                continue
            db_path = _collection_db(root, entry.name)
            stat = db_path.stat()
            info = _collection_stats(db_path)
            collections.append(
                {
                    "name": entry.name,
                    "path": str(entry),
                    "chunk_count": info["chunk_count"],
                    "doc_count": info["doc_count"],
                    "last_updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(stat.st_mtime)),
                    "size_bytes": stat.st_size,
                }
            )
    return {"root": str(root), "collections": collections}


@router.get("/{name}")
async def get_collection(
    name: str,
    request: Request,
    _username: str = Depends(require_session),
) -> dict[str, Any]:
    root = _kb_root()
    db_path = _collection_db(root, name)
    if not db_path.is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"KB '{name}' not found")
    stat = db_path.stat()
    info = _collection_stats(db_path)
    sample_keys: set[str] = set()
    try:
        with _open_readonly(db_path) as conn:
            rows = conn.execute(
                "SELECT metadata FROM chunks WHERE metadata IS NOT NULL LIMIT 50"
            ).fetchall()
            for row in rows:
                try:
                    md = json.loads(row[0] or "{}")
                    sample_keys.update(md.keys())
                except (json.JSONDecodeError, TypeError):
                    logger.debug("Failed to parse KB chunk metadata JSON", exc_info=True)
    except sqlite3.Error:
        logger.warning("Failed to read KB metadata sample from %s", name, exc_info=True)
    has_embeddings = _has_embeddings(db_path)
    search_types = _search_types(has_embeddings)
    return {
        # How the search box can query this KB (1.89.0). ``search_type`` is
        # what a search without one uses: hybrid with embeddings, else keyword.
        "has_embeddings": has_embeddings,
        "search_types": search_types,
        "search_type": "hybrid" if has_embeddings else "keyword",
        "name": name,
        "path": str(db_path.parent),
        "chunk_count": info["chunk_count"],
        "doc_count": info["doc_count"],
        "size_bytes": stat.st_size,
        "last_updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(stat.st_mtime)),
        "metadata_keys": sorted(sample_keys),
    }


@router.get("/{name}/documents")
async def list_documents(
    name: str,
    request: Request,
    page: int = 1,
    page_size: int = 50,
    _username: str = Depends(require_session),
) -> dict[str, Any]:
    """Group chunks by ``metadata.source`` and return one row per document."""
    root = _kb_root()
    db_path = _collection_db(root, name)
    if not db_path.is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"KB '{name}' not found")
    page = max(page, 1)
    page_size = max(min(page_size, 200), 1)
    offset = (page - 1) * page_size
    with _open_readonly(db_path) as conn:
        total_row = conn.execute(
            "SELECT COUNT(DISTINCT COALESCE(json_extract(metadata, '$.source'), id)) FROM chunks"
        ).fetchone()
        total = int(total_row[0]) if total_row and total_row[0] is not None else 0
        rows = conn.execute(
            """
            SELECT
                COALESCE(json_extract(metadata, '$.source'), id) AS source,
                COUNT(*) AS chunk_count,
                MIN(content) AS preview,
                MIN(metadata) AS sample_metadata
            FROM chunks
            GROUP BY source
            ORDER BY source
            LIMIT ? OFFSET ?
            """,
            (page_size, offset),
        ).fetchall()
    documents = []
    for row in rows:
        try:
            md = json.loads(row["sample_metadata"] or "{}")
        except (json.JSONDecodeError, TypeError):
            md = {}
        preview = (row["preview"] or "")[:240]
        documents.append(
            {
                "source": row["source"] or "",
                "chunk_count": int(row["chunk_count"]),
                "preview": preview,
                "metadata": md,
            }
        )
    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "documents": documents,
    }


@router.get("/{name}/chunks")
async def list_chunks_for_document(
    name: str,
    source: str,
    request: Request,
    _username: str = Depends(require_session),
) -> dict[str, Any]:
    """Return every chunk belonging to ``source`` (a single document)."""
    root = _kb_root()
    db_path = _collection_db(root, name)
    if not db_path.is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"KB '{name}' not found")
    with _open_readonly(db_path) as conn:
        rows = conn.execute(
            """
            SELECT id, content, metadata, index_pos, start_char, end_char
            FROM chunks
            WHERE json_extract(metadata, '$.source') = ?
               OR id = ?
            ORDER BY index_pos
            """,
            (source, source),
        ).fetchall()
    chunks = []
    for row in rows:
        try:
            md = json.loads(row["metadata"] or "{}")
        except (json.JSONDecodeError, TypeError):
            md = {}
        chunks.append(
            {
                "id": row["id"],
                "content": row["content"],
                "metadata": md,
                "index": int(row["index_pos"] or 0),
                "start_char": int(row["start_char"] or 0),
                "end_char": int(row["end_char"] or 0),
            }
        )
    return {"source": source, "chunks": chunks}


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4096)
    top_k: int = Field(default=5, ge=1, le=50)
    search_type: Literal["keyword", "vector", "hybrid"] | None = Field(
        default=None,
        description=(
            "How to search. Omitted: ``hybrid`` when the KB stores embeddings, "
            "else ``keyword``. ``vector``/``hybrid`` on a KB without "
            "embeddings is a 400."
        ),
    )


def _has_embeddings(db_path: Path) -> bool:
    """Does any chunk of this KB store an embedding? Read-only, no model load."""
    try:
        with contextlib.closing(_open_readonly(db_path)) as conn:
            row = conn.execute(
                "SELECT 1 FROM chunks WHERE embedding IS NOT NULL "
                "AND embedding NOT IN ('', '[]') LIMIT 1"
            ).fetchone()
            return row is not None
    except sqlite3.Error:
        logger.debug("Could not check KB embeddings in %s", db_path, exc_info=True)
        return False


def _search_types(has_embeddings: bool) -> list[str]:
    """The search types a KB supports: keyword always, vector/hybrid with embeddings."""
    return ["keyword", "vector", "hybrid"] if has_embeddings else ["keyword"]


def _get_kb(name: str, search_type: str | None = None) -> tuple[Any, str]:
    """Instantiate (and cache) a LocalKB for search; returns ``(kb, search_type)``.

    The search type follows what the KB actually holds (1.89.0). The UI used
    to open every KB as ``hybrid``: a keyword-only KB then built the default
    embedder (a model load) for nothing and the response called a pure BM25
    search "hybrid". A KB with no stored embedding opens as ``keyword``, so no
    embedder is built.
    """
    root = _kb_root()
    db_path = _collection_db(root, name)
    if not db_path.is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"KB '{name}' not found")
    has_embeddings = _has_embeddings(db_path)
    if search_type is None:
        search_type = "hybrid" if has_embeddings else "keyword"
    elif search_type not in _search_types(has_embeddings):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"KB '{name}' stores no embeddings, so it cannot run a "
            f"{search_type} search. Use search_type='keyword'.",
        )
    key = (str(root.resolve()), name, search_type)
    cached = _LOCALKB_CACHE.get(key)
    if cached is not None:
        return cached, search_type
    from fastaiagent.kb.local import LocalKB

    try:
        kb = LocalKB(name=name, path=str(root), search_type=cast(Any, search_type))
    except ValueError as exc:
        # e.g. the KB was embedded with another embedder than the default one
        # the UI builds (dimension mismatch). Keyword search still works.
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            f"Could not open KB '{name}' for a {search_type} search: {exc}",
        ) from exc
    _LOCALKB_CACHE[key] = kb
    return kb, search_type


@router.post("/{name}/search")
async def search_collection(
    name: str,
    request: Request,
    body: SearchRequest = Body(...),
    _username: str = Depends(require_session),
) -> dict[str, Any]:
    from fastaiagent.trace.provenance import provenance_scope

    kb, search_type = _get_kb(name, body.search_type)
    try:
        # A search typed into the UI is not a production retrieval: its trace
        # is stamped ``fastaiagent.source=kb_search`` (local-only, kept out of
        # Analytics and production views).
        with provenance_scope("kb_search"):
            results = kb.search(body.query, top_k=body.top_k)
    except Exception as exc:  # noqa: BLE001 — surface backend errors to the UI
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            f"Search failed: {exc}",
        ) from exc
    rows = []
    for res in results:
        chunk = res.chunk
        rows.append(
            {
                "id": chunk.id,
                "content": chunk.content,
                "metadata": chunk.metadata,
                "score": float(res.score),
                "source": (chunk.metadata or {}).get("source"),
                "index": chunk.index,
            }
        )
    return {
        "query": body.query,
        "top_k": body.top_k,
        "search_type": search_type,
        "results": rows,
    }


@router.get("/{name}/lineage")
async def kb_lineage(
    name: str,
    request: Request,
    limit: int = 50,
    _username: str = Depends(require_session),
) -> dict[str, Any]:
    """Traces + agents that retrieved from this KB (via ``retrieval.<name>`` spans)."""
    ctx = get_context(request)
    limit = max(min(limit, 500), 1)
    span_name = f"retrieval.{name}"
    db = ctx.db()
    if ctx.project_id:
        span_rows = db.fetchall(
            """
            SELECT trace_id, span_id, start_time, attributes
            FROM spans
            WHERE name = ? AND project_id = ?
            ORDER BY start_time DESC
            LIMIT ?
            """,
            (span_name, ctx.project_id, limit),
        )
    else:
        span_rows = db.fetchall(
            """
            SELECT trace_id, span_id, start_time, attributes
            FROM spans
            WHERE name = ?
            ORDER BY start_time DESC
            LIMIT ?
            """,
            (span_name, limit),
        )
    from fastaiagent.ui.attrs import attr

    agents: dict[str, int] = {}
    trace_ids: list[str] = []
    seen_traces: set[str] = set()
    for row in span_rows:
        try:
            attrs = json.loads(row["attributes"] or "{}")
        except (json.JSONDecodeError, TypeError):
            attrs = {}
        agent_name = attr(attrs, "agent.name")
        if agent_name:
            agents[agent_name] = agents.get(agent_name, 0) + 1
        tid = row["trace_id"]
        if tid and tid not in seen_traces:
            seen_traces.add(tid)
            trace_ids.append(tid)

    recent_traces = []
    for tid in trace_ids[:20]:
        if ctx.project_id:
            root = db.fetchone(
                "SELECT name, start_time, attributes, status "
                "FROM spans WHERE trace_id = ? AND parent_span_id IS NULL "
                "AND project_id = ? LIMIT 1",
                (tid, ctx.project_id),
            )
        else:
            root = db.fetchone(
                "SELECT name, start_time, attributes, status "
                "FROM spans WHERE trace_id = ? AND parent_span_id IS NULL LIMIT 1",
                (tid,),
            )
        if not root:
            continue
        try:
            attrs = json.loads(root["attributes"] or "{}")
        except (json.JSONDecodeError, TypeError):
            attrs = {}
        recent_traces.append(
            {
                "trace_id": tid,
                "name": root["name"],
                "start_time": root["start_time"],
                "status": root["status"],
                "agent_name": attr(attrs, "agent.name"),
            }
        )
    return {
        "kb_name": name,
        "retrieval_count": len(span_rows),
        "agents": [
            {"agent_name": k, "retrieval_count": v}
            for k, v in sorted(agents.items(), key=lambda x: -x[1])
        ],
        "recent_traces": recent_traces,
    }
