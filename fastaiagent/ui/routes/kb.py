"""Read-only browser + search playground for LocalKB collections.

Scans a directory of LocalKB stores (default ``./.fastaiagent/kb/``, override
with ``FASTAIAGENT_KB_DIR``) and exposes list / detail / documents / search /
lineage endpoints. No mutations — add / delete / re-index stay in code.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from fastaiagent.ui.deps import get_context, require_session

router = APIRouter(prefix="/api/kb", tags=["kb"])

_DEFAULT_KB_DIR = ".fastaiagent/kb"
_LOCALKB_CACHE: dict[tuple[str, str], Any] = {}


def _kb_root() -> Path:
    return Path(os.environ.get("FASTAIAGENT_KB_DIR", _DEFAULT_KB_DIR)).expanduser()


def _collection_db(root: Path, name: str) -> Path:
    return root / name / "kb.sqlite"


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
                    pass
        return {"chunk_count": chunks, "doc_count": docs}
    except sqlite3.Error:
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
                    "last_updated": time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ", time.gmtime(stat.st_mtime)
                    ),
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
                    pass
    except sqlite3.Error:
        pass
    return {
        "name": name,
        "path": str(db_path.parent),
        "chunk_count": info["chunk_count"],
        "doc_count": info["doc_count"],
        "size_bytes": stat.st_size,
        "last_updated": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime(stat.st_mtime)
        ),
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
            "SELECT COUNT(DISTINCT COALESCE(json_extract(metadata, '$.source'), id)) "
            "FROM chunks"
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


def _get_kb(name: str) -> Any:
    """Instantiate (and cache) a LocalKB for read-only search."""
    root = _kb_root()
    db_path = _collection_db(root, name)
    if not db_path.is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"KB '{name}' not found")
    key = (str(root.resolve()), name)
    cached = _LOCALKB_CACHE.get(key)
    if cached is not None:
        return cached
    from fastaiagent.kb.local import LocalKB

    kb = LocalKB(name=name, path=str(root))
    _LOCALKB_CACHE[key] = kb
    return kb


@router.post("/{name}/search")
async def search_collection(
    name: str,
    request: Request,
    body: SearchRequest = Body(...),
    _username: str = Depends(require_session),
) -> dict[str, Any]:
    kb = _get_kb(name)
    try:
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
        "search_type": getattr(kb, "search_type", "hybrid"),
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
