"""SQLite-backed ``MetadataStore`` implementation.

Persists documents and chunks (with optional embeddings) to a local SQLite
database file. This is the default metadata backend used by ``LocalKB``
when ``persist=True``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastaiagent._internal.storage import SQLiteHelper
from fastaiagent.kb.chunking import Chunk
from fastaiagent.kb.document import Document

_DOC_SCHEMA = """
CREATE TABLE IF NOT EXISTS kb_documents (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    metadata TEXT DEFAULT '{}',
    source TEXT DEFAULT ''
)
"""

_CHUNK_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    metadata TEXT DEFAULT '{}',
    index_pos INTEGER DEFAULT 0,
    start_char INTEGER DEFAULT 0,
    end_char INTEGER DEFAULT 0,
    embedding TEXT
)
"""


def _to_python_floats(embedding: list[Any]) -> list[float]:
    """Convert numpy float32 / similar values to Python floats for JSON."""
    return [float(v) for v in embedding]


class SqliteMetadataStore:
    """Durable document and chunk store on SQLite.

    Supplies the canonical record of what is in the knowledge base. Vector
    and keyword indexes can be rebuilt from here on startup.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = SQLiteHelper(str(self.path))
        self._db.execute(_DOC_SCHEMA)
        self._db.execute(_CHUNK_SCHEMA)

    # ---- Documents --------------------------------------------------------

    def put_document(self, doc: Document) -> None:
        doc_id = doc.source or doc.content[:120]  # best-effort id fallback
        self._db.execute(
            "INSERT OR REPLACE INTO kb_documents "
            "(id, content, metadata, source) VALUES (?, ?, ?, ?)",
            (doc_id, doc.content, json.dumps(doc.metadata), doc.source),
        )

    def get_document(self, doc_id: str) -> Document | None:
        row = self._db.fetchone("SELECT * FROM kb_documents WHERE id = ?", (doc_id,))
        if row is None:
            return None
        return Document(
            content=row["content"],
            metadata=json.loads(row["metadata"]),
            source=row["source"],
        )

    def list_documents(self) -> list[Document]:
        rows = self._db.fetchall("SELECT * FROM kb_documents")
        return [
            Document(
                content=r["content"],
                metadata=json.loads(r["metadata"]),
                source=r["source"],
            )
            for r in rows
        ]

    def delete_document(self, doc_id: str) -> None:
        self._db.execute("DELETE FROM kb_documents WHERE id = ?", (doc_id,))

    # ---- Chunks -----------------------------------------------------------

    def put_chunks(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]] | None,
    ) -> None:
        if not chunks:
            return
        rows: list[tuple[Any, ...]] = []
        for i, chunk in enumerate(chunks):
            emb_json = None
            if embeddings and i < len(embeddings):
                emb_json = json.dumps(_to_python_floats(embeddings[i]))
            rows.append(
                (
                    chunk.id,
                    chunk.content,
                    json.dumps(chunk.metadata),
                    chunk.index,
                    chunk.start_char,
                    chunk.end_char,
                    emb_json,
                )
            )
        self._db.executemany(
            """INSERT OR REPLACE INTO chunks
               (id, content, metadata, index_pos, start_char, end_char, embedding)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            rows,  # type: ignore[arg-type]  # SQLiteHelper param type is list-invariant
        )

    def get_chunks(self) -> tuple[list[Chunk], list[list[float]]]:
        rows = self._db.fetchall("SELECT * FROM chunks ORDER BY rowid")
        chunks: list[Chunk] = []
        embeddings: list[list[float]] = []
        for r in rows:
            chunks.append(
                Chunk(
                    id=r["id"],
                    content=r["content"],
                    metadata=json.loads(r["metadata"]),
                    index=r["index_pos"],
                    start_char=r["start_char"],
                    end_char=r["end_char"],
                )
            )
            emb = json.loads(r["embedding"]) if r["embedding"] else []
            embeddings.append(emb)
        return chunks, embeddings

    def delete_chunks(self, chunk_ids: list[str]) -> None:
        if not chunk_ids:
            return
        placeholders = ",".join("?" for _ in chunk_ids)
        self._db.execute(
            f"DELETE FROM chunks WHERE id IN ({placeholders})",
            tuple(chunk_ids),
        )

    # ---- Lifecycle --------------------------------------------------------

    def reset(self) -> None:
        self._db.execute("DELETE FROM chunks")
        self._db.execute("DELETE FROM kb_documents")

    def close(self) -> None:
        self._db.close()
