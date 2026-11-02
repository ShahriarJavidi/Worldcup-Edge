"""Chroma-backed vector store with local sentence-transformer embeddings.

Every document is identified by its ``source`` label (a URL, a file name,
``api_football:<Team>``, ...). Adding a document whose source already exists
replaces the old chunks, so refreshing a team dossier never leaves stale or
duplicated context behind.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # heavy imports (torch, chromadb) are deferred until first use
    from langchain_chroma import Chroma

CHUNK_SIZE = 700
CHUNK_OVERLAP = 120

# Document kinds stored in the `kind` metadata field.
KIND_RAW = "raw_text"
KIND_ANALYSIS = "match_report_analysis"
KIND_TEAM = "team_context"
TEAM_SOURCE_PREFIX = "api_football:"


def kind_of(metadata: dict[str, Any]) -> str:
    """Document kind; chunks stored before ``kind`` existed are inferred from their source."""
    if kind := metadata.get("kind"):
        return str(kind)
    return KIND_TEAM if str(metadata.get("source", "")).startswith(TEAM_SOURCE_PREFIX) else KIND_RAW


def chunk_header(title: str) -> str:
    return f"[{title}]\n"


@dataclass(frozen=True)
class Hit:
    """One retrieved chunk."""

    text: str
    source: str
    kind: str
    distance: float  # lower is more similar
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ReindexResult:
    retitled: int
    duplicates_removed: int


class KnowledgeBase:
    """A persistent collection of football context, searchable by meaning."""

    def __init__(self, path: str | Path, embedding_model: str) -> None:
        self.path = Path(path)
        self.embedding_model = embedding_model

    @property
    def exists(self) -> bool:
        return self.path.exists()

    @cached_property
    def _store(self) -> Chroma:
        # Keep model loading quiet; these only take effect before the libraries are imported.
        for var in ("HF_HUB_DISABLE_PROGRESS_BARS", "TRANSFORMERS_NO_ADVISORY_WARNINGS"):
            os.environ.setdefault(var, "1")
        for var in ("TRANSFORMERS_VERBOSITY", "HF_HUB_VERBOSITY"):
            os.environ.setdefault(var, "error")

        from chromadb.config import Settings as ChromaSettings
        from langchain_chroma import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings

        return Chroma(
            persist_directory=str(self.path),
            embedding_function=HuggingFaceEmbeddings(model_name=self.embedding_model),
            client_settings=ChromaSettings(anonymized_telemetry=False, is_persistent=True),
        )

    # ------------------------------------------------------------------ write
    def add(
        self,
        text: str,
        *,
        source: str,
        kind: str,
        title: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> int:
        """Chunk, embed and store ``text``, replacing any earlier copy of ``source``.

        ``title`` is prepended to every chunk. A chunk cut from the middle of a
        team dossier otherwise never names the team, and its embedding cannot
        match a query like "Canada injuries". Returns the number of chunks written.
        """
        from langchain_core.documents import Document
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        chunks = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP).split_text(text)
        if not chunks:
            return 0

        title = title or source
        base = {
            "source": source,
            "title": title,
            "kind": kind,
            "ingested_at": datetime.now(UTC).isoformat(timespec="seconds"),
            **(metadata or {}),
        }
        docs = [
            Document(page_content=chunk_header(title) + chunk, metadata={**base, "chunk": i, "chunks": len(chunks)})
            for i, chunk in enumerate(chunks)
        ]
        self.delete(source)
        self._store.add_documents(docs)
        return len(docs)

    def delete(self, source: str) -> int:
        """Remove every chunk of ``source``. Returns how many were removed."""
        if not self.exists:
            return 0
        ids = self._store.get(where={"source": source}, include=[])["ids"]
        if ids:
            self._store.delete(ids=ids)
        return len(ids)

    # ------------------------------------------------------------------ read
    def search(self, query: str, k: int = 5, *, where: dict[str, str] | None = None) -> list[Hit]:
        """Return the ``k`` chunks closest to ``query`` (optionally filtered by metadata)."""
        if not self.exists:
            return []
        results = self._store.similarity_search_with_score(query, k=k, filter=where)
        return [
            Hit(
                text=doc.page_content,
                source=str(doc.metadata.get("source", "unknown")),
                kind=kind_of(doc.metadata),
                distance=float(score),
                metadata=dict(doc.metadata),
            )
            for doc, score in results
        ]

    def stats(self) -> dict[str, Any]:
        """Chunk counts per kind and per source."""
        if not self.exists:
            return {"chunks": 0, "by_kind": {}, "by_source": {}}
        metadatas = self._store.get(include=["metadatas"])["metadatas"]
        by_kind: dict[str, int] = {}
        by_source: dict[str, int] = {}
        for meta in metadatas:
            meta = meta or {}
            kind, source = kind_of(meta), str(meta.get("source", "unknown"))
            by_kind[kind] = by_kind.get(kind, 0) + 1
            by_source[source] = by_source.get(source, 0) + 1
        return {"chunks": len(metadatas), "by_kind": by_kind, "by_source": by_source}

    # ------------------------------------------------------------------ maintenance
    def reindex(self) -> ReindexResult:
        """Upgrade chunks written before titles existed.

        Exact duplicate chunks (e.g. ``api_football:Iran`` and ``api_football:iran``)
        are removed, and every remaining legacy chunk gets its title header,
        ``title`` and ``kind`` metadata, and a fresh embedding.
        """
        from langchain_core.documents import Document

        from wm.rag.analysis import match_title

        if not self.exists:
            return ReindexResult(0, 0)
        data = self._store.get(include=["documents", "metadatas"])
        rows = [(i, d, m or {}) for i, d, m in zip(data["ids"], data["documents"], data["metadatas"], strict=True)]
        legacy = [r for r in rows if "title" not in r[2]]

        seen = {(str(m.get("source", "")).lower(), d) for _, d, m in rows if "title" in m}
        duplicates, keep = [], []
        for row in legacy:
            key = (str(row[2].get("source", "")).lower(), row[1])
            (duplicates if key in seen else keep).append(row)
            seen.add(key)
        if duplicates:
            self._store.delete(ids=[r[0] for r in duplicates])

        titles: dict[str, str] = {}
        for _, text, meta in keep:
            source = str(meta.get("source", "unknown"))
            if source.startswith(TEAM_SOURCE_PREFIX):
                titles[source] = f"Team dossier: {source.removeprefix(TEAM_SOURCE_PREFIX)}"
            elif source not in titles or titles[source] == source:
                titles[source] = match_title(text) or source

        if keep:
            docs = []
            for _, text, meta in keep:
                source = str(meta.get("source", "unknown"))
                title = titles[source]
                docs.append(
                    Document(
                        page_content=chunk_header(title) + text,
                        metadata={**meta, "title": title, "kind": kind_of(meta)},
                    )
                )
            self._store.update_documents(ids=[r[0] for r in keep], documents=docs)
        return ReindexResult(retitled=len(keep), duplicates_removed=len(duplicates))
