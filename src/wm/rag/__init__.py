"""Local retrieval-augmented knowledge base for football context.

Pipeline::

    sources  (url / file / stdin / API-Football)
       │
       ▼
    analysis (optional: Claude rewrites a raw match report into structured notes)
       │
       ▼
    store    (chunk → embed with MiniLM → persist in Chroma)
       │
       ▼
    retrieval for predictions (multi-query, de-duplicated, source-labelled)
"""

from wm.rag.ingest import ingest_text
from wm.rag.store import Hit, KnowledgeBase

__all__ = ["Hit", "KnowledgeBase", "ingest_text"]
