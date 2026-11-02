"""Single entry point for writing anything into the knowledge base."""

from __future__ import annotations

from dataclasses import dataclass

from wm.llm import Claude
from wm.rag.analysis import analyze_report, match_title
from wm.rag.store import KIND_ANALYSIS, KIND_RAW, KnowledgeBase


@dataclass(frozen=True)
class IngestResult:
    source: str
    kind: str
    chunks: int
    title: str | None = None


def ingest_text(
    kb: KnowledgeBase,
    text: str,
    *,
    source: str,
    llm: Claude | None = None,
    kind: str | None = None,
    title: str | None = None,
    metadata: dict[str, str] | None = None,
) -> IngestResult:
    """Store ``text`` under ``source`` (replacing any earlier copy).

    With an ``llm`` the text is first rewritten as structured match notes, and the
    match line from those notes becomes the chunk title.
    """
    if llm is not None:
        text = analyze_report(llm, text)
        kind = kind or KIND_ANALYSIS
        title = title or match_title(text)
    kind = kind or KIND_RAW
    chunks = kb.add(text, source=source, kind=kind, title=title, metadata=metadata)
    return IngestResult(source=source, kind=kind, chunks=chunks, title=title)
