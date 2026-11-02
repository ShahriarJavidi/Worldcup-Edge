"""Retrieval strategy for match predictions."""

from __future__ import annotations

from wm.rag.store import Hit, KnowledgeBase


def match_context(kb: KnowledgeBase, team_a: str, team_b: str, k: int = 8) -> list[Hit]:
    """Gather context for a fixture from several angles and merge the results.

    A single "A vs B" query tends to be dominated by whichever team has more
    stored text, so each team is also queried on its own. Results are
    de-duplicated and ordered by similarity.
    """
    per_query = max(2, k // 2)
    queries = [f"{team_a} vs {team_b}", f"{team_a} form injuries squad", f"{team_b} form injuries squad"]

    best: dict[str, Hit] = {}
    for query in queries:
        for hit in kb.search(query, k=per_query):
            seen = best.get(hit.text)
            if seen is None or hit.distance < seen.distance:
                best[hit.text] = hit
    return sorted(best.values(), key=lambda h: h.distance)[:k]


def format_context(hits: list[Hit]) -> str:
    """Render hits as source-labelled documents for the prompt."""
    return "\n\n".join(
        f'<document index="{i}" source="{hit.source}" kind="{hit.kind}">\n{hit.text}\n</document>'
        for i, hit in enumerate(hits, start=1)
    )
