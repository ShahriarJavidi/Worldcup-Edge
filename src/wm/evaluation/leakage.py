"""Keep the evaluation honest: retrieval may only see what existed before kickoff.

The knowledge base is built over time, so a later dossier or match report can
contain the very result being predicted. A chunk is visible for a fixture only
if it was ingested before kickoff.

Chunks written before ``ingested_at`` existed have no date. Team dossiers
from that era were fetched before the tournament (their newest fixture is
2026-06-03), so they stay visible. Undated match reports and raw text are
hidden, because there is no way to tell what they reveal.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

from wm.rag.store import KIND_TEAM, Hit


def _parse(ts: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(ts)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def visible_before(kickoff: datetime) -> Callable[[Hit], bool]:
    def keep(hit: Hit) -> bool:
        ingested = _parse(str(hit.metadata.get("ingested_at") or ""))
        if ingested is None:
            return hit.kind == KIND_TEAM
        return ingested < kickoff

    return keep
