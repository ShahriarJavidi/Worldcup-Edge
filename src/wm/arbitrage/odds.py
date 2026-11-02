"""Loading a scraped odds sheet (``market,selection,decimal_odds``)."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

# Element-id fragments the scraper sometimes leaves in selection labels.
_LABEL_NOISE = ("_sr:", " 40906c97", "_selection")


def clean_selection(label: str) -> str:
    for marker in _LABEL_NOISE:
        cut = label.find(marker)
        if cut >= 0:
            label = label[:cut]
    return label.strip()


@dataclass
class OddsBook:
    """All selections offered for one event, grouped by market (insertion-ordered)."""

    markets: dict[str, list[tuple[str, float]]] = field(default_factory=dict)

    def add(self, market: str, selection: str, odds: float) -> None:
        self.markets.setdefault(market, []).append((selection, odds))

    def get(self, market: str) -> list[tuple[str, float]]:
        return self.markets.get(market, [])

    def __len__(self) -> int:
        return sum(len(s) for s in self.markets.values())


def load_odds_csv(path: str | Path) -> OddsBook:
    book = OddsBook()
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            book.add(row["market"].strip(), clean_selection(row["selection"]), float(row["decimal_odds"]))
    return book
