"""Cross-book arbitrage on a three-way (1X2) market."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

OUTCOMES = ("1", "X", "2")


@dataclass(frozen=True)
class Dutch:
    books: tuple[int, int, int]  # which book (0-based) each outcome is backed at
    odds: tuple[float, float, float]

    @property
    def implied(self) -> float:
        return sum(1 / o for o in self.odds)

    @property
    def is_arb(self) -> bool:
        return self.implied < 1

    @property
    def roi(self) -> float:
        return 1 / self.implied - 1

    def stakes(self, bankroll: float) -> tuple[float, float, float]:
        """Stakes that return the same amount whatever the result."""
        return tuple(bankroll / (o * self.implied) for o in self.odds)  # type: ignore[return-value]


def combinations(books: list[tuple[float, float, float]]) -> list[Dutch]:
    """Every way to back 1, X and 2 across ``books``, best first."""
    if any(o <= 1 for book in books for o in book):
        raise ValueError("Decimal odds must be greater than 1.")
    combos = [
        Dutch(books=choice, odds=tuple(books[b][i] for i, b in enumerate(choice)))  # type: ignore[arg-type]
        for choice in product(range(len(books)), repeat=3)
    ]
    return sorted(combos, key=lambda d: d.implied)
