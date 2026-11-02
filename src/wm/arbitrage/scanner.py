"""Three-layer arbitrage scan of a single bookmaker's back-only odds.

1. **Overround** of every self-contained market: ``sum(1/odds) < 1`` is a
   single-market arb.
2. **Ladder monotonicity**: implied P(Over) must fall as the line rises.
3. **Global LP** over the full-time scoreline grid::

       minimise   sum_k stake_k
       subject to sum_k stake_k * odds_k * [k wins on o] >= 1   for every scoreline o
                  stake_k >= 0

   If the cheapest portfolio that guarantees a return of 1 costs less than 1,
   a cross-market arbitrage exists. This subsumes every dutching combination.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linprog

from wm.arbitrage.markets import (
    GRID,
    LADDER_MARKETS,
    PARTITION_MARKETS,
    THREE_WAY_HANDICAP_MARKETS,
    over_under,
    winning_scores,
)
from wm.arbitrage.odds import OddsBook

EPS = 1e-9


@dataclass(frozen=True)
class Overround:
    market: str
    book: float  # sum of implied probabilities

    @property
    def vig(self) -> float:
        return self.book - 1

    @property
    def is_arb(self) -> bool:
        return self.book < 1 - EPS


@dataclass(frozen=True)
class Ladder:
    market: str
    points: list[tuple[float, float]]  # (line, implied P(Over)), ascending line

    @property
    def violated(self) -> bool:
        return any(self.points[i][1] > self.points[i - 1][1] + EPS for i in range(1, len(self.points)))


@dataclass(frozen=True)
class Leg:
    market: str
    selection: str
    odds: float
    stake: float  # per 1.0 of guaranteed return


@dataclass
class PortfolioResult:
    solved: bool
    message: str = ""
    cost: float = float("nan")  # cheapest outlay that returns >= 1 on every scoreline
    legs: list[Leg] = field(default_factory=list)
    tightest: list[tuple[tuple[int, int], float]] = field(default_factory=list)
    markets_used: set[str] = field(default_factory=set)
    markets_skipped: set[str] = field(default_factory=set)
    n_legs: int = 0

    @property
    def is_arb(self) -> bool:
        return self.solved and self.cost < 1 - 1e-6

    @property
    def roi(self) -> float:
        return (1 - self.cost) / self.cost


@dataclass
class ScanReport:
    overrounds: list[Overround]
    ladders: list[Ladder]
    portfolio: PortfolioResult


def overrounds(book: OddsBook) -> list[Overround]:
    out = [Overround(m, sum(1 / o for _, o in book.get(m))) for m in book.markets if m in PARTITION_MARKETS]

    # Over/Under-style markets carry many lines; each line pair is its own book.
    for market, sels in book.markets.items():
        by_line: dict[float, list[float]] = defaultdict(list)
        for sel, odds in sels:
            if parsed := over_under(sel):
                by_line[parsed[1]].append(odds)
        out += [
            Overround(f"{market} [O/U {line}]", sum(1 / o for o in os)) for line, os in by_line.items() if len(os) == 2
        ]

    # 3-way European handicap: one book per handicap.
    for market in THREE_WAY_HANDICAP_MARKETS:
        by_hcp: dict[str, list[float]] = defaultdict(list)
        for sel, odds in book.get(market):
            if m := re.match(r"^[1X2]\s+(\d+:\d+)$", sel):
                by_hcp[m[1]].append(odds)
        out += [Overround(f"{market} [{hcp}]", sum(1 / o for o in os)) for hcp, os in by_hcp.items() if len(os) == 3]

    return sorted(out, key=lambda r: r.book)


def ladders(book: OddsBook) -> list[Ladder]:
    out = []
    for market in LADDER_MARKETS:
        points = sorted(
            (parsed[1], 1 / odds) for sel, odds in book.get(market) if (parsed := over_under(sel)) and parsed[0] is True
        )
        if points:
            out.append(Ladder(market, points))
    return out


def best_portfolio(book: OddsBook, show_tightest: int = 6) -> PortfolioResult:
    index = {score: i for i, score in enumerate(GRID)}
    legs: list[tuple[str, str, float]] = []
    columns: list[np.ndarray] = []
    used, skipped = set(), set()

    for market, sels in book.markets.items():
        for sel, odds in sels:
            wins = winning_scores(market, sel)
            if not wins:
                continue
            payout = np.zeros(len(GRID))
            payout[[index[w] for w in wins]] = odds
            legs.append((market, sel, odds))
            columns.append(payout)
            used.add(market)
        if market not in used:
            skipped.add(market)

    result = PortfolioResult(solved=False, markets_used=used, markets_skipped=skipped, n_legs=len(legs))
    if not legs:
        result.message = "no full-time markets to combine"
        return result

    payoff = np.array(columns).T  # scorelines x legs
    lp = linprog(
        c=np.ones(len(legs)),
        A_ub=-payoff,
        b_ub=-np.ones(len(GRID)),
        bounds=[(0, None)] * len(legs),
        method="highs",
    )
    if not lp.success:
        result.message = lp.message
        return result

    result.solved = True
    result.cost = float(lp.fun)
    result.legs = [Leg(*legs[k], stake=float(lp.x[k])) for k in np.argsort(-lp.x) if lp.x[k] > 1e-6]
    cover = payoff @ lp.x
    result.tightest = [(GRID[i], float(cover[i] / lp.fun)) for i in np.argsort(cover)[:show_tightest]]
    return result


def scan(book: OddsBook) -> ScanReport:
    return ScanReport(overrounds=overrounds(book), ladders=ladders(book), portfolio=best_portfolio(book))
