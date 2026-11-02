"""Market semantics: which full-time scorelines make each selection win.

Every market that settles purely on the 90-minute score is mapped to the set
of ``(home_goals, away_goals)`` cells it wins on a capped grid. Market names
are the exact strings used by Swisslos Sporttip; a new bookmaker or market
needs a new branch in :func:`winning_scores`.
"""

from __future__ import annotations

import re
from collections.abc import Callable

CAP = 7  # goals per team; every market boundary sits well below this
GRID: list[tuple[int, int]] = [(h, a) for h in range(CAP + 1) for a in range(CAP + 1)]

Score = tuple[int, int]
ScorePredicate = Callable[[int, int], bool]
CountPredicate = Callable[[int], bool]

# Markets whose listed selections form a mutually exclusive, exhaustive partition.
PARTITION_MARKETS = {
    "Final Result",
    "Both Teams to score",
    "Goal Range (0-1, 2-3, 4-5, 6+)",
    "Exact Goals (0-6+)",
    "Team 1 - Goals",
    "Team 2 - Goals",
    "Which Team To Score",
    "Number of Goals - Odd/Even",
    "Team 1 - Odd/Even number of Goals",
    "Team 2 - Odd/Even number of Goals",
    "Team 1 Clean Sheet",
    "Team 2 Clean Sheet",
    "Home Win to Nil",
    "Away Win to Nil",
    "Halftime / Fulltime",
    "Who wins the 1st Half",
    "Who wins the 2nd Half",
    "Which Half will produce most Goals?",
    "Matchbet and Over/Under 1.5",
    "Matchbet and Over/Under 2.5",
    "Matchbet and Over/Under 3.5",
    "Matchbet and Over/Under 4.5",
    "Matchbet and both Teams to score",
    "Both Teams to score and Over/Under 2.5",
    "1st/2nd Half - Both Teams to Score",
}

# Over/Under ladders whose implied P(Over) must fall as the line rises.
LADDER_MARKETS = [
    "Over/Under",
    "1st half – Over/Under",
    "2nd Half - Over/Under",
    "How many Goals will Team 1 score - Over/Under",
    "How many Goals will Team 2 score - Over/Under",
]

THREE_WAY_HANDICAP_MARKETS = ["Handicap", "1st Half - Handicap", "2nd Half - Handicap"]

PUSH = "PUSH"


# --------------------------------------------------------------------------- label parsing
def result(h: int, a: int) -> str:
    return "1" if h > a else ("X" if h == a else "2")


def btts(h: int, a: int) -> bool:
    return h >= 1 and a >= 1


DOUBLE_CHANCE: dict[str, ScorePredicate] = {
    "1X": lambda h, a: h >= a,
    "12": lambda h, a: h != a,
    "X2": lambda h, a: h <= a,
}

WHICH_TEAM_SCORES: dict[str, ScorePredicate] = {
    "Only Team 1": lambda h, a: h >= 1 and a == 0,
    "Only Team 2": lambda h, a: a >= 1 and h == 0,
    "Both Teams": lambda h, a: h >= 1 and a >= 1,
    "None": lambda h, a: h == 0 and a == 0,
}


def count_band(label: str) -> CountPredicate | None:
    """Predicate over a goal count from labels like ``1-2``, ``4 or more``, ``3+``, ``0``, ``No Goal``."""
    label = label.strip()
    if label.lower() in ("no goal", "none"):
        return lambda n: n == 0
    if m := re.match(r"^(\d+)\s*-\s*(\d+)$", label):
        lo, hi = int(m[1]), int(m[2])
        return lambda n: lo <= n <= hi
    if m := re.match(r"^(\d+)\s*(?:or more|\+)", label):
        lo = int(m[1])
        return lambda n: n >= lo
    if m := re.match(r"^(\d+)$", label):
        exact = int(m[1])
        return lambda n: n == exact
    return None


def over_under(label: str) -> tuple[bool | str, float] | None:
    """``Over 2.5`` -> ``(True, 2.5)``; integer lines (which can push) -> ``("PUSH", line)``."""
    m = re.match(r"^(Over|Under)\s+([0-9.]+)$", label)
    if not m:
        return None
    line = float(m[2])
    if line == int(line):
        return (PUSH, line)
    return (m[1] == "Over", line)


# --------------------------------------------------------------------------- indicator
def _where(pred: ScorePredicate) -> list[Score]:
    return [(h, a) for h, a in GRID if pred(h, a)]


def _ou_where(sel: str, count: Callable[[int, int], int]) -> list[Score] | None:
    parsed = over_under(sel)
    if not parsed or parsed[0] == PUSH:
        return None
    over, line = parsed
    return _where(lambda h, a: (count(h, a) > line) == over)


def _band_where(sel: str, count: Callable[[int, int], int]) -> list[Score] | None:
    band = count_band(sel)
    return _where(lambda h, a: band(count(h, a))) if band else None


def _yes_where(sel: str, pred: ScorePredicate) -> list[Score]:
    yes = sel == "Yes"
    return _where(lambda h, a: pred(h, a) == yes)


def _odd_where(sel: str, count: Callable[[int, int], int]) -> list[Score]:
    odd = sel == "Odd"
    return _where(lambda h, a: (count(h, a) % 2 == 1) == odd)


TOTAL = lambda h, a: h + a  # noqa: E731
HOME = lambda h, a: h  # noqa: E731
AWAY = lambda h, a: a  # noqa: E731

_BAND_MARKETS = {
    "Goal Range (0-1, 2-3, 4-5, 6+)": TOTAL,
    "Exact Goals (0-6+)": TOTAL,
    "Multigoals": TOTAL,
    "Home Multigoals": HOME,
    "Away Multigoals": AWAY,
    "Team 1 - Goals": HOME,
    "Team 2 - Goals": AWAY,
}
_OU_MARKETS = {
    "Over/Under": TOTAL,
    "How many Goals will Team 1 score - Over/Under": HOME,
    "How many Goals will Team 2 score - Over/Under": AWAY,
}
_YES_NO_MARKETS: dict[str, ScorePredicate] = {
    "Both Teams to score": btts,
    "Team 1 Clean Sheet": lambda h, a: a == 0,
    "Team 2 Clean Sheet": lambda h, a: h == 0,
    "Home Win to Nil": lambda h, a: h > a and a == 0,
    "Away Win to Nil": lambda h, a: a > h and h == 0,
}
_ODD_EVEN_MARKETS = {
    "Number of Goals - Odd/Even": TOTAL,
    "Team 1 - Odd/Even number of Goals": HOME,
    "Team 2 - Odd/Even number of Goals": AWAY,
}


def winning_scores(market: str, sel: str) -> list[Score] | None:
    """Scorelines on which ``sel`` wins, or ``None`` if the selection is not modelled.

    ``None`` covers markets that can push/refund or that do not settle on the
    full-time score (halves, HT/FT, scorers, timing). Leaving them out can only
    make the arbitrage LP more conservative, never produce a false positive.
    """
    if market == "Final Result":
        return _where(lambda h, a: result(h, a) == sel)
    if market == "Double Chance":
        pred = DOUBLE_CHANCE.get(sel)
        return _where(pred) if pred else None
    if market == "Which Team To Score":
        pred = WHICH_TEAM_SCORES.get(sel)
        return _where(pred) if pred else None
    if market == "Correct Score":
        m = re.match(r"^(\d+):(\d+)", sel)
        if not m:
            return None  # "Other" is not modelled
        score = (int(m[1]), int(m[2]))
        return [score] if score in GRID else []
    if market in _BAND_MARKETS:
        return _band_where(sel, _BAND_MARKETS[market])
    if market in _OU_MARKETS:
        return _ou_where(sel, _OU_MARKETS[market])
    if market in _YES_NO_MARKETS:
        return _yes_where(sel, _YES_NO_MARKETS[market])
    if market in _ODD_EVEN_MARKETS:
        return _odd_where(sel, _ODD_EVEN_MARKETS[market])
    if market == "Handicap":
        return _handicap(sel)
    return _combo(market, sel)


def _handicap(sel: str) -> list[Score] | None:
    """One market mixes 3-way European lines (``2 0:3``) and Asian half lines (``-2.5 Team 1``)."""
    if m := re.match(r"^([1X2])\s+(\d+):(\d+)$", sel):
        pick, gh, ga = m[1], int(m[2]), int(m[3])
        return _where(lambda h, a: result(h + gh, a + ga) == pick)
    if m := re.match(r"^([+-]?\d+\.5)\s+Team\s+([12])\b", sel):
        adj, team = float(m[1]), m[2]
        if team == "1":
            return _where(lambda h, a: h + adj > a)
        return _where(lambda h, a: a + adj > h)
    return None


def _combo(market: str, sel: str) -> list[Score] | None:
    if " and " not in sel:
        return None
    left, right = sel.split(" and ", 1)

    if market.startswith("Matchbet and Over/Under"):
        line, over = float(market.split()[-1]), right.startswith("Over")
        return _where(lambda h, a: result(h, a) == left and (h + a > line) == over)
    if market == "Matchbet and both Teams to score":
        yes = right == "Yes"
        return _where(lambda h, a: result(h, a) == left and btts(h, a) == yes)
    if market.startswith("Double Chance and Over/Under"):
        line, over, dc = float(market.split()[-1]), right.startswith("Over"), DOUBLE_CHANCE.get(left)
        return _where(lambda h, a: dc(h, a) and (h + a > line) == over) if dc else None
    if market == "Double Chance and both Teams to Score":
        yes, dc = right == "Yes", DOUBLE_CHANCE.get(left)
        return _where(lambda h, a: dc(h, a) and btts(h, a) == yes) if dc else None
    if market == "Both Teams to score and Over/Under 2.5":
        over, yes = left.startswith("Over"), right.startswith("Yes")
        return _where(lambda h, a: (h + a > 2.5) == over and btts(h, a) == yes)
    return None
