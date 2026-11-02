"""Proper scoring rules for probabilistic forecasts, plus a paired bootstrap.

Lower is better for every score here. Reference points for a forecaster that
knows nothing (1/3 each for 1X2, 50% for binary markets):

* 1X2 Brier (summed over the three outcomes, range 0-2): 0.667
* 1X2 log loss (natural log): ln 3 = 1.099
* binary Brier: 0.25
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from wm.evaluation.fixtures import Fixture
from wm.predict import Prediction

UNIFORM_BRIER_1X2 = 2 / 3
UNIFORM_LOGLOSS_1X2 = math.log(3)
EPS = 1e-6  # log loss of a 0% forecast that happens would otherwise be infinite


def brier_1x2(p: Prediction, f: Fixture) -> float:
    probs = (p.p_a, p.p_draw, p.p_b)
    return sum((q - (i == f.outcome)) ** 2 for i, q in enumerate(probs))


def logloss_1x2(p: Prediction, f: Fixture) -> float:
    return -math.log(max((p.p_a, p.p_draw, p.p_b)[f.outcome], EPS))


def brier_binary(prob: float, happened: bool) -> float:
    return (prob - happened) ** 2


def per_match_scores(p: Prediction, f: Fixture) -> dict[str, float]:
    probs = (p.p_a, p.p_draw, p.p_b)
    return {
        "brier_1x2": brier_1x2(p, f),
        "logloss_1x2": logloss_1x2(p, f),
        # Ties for the favourite count as a miss; with continuous probabilities they are rare.
        "hit_1x2": float(probs.index(max(probs)) == f.outcome and probs.count(max(probs)) == 1),
        "brier_over_2_5": brier_binary(p.p_over_2_5, f.total_goals >= 3),
        "brier_over_3_5": brier_binary(p.p_over_3_5, f.total_goals >= 4),
        "brier_btts": brier_binary(p.p_btts, f.btts),
    }


def mean_scores(rows: Sequence[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    return {key: sum(r[key] for r in rows) / len(rows) for key in rows[0]}


@dataclass(frozen=True)
class PairedDiff:
    """Mean of (B - A) over the matches both runs scored, with a bootstrap 95% interval."""

    n: int
    mean: float
    low: float
    high: float

    @property
    def significant(self) -> bool:
        return self.low > 0 or self.high < 0


def paired_bootstrap(a: Sequence[float], b: Sequence[float], *, resamples: int = 10_000, seed: int = 0) -> PairedDiff:
    """Bootstrap the mean per-match difference ``b - a``, resampling matches.

    Pairing matters: some matches are hard for every model, so comparing two
    independent means would hide real differences under that shared noise.
    """
    diffs = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    if diffs.size == 0:
        return PairedDiff(0, math.nan, math.nan, math.nan)
    rng = np.random.default_rng(seed)
    means = diffs[rng.integers(0, diffs.size, size=(resamples, diffs.size))].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return PairedDiff(int(diffs.size), float(diffs.mean()), float(low), float(high))
