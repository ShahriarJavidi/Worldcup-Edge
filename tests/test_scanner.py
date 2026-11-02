import pytest

from wm.arbitrage.dutch import combinations
from wm.arbitrage.odds import OddsBook, clean_selection
from wm.arbitrage.scanner import best_portfolio, ladders, overrounds


def book(rows):
    b = OddsBook()
    for market, sel, odds in rows:
        b.add(market, sel, odds)
    return b


def test_overround_flags_single_market_arb():
    result = overrounds(book([("Final Result", "1", 3.2), ("Final Result", "X", 3.5), ("Final Result", "2", 3.4)]))
    assert result[0].market == "Final Result"
    assert result[0].is_arb


def test_overround_splits_over_under_lines():
    b = book([("Over/Under", "Over 2.5", 1.9), ("Over/Under", "Under 2.5", 1.9), ("Over/Under", "Over 1.5", 1.3)])
    names = [r.market for r in overrounds(b)]
    assert names == ["Over/Under [O/U 2.5]"]  # 1.5 has only one side


def test_ladder_violation():
    b = book([("Over/Under", "Over 1.5", 1.5), ("Over/Under", "Over 2.5", 1.4)])
    assert ladders(b)[0].violated


def test_lp_finds_cross_market_arb():
    # 1X2 with a fat book, but "1" is mispriced against Double Chance X2.
    b = book(
        [
            ("Final Result", "1", 2.2),
            ("Final Result", "X", 3.0),
            ("Final Result", "2", 3.0),
            ("Double Chance", "X2", 2.1),
        ]
    )
    result = best_portfolio(b)
    assert result.is_arb
    assert {leg.selection for leg in result.legs} == {"1", "X2"}
    assert result.cost == pytest.approx(1 / 2.2 + 1 / 2.1)


def test_lp_no_arb_on_fair_book():
    b = book([("Final Result", "1", 2.0), ("Final Result", "X", 3.5), ("Final Result", "2", 4.0)])
    result = best_portfolio(b)
    assert result.solved and not result.is_arb


def test_clean_selection_strips_scraper_noise():
    assert clean_selection("Over 2.5_sr:abc") == "Over 2.5"


def test_dutch_finds_best_combo_and_equal_returns():
    best = combinations([(1.90, 3.65, 4.20), (2.10, 3.40, 4.60)])[0]
    assert best.odds == (2.10, 3.65, 4.60) and best.is_arb
    returns = [s * o for s, o in zip(best.stakes(100), best.odds, strict=True)]
    assert max(returns) == pytest.approx(min(returns))
    assert sum(best.stakes(100)) == pytest.approx(100)
