from wm.arbitrage.markets import GRID, count_band, over_under, winning_scores


def test_final_result_partitions_the_grid():
    wins = [set(winning_scores("Final Result", s)) for s in ("1", "X", "2")]
    assert set().union(*wins) == set(GRID)
    assert sum(len(w) for w in wins) == len(GRID)


def test_over_under_half_line():
    over = set(winning_scores("Over/Under", "Over 2.5"))
    under = set(winning_scores("Over/Under", "Under 2.5"))
    assert (1, 2) in over and (1, 1) in under
    assert over.isdisjoint(under) and over | under == set(GRID)


def test_integer_lines_push_and_are_skipped():
    assert over_under("Over 2") == ("PUSH", 2.0)
    assert winning_scores("Over/Under", "Over 2") is None


def test_count_bands():
    assert count_band("1-2")(2) and not count_band("1-2")(3)
    assert count_band("4 or more")(9) and count_band("3+")(3)
    assert count_band("No Goal")(0) and not count_band("No Goal")(1)
    assert count_band("Other") is None


def test_correct_score_and_other():
    assert winning_scores("Correct Score", "2:1") == [(2, 1)]
    assert winning_scores("Correct Score", "Other") is None


def test_european_handicap():
    # Team 2 receives a 0:1 start; "X 0:1" wins when team 1 wins by exactly one.
    wins = set(winning_scores("Handicap", "X 0:1"))
    assert (2, 1) in wins and (1, 1) not in wins and (3, 1) not in wins


def test_asian_half_line_handicap():
    wins = set(winning_scores("Handicap", "-1.5 Team 1"))
    assert (2, 0) in wins and (1, 0) not in wins


def test_combo_market():
    wins = set(winning_scores("Matchbet and Over/Under 2.5", "1 and Over 2.5"))
    assert (2, 1) in wins and (1, 0) not in wins and (1, 2) not in wins


def test_unknown_market_is_not_modelled():
    assert winning_scores("Halftime / Fulltime", "1/1") is None
