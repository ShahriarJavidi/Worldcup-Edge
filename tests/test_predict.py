import pytest

from wm.predict import build_prompt, from_model_output
from wm.rag.store import Hit
from wm.teams import host_context

RAW = {"team_a_win": 50, "draw": 30, "team_b_win": 30, "over_2_5": 45, "over_3_5": 60, "btts": 52, "rationale": "x"}


def test_1x2_is_normalised_and_ladder_enforced():
    p = from_model_output("Canada", "Qatar", RAW, [])
    assert p.p_a + p.p_draw + p.p_b == pytest.approx(1)
    assert p.p_over_3_5 <= p.p_over_2_5


def test_derived_markets_are_consistent():
    p = from_model_output("Canada", "Qatar", RAW, [])
    markets = dict(p.markets())
    dc = dict(markets["Double chance"])
    assert dc["1X"] == pytest.approx(p.p_a + p.p_draw)
    dnb = dict(markets["Draw no bet"])
    assert sum(dnb.values()) == pytest.approx(1)


def test_host_context():
    assert "Canada is a 2026 host" in host_context("Canada", "Qatar")
    assert "Neutral venue" in host_context("Spain", "Uruguay")
    assert "Both teams" in host_context("USA", "Mexico")


def test_prompt_includes_documents():
    hit = Hit(text="Canada lost their captain to injury.", source="s", kind="team_context", distance=0.1, metadata={})
    prompt = build_prompt("Canada", "Qatar", [hit])
    assert '<document index="1"' in prompt and "captain" in prompt
    assert "No retrieved context" in build_prompt("Canada", "Qatar", [])
