import json
import math
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from wm.config import DEFAULT_EFFORT, DEFAULT_MODEL, Settings
from wm.evaluation import report, runner
from wm.evaluation.fixtures import Fixture, load_fixtures, parse_api_fixtures, save_fixtures
from wm.evaluation.leakage import visible_before
from wm.evaluation.metrics import UNIFORM_BRIER_1X2, brier_1x2, logloss_1x2, paired_bootstrap, per_match_scores
from wm.llm import Claude
from wm.predict import from_model_output
from wm.rag.store import Hit


def api_item(fid, home, away, ft, status="FT", date="2026-06-12T19:00:00+00:00", rnd="Group Stage - 1"):
    return {
        "fixture": {"id": fid, "date": date, "status": {"short": status}},
        "league": {"round": rnd},
        "teams": {"home": {"name": home}, "away": {"name": away}},
        "score": {"fulltime": {"home": ft[0], "away": ft[1]}},
    }


def test_parse_api_fixtures():
    items = [
        api_item(2, "USA", "Paraguay", (2, 1), date="2026-06-13T01:00:00+00:00"),
        api_item(1, "Canada", "Bosnia & Herzegovina", (1, 1)),
        api_item(3, "Spain", "Uruguay", (1, 1), status="PEN", date="2026-07-04T19:00:00+00:00", rnd="Round of 16"),
        api_item(4, "Brazil", "Morocco", (None, None), status="NS"),
    ]
    fixtures, warnings = parse_api_fixtures(items)
    assert [f.match_id for f in fixtures] == ["1", "2", "3"]  # sorted by kickoff
    assert fixtures[0].team_b == "Bosnia and Herzegovina" and fixtures[1].team_a == "United States"
    assert fixtures[1].outcome == 0 and fixtures[0].outcome == 1 and fixtures[2].outcome == 1
    assert fixtures[0].is_group and not fixtures[2].is_group
    assert len(warnings) == 1 and "NS" in warnings[0]


def test_extra_time_score_is_rejected():
    with pytest.raises(ValueError, match="non-level"):
        parse_api_fixtures([api_item(5, "France", "Norway", (2, 1), status="AET")])


def test_fixtures_round_trip(tmp_path):
    fixtures, _ = parse_api_fixtures([api_item(1, "Canada", "Qatar", (3, 0))])
    save_fixtures(fixtures, tmp_path / "r.csv")
    assert load_fixtures(tmp_path / "r.csv") == fixtures


# --------------------------------------------------------------------------- metrics
def fixture(match_id="1", goals=(1, 0), stage="Group Stage - 1"):
    return Fixture(match_id, "2026-06-12T19:00:00+00:00", stage, "Canada", "Qatar", *goals, "FT")


def pred(a, x, b, o25=50, o35=30, btts=50):
    raw = {"team_a_win": a, "draw": x, "team_b_win": b, "over_2_5": o25, "over_3_5": o35, "btts": btts}
    return from_model_output("Canada", "Qatar", {**raw, "rationale": ""}, [])


def test_scores():
    f = fixture(goals=(1, 0))
    assert brier_1x2(pred(1, 1, 1), f) == pytest.approx(UNIFORM_BRIER_1X2)
    assert logloss_1x2(pred(1, 1, 1), f) == pytest.approx(math.log(3))
    assert brier_1x2(pred(100, 0, 0), f) == 0
    assert brier_1x2(pred(0, 0, 100), f) == pytest.approx(2)
    assert logloss_1x2(pred(0, 0, 100), f) < 20  # clipped, not infinite
    s = per_match_scores(pred(60, 25, 15, o25=30, btts=20), f)
    assert s["hit_1x2"] == 1 and s["brier_over_2_5"] == pytest.approx(0.09) and s["brier_btts"] == pytest.approx(0.04)


def test_paired_bootstrap():
    d = paired_bootstrap([0.5] * 50, [0.4 + 0.01 * (i % 3) for i in range(50)])
    assert d.mean < 0 and d.high < 0 and d.significant
    assert not paired_bootstrap([0.5, 0.6], [0.6, 0.5]).significant


def test_leakage_filter():
    keep = visible_before(datetime(2026, 6, 20, tzinfo=UTC))
    hit = lambda kind, **meta: Hit("t", "s", kind, 0.1, meta)  # noqa: E731
    assert keep(hit("team_context"))  # legacy, pre-tournament dossier
    assert not keep(hit("match_report_analysis"))  # undated report: could contain anything
    assert keep(hit("match_report_analysis", ingested_at="2026-06-19T10:00:00+00:00"))
    assert not keep(hit("team_context", ingested_at="2026-06-21T10:00:00+00:00"))


# --------------------------------------------------------------------------- runner
class FakeMessages:
    def __init__(self, answers):
        self.answers, self.calls = answers, []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        teams = kwargs["messages"][0]["content"].splitlines()[0]
        answer = self.answers(teams)
        if isinstance(answer, Exception):
            raise answer
        return SimpleNamespace(
            stop_reason="end_turn",
            model=kwargs["model"],
            content=[SimpleNamespace(type="text", text=json.dumps(answer))],
            usage=SimpleNamespace(input_tokens=1000, output_tokens=200, cache_read_input_tokens=0),
        )


def fake_llm(answers, model=DEFAULT_MODEL, effort=DEFAULT_EFFORT):
    messages = FakeMessages(answers)
    settings = Settings(model, effort, None, "", "key", None)
    return Claude(settings, client=SimpleNamespace(beta=SimpleNamespace(messages=messages))), messages


GOOD = {"team_a_win": 60, "draw": 25, "team_b_win": 15, "over_2_5": 45, "over_3_5": 20, "btts": 40, "rationale": ""}
FIXTURES = [fixture("1", (2, 0)), fixture("2", (1, 0)), fixture("3", (1, 1), stage="Final")]


def test_run_records_costs_and_resumes(tmp_path):
    config = runner.RunConfig(DEFAULT_MODEL, DEFAULT_EFFORT, rag=True)
    llm, messages = fake_llm(lambda _: GOOD)
    retrieved = []
    retrieve = lambda f: retrieved.append(f.match_id) or [Hit("doc", "api_football:Canada", "team_context", 0.1, {})]  # noqa: E731
    path = tmp_path / "run.jsonl"

    records = runner.run(llm, config, FIXTURES, path, retrieve=retrieve, workers=2)
    assert len(records) == 3 and all(r.ok for r in records)
    assert sorted(retrieved) == ["1", "2", "3"]
    assert '<document index="1"' in messages.calls[0]["messages"][0]["content"]
    rec = runner.load_records(path)["1"]
    assert rec.sources == ["api_football:Canada"] and rec.latency_s is not None
    assert rec.cost_usd == pytest.approx(runner.cost_usd(DEFAULT_MODEL, 1000, 200))
    assert len(path.read_text().splitlines()) == 3

    runner.run(llm, config, FIXTURES, path, retrieve=retrieve)
    assert len(messages.calls) == 3  # nothing left to do


def test_failures_are_recorded_and_retried(tmp_path):
    config = runner.RunConfig(DEFAULT_MODEL, DEFAULT_EFFORT, rag=False)
    bad = {**GOOD}
    del bad["btts"]
    llm, messages = fake_llm(lambda teams: bad if "Qatar" in teams else GOOD)
    records = runner.run(llm, config, FIXTURES[:1], tmp_path / "r.jsonl")
    assert records[0].error and not records[0].ok

    llm, _ = fake_llm(lambda _: GOOD)
    records = runner.run(llm, config, FIXTURES[:1], tmp_path / "r.jsonl")
    assert records[0].ok and len(runner.load_records(tmp_path / "r.jsonl")) == 1


def test_fingerprint_tracks_prompt_and_config(monkeypatch):
    config = runner.RunConfig(DEFAULT_MODEL, DEFAULT_EFFORT, rag=True)
    fp = runner.fingerprint(config)
    assert fp == runner.fingerprint(config)
    assert fp != runner.fingerprint(runner.RunConfig("claude-haiku-5-5", DEFAULT_EFFORT, rag=True))
    assert fp != runner.fingerprint(runner.RunConfig(DEFAULT_MODEL, DEFAULT_EFFORT, rag=False))
    monkeypatch.setattr(runner, "SYSTEM", runner.SYSTEM + " Be bold.")
    assert fp != runner.fingerprint(config)


# --------------------------------------------------------------------------- report and gate
def make_run(tmp_path, config, answer):
    llm, _ = fake_llm(lambda _: answer, model=config.model, effort=config.effort)
    path = runner.run_path(tmp_path / "runs", config.label)
    runner.run(llm, config, FIXTURES, path, retrieve=lambda f: [])
    return report.score_run(path, FIXTURES)


def test_report_compares_rag_and_models(tmp_path):
    opus_rag = make_run(tmp_path, runner.RunConfig(DEFAULT_MODEL, DEFAULT_EFFORT, True), GOOD)
    opus = make_run(tmp_path, runner.RunConfig(DEFAULT_MODEL, DEFAULT_EFFORT, False), {**GOOD, "team_a_win": 34})
    haiku = make_run(tmp_path, runner.RunConfig("claude-haiku-5-5", DEFAULT_EFFORT, False), GOOD)
    text = report.render([opus_rag, opus, haiku], FIXTURES)
    assert "Does retrieval help?" in text and "Models compared with claude-opus-5-5" in text
    assert "40.0× cheaper" in text  # $4/$20 vs $0.10/$0.50 per MTok
    assert "Group stage vs knockouts" in text
    summary = report.summary_json([opus_rag])
    assert summary[opus_rag.label]["n"] == 3 and summary[opus_rag.label]["cost_usd"] > 0


def test_gate(tmp_path, monkeypatch):
    runs, gate = tmp_path / "runs", tmp_path / "gate.json"
    prod = report.production_config()

    score = make_run(tmp_path, prod, GOOD)
    report.write_gate(gate, score, tolerance=0.01)
    assert report.check(gate, runs, FIXTURES) == []

    # A worse prompt: re-run, and the gate catches the regression.
    monkeypatch.setattr(runner, "SYSTEM", runner.SYSTEM + " Always back the first team.")
    assert any("different prompt" in p for p in report.check(gate, runs, FIXTURES))
    make_run(tmp_path, prod, {**GOOD, "team_a_win": 95, "draw": 3, "team_b_win": 2})
    problems = report.check(gate, runs, FIXTURES)
    assert any(p.startswith("brier_1x2 regressed") for p in problems)


def test_gate_requires_a_production_run(tmp_path):
    gate = tmp_path / "gate.json"
    gate.write_text(json.dumps({"run": "x", "max": {}, "baseline": {}, "tolerance": 0.01}))
    assert "No evaluation run" in report.check(gate, tmp_path / "runs", FIXTURES)[0]


def test_cli_check_without_gate_passes(tmp_path, capsys):
    from wm.cli import main

    assert main(["eval", "--eval-dir", str(tmp_path), "check"]) == 0  # no gate and no results yet: CI passes
    assert "nothing to check" in capsys.readouterr().out
    save_fixtures(FIXTURES, tmp_path / "wc2026_results.csv")
    assert main(["eval", "--eval-dir", str(tmp_path), "run", "--no-rag", "--dry-run"]) == 0
    assert "3 of 3 matches" in capsys.readouterr().out
