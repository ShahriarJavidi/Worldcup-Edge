"""Score run files against results, compare configurations, and gate CI."""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from wm.config import DEFAULT_EFFORT, DEFAULT_MODEL
from wm.evaluation.fixtures import Fixture
from wm.evaluation.metrics import (
    UNIFORM_BRIER_1X2,
    PairedDiff,
    mean_scores,
    paired_bootstrap,
    per_match_scores,
)
from wm.evaluation.runner import RunConfig, fingerprint, load_records


@dataclass
class RunScore:
    label: str
    config: RunConfig
    n_fixtures: int
    per_match: dict[str, dict[str, float]]  # match_id -> scores
    stale: int  # records whose fingerprint no longer matches the code
    failed: int
    cost_usd: float | None
    latencies: list[float] = field(default_factory=list)
    served_models: dict[str, int] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.per_match)

    @property
    def coverage(self) -> float:
        return self.n / self.n_fixtures if self.n_fixtures else 0.0

    def mean(self, only: set[str] | None = None) -> dict[str, float]:
        rows = [s for m, s in self.per_match.items() if only is None or m in only]
        return mean_scores(rows)

    @property
    def cost_per_match(self) -> float | None:
        return self.cost_usd / self.n if self.cost_usd is not None and self.n else None

    @property
    def median_latency(self) -> float | None:
        return statistics.median(self.latencies) if self.latencies else None

    @property
    def p90_latency(self) -> float | None:
        if len(self.latencies) < 2:
            return self.median_latency
        return statistics.quantiles(self.latencies, n=10)[-1]


def score_run(path: str | Path, fixtures: list[Fixture]) -> RunScore:
    """Score every successful, current-fingerprint record in one run file."""
    records = load_records(path)
    if not records:
        raise ValueError(f"{path} has no records.")
    first = next(iter(records.values()))
    config = RunConfig(model=first.model, effort=first.effort, rag=first.rag, k=first.k)
    fp = fingerprint(config)
    by_id = {f.match_id: f for f in fixtures}

    per_match: dict[str, dict[str, float]] = {}
    stale = failed = 0
    costs: list[float | None] = []
    latencies: list[float] = []
    served: dict[str, int] = {}
    for match_id, rec in records.items():
        fixture = by_id.get(match_id)
        if fixture is None:
            continue
        if rec.fingerprint != fp:
            stale += 1
            continue
        if not rec.ok:
            failed += 1
            continue
        per_match[match_id] = per_match_scores(rec.prediction(fixture), fixture)
        costs.append(rec.cost_usd)
        if rec.latency_s is not None:
            latencies.append(rec.latency_s)
        if rec.served_model:
            served[rec.served_model] = served.get(rec.served_model, 0) + 1
    total_cost = None if not costs or any(c is None for c in costs) else sum(c for c in costs if c is not None)
    return RunScore(
        label=Path(path).stem,
        config=config,
        n_fixtures=len(fixtures),
        per_match=per_match,
        stale=stale,
        failed=failed,
        cost_usd=total_cost,
        latencies=latencies,
        served_models=served,
    )


def compare(a: RunScore, b: RunScore, metric: str = "brier_1x2") -> PairedDiff:
    """Paired difference ``b - a`` on the matches both runs scored (negative means b is better)."""
    common = sorted(a.per_match.keys() & b.per_match.keys())
    return paired_bootstrap([a.per_match[m][metric] for m in common], [b.per_match[m][metric] for m in common])


# --------------------------------------------------------------------------- text report
def _f(x: float | None, fmt: str) -> str:
    return "—" if x is None else format(x, fmt)


def render(scores: list[RunScore], fixtures: list[Fixture]) -> str:
    """A Markdown report: the leaderboard, RAG vs no-RAG, model vs model, and group vs knockout."""
    out: list[str] = []
    n_fix = len(fixtures)
    out.append(f"## Evaluation on {n_fix} World Cup 2026 matches (90-minute results)\n")
    out.append(
        "Lower is better for Brier and log loss. A know-nothing forecast (1/3 each, 50% for binary markets) "
        f"scores 1X2 Brier {UNIFORM_BRIER_1X2:.3f}, log loss 1.099 and binary Brier 0.250.\n"
    )
    out.append(
        "| run | n | 1X2 Brier | 1X2 log loss | favourite hit | O2.5 Brier | O3.5 Brier | BTTS Brier "
        "| cost total | cost/match | median s | p90 s |"
    )
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for s in sorted(scores, key=lambda s: s.mean().get("brier_1x2", 9)):
        m = s.mean()
        flags = []
        if s.n < s.n_fixtures:
            flags.append(f"{s.n_fixtures - s.n} missing")
        if s.stale:
            flags.append(f"{s.stale} stale")
        note = f" ({', '.join(flags)})" if flags else ""
        out.append(
            f"| {s.label}{note} | {s.n} | {_f(m.get('brier_1x2'), '.4f')} | {_f(m.get('logloss_1x2'), '.4f')} "
            f"| {_f(100 * m['hit_1x2'] if m else None, '.1f')}% | {_f(m.get('brier_over_2_5'), '.4f')} "
            f"| {_f(m.get('brier_over_3_5'), '.4f')} | {_f(m.get('brier_btts'), '.4f')} "
            f"| {_f(s.cost_usd, '.2f')} | {_f(s.cost_per_match, '.4f')} "
            f"| {_f(s.median_latency, '.1f')} | {_f(s.p90_latency, '.1f')} |"
        )
    out.append("\nCosts are in USD at list price; latency is wall-clock seconds per request, including retries.")

    by_key = {(s.config.model, s.config.effort, s.config.rag): s for s in scores}
    pairs = [(by_key[(m, e, False)], s) for (m, e, rag), s in sorted(by_key.items()) if rag and (m, e, False) in by_key]
    if pairs:
        out.append("\n### Does retrieval help? (RAG minus no-RAG, paired over the same matches)\n")
        out.append("| model | 1X2 Brier: no RAG → RAG | difference [95% CI] | log loss difference [95% CI] |")
        out.append("|---|---|---|---|")
        for base, rag in pairs:
            d, dl = compare(base, rag), compare(base, rag, "logloss_1x2")
            common = base.per_match.keys() & rag.per_match.keys()
            out.append(
                f"| {rag.config.model} ({rag.config.effort}) "
                f"| {base.mean(common)['brier_1x2']:.4f} → {rag.mean(common)['brier_1x2']:.4f} "
                f"| {_diff(d)} | {_diff(dl)} |"
            )

    groups: dict[tuple[str, bool], list[RunScore]] = {}
    for s in scores:
        groups.setdefault((s.config.effort, s.config.rag), []).append(s)
    for (effort, rag), runs in sorted(groups.items()):
        if len(runs) < 2:
            continue
        ref = max(runs, key=lambda s: (s.config.model == DEFAULT_MODEL, -s.mean().get("brier_1x2", 9)))
        out.append(f"\n### Models compared with {ref.config.model} ({effort}, {'RAG' if rag else 'no RAG'})\n")
        out.append("| model | 1X2 Brier difference [95% CI] | relative | cost ratio | median latency ratio |")
        out.append("|---|---|---|---|---|")
        for s in runs:
            if s is ref:
                continue
            d = compare(ref, s)
            ref_brier = ref.mean(ref.per_match.keys() & s.per_match.keys())["brier_1x2"]
            cost_ratio = (
                f"{ref.cost_per_match / s.cost_per_match:.1f}× cheaper"
                if ref.cost_per_match and s.cost_per_match
                else "—"
            )
            lat = f"{s.median_latency / ref.median_latency:.2f}×" if s.median_latency and ref.median_latency else "—"
            out.append(f"| {s.config.model} | {_diff(d)} | {100 * d.mean / ref_brier:+.1f}% | {cost_ratio} | {lat} |")

    group_ids = {f.match_id for f in fixtures if f.is_group}
    ko_ids = {f.match_id for f in fixtures if not f.is_group}
    if group_ids and ko_ids:
        out.append("\n### Group stage vs knockouts (1X2 Brier)\n")
        out.append(
            "Later matches are the cleaner test: the further past a model's training cutoff a match is, the less "
            "chance the model has seen its result. A large group-stage advantage over knockouts is a warning sign.\n"
        )
        out.append(f"| run | group ({len(group_ids)}) | knockout ({len(ko_ids)}) |")
        out.append("|---|---|---|")
        for s in sorted(scores, key=lambda s: s.label):
            g, k = s.mean(group_ids), s.mean(ko_ids)
            out.append(f"| {s.label} | {_f(g.get('brier_1x2'), '.4f')} | {_f(k.get('brier_1x2'), '.4f')} |")
    return "\n".join(out) + "\n"


def _diff(d: PairedDiff) -> str:
    star = " *" if d.significant else ""
    return f"{d.mean:+.4f} [{d.low:+.4f}, {d.high:+.4f}]{star}"


def summary_json(scores: list[RunScore]) -> dict[str, Any]:
    return {
        s.label: {
            "model": s.config.model,
            "effort": s.config.effort,
            "rag": s.config.rag,
            "n": s.n,
            "stale": s.stale,
            "failed": s.failed,
            **{k: round(v, 5) for k, v in s.mean().items()},
            "cost_usd": None if s.cost_usd is None else round(s.cost_usd, 4),
            "cost_per_match_usd": None if s.cost_per_match is None else round(s.cost_per_match, 6),
            "median_latency_s": s.median_latency,
            "p90_latency_s": s.p90_latency,
            "served_models": s.served_models,
        }
        for s in scores
    }


# --------------------------------------------------------------------------- CI gate
GATED_METRICS = ("brier_1x2", "logloss_1x2", "brier_over_2_5", "brier_btts")


def production_config(k: int = 8) -> RunConfig:
    """What `wm predict` does by default: the default model and effort, with retrieval."""
    return RunConfig(model=DEFAULT_MODEL, effort=DEFAULT_EFFORT, rag=True, k=k)


def write_gate(path: str | Path, score: RunScore, tolerance: float) -> dict[str, Any]:
    """Record the current scores of the production run, plus ``tolerance``, as the CI ceiling."""
    means = score.mean()
    gate = {
        "run": score.label,
        "fingerprint": fingerprint(score.config),
        "tolerance": tolerance,
        "baseline": {m: round(means[m], 5) for m in GATED_METRICS},
        "max": {m: round(means[m] + tolerance, 5) for m in GATED_METRICS},
    }
    Path(path).write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    return gate


def check(gate_path: str | Path, runs_dir: str | Path, fixtures: list[Fixture]) -> list[str]:
    """Problems that should fail CI; an empty list means the gate passes.

    It fails if the production configuration changed (model, effort, prompt,
    schema or retrieval code) without a fresh evaluation run, if that run does
    not cover every fixture, or if it scores worse than the committed ceiling.
    """
    gate = json.loads(Path(gate_path).read_text(encoding="utf-8"))
    config = production_config()
    expected_label, expected_fp = config.label, fingerprint(config)
    path = Path(runs_dir) / f"{expected_label}.jsonl"
    if not path.exists():
        return [
            f"No evaluation run for the production configuration ({expected_label}). "
            f"Run `wm eval run` and then `wm eval check --set-baseline` if the scores are acceptable."
        ]

    problems = []
    score = score_run(path, fixtures)
    if score.stale:
        problems.append(
            f"{score.stale} prediction(s) in {path} were made with a different prompt, schema, model or "
            f"retrieval code (current fingerprint {expected_fp}). Re-run `wm eval run` to refresh them."
        )
    if score.n < len(fixtures):
        problems.append(f"{path} covers {score.n} of {len(fixtures)} fixtures ({score.failed} failed).")
    if gate.get("run") != expected_label:
        problems.append(
            f"gate.json is for {gate.get('run')} but production is now {expected_label}. Review the new run's "
            "scores and update the gate with `wm eval check --set-baseline`."
        )
    if problems:
        return problems

    means = score.mean()
    for metric, ceiling in gate["max"].items():
        if means[metric] > ceiling:
            problems.append(
                f"{metric} regressed: {means[metric]:.4f} > ceiling {ceiling:.4f} "
                f"(baseline {gate['baseline'][metric]:.4f} + tolerance {gate['tolerance']})."
            )
    if means["brier_1x2"] >= UNIFORM_BRIER_1X2:
        problems.append(f"1X2 Brier {means['brier_1x2']:.4f} is no better than guessing 1/3 each.")
    return problems
