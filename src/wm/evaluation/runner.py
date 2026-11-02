"""Replay ``wm predict`` over every finished fixture and record what came back.

Each configuration (model, effort, RAG on/off, k) writes one JSONL file under
``eval/runs/``. A record keeps the model's raw output rather than derived
probabilities, so scoring always uses the current ``from_model_output``.
Records carry a *fingerprint* of everything that shapes the request (system
prompt, schema, prompt template, retrieval code, model, effort), which makes
runs resumable and lets CI detect predictions made by an older prompt.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import threading
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from wm.evaluation.fixtures import Fixture
from wm.llm import Claude, LLMError, parse_json
from wm.predict import SCHEMA, SYSTEM, Prediction, build_prompt, from_model_output
from wm.rag.store import KIND_TEAM, Hit

# Bump when the evaluation procedure itself changes in a way that invalidates old runs.
EVAL_VERSION = 1

# USD per million tokens: (input, output, cache read). Anthropic first-party rates, 2026-10.
PRICES: dict[str, tuple[float, float, float]] = {
    "claude-opus-5-5": (4.00, 20.00, 0.20),
    "claude-sonnet-5-5": (2.00, 10.00, 0.20),
    "claude-haiku-5-5": (0.10, 0.50, 0.01),
}


def cost_usd(model: str, input_tokens: int, output_tokens: int, cache_read_tokens: int = 0) -> float | None:
    """Price of one call at the served model's rates, or None for a model not in ``PRICES``.

    Thinking tokens are billed as output tokens and are already in ``output_tokens``.
    """
    if model not in PRICES:
        return None
    p_in, p_out, p_cache = PRICES[model]
    return (input_tokens * p_in + output_tokens * p_out + cache_read_tokens * p_cache) / 1e6


@dataclass(frozen=True)
class RunConfig:
    model: str
    effort: str
    rag: bool
    k: int = 8

    @property
    def label(self) -> str:
        return f"{self.model.removeprefix('claude-')}_{self.effort}_{'rag' if self.rag else 'norag'}"


def fingerprint(config: RunConfig) -> str:
    """Hash of everything in the code that shapes a request for ``config``.

    The prompt is rendered for two placeholder fixtures (one with a host
    nation, one neutral) so that changes to the template, the venue sentence or
    the document formatting all change the hash. With RAG on, the retrieval
    module's source is included too. The knowledge base's contents are not:
    they are local data that CI cannot see.
    """
    from wm.evaluation import leakage
    from wm.rag import retrieval

    doc = [Hit(text="[Team dossier: X]\n...", source="api_football:X", kind=KIND_TEAM, distance=0.0, metadata={})]
    hits = doc if config.rag else []
    payload = {
        "eval_version": EVAL_VERSION,
        "system": SYSTEM,
        "schema": SCHEMA,
        "prompts": [build_prompt("Canada", "Team B", hits), build_prompt("Team A", "Team B", hits)],
        "model": config.model,
        "effort": config.effort,
        "rag": config.rag,
        "k": config.k if config.rag else None,
        "retrieval": inspect.getsource(retrieval) + inspect.getsource(leakage) if config.rag else None,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


@dataclass
class Record:
    match_id: str
    label: str
    fingerprint: str
    model: str
    effort: str
    rag: bool
    k: int
    served_model: str | None = None
    output: dict[str, Any] | None = None
    error: str | None = None
    sources: list[str] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    latency_s: float | None = None
    cost_usd: float | None = None
    created_at: str = ""

    @property
    def ok(self) -> bool:
        return self.output is not None and self.error is None

    def prediction(self, fixture: Fixture) -> Prediction:
        assert self.output is not None
        return from_model_output(fixture.team_a, fixture.team_b, self.output, [])


def run_path(runs_dir: str | Path, label: str) -> Path:
    return Path(runs_dir) / f"{label}.jsonl"


def load_records(path: str | Path) -> dict[str, Record]:
    """Latest record per match in a run file (later lines win, so retries replace failures)."""
    path = Path(path)
    records: dict[str, Record] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = Record(**json.loads(line))
                records[rec.match_id] = rec
    return records


def compact(path: str | Path) -> None:
    """Rewrite a run file with one line per match, in a stable order, for a clean diff."""
    records = load_records(path)
    lines = [json.dumps(asdict(r), sort_keys=True) for _, r in sorted(records.items())]
    Path(path).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def pending(fixtures: Iterable[Fixture], records: dict[str, Record], fp: str) -> list[Fixture]:
    """Fixtures without a successful record for the current fingerprint."""
    return [f for f in fixtures if not ((r := records.get(f.match_id)) and r.ok and r.fingerprint == fp)]


Retriever = Callable[[Fixture], list[Hit]]


def predict_one(llm: Claude, config: RunConfig, fp: str, fixture: Fixture, hits: list[Hit]) -> Record:
    rec = Record(
        match_id=fixture.match_id,
        label=config.label,
        fingerprint=fp,
        model=config.model,
        effort=config.effort,
        rag=config.rag,
        k=config.k,
        sources=sorted({h.source for h in hits}),
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    try:
        done = llm.run(system=SYSTEM, prompt=build_prompt(fixture.team_a, fixture.team_b, hits), schema=SCHEMA)
        rec.served_model = done.model
        rec.input_tokens, rec.output_tokens = done.input_tokens, done.output_tokens
        rec.cache_read_tokens, rec.latency_s = done.cache_read_tokens, round(done.latency_s, 3)
        rec.cost_usd = cost_usd(done.model, done.input_tokens, done.output_tokens, done.cache_read_tokens)
        output = parse_json(done.text)
        from_model_output(fixture.team_a, fixture.team_b, output, [])  # reject unusable output now, not at scoring
        rec.output = output
    except (LLMError, KeyError, TypeError, ValueError) as exc:
        rec.error = str(exc) or type(exc).__name__
    return rec


def run(
    llm: Claude,
    config: RunConfig,
    fixtures: list[Fixture],
    path: str | Path,
    *,
    retrieve: Retriever | None = None,
    workers: int = 4,
    on_record: Callable[[Record], None] | None = None,
) -> list[Record]:
    """Predict every pending fixture and append each record to ``path`` as soon as it arrives.

    Retrieval runs first, sequentially (the embedding model is not thread-safe
    and it is fast); the API calls then run on ``workers`` threads.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fp = fingerprint(config)
    todo = pending(fixtures, load_records(path), fp)
    jobs = [(f, retrieve(f) if (config.rag and retrieve) else []) for f in todo]

    lock = threading.Lock()
    results: list[Record] = []
    with path.open("a", encoding="utf-8") as out, ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(predict_one, llm, config, fp, f, hits) for f, hits in jobs]
        for future in as_completed(futures):
            rec = future.result()
            with lock:
                out.write(json.dumps(asdict(rec), sort_keys=True) + "\n")
                out.flush()
                results.append(rec)
            if on_record:
                on_record(rec)
    compact(path)
    return results
