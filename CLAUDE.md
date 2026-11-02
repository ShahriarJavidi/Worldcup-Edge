# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`wm` is a Python package plus CLI for World Cup 2026 betting analysis. It has two independent halves: RAG-grounded match predictions from Claude, and arbitrage scanning of scraped bookmaker odds. See README.md for usage and the design rationale.

## Commands

```bash
source .venv/bin/activate          # Python 3.14 venv; package is installed editable (pip install -e ".[dev]")
pytest                             # whole suite, offline, no API keys needed
pytest tests/test_markets.py::test_european_handicap   # single test
ruff check src tests && ruff format src tests          # lint + format (CI enforces both)
wm --help                          # ingest | context | search | stats | forget | reindex | predict | eval | arb scan | arb dutch
wm arb scan examples/canada_bosnia_odds.csv            # quick end-to-end check, no network
node scraper/scrape_swiss.mjs "<event-url>"            # needs `npm install` in scraper/ and system Chrome
```

`wm predict`, `wm eval run` and `wm ingest` without `--raw` call the Anthropic API, and `wm context` and `wm eval fetch` call API-Football. Both cost money, so don't run them just to check a change.

## Architecture

- `src/wm/cli.py` imports heavy modules (torch/chromadb, scipy) lazily inside each `cmd_*` function, so `wm arb` starts fast. Keep it that way.
- `llm.Claude.run()` (wrapped by `complete()`/`complete_json()`) is the only place the Anthropic SDK is called; it also returns token usage, the served model and latency. It uses `client.beta.messages.create` with `output_config` (effort plus an optional JSON schema). For models in `_FALLBACK_MODELS` it also sends `fallbacks: "default"` via `extra_body`, because the pinned SDK version has no `fallbacks` kwarg. Don't add `temperature` or a thinking budget; current models reject them.
- RAG (`src/wm/rag/`): documents are keyed by `source`, and `KnowledgeBase.add()` deletes any existing chunks of that source first (replace, not append). It also prefixes every chunk with `[title]`; retrieval quality depends on this, so pass a `title` for any new kind of document. Team dossiers use source `api_football:<API canonical name>`. Analysed reports get their title from `analysis.match_title()`, which reads the `Title:` line the analysis prompt asks for, or the `Teams` section of older analyses.
- Chunk metadata carries `kind` (`team_context`, `match_report_analysis` or `raw_text`). Chunks written by pre-0.2 code have no `title`/`kind`; `wm reindex` (`KnowledgeBase.reindex()`) migrates them, and `store.kind_of()` infers `kind` until then. The user's real `rag_chroma_db/` is irreplaceable local data: test migrations on a copy (`wm --db /tmp/copy ...`).
- `predict.py`: Claude returns only 1X2, O2.5, O3.5 and BTTS. All other markets are derived in `Prediction.markets()`, so add new derived markets there rather than to the schema. The system prompt tells the model to ignore stored reports of the fixture's own result, because the KB often contains them.
- Arbitrage: `markets.winning_scores()` maps exact Swisslos market and selection strings to winning scorelines on a 0–7 grid. It returns `None` for anything that can push or doesn't settle on full time, which keeps the LP in `scanner.best_portfolio()` sound. New markets go there.
- Evaluation (`src/wm/evaluation/`): `wm eval run` writes one JSONL per config to `eval/runs/`, storing the model's raw output so scoring always uses the current `from_model_output()`. Every record carries `runner.fingerprint()`, a hash of SYSTEM, SCHEMA, the rendered prompt, the retrieval and leakage source, model and effort. Any change to those makes the production run stale, so `wm eval check` fails in CI until someone re-runs `wm eval run` and, if the scores change, `wm eval check --set-baseline`. That is intended. Retrieval during evaluation goes through `leakage.visible_before()`, so the KB can't leak results.
- `data/`, `rag_chroma_db/` and `.env` are git-ignored local state. `examples/` holds the only committed odds data; `eval/` holds committed results and prediction runs.
