# worldcup-edge

[![CI](https://github.com/ShahriarJavidi/worldcup-edge/actions/workflows/ci.yml/badge.svg)](https://github.com/ShahriarJavidi/worldcup-edge/actions/workflows/ci.yml)

A command-line toolkit (`wm`) for World Cup 2026 football betting analysis. It does two things:

- **Forecasts matches with Claude, grounded in a local knowledge base.** You feed it match reports and API-Football team data, and it returns calibrated probabilities for the common markets, with fair odds.
- **Finds arbitrage in bookmaker odds.** It scrapes a full market board, then uses a linear program to search every combination of bets for a guaranteed profit.

```text
$ wm predict Canada "Bosnia and Herzegovina"

Canada vs Bosnia and Herzegovina
────────────────────────────────────────────────────────────────────────
Market                Selection                        Prob  Fair odds
1X2                   Canada                          50.0%       2.00
                      Draw                            27.0%       3.70
                      Bosnia and Herzegovina          23.0%       4.35
Double chance         1X                              77.0%       1.30
...
Over/Under 2.5        Over                            45.0%       2.22
                      Under                           55.0%       1.82
Both teams to score   Yes                             48.0%       2.08
                      No                              52.0%       1.92
────────────────────────────────────────────────────────────────────────
Canada are the stronger side under Marsch, ranked well above Bosnia, and
play at home in Toronto as co-hosts, while Bosnia's recent form is
patchy and draw-heavy...
```

## Quick start

```bash
git clone https://github.com/ShahriarJavidi/worldcup-edge.git && cd worldcup-edge
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env        # then put your own keys in .env
wm --help
```

You need your own API keys, set in `.env` (which git ignores):

| Key | Used by | Get one |
|---|---|---|
| `ANTHROPIC_API_KEY` | `wm predict`, `wm ingest` (without `--raw`) | [console.anthropic.com](https://console.anthropic.com) |
| `FOOTBALL_API_KEY` | `wm context` | [api-football.com](https://www.api-football.com) (free tier available) |

The arbitrage tools, `wm search` and `wm stats` need no keys. The first command that touches the knowledge base downloads the embedding model (`all-MiniLM-L6-v2`, about 90 MB); after that, embeddings run locally.

## Usage

### 1. Build the knowledge base

```bash
# Team dossiers from API-Football: profile, form, ranking, fixtures, key players, injuries
wm context Canada "Bosnia and Herzegovina"
wm context --groups B G
wm context --all-groups

# Match reports: Claude rewrites them as structured notes before storing
wm ingest --url https://www.skysports.com/football/canada-vs-bos-and-herz/report/549768
wm ingest --file report.txt
cat report.txt | wm ingest --file -
wm ingest --text "Spain pressed high all game..." --raw     # store as-is, no LLM call

# Inspect and maintain it
wm stats                                              # chunk counts by kind and source
wm search "Canada injuries" -k 3
wm search "set pieces" --kind match_report_analysis
wm forget api_football:Qatar                          # remove one source
```

Every document is keyed by its **source**: the URL, the file name, `api_football:<Team>`, or `--source LABEL`. Adding a source that already exists replaces it, so re-running `wm context` refreshes dossiers instead of piling up duplicates. Inline `--text` and stdin get a content-hash source, so separate snippets never overwrite each other.

**Upgrading from an older version?** Knowledge bases built before chunk titles existed should be migrated once:

```bash
wm reindex      # adds titles, re-embeds, removes exact duplicates; safe to run again
```

### 2. Predict

```bash
wm predict Canada "Bosnia and Herzegovina"
wm predict Spain Uruguay --show-sources     # list the chunks that were retrieved
wm predict Spain Uruguay --no-rag           # Claude's general knowledge only
```

### 3. Scan for arbitrage

```bash
# Scrape every market for one event from Swisslos Sporttip (needs Node and Chrome)
(cd scraper && npm install)
node scraper/scrape_swiss.mjs "<event-url>"          # → data/swiss_scraped.csv

wm arb scan data/swiss_scraped.csv
wm arb scan examples/canada_bosnia_odds.csv --all    # bundled sample

# 1X2 across several bookmakers, with stakes for a 100 bankroll
wm arb dutch --book 1.90 3.65 4.20 --book 2.10 3.40 4.60 --bankroll 100
```

## How it works

```mermaid
flowchart LR
    subgraph Ingest
        U[URL / file / stdin] --> S[sources<br/>HTML → text]
        S --> A[analysis<br/>Claude → structured notes]
        F[API-Football] --> D[team dossier]
    end
    A --> KB[(Chroma<br/>MiniLM embeddings)]
    D --> KB
    S -- --raw --> KB
    KB --> R[retrieval<br/>multi-query, de-duplicated]
    R --> P[predict<br/>Claude, JSON schema]
    P --> M[derived markets<br/>+ fair odds]

    O[Swisslos scraper] --> CSV[odds CSV] --> AR[arbitrage<br/>overround · ladders · LP]
```

### The RAG pipeline

| Stage | Module | What it does |
|---|---|---|
| Sources | `wm.rag.sources` | Fetches URLs, strips HTML (`script`/`style` removed, one line per block element), reads files or stdin. |
| Analysis | `wm.rag.analysis` | Claude rewrites a raw report under fixed headings (*Result*, *Injuries And Absences*, *Prediction-Relevant Takeaways*, …), starting with a `Title: Home vs Away, competition, date` line. It is told to write "not stated" rather than guess, because invented facts would poison every later prediction. The original text is kept below the notes. |
| Storage | `wm.rag.store` | Splits text into 700-character chunks, embeds them locally and persists them in Chroma. Every chunk is **prefixed with its document's title** (`[Team dossier: Canada]`, `[Canada vs Bosnia and Herzegovina, …]`), so a chunk from the middle of a dossier still matches a query like "Canada injuries". Re-adding a source replaces it. |
| Retrieval | `wm.rag.retrieval` | Runs three queries (the fixture, then each team on its own) so one well-covered team can't crowd out the other. Results are merged, de-duplicated and ranked. |
| Prediction | `wm.predict` | Claude sees the chunks as source-labelled `<document>`s and returns JSON constrained by a schema. It is told to forecast as of kickoff and to ignore any stored report of the fixture's own result. |

**Claude estimates only the independent quantities:** 1X2, Over 2.5, Over 3.5 and BTTS. Double chance, draw-no-bet, the ±0.5 handicap and every *Under* are derived from those numbers, so the markets can never contradict each other. 1X2 is normalised to 100%, and P(Over 3.5) is capped at P(Over 2.5).

### The arbitrage scanner

`wm arb scan` runs three checks on one bookmaker's back-only odds:

1. **Overround:** for every self-contained market, `Σ 1/odds < 1` means a single-market arb.
2. **Ladder monotonicity:** the implied P(Over) must fall as the goal line rises.
3. **Global LP:** each selection that settles on the 90-minute score is mapped to the scorelines it wins, on a 0–7 goals-per-team grid. The scanner then solves:

   ```
   minimise    Σ stake_k
   subject to  Σ stake_k · odds_k · [k wins on scoreline o]  ≥  1     for every scoreline o
               stake_k ≥ 0
   ```

   If the cheapest portfolio that guarantees a return of 1 costs less than 1, there is a cross-market arbitrage, and the solution gives the stakes. This covers every dutching combination at once.

Markets that can push or refund (integer Asian lines, draw-no-bet) are left out of the LP, as are markets that don't settle on the full-time score (halves, HT/FT, scorers). Leaving bets out can only make the LP more conservative, so any arb it reports is real.

Market names are matched exactly against Swisslos' labels in `wm/arbitrage/markets.py`. To support another bookmaker, add branches to `winning_scores()`.

## Configuration

All settings come from environment variables or `.env`:

| Variable | Default | |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | See [Quick start](#quick-start). Not needed if you've run `ant auth login`. |
| `FOOTBALL_API_KEY` | — | See [Quick start](#quick-start). |
| `WM_MODEL` | `claude-opus-5-5` | Any Claude model ID. |
| `WM_EFFORT` | `medium` | `low` … `max`; trades depth for cost and latency. |
| `WM_DB_PATH` | `rag_chroma_db` | Knowledge-base directory (or pass `wm --db PATH`). |
| `WM_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Changing it requires rebuilding the knowledge base. |

## Project layout

```
src/wm/
├── cli.py              the `wm` command
├── config.py           settings from env / .env
├── llm.py              Anthropic SDK wrapper (structured output, refusal fallback, errors)
├── teams.py            groups, hosts, API-Football name aliases
├── predict.py          prediction prompt, schema, derived markets
├── rag/                sources → analysis → store → retrieval
├── football/           API-Football client and dossier builder
└── arbitrage/          odds loading, market semantics, scanner, cross-book dutching
scraper/                Puppeteer scraper for Swisslos Sporttip
examples/               sample odds sheet
tests/
```

## Development

```bash
pytest                                   # whole suite; no API keys, nothing billed
pytest tests/test_markets.py -k handicap # a subset
ruff check src tests && ruff format src tests
```

The RAG tests build a throwaway knowledge base with the real embedding model, and are skipped if it can't be downloaded. CI (`.github/workflows/ci.yml`) runs lint and tests on every push and pull request.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `error: FOOTBALL_API_KEY is not set` | Copy `.env.example` to `.env` and add the key. Run `wm` from the folder that contains `.env`. |
| `Anthropic authentication failed` | Check `ANTHROPIC_API_KEY` in `.env`, or run `ant auth login`. |
| `wm search` returns other teams' chunks | The knowledge base predates chunk titles; run `wm reindex`. |
| `wm context` skips a team ("Team not found") | API-Football uses a different name; add it to `TEAM_NAME_ALIASES` in `src/wm/teams.py`. |
| The scraper can't find Chrome | Set `CHROME_PATH=/path/to/chrome`. |

## Disclaimer

This is a personal research project, not betting advice. Model probabilities are estimates, and bookmakers limit accounts that consistently take arbitrage. Check a site's terms of service before scraping it.

## License

[MIT](LICENSE)
