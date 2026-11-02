"""``wm`` command-line interface.

wm ingest      add a match report (URL, file or text) to the knowledge base
wm context     fetch API-Football team dossiers into the knowledge base
wm search      query the knowledge base
wm stats       show what the knowledge base contains
wm forget      delete one source from the knowledge base
wm reindex     upgrade chunks stored by older versions
wm predict     probabilities for a fixture, grounded in the knowledge base
wm arb scan    arbitrage scan of a scraped odds CSV
wm arb dutch   cross-book 1X2 arbitrage
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import textwrap

from wm import __version__
from wm.config import Settings

RULE = "─" * 72


# --------------------------------------------------------------------------- helpers
def _kb(settings: Settings, path: str | None):
    from wm.rag import KnowledgeBase

    return KnowledgeBase(path or settings.db_path, settings.embedding_model)


def _llm(settings: Settings):
    from wm.llm import Claude

    return Claude(settings)


def _heading(title: str) -> None:
    print(f"\n{title}\n{RULE}")


def _pct(p: float) -> str:
    return f"{100 * p:5.1f}%"


def _fair(p: float) -> str:
    return f"{1 / p:6.2f}" if p > 0 else "     ∞"


# --------------------------------------------------------------------------- knowledge base
def cmd_ingest(args: argparse.Namespace, settings: Settings) -> None:
    from wm.rag import ingest_text
    from wm.rag.sources import fetch_url, read_file

    if args.url:
        text, default_source = fetch_url(args.url), args.url
    elif args.file and args.file != "-":
        text, default_source = read_file(args.file), args.file
    else:
        text = read_file("-") if args.file else args.text.strip()
        if not text:
            raise ValueError("Nothing to ingest: the input text is empty.")
        # Content-addressed, so separate snippets never replace each other.
        default_source = "text:" + hashlib.sha1(text.encode()).hexdigest()[:10]
    source = args.source or default_source

    metadata = {"url": args.url} if args.url else None
    llm = None if args.raw else _llm(settings)
    result = ingest_text(_kb(settings, args.db), text, source=source, llm=llm, metadata=metadata)
    print(f"Stored {result.chunks} chunk(s) from {result.source} as {result.kind}.")
    if result.title:
        print(f"Title: {result.title}")


def cmd_context(args: argparse.Namespace, settings: Settings) -> None:
    from wm import teams
    from wm.football import APIFootballClient, build_team_context
    from wm.rag.store import KIND_TEAM, TEAM_SOURCE_PREFIX

    if args.all_groups:
        names = teams.all_teams()
    elif args.groups:
        names = teams.teams_in_groups(args.groups)
    else:
        names = args.teams
    if not names:
        raise ValueError("Give team names, --groups A B, or --all-groups.")

    client = APIFootballClient(settings.require_football_key())
    kb = _kb(settings, args.db)
    for name in names:
        try:
            label, text = build_team_context(client, name)
        except Exception as exc:  # one bad team should not stop a batch
            print(f"✗ {name}: {exc}")
            continue
        # Keyed by the API's canonical name, so re-running refreshes the dossier in place.
        chunks = kb.add(
            text,
            source=f"{TEAM_SOURCE_PREFIX}{label}",
            kind=KIND_TEAM,
            title=f"Team dossier: {label}",
            metadata={"team": label},
        )
        print(f"✓ {label}: {chunks} chunk(s)")
        if args.verbose:
            print(textwrap.indent(text, "    ") + "\n")


def cmd_search(args: argparse.Namespace, settings: Settings) -> None:
    kb = _kb(settings, args.db)
    hits = kb.search(args.query, k=args.k, where={"kind": args.kind} if args.kind else None)
    if not hits:
        print("No results. Is the knowledge base empty? Try `wm stats`.")
    for i, hit in enumerate(hits, 1):
        _heading(f"[{i}] {hit.source}  ·  {hit.kind}  ·  distance {hit.distance:.3f}")
        print(textwrap.shorten(hit.text, width=args.width, placeholder=" …") if args.width else hit.text)


def cmd_stats(args: argparse.Namespace, settings: Settings) -> None:
    kb = _kb(settings, args.db)
    stats = kb.stats()
    print(f"Knowledge base: {kb.path}  ({stats['chunks']} chunks)")
    for title, counts in (("By kind", stats["by_kind"]), ("By source", stats["by_source"])):
        if counts:
            _heading(title)
            for name, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
                print(f"  {n:5d}  {name}")


def cmd_forget(args: argparse.Namespace, settings: Settings) -> None:
    removed = _kb(settings, args.db).delete(args.source)
    if not removed:
        raise ValueError(f"No chunks with source {args.source!r}. `wm stats` lists every source.")
    print(f"Removed {removed} chunk(s) of {args.source}.")


def cmd_reindex(args: argparse.Namespace, settings: Settings) -> None:
    result = _kb(settings, args.db).reindex()
    print(f"Re-titled and re-embedded {result.retitled} chunk(s); removed {result.duplicates_removed} duplicate(s).")


# --------------------------------------------------------------------------- prediction
def cmd_predict(args: argparse.Namespace, settings: Settings) -> None:
    from wm.predict import predict_match
    from wm.rag.retrieval import match_context

    hits = []
    if not args.no_rag:
        kb = _kb(settings, args.db)
        hits = match_context(kb, args.team_a, args.team_b, k=args.k) if kb.exists else []

    p = predict_match(_llm(settings), args.team_a, args.team_b, hits)

    print(f"\n{p.team_a} vs {p.team_b}")
    print(RULE)
    print(f"{'Market':<22}{'Selection':<30}{'Prob':>7}{'Fair odds':>11}")
    for market, selections in p.markets():
        for i, (sel, prob) in enumerate(selections):
            print(f"{market if i == 0 else '':<22}{sel:<30}{_pct(prob):>7}{_fair(prob):>11}")
    print(RULE)
    if p.rationale:
        print(textwrap.fill(p.rationale, width=72))
    if args.show_sources:
        _heading(f"Retrieved context ({len(p.sources)} chunks)")
        for hit in p.sources:
            print(f"  {hit.distance:.3f}  {hit.kind:<22} {hit.source}")
    elif not args.no_rag:
        print(f"\n({len(hits)} knowledge-base chunks used; --show-sources to list them)")


# --------------------------------------------------------------------------- arbitrage
def cmd_arb_scan(args: argparse.Namespace, settings: Settings) -> None:
    from wm.arbitrage import load_odds_csv, scan

    book = load_odds_csv(args.csv)
    report = scan(book)
    print(f"{args.csv}: {len(book)} selections across {len(book.markets)} markets")

    _heading("1. Per-market overround   (book = Σ 1/odds; negative vig ⇒ arbitrage)")
    rows = report.overrounds if args.all else report.overrounds[: args.top]
    for r in rows:
        flag = "  ◀ ARB" if r.is_arb else ""
        print(f"  {r.market:<52} book={r.book:6.4f}  vig={100 * r.vig:+6.2f}%{flag}")
    if not args.all and len(report.overrounds) > args.top:
        print(f"  … {len(report.overrounds) - args.top} more (use --all)")
    arbs = sum(r.is_arb for r in report.overrounds)
    if report.overrounds:
        best = report.overrounds[0]
        print(f"\n  tightest book: {best.book:.4f} ({best.market});  single-market arbs: {arbs}")

    _heading("2. Over/Under ladder monotonicity   (P(Over) must fall as the line rises)")
    for lad in report.ladders:
        line = "  ".join(f"O{goals}={p:.3f}" for goals, p in lad.points)
        print(f"  {lad.market:<46} {line}   {'◀ VIOLATION' if lad.violated else 'ok'}")

    _heading("3. Global cross-market arbitrage   (LP over the full-time scoreline grid)")
    pf = report.portfolio
    print(f"  markets in LP: {len(pf.markets_used)}   legs: {pf.n_legs}")
    print(f"  markets skipped (halves, HT/FT, pushes, scorers, timing): {len(pf.markets_skipped)}")
    if not pf.solved:
        print(f"  LP failed: {pf.message}")
        return
    print(f"\n  Cheapest guaranteed return of 1.00 costs: {pf.cost:.5f}")
    if pf.is_arb:
        print(f"  ◀ ARBITRAGE: guaranteed ROI {100 * pf.roi:.2f}%. Stake per 1.00 returned:")
        for leg in pf.legs:
            print(f"      {leg.stake:7.4f} @ {leg.odds:6.2f}  {leg.market} :: {leg.selection}")
    else:
        print(f"  No arbitrage: a sure payout costs {100 * (pf.cost - 1):.2f}% over fair.")
        print("  Tightest-covered scorelines (return per 1.00 staked at the optimum):")
        for (h, a), ret in pf.tightest:
            print(f"      {h}:{a}  →  {ret:.4f}")


def cmd_arb_dutch(args: argparse.Namespace, settings: Settings) -> None:
    from wm.arbitrage.dutch import OUTCOMES, combinations

    combos = combinations([tuple(b) for b in args.book])
    arbs = [c for c in combos if c.is_arb]
    print(f"{len(args.book)} books, {len(combos)} combinations, {len(arbs)} arbitrage(s)")
    for c in arbs or combos[:1]:
        _heading(f"{'ARB' if c.is_arb else 'Best (no arb)'}: Σ 1/odds = {c.implied:.4f}   ROI {100 * c.roi:+.2f}%")
        for outcome, b, o, stake in zip(OUTCOMES, c.books, c.odds, c.stakes(args.bankroll), strict=True):
            print(f"  {outcome}  book {b + 1}  @ {o:5.2f}   stake {stake:8.2f}  →  returns {stake * o:8.2f}")


# --------------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="wm", description="World Cup 2026 predictions and arbitrage.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--db", help="knowledge-base directory (default: $WM_DB_PATH or ./rag_chroma_db)")
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    p = sub.add_parser("ingest", help="add a match report to the knowledge base")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--url", help="fetch an article (HTML is converted to text)")
    src.add_argument("--file", help="read a text file, or '-' for stdin")
    src.add_argument("--text", help="inline text")
    p.add_argument("--source", help="label stored with the text; re-using a label replaces that document")
    p.add_argument("--raw", action="store_true", help="store as-is, skipping Claude's structured analysis")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("context", help="fetch API-Football team dossiers into the knowledge base")
    p.add_argument("teams", nargs="*", help="team names, e.g. Canada 'Bosnia and Herzegovina'")
    p.add_argument("--groups", nargs="+", metavar="LETTER", help="every team in these World Cup groups")
    p.add_argument("--all-groups", action="store_true", help="all 48 teams")
    p.add_argument("-v", "--verbose", action="store_true", help="print each dossier")
    p.set_defaults(func=cmd_context)

    p = sub.add_parser("search", help="query the knowledge base")
    p.add_argument("query")
    p.add_argument("-k", type=int, default=5, help="number of results")
    p.add_argument("--kind", help="filter: team_context, match_report_analysis or raw_text")
    p.add_argument("--width", type=int, default=600, help="truncate each hit (0 = full text)")
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("stats", help="summarise the knowledge base")
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("forget", help="delete one source from the knowledge base")
    p.add_argument("source", help="exact source label, as shown by `wm stats`")
    p.set_defaults(func=cmd_forget)

    p = sub.add_parser("reindex", help="add titles to chunks stored by older versions and drop duplicates")
    p.set_defaults(func=cmd_reindex)

    p = sub.add_parser("predict", help="probabilities for a fixture")
    p.add_argument("team_a")
    p.add_argument("team_b")
    p.add_argument("--no-rag", action="store_true", help="do not use the knowledge base")
    p.add_argument("-k", type=int, default=8, help="chunks of context to retrieve")
    p.add_argument("--show-sources", action="store_true", help="list retrieved chunks")
    p.set_defaults(func=cmd_predict)

    arb = sub.add_parser("arb", help="arbitrage tools").add_subparsers(
        dest="arb_command", required=True, metavar="TOOL"
    )

    p = arb.add_parser("scan", help="scan a scraped odds CSV (market,selection,decimal_odds)")
    p.add_argument("csv")
    p.add_argument("--top", type=int, default=15, help="overround rows to show")
    p.add_argument("--all", action="store_true", help="show every overround row")
    p.set_defaults(func=cmd_arb_scan)

    p = arb.add_parser("dutch", help="1X2 arbitrage across bookmakers")
    p.add_argument(
        "--book",
        nargs=3,
        type=float,
        action="append",
        required=True,
        metavar=("ODDS_1", "ODDS_X", "ODDS_2"),
        help="one bookmaker's 1X2 odds (repeatable)",
    )
    p.add_argument("--bankroll", type=float, default=100.0, help="total amount to stake")
    p.set_defaults(func=cmd_arb_dutch)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = Settings.from_env()
    try:
        args.func(args, settings)
    # ConfigError, LLMError and APIFootballError are RuntimeErrors; SourceError is a ValueError.
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
