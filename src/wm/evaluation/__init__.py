"""Backtest ``wm predict`` on the finished World Cup 2026.

fixtures  real 90-minute results (fetched once from API-Football, then committed)
leakage   retrieval only sees chunks ingested before kickoff
runner    one JSONL run file per (model, effort, RAG, k), resumable and fingerprinted
metrics   Brier score, log loss, paired bootstrap
report    leaderboard, RAG and model comparisons, and the CI gate
"""
