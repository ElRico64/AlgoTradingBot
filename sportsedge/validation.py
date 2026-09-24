"""Reproducible check that the pattern engine earns its place.

For each league, simulate seasons twice: plain, and with planted patterns
(hot/cold regimes and a travel-fatigue interaction) that the structural
models cannot represent. Backtest each walk-forward with the pattern engine
on and off, and report out-of-sample log loss. Lower is better; a difference
of 0.002 or more on ~2,000 games is meaningful here.

Expected outcome for an honest pattern engine:
  * plain leagues: about the same log loss with or without it (no harm)
  * patterned leagues: clearly lower log loss with it (it finds the patterns)
"""
from __future__ import annotations

from .backtest import walk_forward
from .engine import EngineSettings
from .picks import PickPolicy
from .synthetic import generate


def compare(league: str, seasons: int = 2, seed: int = 5, n_draws: int = 150) -> list[dict]:
    rows = []
    for planted in (False, True):
        games = generate(league, seasons=seasons, seed=seed, patterns=planted)
        for use in (False, True):
            res = walk_forward(league, games, PickPolicy(require_positive_ev=False),
                               settings=EngineSettings(n_draws=n_draws, use_patterns=use), n_draws=n_draws)
            s = res.summary
            rows.append({"league": league, "planted_patterns": planted, "pattern_engine": use,
                         "games": s["n_games"], "logloss_model_only": s["logloss_model_only"],
                         "logloss_with_market": s["logloss_model"], "logloss_market": s.get("logloss_market"),
                         "ece": s["ece_model"], "picks_70": s["picks"]["n"],
                         "hit_rate_70": s["picks"].get("hit_rate")})
    return rows


def format_rows(rows: list[dict]) -> str:
    out = [f"{'league':6s} {'patterns':9s} {'engine':7s} {'games':>6s} {'LL model':>9s} {'LL +mkt':>8s} "
           f"{'LL mkt':>7s} {'70% picks':>9s} {'hit':>6s}"]
    for r in rows:
        hit = "—" if r["hit_rate_70"] is None else f"{r['hit_rate_70']:.1%}"
        out.append(f"{r['league']:6s} {'planted' if r['planted_patterns'] else 'none':9s} "
                   f"{'on' if r['pattern_engine'] else 'off':7s} {r['games']:6d} {r['logloss_model_only']:9.4f} "
                   f"{r['logloss_with_market']:8.4f} {r['logloss_market'] or float('nan'):7.4f} "
                   f"{r['picks_70']:9d} {hit:>6s}")
    return "\n".join(out)
