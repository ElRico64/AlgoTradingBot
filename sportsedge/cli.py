"""Command-line interface.

  sportsedge demo          --league NBA                 synthetic end-to-end check
  sportsedge fetch-history --league NBA --start 2023-10-01 --end 2024-06-30 --out nba.csv
  sportsedge backtest      --league NBA --history nba.csv [--json report.json]
  sportsedge train         --league NBA --history nba.csv --state nba.pkl [--tune]
  sportsedge picks         --league NBA --state nba.pkl [--odds-api-key KEY] [--llm]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
from datetime import date, datetime, timezone

from .backtest import walk_forward
from .config import LEAGUES
from .engine import EngineSettings, LeagueEngine
from .picks import PickPolicy

DISCLAIMER = ("Probabilities are model estimates, not guarantees. Past (back-tested) performance does not "
              "guarantee future results. Sports betting involves risk of loss; check local laws.")


def _policy(a) -> PickPolicy:
    return PickPolicy(min_probability=a.min_prob, min_lower_bound=a.min_lower,
                      require_positive_ev=not a.no_ev, min_edge=a.min_edge)


def _add_policy_args(p):
    p.add_argument("--min-prob", type=float, default=0.70, help="confidence gate (default 0.70)")
    p.add_argument("--min-lower", type=float, default=0.62, help="lower 80%% credible bound gate")
    p.add_argument("--min-edge", type=float, default=0.015, help="min edge vs no-vig market")
    p.add_argument("--no-ev", action="store_true", help="do not require positive expected value")


def cmd_demo(a):
    from .synthetic import generate

    for lg in a.league:
        seasons = a.seasons or (4 if lg == "NFL" else 2)
        games = generate(lg, seasons=seasons, seed=a.seed)
        res = walk_forward(lg, games, _policy(a))
        print(res.format())
        print()
    print("NOTE: synthetic leagues verify the statistics, not real-world edge.")


def cmd_fetch_history(a):
    from .data.csv_io import write_results
    from .data.espn import fetch_history

    games = fetch_history(a.league, date.fromisoformat(a.start), date.fromisoformat(a.end))
    write_results(a.out, games)
    print(f"wrote {len(games)} games to {a.out}")


def cmd_backtest(a):
    from .data.csv_io import read_results

    games = read_results(a.history, a.league)
    res = walk_forward(a.league, games, _policy(a))
    print(res.format())
    if a.json:
        with open(a.json, "w") as f:
            f.write(res.to_json())
        print(f"report written to {a.json}")


def cmd_train(a):
    from .data.csv_io import read_results

    games = read_results(a.history, a.league)
    eng = LeagueEngine(a.league, EngineSettings())
    if a.tune:
        eng.tune(games)
        print(f"tuned margin obs_sd={eng.kf_margin.obs_sd:.3f} q={eng.kf_margin.process_sd_per_day:.4f}")
    eng.fit(games)
    with open(a.state, "wb") as f:
        pickle.dump(eng, f)
    print(f"trained on {len(games)} games; calibrated={eng.calibrated}; saved {a.state}")
    for k, v in eng.stacker.weights().items():
        print(f"  stack weight {k}: {v:+.3f}")


def cmd_picks(a):
    from .news.feeds import gather_news
    from .news.impact import PlayerImpactRegistry, build_factors

    if a.state:
        with open(a.state, "rb") as f:
            eng: LeagueEngine = pickle.load(f)
    elif a.history:
        from .data.csv_io import read_results

        eng = LeagueEngine(a.league).fit(read_results(a.history, a.league))
    else:
        sys.exit("need --state or --history")
    if a.slate:
        from .data.csv_io import read_slate

        slate = read_slate(a.slate, a.league)
    else:
        from .data.espn import fetch_slate

        slate = fetch_slate(a.league, date.fromisoformat(a.date) if a.date else None)
    if not slate:
        print("no upcoming games found")
        return
    odds_map = {}
    key = a.odds_api_key or os.environ.get("ODDS_API_KEY")
    if key:
        from .data.oddsapi import fetch_odds

        odds_map = fetch_odds(a.league, key)
    factors = []
    if not a.no_news:
        analyzer = None
        if a.llm:
            from .news.llm import ClaudeNewsAnalyzer

            analyzer = ClaudeNewsAnalyzer(model=a.llm_model)
        events = gather_news(a.league, a.rss or [], use_espn=True, analyzer=analyzer)
        reg = PlayerImpactRegistry.from_csv(a.impacts) if a.impacts else None
        factors = build_factors(events, eng.cfg, datetime.now(timezone.utc), reg)
        print(f"news: {len(events)} events -> {len(factors)} availability factors")
    preds = []
    for g in slate:
        odds = odds_map.get((g.home, g.away)) or g.extras.pop("odds", None)
        preds.append(eng.predict(g, odds, factors))
    policy = _policy(a)
    picks = policy.select(preds)
    print(f"\n{eng.league} slate — {len(preds)} games (calibrated={eng.calibrated})")
    for p in preds:
        print(f"  {p.game.away:>4} @ {p.game.home:<4} P(home)={p.home_win_prob:.1%} "
              f"[{p.home_win_lower:.1%}, {p.home_win_upper:.1%}]  E[margin]={p.expected_margin:+.1f} "
              f"E[total]={p.expected_total:.1f}")
    print(f"\n=== PICKS (confidence ≥ {policy.min_probability:.0%}"
          f"{', +EV required' if policy.require_positive_ev else ''}): {len(picks)} ===")
    if not picks:
        print("  No game clears the gate today. Passing is a position.")
    for pk in picks:
        print("  " + pk.describe() + f"  stake {pk.stake_fraction:.2%} of bankroll")
        for r in pk.reasons:
            print("      - " + r)
    if a.json:
        out = [{"game": f"{pk.game.away}@{pk.game.home}", "start": pk.game.start_time.isoformat(),
                "market": pk.market.market, "side": pk.market.side, "line": pk.market.line,
                "price": pk.market.price, "probability": pk.market.probability, "lower": pk.market.lower,
                "upper": pk.market.upper, "edge": pk.market.edge, "ev": pk.market.expected_value,
                "stake": pk.stake_fraction, "reasons": pk.reasons} for pk in picks]
        with open(a.json, "w") as f:
            json.dump(out, f, indent=2)
    print("\n" + DISCLAIMER)


def main(argv=None):
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(prog="sportsedge", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    leagues = sorted(LEAGUES)

    p = sub.add_parser("demo", help="synthetic-league walk-forward check")
    p.add_argument("--league", nargs="+", default=leagues, choices=leagues)
    p.add_argument("--seasons", type=int)
    p.add_argument("--seed", type=int, default=1)
    _add_policy_args(p)
    p.set_defaults(fn=cmd_demo)

    p = sub.add_parser("fetch-history", help="download results + odds from ESPN")
    p.add_argument("--league", required=True, choices=leagues)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_fetch_history)

    p = sub.add_parser("backtest", help="walk-forward backtest on a results CSV")
    p.add_argument("--league", required=True, choices=leagues)
    p.add_argument("--history", required=True)
    p.add_argument("--json")
    _add_policy_args(p)
    p.set_defaults(fn=cmd_backtest)

    p = sub.add_parser("train", help="fit models + calibrators and save state")
    p.add_argument("--league", required=True, choices=leagues)
    p.add_argument("--history", required=True)
    p.add_argument("--state", required=True)
    p.add_argument("--tune", action="store_true", help="MLE-tune Kalman noise parameters first")
    p.set_defaults(fn=cmd_train)

    p = sub.add_parser("picks", help="generate today's picks")
    p.add_argument("--league", required=True, choices=leagues)
    p.add_argument("--state")
    p.add_argument("--history")
    p.add_argument("--slate", help="CSV of upcoming games instead of ESPN")
    p.add_argument("--date")
    p.add_argument("--odds-api-key")
    p.add_argument("--no-news", action="store_true")
    p.add_argument("--rss", nargs="*")
    p.add_argument("--impacts", help="player impact CSV")
    p.add_argument("--llm", action="store_true", help="use Claude to read the news")
    p.add_argument("--llm-model", default="claude-opus-5")
    p.add_argument("--json")
    _add_policy_args(p)
    p.set_defaults(fn=cmd_picks)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
