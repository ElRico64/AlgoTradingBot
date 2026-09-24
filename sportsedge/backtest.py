"""Walk-forward backtesting with strict no-lookahead.

For each calendar day, in order:
  1. (every `recalibrate_days`) refit calibrators on out-of-sample forecasts
     made strictly *before* that day;
  2. predict every game of the day from the current model state;
  3. apply the pick gate and settle picks at the recorded prices;
  4. only then feed the day's results into the models.

Reported statistics are the ones an investor due-diligence team should ask
for: proper scoring rules vs. the market, calibration (reliability table,
ECE, Poisson-binomial z), pick hit rate with a Wilson interval, an exact
binomial test against 70%, flat-stake ROI with a bootstrap interval, and a
Kelly bankroll path with maximum drawdown.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from itertools import groupby
from typing import Optional, Sequence

import numpy as np

from .engine import EngineSettings, LeagueEngine
from .picks import PickPolicy
from .stats.calibration import (binomial_test_greater, brier, expected_calibration_error, log_loss,
                                poisson_binomial_z, reliability_table, wilson_interval)
from .stats.odds import american_to_decimal
from .types import GameResult, Pick


def settle(pick: Pick, r: GameResult) -> float:
    """Profit per 1 unit staked (0 on a push)."""
    m = pick.market
    if m.market == "moneyline":
        edge = r.margin if m.side == "home" else -r.margin
    elif m.market == "spread":
        edge = (r.margin + m.line) if m.side == "home" else -(r.margin + m.line)
    else:
        edge = (r.total - m.line) if m.side == "over" else (m.line - r.total)
    if edge == 0:
        return 0.0
    return american_to_decimal(m.price) - 1.0 if edge > 0 else -1.0


@dataclass
class BacktestResult:
    league: str
    summary: dict
    reliability: list[dict]
    picks: list[dict] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps({"league": self.league, "summary": self.summary, "reliability": self.reliability,
                           "picks": self.picks}, indent=2, default=str)

    def format(self) -> str:
        s = self.summary
        lines = [f"=== Walk-forward backtest: {self.league} ===",
                 f"games evaluated (post-warm-up, calibrated): {s['n_games']}"]
        lines.append(f"moneyline log loss  model {s['logloss_model']:.4f} | market {s.get('logloss_market', float('nan')):.4f}"
                     f" | model-only (no market) {s['logloss_model_only']:.4f}")
        lines.append(f"moneyline Brier     model {s['brier_model']:.4f} | market {s.get('brier_market', float('nan')):.4f}")
        lines.append(f"calibration ECE     {s['ece_model']:.4f}")
        p = s["picks"]
        lines.append(f"--- picks passing the gate: {p['n']} ({p['per_100_games']:.1f} per 100 games) ---")
        if p["n"]:
            lines.append(f"record {p['wins']}-{p['losses']}-{p['pushes']}  hit rate {p['hit_rate']:.1%} "
                         f"(95% Wilson {p['wilson_low']:.1%}–{p['wilson_high']:.1%}); mean claimed p {p['mean_p']:.1%}")
            lines.append(f"calibration z (hits vs claimed) {p['calibration_z']:+.2f}   "
                         f"P(hit rate<=70% | data) one-sided p={p['p_value_vs_70']:.3f}")
            lines.append(f"flat ROI {p['roi']:+.2%} (95% bootstrap {p['roi_ci'][0]:+.2%}..{p['roi_ci'][1]:+.2%}), "
                         f"avg price {p['avg_price']:+.0f}")
            lines.append(f"fractional-Kelly bankroll x{p['kelly_final']:.3f}, max drawdown {p['kelly_max_dd']:.1%}")
            for mk, v in p["by_market"].items():
                lines.append(f"  {mk:9s} n={v['n']:4d} hit={v['hit_rate']:.1%} roi={v['roi']:+.2%}")
        lines.append("reliability (predicted vs observed home-win rate):")
        for r in self.reliability:
            lines.append(f"  {r['bin']}: n={r['n']:5d} pred {r['mean_pred']:.3f} obs {r['observed']:.3f} "
                         f"[{r['ci_low']:.3f},{r['ci_high']:.3f}]")
        return "\n".join(lines)


def walk_forward(league: str, results: Sequence[GameResult], policy: Optional[PickPolicy] = None,
                 settings: Optional[EngineSettings] = None, recalibrate_days: int = 7,
                 n_draws: int = 300, seed: int = 11) -> BacktestResult:
    policy = policy or PickPolicy()
    settings = settings or EngineSettings(n_draws=n_draws)
    engine = LeagueEngine(league, settings)
    results = sorted(results, key=lambda r: r.start_time)
    rows, picks = [], []
    last_cal = None
    for day, it in groupby(results, key=lambda r: r.start_time.date()):
        games = list(it)
        t0 = min(g.start_time for g in games)
        if last_cal is None or (day - last_cal).days >= recalibrate_days:
            if engine.fit_calibrators(before=t0):
                last_cal = day
        if engine.calibrated:
            for g in games:
                pred = engine.predict(g, g.odds, n_draws=n_draws)
                if g.margin != 0:
                    ml = next(m for m in pred.markets if m.market == "moneyline" and m.side == "home")
                    model_only = engine.stacker.transform(pred.components, None)
                    rows.append({"p": pred.home_win_prob, "p_model_only": model_only,
                                 "market": ml.fair_market_prob, "y": float(g.margin > 0)})
                for pk in policy.evaluate(pred):
                    if pk.market.price is None:
                        continue
                    picks.append({"date": g.start_time.date().isoformat(), "game": f"{g.away}@{g.home}",
                                  "game_id": g.game_id,
                                  "market": pk.market.market, "side": pk.market.side, "line": pk.market.line,
                                  "price": pk.market.price, "p": pk.market.probability, "lower": pk.market.lower,
                                  "fair_market": pk.market.fair_market_prob, "stake": pk.stake_fraction,
                                  "profit": settle(pk, g)})
        engine.observe(games)
    return BacktestResult(league, _summarize(rows, picks, seed), reliability_table(
        [r["p"] for r in rows], [r["y"] for r in rows]) if rows else [], picks)


def _summarize(rows: list[dict], picks: list[dict], seed: int) -> dict:
    s: dict = {"n_games": len(rows)}
    if rows:
        p = np.array([r["p"] for r in rows])
        y = np.array([r["y"] for r in rows])
        s.update(logloss_model=log_loss(p, y), brier_model=brier(p, y), ece_model=expected_calibration_error(p, y),
                 logloss_model_only=log_loss([r["p_model_only"] for r in rows], y))
        mk = [(r["market"], r["y"], r["p"]) for r in rows if r["market"] is not None]
        if mk:
            m = np.array([a for a, _, _ in mk])
            ym = np.array([b for _, b, _ in mk])
            s.update(logloss_market=log_loss(m, ym), brier_market=brier(m, ym),
                     logloss_model_on_market_games=log_loss([c for _, _, c in mk], ym))
    else:
        s.update(logloss_model=float("nan"), brier_model=float("nan"), ece_model=float("nan"),
                 logloss_model_only=float("nan"))
    s["picks"] = _pick_stats(picks, len(rows), seed)
    return s


def _pick_stats(picks: list[dict], n_games: int, seed: int) -> dict:
    n = len(picks)
    out: dict = {"n": n, "per_100_games": 100.0 * n / max(1, n_games)}
    if not n:
        return out
    prof = np.array([p["profit"] for p in picks])
    decided = prof != 0
    wins = int((prof > 0).sum())
    losses = int((prof < 0).sum())
    lo, hi = wilson_interval(wins, wins + losses)
    pd = np.array([p["p"] for p in picks])[decided]
    yd = (prof[decided] > 0).astype(float)
    rng = np.random.default_rng(seed)
    boots = [rng.choice(prof, n, replace=True).mean() for _ in range(2000)]
    out.update(wins=wins, losses=losses, pushes=int(n - wins - losses),
               hit_rate=wins / max(1, wins + losses), wilson_low=lo, wilson_high=hi,
               mean_p=float(pd.mean()) if len(pd) else float("nan"),
               calibration_z=poisson_binomial_z(pd, yd) if len(pd) else 0.0,
               p_value_vs_70=binomial_test_greater(wins, wins + losses, 0.70),
               roi=float(prof.mean()), roi_ci=(float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))),
               avg_price=float(np.mean([p["price"] for p in picks])))
    # fractional-Kelly bankroll, stakes sized on start-of-day bankroll, daily exposure capped at 25%
    bank, peak, max_dd = 1.0, 1.0, 0.0
    for _, day in groupby(picks, key=lambda p: p["date"]):
        day = list(day)
        stakes = np.array([p["stake"] for p in day])
        if stakes.sum() > 0.25:
            stakes *= 0.25 / stakes.sum()
        bank += bank * float(np.sum(stakes * np.array([p["profit"] for p in day])))
        peak = max(peak, bank)
        max_dd = max(max_dd, 1 - bank / peak)
    out.update(kelly_final=bank, kelly_max_dd=max_dd)
    by = {}
    for mk in sorted({p["market"] for p in picks}):
        pr = np.array([p["profit"] for p in picks if p["market"] == mk])
        w, l_ = int((pr > 0).sum()), int((pr < 0).sum())
        by[mk] = {"n": len(pr), "hit_rate": w / max(1, w + l_), "roi": float(pr.mean())}
    out["by_market"] = by
    return out
