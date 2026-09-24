"""The prediction engine.

Pipeline for one game
---------------------
1. Component models, each giving a home-win probability:
     * Kalman state-space margin ratings (all leagues)
     * margin-of-victory Elo (all leagues)
     * attack/defence/park Poisson / negative-binomial model (NHL, MLB)
2. Situational adjustments (rest, travel, altitude, starters, weather).
3. News: every uncertain player becomes a Bernoulli availability mixture.
4. Monte Carlo over parameter posteriors x availability scenarios gives a
   *distribution* of each component's probability.
5. A stacking calibrator — penalised logistic regression fit only on
   out-of-sample (walk-forward) forecasts — pools the components and the
   de-vigged market price in log-odds space and calibrates the result.
6. Output: calibrated probabilities with an 80% credible band for moneyline,
   spread and total, plus edge / EV against the offered price.

`PickPolicy` (picks.py) then applies the >= 70% confidence gate.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from datetime import datetime
from itertools import groupby
from typing import Iterable, Optional, Sequence

import numpy as np

from .config import LeagueConfig, get_config
from .models.count_models import CountRatings
from .models.distributions import (MarginDist, TotalDist, count_final_distributions, gaussian_margin_dist,
                                   gaussian_total_dist, joint_score_matrix)
from .models.elo import EloRatings
from .models.kalman import KalmanRatings
from .models.patterns import PatternModel
from .models.situational import Adjustment, ScheduleTracker, compute_adjustment
from .news.impact import AvailabilityFactor
from .stats.calibration import StackingCalibrator
from .stats.odds import devig
from .types import Game, GameResult, MarketOdds, MarketProbability, Prediction

def _logit(p: float) -> float:
    p = min(max(float(p), 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


WARMUP_GAMES = {"NFL": 96, "NBA": 250, "NHL": 250, "MLB": 350}


@dataclass
class EngineSettings:
    n_draws: int = 2000
    seed: int = 7
    band: tuple[float, float] = (0.10, 0.90)
    devig_method: str = "shin"
    count_refit_days: int = 1
    dispersion_refit_days: int = 30
    stack_l2: float = 2.0
    min_calibration_samples: int = 300
    warmup_games: Optional[int] = None
    use_patterns: bool = True  # regime HMM + residual gradient boosting component


@dataclass
class _Point:
    comp: dict[str, float]
    margin: MarginDist
    total: TotalDist
    mu_margin: float
    mu_total: float
    k_mu: float = 0.0  # Kalman margin mean (with situational shift)
    k_var: float = 0.0  # Kalman parameter variance
    k_sd: float = 1.0  # Kalman predictive sd
    pace: float = 0.0  # standardised expected total


class LeagueEngine:
    def __init__(self, league: str, settings: Optional[EngineSettings] = None,
                 config: Optional[LeagueConfig] = None):
        self.cfg = config or get_config(league)
        self.league = self.cfg.league
        self.s = settings or EngineSettings()
        self.rng = np.random.default_rng(self.s.seed)
        c = self.cfg
        self.kf_margin = KalmanRatings(
            obs_sd=c.obs_sd, prior_sd=c.prior_rating_sd, process_sd_per_day=c.process_sd_per_day,
            intercept_prior=c.home_adv_prior, intercept_prior_sd=max(0.05, 0.5 * c.home_adv_prior),
            season_carryover=c.season_carryover, offseason_days=c.offseason_days,
            clip_sds=c.margin_cap_sds, mode="margin")
        ratio = c.total_obs_sd / c.obs_sd
        self.kf_total = KalmanRatings(
            obs_sd=c.total_obs_sd, prior_sd=0.25 * c.total_obs_sd,
            process_sd_per_day=c.process_sd_per_day * ratio, intercept_prior=c.league_avg_total,
            intercept_prior_sd=0.1 * c.league_avg_total, season_carryover=c.season_carryover,
            offseason_days=c.offseason_days, clip_sds=c.margin_cap_sds, mode="total")
        self.elo = EloRatings(k=c.elo_k, home_adv=c.elo_home_adv, offseason_days=c.offseason_days)
        self.count: Optional[CountRatings] = None
        if c.score_model == "count":
            self.count = CountRatings(
                decay_per_day=c.count_decay_per_day, ridge_sd=c.count_ridge_sd, park_sd=c.count_park_sd,
                window_days=c.count_window_days, dispersion=c.count_dispersion,
                estimate_rho=c.count_dispersion is None)
        self.components = ["count", "kalman", "elo"] if self.count else ["kalman", "elo"]
        self.patterns: Optional[PatternModel] = None
        if self.s.use_patterns:
            self.patterns = PatternModel(offseason_days=c.offseason_days)
            self.components.append("pattern")
        self.sched = ScheduleTracker()
        self.history: list[GameResult] = []
        self.oos: list[dict] = []
        self.stacker = StackingCalibrator(self.components, l2=self.s.stack_l2)
        # spread / total: model probability pooled with the market's, like moneyline
        self.spread_cal: Optional[StackingCalibrator] = None
        self.total_cal: Optional[StackingCalibrator] = None
        self._last_dispersion_fit: Optional[datetime] = None
        self.n_observed = 0

    # ------------------------------------------------------------------ utils
    @property
    def half_total(self) -> float:
        return self.cfg.league_avg_total / 2.0

    @property
    def calibrated(self) -> bool:
        return self.stacker.fitted

    def _ot_home(self, lh=None, la=None):
        base = 0.5 + self.cfg.extra_time_home_edge
        if lh is None:
            return base
        return np.clip(base + 0.5 * (lh - la) / (lh + la), 0.05, 0.95)

    def _margin_dist(self, mu, sd, ot=None) -> MarginDist:
        c = self.cfg
        if c.league == "NFL":
            return gaussian_margin_dist(mu, sd, c.key_number_weights, c.tie_mass_factor,
                                        ot if ot is not None else self._ot_home(), ties_allowed=True, span=80)
        span = 90 if c.league == "NBA" else 25
        return gaussian_margin_dist(mu, sd, None, 1.0, ot if ot is not None else self._ot_home(),
                                    ties_allowed=False, span=span)

    def _ensure_count_fit(self, as_of: datetime) -> None:
        if not self.count:
            return
        fa = self.count.fitted_at
        if fa is None or (as_of - fa).total_seconds() / 86400.0 >= self.s.count_refit_days:
            upd = self._last_dispersion_fit is None or (as_of - self._last_dispersion_fit).days >= self.s.dispersion_refit_days
            self.count.fit(self.history, as_of, update_dispersion=upd)
            if upd:
                self._last_dispersion_fit = as_of

    def _models_at(self, when: datetime) -> tuple[KalmanRatings, KalmanRatings, EloRatings]:
        """Copies of the rating models rolled forward to `when` (drift variance,
        season regression) without mutating the learned state."""
        kf_m, kf_t, elo = copy.deepcopy((self.kf_margin, self.kf_total, self.elo))
        kf_m.advance(when)
        kf_t.advance(when)
        elo.advance(when)
        return kf_m, kf_t, elo

    def _margin_shift(self, adj: Adjustment) -> float:
        """All situational effects expressed in margin units (home perspective)."""
        return adj.margin + self.half_total * (adj.log_rate_home - adj.log_rate_away)

    def _log_shifts(self, adj: Adjustment) -> tuple[float, float]:
        k = adj.margin / (2.0 * self.half_total)
        return adj.log_rate_home + k, adj.log_rate_away - k

    # ------------------------------------------------------------- point path
    def _point(self, g: Game, adj: Adjustment, models=None) -> _Point:
        c = self.cfg
        kf_m, kf_t, elo = models or self._models_at(g.start_time)
        shift = self._margin_shift(adj)
        mu, var, obs = kf_m.predict(g.home, g.away, g.neutral)
        mu += shift
        sd = math.sqrt(var + obs)
        md_k = self._margin_dist(np.array([mu]), sd)
        comp = {"kalman": float(md_k.win()[0] / (1 - md_k.tie()[0]))}
        elo_d = elo.diff(g.home, g.away, g.neutral) + shift * 100.0 / c.elo_points_per_100
        comp["elo"] = float(1.0 / (1.0 + 10 ** (-elo_d / 400.0)))
        if self.count:
            m, S = self.count.predict_log_rates(g.home, g.away, g.neutral)
            dh, da = self._log_shifts(adj)
            lh = np.array([math.exp(m[0] + 0.5 * S[0, 0] + dh)])
            la = np.array([math.exp(m[1] + 0.5 * S[1, 1] + da)])
            J = joint_score_matrix(lh, la, c.max_score, self.count.dispersion, self.count.rho)
            md, td = count_final_distributions(J, self._ot_home(lh, la))
            comp["count"] = float(md.win()[0])
            pace = (float(td.mean()[0]) - c.league_avg_total) / c.total_obs_sd
            return _Point(comp, md, td, float(md.mean()[0]), float(td.mean()[0]), mu, var, sd, pace)
        mt, vt, ot = kf_t.predict(g.home, g.away, g.neutral)
        mt += adj.total
        td = gaussian_total_dist(np.array([mt]), math.sqrt(vt + ot))
        return _Point(comp, md_k, td, mu, mt, mu, var, sd, (mt - c.league_avg_total) / c.total_obs_sd)

    def _pattern_x(self, g: Game, pt: _Point) -> np.ndarray:
        s = self.sched
        rest = (s.rest_days(g.home, g.start_time), s.rest_days(g.away, g.start_time))
        da, tza = s.travel(self.league, g.away, g.home)
        dh, _ = s.travel(self.league, g.home, g.home)
        return self.patterns.features(g.home, g.away, g.start_time, g.neutral, _logit(pt.comp["kalman"]),
                                      math.sqrt(max(pt.k_var, 0.0)) / self.cfg.obs_sd, rest,
                                      (da / 1000.0, tza, dh / 1000.0), pt.pace)

    def _market_fair(self, odds: Optional[MarketOdds]) -> dict[str, Optional[float]]:
        out: dict[str, Optional[float]] = {"ml": None, "spread": None, "total": None}
        if odds is None:
            return out
        try:
            if odds.fair_home_prob is not None:
                out["ml"] = odds.fair_home_prob
            elif odds.has_moneyline():
                out["ml"] = float(devig([odds.home_ml, odds.away_ml], self.s.devig_method)[0])
            if odds.spread is not None:
                out["spread"] = odds.fair_home_cover_prob if odds.fair_home_cover_prob is not None else \
                    float(devig([odds.home_spread_price or -110, odds.away_spread_price or -110], self.s.devig_method)[0])
            if odds.total is not None:
                out["total"] = odds.fair_over_prob if odds.fair_over_prob is not None else \
                    float(devig([odds.over_price or -110, odds.under_price or -110], self.s.devig_method)[0])
        except (ValueError, ZeroDivisionError):
            pass
        return out

    # ---------------------------------------------------------------- learning
    def observe(self, results: Iterable[GameResult]) -> None:
        """Assimilate results in chronological order. For each calendar day,
        out-of-sample forecasts for *all* that day's games are recorded before
        any of them updates the models (no same-day leakage)."""
        results = sorted(results, key=lambda r: r.start_time)
        warm = self.s.warmup_games if self.s.warmup_games is not None else WARMUP_GAMES.get(self.league, 200)
        for _, day_iter in groupby(results, key=lambda r: r.start_time.date()):
            day = list(day_iter)
            t0 = min(r.start_time for r in day)
            self._ensure_count_fit(t0)
            if self.patterns is not None:
                self.patterns.maybe_fit(t0)  # trained only on games before today
            pending = []
            for r in day:
                adj = compute_adjustment(self.cfg, r, self.sched)
                pt = self._point(r, adj)
                x = None
                if self.patterns is not None:
                    x = self._pattern_x(r, pt)
                    pt.comp["pattern"] = float(_sigmoid(_logit(pt.comp["kalman"]) + self.patterns.correction(x)))
                if self.n_observed >= warm:
                    self.oos.append(self._oos_record(r, pt))
                self.n_observed += 1
                pending.append((r, pt, x))
            for r in day:
                self.kf_margin.update(r.home, r.away, r.margin, r.neutral, r.start_time)
                self.kf_total.update(r.home, r.away, r.total, r.neutral, r.start_time)
                self.elo.update(r.home, r.away, r.margin, r.neutral, r.start_time)
                self.sched.record(r)
                self.history.append(r)
            if self.patterns is not None:
                for r, pt, x in pending:
                    z = (r.margin - pt.k_mu) / pt.k_sd
                    y = None if r.margin == 0 else float(r.margin > 0)
                    self.patterns.record(r.home, r.away, r.start_time, x, _logit(pt.comp["kalman"]), y, z, r.margin)

    def _oos_record(self, r: GameResult, pt: _Point) -> dict:
        fair = self._market_fair(r.odds)
        rec = {"t": r.start_time, "comp": pt.comp, "market": fair["ml"],
               "y": None if r.margin == 0 else float(r.margin > 0)}
        o = r.odds
        if o is not None and o.spread is not None:
            p, push = pt.margin.cover(o.spread)
            adj_m = r.margin + o.spread
            rec["spread"] = (float(p[0] / max(1e-9, 1 - push[0])), None if adj_m == 0 else float(adj_m > 0),
                             fair["spread"])
        if o is not None and o.total is not None:
            p, push = pt.total.over(o.total)
            rec["total"] = (float(p[0] / max(1e-9, 1 - push[0])),
                            None if r.total == o.total else float(r.total > o.total), fair["total"])
        return rec

    def fit(self, results: Sequence[GameResult]) -> "LeagueEngine":
        self.observe(results)
        self.fit_calibrators()
        return self

    def fit_calibrators(self, before: Optional[datetime] = None) -> bool:
        recs = [r for r in self.oos if before is None or r["t"] < before]
        ml = [r for r in recs if r["y"] is not None]
        if len(ml) < self.s.min_calibration_samples:
            return False
        self.stacker.fit([r["comp"] for r in ml], [r["market"] for r in ml], [r["y"] for r in ml])
        for key, attr in (("spread", "spread_cal"), ("total", "total_cal")):
            rows = [r[key] for r in recs if key in r and r[key][1] is not None]
            if len(rows) >= self.s.min_calibration_samples:
                cal = StackingCalibrator(["model"], l2=self.s.stack_l2, features="beta")
                setattr(self, attr, cal.fit([{"model": p} for p, _, _ in rows], [m for _, _, m in rows],
                                            [y for _, y, _ in rows]))
        return True

    def tune(self, results: Sequence[GameResult]) -> None:
        """Maximum-likelihood re-estimation of the Kalman noise parameters."""
        results = sorted(results, key=lambda r: r.start_time)
        self.kf_margin = KalmanRatings.tune(results, self.kf_margin, value=lambda g: g.margin)
        self.kf_total = KalmanRatings.tune(results, self.kf_total, value=lambda g: g.total)

    # --------------------------------------------------------------- inference
    def _simulate(self, g: Game, adj: Adjustment, factors: Sequence[AvailabilityFactor], n: int, models):
        c = self.cfg
        kf_m, kf_t, elo = models
        rng = self.rng
        news_m = np.zeros(n)  # margin units, home perspective
        news_lh = np.zeros(n)
        news_la = np.zeros(n)
        for f in factors:
            if f.team not in (g.home, g.away):
                continue
            misses = rng.random(n) >= f.play_prob
            loss = misses * np.maximum(0.0, rng.normal(f.impact, f.impact_sd, n))
            sign = -1.0 if f.team == g.home else 1.0
            news_m += sign * loss
            dl = loss / self.half_total
            if f.defensive:  # goalie / pitcher: opponent scores more
                if f.team == g.home:
                    news_la += dl
                else:
                    news_lh += dl
            else:
                if f.team == g.home:
                    news_lh -= dl
                else:
                    news_la -= dl
        shift = self._margin_shift(adj)
        mu_draws = kf_m.sample_means(g.home, g.away, g.neutral, n, rng) + shift + news_m
        md_k = self._margin_dist(mu_draws, c.obs_sd)
        comps = {"kalman": md_k.win() / np.clip(1 - md_k.tie(), 1e-9, None)}
        elo_d = elo.diff(g.home, g.away, g.neutral) + (shift + news_m) * 100.0 / c.elo_points_per_100
        comps["elo"] = 1.0 / (1.0 + 10 ** (-elo_d / 400.0))
        if self.count:
            lh, la = self.count.sample_rates(g.home, g.away, g.neutral, n, rng)
            dh, da = self._log_shifts(adj)
            lh = lh * np.exp(dh + news_lh)
            la = la * np.exp(da + news_la)
            J = joint_score_matrix(lh, la, c.max_score, self.count.dispersion, self.count.rho)
            md, td = count_final_distributions(J, self._ot_home(lh, la))
            comps["count"] = md.win()
        else:
            md = md_k
            mt = kf_t.sample_means(g.home, g.away, g.neutral, n, rng) + adj.total
            td = gaussian_total_dist(mt, c.total_obs_sd)
        return comps, md, td, news_m

    def predict(self, game: Game, odds: Optional[MarketOdds] = None,
                factors: Sequence[AvailabilityFactor] = (), n_draws: Optional[int] = None) -> Prediction:
        n = n_draws or self.s.n_draws
        self._ensure_count_fit(game.start_time)
        adj = compute_adjustment(self.cfg, game, self.sched)
        models = self._models_at(game.start_time)
        comps, md, td, news_m = self._simulate(game, adj, factors, n, models)
        pattern_notes = []
        if self.patterns is not None:
            # the learned correction rides on every draw, so news still flows through it
            x = self._pattern_x(game, self._point(game, adj, models))
            delta = self.patterns.correction(x)
            comps["pattern"] = _sigmoid(np.log(np.clip(comps["kalman"], 1e-6, 1 - 1e-6) /
                                               np.clip(1 - comps["kalman"], 1e-6, 1)) + delta)
            pattern_notes = self.patterns.describe(game.home, game.away, x)
        fair = self._market_fair(odds)
        lo_q, hi_q = self.s.band
        comp_mean = {k: float(np.mean(v)) for k, v in comps.items()}
        # stacked, calibrated home-win probability (+ its distribution across draws)
        P = np.column_stack([comps[k] for k in self.components])
        p_draws = self.stacker.predict_many(P, fair["ml"])
        p_home = self.stacker.transform(comp_mean, fair["ml"]) if self.calibrated else float(p_draws.mean())
        lo, hi = float(np.quantile(p_draws, lo_q)), float(np.quantile(p_draws, hi_q))
        lo, hi = min(lo, p_home), max(hi, p_home)
        tie_ml = float(np.mean(md.tie())) if self.league == "NFL" else 0.0
        ml_aware = self.stacker.with_market is not None
        pred = Prediction(game=game, home_win_prob=p_home, home_win_lower=lo, home_win_upper=hi,
                          expected_margin=float(np.mean(md.mean())), expected_total=float(np.mean(td.mean())),
                          components=comp_mean, adjustments=dict(adj.parts))
        pred.notes.extend(pattern_notes)
        if self.patterns is not None and "pattern" in comp_mean:
            pred.adjustments["pattern_logit"] = float(_logit(comp_mean["pattern"]) - _logit(comp_mean["kalman"]))
        if factors:
            pred.adjustments["news_margin_mean"] = float(news_m.mean())
            pred.notes.extend(f.label for f in factors if f.team in (game.home, game.away))
        o = odds
        pred.markets.append(MarketProbability("moneyline", "home", None, p_home, lo, hi,
                                              o.home_ml if o else None, fair["ml"], tie_ml, self.calibrated, ml_aware))
        pred.markets.append(MarketProbability("moneyline", "away", None, 1 - p_home, 1 - hi, 1 - lo,
                                              o.away_ml if o else None,
                                              None if fair["ml"] is None else 1 - fair["ml"], tie_ml, self.calibrated, ml_aware))
        if o is not None and o.spread is not None:
            p, push = md.cover(o.spread)
            self._add_two_way(pred, "spread", o.spread, p, push, self.spread_cal,
                              (o.home_spread_price, o.away_spread_price), fair["spread"], ("home", "away"))
        if o is not None and o.total is not None:
            p, push = td.over(o.total)
            self._add_two_way(pred, "total", o.total, p, push, self.total_cal,
                              (o.over_price, o.under_price), fair["total"], ("over", "under"))
        return pred

    def _add_two_way(self, pred, market, line, p, push, cal, prices, fair, sides):
        cond = p / np.clip(1 - push, 1e-9, None)
        if cal is not None:
            cond = cal.predict_many(cond[:, None], fair)
        mean = float(np.mean(cond))
        lo, hi = float(np.quantile(cond, self.s.band[0])), float(np.quantile(cond, self.s.band[1]))
        pp = float(np.mean(push))
        ok = cal is not None
        aware = cal is not None and cal.with_market is not None
        pred.markets.append(MarketProbability(market, sides[0], line, mean, lo, hi, prices[0], fair, pp, ok, aware))
        pred.markets.append(MarketProbability(market, sides[1], line, 1 - mean, 1 - hi, 1 - lo, prices[1],
                                              None if fair is None else 1 - fair, pp, ok, aware))
