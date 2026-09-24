"""The pattern engine: learns what the structural models miss.

Two layers:

1. **Regime detection** (models/regimes.py). A hidden Markov model on each
   team's rating surprises flags hot and cold stretches earlier than the
   smooth Kalman drift can.
2. **Residual learning** in two stages, both starting from the structural
   model's logit so they only learn corrections:
   (a) ridge-logistic main effects, penalty chosen on the most recent games;
   (b) gradient-boosted trees (models/boosting.py) for interactions on top.
   Each stage is kept only if it improves held-out log loss. The features
   are a pre-game vector:

   * the structural forecast and its uncertainty
   * rest and back-to-backs
   * travel and time-zone shifts
   * regime state and short-term form
   * home and road splits
   * head-to-head history
   * streaks and volatility
   * season phase and pace

   These cover interactions (e.g. "a road team on no rest after a long flight
   against a hot home team") that a linear rating cannot express.

Leakage control:

* Every feature uses only information available before the game.
* The trees are refit on a schedule using only games already played.
* Their forecasts reach the stacker only as out-of-sample predictions.
* The stacker's non-negative weight then decides how much, if anything, the
  pattern engine is trusted. With no real pattern to find, the boosting stops
  early and its correction is about zero.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np

from .boosting import HistGradientBoosting
from .regimes import RegimeHMM

FEATURES = [
    "structural_logit", "rating_uncertainty", "rest_home", "rest_away", "b2b_home", "b2b_away",
    "travel_away", "timezones_away", "travel_home", "form_home", "form_away", "hot_home", "hot_away",
    "recent_home", "recent_away", "home_split_home", "road_split_away", "head_to_head",
    "streak_home", "streak_away", "volatility_home", "volatility_away", "season_phase",
    "games_home", "games_away", "pace", "neutral",
]
LABELS = {
    "structural_logit": "strength gap", "rating_uncertainty": "rating uncertainty", "rest_home": "home rest",
    "rest_away": "away rest", "b2b_home": "home back-to-back", "b2b_away": "away back-to-back",
    "travel_away": "away travel", "timezones_away": "away time zones", "travel_home": "home travel",
    "form_home": "home form regime", "form_away": "away form regime", "hot_home": "home hot/cold",
    "hot_away": "away hot/cold", "recent_home": "home recent form", "recent_away": "away recent form",
    "home_split_home": "home-court split", "road_split_away": "road split", "head_to_head": "head-to-head",
    "streak_home": "home streak", "streak_away": "away streak", "volatility_home": "home volatility",
    "volatility_away": "away volatility", "season_phase": "season phase", "games_home": "home games played",
    "games_away": "away games played", "pace": "pace", "neutral": "neutral site",
}
EWMA_FAST = 0.30
EWMA_SPLIT = 0.15
EWMA_H2H = 0.35


@dataclass
class TeamState:
    predictive: Optional[np.ndarray] = None
    recent: float = 0.0
    home_split: float = 0.0
    road_split: float = 0.0
    volatility: float = 0.8
    streak: int = 0
    games: int = 0
    season_start: Optional[datetime] = None
    last_time: Optional[datetime] = None
    zs: list = field(default_factory=list)


@dataclass
class PatternModel:
    offseason_days: int
    refit_days: int = 21
    min_rows: int = 500
    window: int = 6000
    hmm: RegimeHMM = field(default_factory=RegimeHMM)
    gbm: Optional[HistGradientBoosting] = None
    teams: dict = field(default_factory=lambda: defaultdict(TeamState))
    h2h: dict = field(default_factory=dict)
    rows_X: list = field(default_factory=list)
    rows_off: list = field(default_factory=list)
    rows_y: list = field(default_factory=list)
    fitted_at: Optional[datetime] = None
    n_fits: int = 0
    lin_coef: Optional[np.ndarray] = None  # ridge-logistic main effects on standardised features
    x_mean: Optional[np.ndarray] = None
    x_sd: Optional[np.ndarray] = None
    lin_lambda: Optional[float] = None
    val_gain: dict = field(default_factory=dict)
    # earn-your-place gate: the correction is applied only when its cross-validated
    # held-out log-loss gain on this league's own history reaches this level
    min_gain: float = 0.0015

    # ------------------------------------------------------------------ state
    def _team(self, t: str, when: datetime) -> TeamState:
        s = self.teams[t]
        if s.predictive is None:
            s.predictive = self.hmm.stationary()
        if s.last_time is not None and (when - s.last_time).days >= self.offseason_days:
            s.predictive = self.hmm.stationary()
            s.recent *= 0.3
            s.home_split *= 0.5
            s.road_split *= 0.5
            s.streak, s.games, s.season_start = 0, 0, None
            s.last_time = None  # reset once, however many times features are requested
        if s.season_start is None:
            s.season_start = when
        return s

    def features(self, home: str, away: str, when: datetime, neutral: bool, structural_logit: float,
                 rating_sd: float, rest: tuple, travel: tuple, pace: float) -> np.ndarray:
        h, a = self._team(home, when), self._team(away, when)
        fh, hh = self.hmm.form(h.predictive)
        fa, ha = self.hmm.form(a.predictive)
        rh, ra = rest
        key = (home, away) if home < away else (away, home)
        sign = 1.0 if home < away else -1.0
        start = min(h.season_start, a.season_start)
        return np.array([
            structural_logit, rating_sd,
            min(rh if rh is not None else 3.0, 7.0), min(ra if ra is not None else 3.0, 7.0),
            float(rh is not None and rh < 1.5), float(ra is not None and ra < 1.5),
            travel[0], travel[1], travel[2],
            fh, fa, hh, ha, h.recent, a.recent, h.home_split, a.road_split,
            sign * self.h2h.get(key, 0.0),
            max(-8, min(8, h.streak)) / 8.0, max(-8, min(8, a.streak)) / 8.0,
            h.volatility, a.volatility,
            min((when - start).days / 180.0, 1.5),
            min(h.games, 82) / 82.0, min(a.games, 82) / 82.0, pace, float(neutral),
        ])

    def record(self, home: str, away: str, when: datetime, x: Optional[np.ndarray], offset: float,
               y: Optional[float], z: float, margin: float) -> None:
        """After the game: store the training row, then update every team state."""
        if x is not None and y is not None:
            self.rows_X.append(x)
            self.rows_off.append(offset)
            self.rows_y.append(y)
            if len(self.rows_y) > self.window:
                del self.rows_X[0], self.rows_off[0], self.rows_y[0]
        # credit attribution: a game's surprise mixes both teams' form, so each team's
        # regime filter sees the surprise net of the opponent's current expected form
        fh = self.hmm.form(self._team(home, when).predictive)[0]
        fa = self.hmm.form(self._team(away, when).predictive)[0]
        for team, zt, is_home, opp_form in ((home, z, True, fa), (away, -z, False, fh)):
            s = self._team(team, when)
            s.predictive = self.hmm.step(s.predictive, zt + opp_form)
            s.recent = (1 - EWMA_FAST) * s.recent + EWMA_FAST * zt
            s.volatility = (1 - EWMA_FAST) * s.volatility + EWMA_FAST * abs(zt)
            if is_home:
                s.home_split = (1 - EWMA_SPLIT) * s.home_split + EWMA_SPLIT * zt
            else:
                s.road_split = (1 - EWMA_SPLIT) * s.road_split + EWMA_SPLIT * zt
            won = (margin > 0) == is_home if margin != 0 else None
            if won is not None:
                s.streak = (s.streak + 1 if s.streak >= 0 else 1) if won else (s.streak - 1 if s.streak <= 0 else -1)
            s.games += 1
            s.last_time = when
            s.zs.append(zt)
            if len(s.zs) > 400:
                del s.zs[0]
        key = (home, away) if home < away else (away, home)
        sign = 1.0 if home < away else -1.0
        self.h2h[key] = (1 - EWMA_H2H) * self.h2h.get(key, 0.0) + EWMA_H2H * sign * z

    # ---------------------------------------------------------------- learning
    def maybe_fit(self, as_of: datetime) -> bool:
        if len(self.rows_y) < self.min_rows:
            return False
        if self.fitted_at is not None and (as_of - self.fitted_at).days < self.refit_days:
            return False
        self.hmm.fit([np.array(s.zs) for s in self.teams.values()])
        X, y, off = np.array(self.rows_X), np.array(self.rows_y), np.array(self.rows_off)
        n = len(y)
        # expanding-window time-series CV: train on everything before each test block
        folds = [(int(n * a), int(n * b)) for a, b in ((0.55, 0.70), (0.70, 0.85), (0.85, 1.0))]
        mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-9
        Z = (X - mu) / sd

        # stage 1: ridge-logistic main effects; penalty = best mean held-out gain
        lams = (3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)
        gains = np.zeros(len(lams))
        for a, b in folds:
            base = _ll(off[a:b], y[a:b])
            for i, lam in enumerate(lams):
                coef = _ridge_logit(Z[:a], y[:a], off[:a], lam)
                gains[i] += (base - _ll(off[a:b] + Z[a:b] @ coef, y[a:b])) / len(folds)
        i = int(np.argmax(gains))
        use_lin = gains[i] > 1e-4
        self.val_gain = {"linear": float(gains[i]) if use_lin else 0.0}
        self.x_mean, self.x_sd = mu, sd
        self.lin_lambda = lams[i] if use_lin else None
        self.lin_coef = _ridge_logit(Z, y, off, lams[i]) if use_lin else None

        # stage 2: boosted trees for interactions; tree count = best mean held-out path
        curves = []
        for a, b in folds:
            lin_tr = Z[:a] @ _ridge_logit(Z[:a], y[:a], off[:a], lams[i]) if use_lin else 0.0
            coef_a = _ridge_logit(Z[:a], y[:a], off[:a], lams[i]) if use_lin else None
            lin_va = Z[a:b] @ coef_a if use_lin else 0.0
            path = HistGradientBoosting(seed=self.n_fits, n_estimators=120, learning_rate=0.08).fit_path(
                X[:a], y[:a], off[:a] + lin_tr, X[a:b], y[a:b], off[a:b] + lin_va)
            curves.append(path[0] - path)  # gain vs. stage 1 alone
        mean_gain = np.mean(curves, axis=0)
        m = int(np.argmax(mean_gain))
        self.val_gain["trees"] = float(mean_gain[m]) if mean_gain[m] > 1e-4 else 0.0
        lin_all = Z @ self.lin_coef if use_lin else np.zeros(n)
        self.gbm = HistGradientBoosting(seed=self.n_fits, learning_rate=0.08).fit_fixed(
            X, y, off + lin_all, m if mean_gain[m] > 1e-4 else 0)
        self.fitted_at = as_of
        self.n_fits += 1
        return True

    @property
    def active(self) -> bool:
        return sum(self.val_gain.values()) >= self.min_gain

    def correction(self, x: np.ndarray) -> float:
        """Learned logit correction to the structural forecast. Zero before the first
        fit and whenever the engine has not earned its place on held-out games."""
        if not self.active:
            return 0.0
        out = 0.0
        if self.lin_coef is not None:
            out += float(((x - self.x_mean) / self.x_sd) @ self.lin_coef)
        if self.gbm is not None:
            out += float(self.gbm.correction(x[None, :])[0])
        return out

    def importance(self, top: int = 8) -> list[tuple[str, float]]:
        """Share of learned signal per feature (|standardised linear coef| + tree split gain)."""
        imp = np.zeros(len(FEATURES))
        if self.lin_coef is not None:
            imp += np.abs(self.lin_coef) / max(np.abs(self.lin_coef).sum(), 1e-12)
        if self.gbm is not None and self.gbm.importance_ is not None and self.gbm.importance_.sum() > 0:
            imp += self.gbm.importance_ / self.gbm.importance_.sum()
        if not imp.sum():
            return []
        imp = imp / imp.sum()
        order = np.argsort(imp)[::-1][:top]
        return [(LABELS[FEATURES[i]], float(imp[i])) for i in order if imp[i] > 0]

    def describe(self, home: str, away: str, x: np.ndarray) -> list[str]:
        """Plain-language notes for strong regime signals."""
        notes = []
        for team, hot in ((home, x[FEATURES.index("hot_home")]), (away, x[FEATURES.index("hot_away")])):
            if hot > 0.45:
                notes.append(f"Form: hot stretch ({team}, regime signal {hot:+.2f})")
            elif hot < -0.45:
                notes.append(f"Form: cold stretch ({team}, regime signal {hot:+.2f})")
        return notes


def _ll(F: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.logaddexp(0.0, F) - y * F))


def _ridge_logit(Z: np.ndarray, y: np.ndarray, off: np.ndarray, lam: float, iters: int = 50) -> np.ndarray:
    """Logistic regression with a fixed offset and an L2 penalty (Newton / IRLS), no intercept:
    the structural offset already carries the level."""
    b = np.zeros(Z.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(off + Z @ b, -30, 30)))
        g = Z.T @ (y - p) - lam * b
        H = (Z.T * (p * (1 - p))) @ Z + lam * np.eye(Z.shape[1])
        step = np.linalg.solve(H, g)
        b += step
        if np.max(np.abs(step)) < 1e-8:
            break
    return b


def logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))
