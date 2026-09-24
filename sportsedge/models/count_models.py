"""Time-weighted attack/defence/park count model for NHL goals and MLB runs.

    log lambda_home = mu + h + att_home + def_away + park_venue
    log lambda_away = mu     + att_away + def_home + park_venue

Goals/runs ~ Poisson(lambda) (NHL) or NegBin2(lambda, r) (MLB, overdispersed).
Parameters are the MAP estimate under Gaussian priors (ridge) with the
Dixon & Coles (1997) exponential time-decay weighting w = exp(-xi * age_days),
solved by Fisher scoring (IRLS). The inverse Fisher information gives a
Laplace-approximate posterior covariance, used to sample (lambda_h, lambda_a)
jointly for uncertainty propagation. The NB dispersion r is re-estimated by
weighted method of moments and the Dixon-Coles rho by profile likelihood.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Sequence

import numpy as np
from scipy.optimize import minimize_scalar

from ..types import GameResult


@dataclass
class CountRatings:
    decay_per_day: float
    ridge_sd: float
    park_sd: float
    window_days: int
    dispersion: Optional[float]  # None => Poisson
    estimate_dispersion: bool = True
    estimate_rho: bool = False
    estimate_prior: bool = True  # empirical-Bayes (EM) estimate of ridge_sd

    teams: dict[str, int] = field(default_factory=dict)
    beta: Optional[np.ndarray] = None
    cov: Optional[np.ndarray] = None
    rho: float = 0.0
    fitted_at: Optional[datetime] = None
    n_games: int = 0

    # layout: [mu, home, att(n), def(n), park(n)]
    def _n(self) -> int:
        return len(self.teams)

    def _idx(self, kind: str, team: str) -> int:
        n = self._n()
        base = {"att": 2, "def": 2 + n, "park": 2 + 2 * n}[kind]
        return base + self.teams[team]

    def _row(self, scorer: str, opp: str, venue: str, is_home: bool, neutral: bool) -> np.ndarray:
        r = np.zeros(2 + 3 * self._n())
        r[0] = 1.0
        r[1] = 1.0 if (is_home and not neutral) else 0.0
        r[self._idx("att", scorer)] = 1.0
        r[self._idx("def", opp)] = 1.0
        if not neutral:
            r[self._idx("park", venue)] = 1.0
        return r

    def fit(self, games: Sequence[GameResult], as_of: datetime, update_dispersion: bool = True) -> "CountRatings":
        games = [g for g in games if (as_of - g.start_time).days <= self.window_days and g.start_time < as_of]
        for g in games:
            for t in (g.home, g.away):
                if t not in self.teams:
                    self.teams[t] = len(self.teams)
        n = self._n()
        d = 2 + 3 * n
        if not games:
            self.beta = np.zeros(d)
            self.beta[0] = math.log(3.0)
            self.cov = np.diag([1.0, 0.1] + [self.ridge_sd ** 2] * 2 * n + [self.park_sd ** 2] * n)
            self.fitted_at = as_of
            return self
        G = len(games)
        hi = np.array([self.teams[g.home] for g in games])
        ai = np.array([self.teams[g.away] for g in games])
        neutral = np.array([g.neutral for g in games])
        rows = np.arange(2 * G)
        scorer = np.empty(2 * G, int)
        opp = np.empty(2 * G, int)
        scorer[0::2], scorer[1::2] = hi, ai
        opp[0::2], opp[1::2] = ai, hi
        X = np.zeros((2 * G, d))
        X[:, 0] = 1.0
        X[0::2, 1] = (~neutral).astype(float)
        X[rows, 2 + scorer] = 1.0
        X[rows, 2 + n + opp] = 1.0
        venue_rows = np.repeat(~neutral, 2)
        X[rows[venue_rows], 2 + 2 * n + np.repeat(hi, 2)[venue_rows]] = 1.0
        y = np.empty(2 * G)
        y[0::2] = [g.home_score for g in games]
        y[1::2] = [g.away_score for g in games]
        age = np.array([(as_of - g.start_time).total_seconds() / 86400.0 for g in games])
        w = np.repeat(np.exp(-self.decay_per_day * age), 2)
        if self.beta is not None and len(self.beta) == d:
            beta = self.beta.copy()  # warm start from the previous fit
        else:
            beta = np.zeros(d)
            beta[0] = math.log(max(y.mean(), 0.1))
        r = self.dispersion
        est_r = r is not None and self.estimate_dispersion and update_dispersion
        est_tau = self.estimate_prior and update_dispersion
        team_fx = slice(2, 2 + 2 * n)
        for outer in range(4 if (est_r or est_tau) else 1):
            prec = np.concatenate([[1e-4, 1e-2], np.full(2 * n, 1 / self.ridge_sd ** 2),
                                   np.full(n, 1 / self.park_sd ** 2)])
            for _ in range(30):
                lam = np.exp(X @ beta)
                a = np.ones_like(lam) if r is None else r / (r + lam)
                g = X.T @ (w * (y - lam) * a) - prec * beta
                H = (X.T * (w * lam * a)) @ X + np.diag(prec)
                step = np.linalg.solve(H, g)
                beta += step
                if np.max(np.abs(step)) < 1e-7:
                    break
            if est_r:
                lam = np.exp(X @ beta)
                num = np.sum(w * lam ** 2)
                den = np.sum(w * ((y - lam) ** 2 - lam))
                r = float(np.clip(num / den, 1.5, 200.0)) if den > 0 else 200.0
            if est_tau:
                # EM step for the hierarchical prior: tau^2 = E[beta^2] under the Laplace posterior
                cov = np.linalg.inv(H)
                tau2 = np.mean(beta[team_fx] ** 2 + np.diag(cov)[team_fx])
                self.ridge_sd = float(np.clip(math.sqrt(tau2), 0.03, 0.6))
        lam = np.exp(X @ beta)
        a = np.ones_like(lam) if r is None else r / (r + lam)
        H = (X.T * (w * lam * a)) @ X + np.diag(prec)
        self.beta = beta
        self.cov = np.linalg.inv(H)
        self.dispersion = r
        self.fitted_at = as_of
        self.n_games = len(games)
        if self.estimate_rho and r is None:
            self.rho = self._fit_rho(games, lam, w)
        return self

    @staticmethod
    def _fit_rho(games, lam, w) -> float:
        lh, la = lam[0::2], lam[1::2]
        ww = w[0::2]
        hs = np.array([g.home_score for g in games])
        as_ = np.array([g.away_score for g in games])

        def nll(rho):
            tau = np.ones_like(lh)
            m00 = (hs == 0) & (as_ == 0)
            m01 = (hs == 0) & (as_ == 1)
            m10 = (hs == 1) & (as_ == 0)
            m11 = (hs == 1) & (as_ == 1)
            tau[m00] = 1 - lh[m00] * la[m00] * rho
            tau[m01] = 1 + lh[m01] * rho
            tau[m10] = 1 + la[m10] * rho
            tau[m11] = 1 - rho
            if np.any(tau <= 0):
                return 1e9
            return -float(np.sum(ww * np.log(tau)))

        return float(minimize_scalar(nll, bounds=(-0.2, 0.2), method="bounded").x)

    # ------------------------------------------------------------ prediction
    def _pred_rows(self, home: str, away: str, neutral: bool):
        for t in (home, away):
            if t not in self.teams:
                # unseen team: prior mean 0 effects; widen by refitting later
                self.teams[t] = len(self.teams)
                self._grow()
        xh = self._row(home, away, home, True, neutral)
        xa = self._row(away, home, home, False, neutral)
        return xh, xa

    def _grow(self) -> None:
        """Insert zero-mean prior entries for a newly seen team."""
        n_old = self._n() - 1
        old_beta, old_cov = self.beta, self.cov
        d = 2 + 3 * self._n()
        beta = np.zeros(d)
        cov = np.zeros((d, d))
        old_idx = [0, 1] + [2 + i for i in range(n_old)] + [2 + n_old + i for i in range(n_old)] + \
                  [2 + 2 * n_old + i for i in range(n_old)]
        new_idx = [0, 1] + [2 + i for i in range(n_old)] + [2 + self._n() + i for i in range(n_old)] + \
                  [2 + 2 * self._n() + i for i in range(n_old)]
        beta[new_idx] = old_beta[old_idx]
        cov[np.ix_(new_idx, new_idx)] = old_cov[np.ix_(old_idx, old_idx)]
        k = self._n() - 1
        cov[2 + k, 2 + k] = self.ridge_sd ** 2
        cov[2 + self._n() + k, 2 + self._n() + k] = self.ridge_sd ** 2
        cov[2 + 2 * self._n() + k, 2 + 2 * self._n() + k] = self.park_sd ** 2
        self.beta, self.cov = beta, cov

    def predict_log_rates(self, home: str, away: str, neutral: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Mean and 2x2 covariance of (log lambda_home, log lambda_away)."""
        xh, xa = self._pred_rows(home, away, neutral)
        Xp = np.vstack([xh, xa])
        return Xp @ self.beta, Xp @ self.cov @ Xp.T

    def sample_rates(self, home: str, away: str, neutral: bool, n: int,
                     rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        m, S = self.predict_log_rates(home, away, neutral)
        draws = rng.multivariate_normal(m, S + 1e-12 * np.eye(2), size=n)
        return np.exp(draws[:, 0]), np.exp(draws[:, 1])
