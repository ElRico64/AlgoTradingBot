"""State-space team ratings estimated with a Kalman filter.

Model (margin mode, all quantities in points / goals / runs):

    state      x_t = [h, theta_1, ..., theta_n]
    transition theta_{t+dt} = theta_t + w,   w ~ N(0, q^2 dt I)
               at a season break: theta <- rho theta,
                                  P     <- rho^2 P + (1 - rho^2) tau^2 I
    observation y = H x + e,  H = [1{not neutral}, +1 at home, -1 at away],
               e ~ N(0, sigma^2)

In totals mode H = [1, +1 at home, +1 at away] and state[0] is the league
scoring level, so theta becomes each team's contribution to game totals.

Because the filter carries a full posterior covariance P, predictions come
with *epistemic* uncertainty: Var[H x] = H P H'. The posterior predictive
win probability is Phi(mu / sqrt(sigma^2 + H P H')), and drawing
x ~ N(x_hat, P) gives a distribution over the win probability itself, which
the engine uses for its lower-bound confidence gate.

Blowouts are handled with a Huber-style clip on the innovation (robust
Kalman filtering), and the hyperparameters (sigma, q) can be estimated by
maximising the prediction-error-decomposition log-likelihood (`tune`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable, Optional

import numpy as np
from scipy.optimize import minimize


@dataclass
class KalmanRatings:
    obs_sd: float
    prior_sd: float
    process_sd_per_day: float
    intercept_prior: float
    intercept_prior_sd: float
    season_carryover: float = 0.7
    offseason_days: int = 90
    clip_sds: float = 2.5
    mode: str = "margin"  # or "total"
    intercept_process_ratio: float = 0.05

    index: dict[str, int] = field(default_factory=dict)
    x: np.ndarray = field(default=None)  # type: ignore[assignment]
    P: np.ndarray = field(default=None)  # type: ignore[assignment]
    last_time: Optional[datetime] = None
    n_updates: int = 0
    loglik: float = 0.0

    def __post_init__(self):
        if self.x is None:
            self.x = np.array([self.intercept_prior], float)
            self.P = np.array([[self.intercept_prior_sd ** 2]], float)

    # ------------------------------------------------------------------ state
    def _ensure(self, team: str) -> int:
        if team not in self.index:
            k = len(self.x)
            self.index[team] = k
            self.x = np.append(self.x, 0.0)
            P = np.zeros((k + 1, k + 1))
            P[:k, :k] = self.P
            P[k, k] = self.prior_sd ** 2
            self.P = P
        return self.index[team]

    def advance(self, when: datetime) -> None:
        if self.last_time is None:
            self.last_time = when
            return
        dt = (when - self.last_time).total_seconds() / 86400.0
        if dt <= 0:
            return
        n = len(self.x)
        if dt >= self.offseason_days:
            rho = self.season_carryover
            teams = slice(1, n)
            self.x[teams] *= rho
            self.x[teams] -= self.x[teams].mean() if self.mode == "margin" and n > 1 else 0.0
            T = np.eye(n)
            T[1:, 1:] *= rho
            self.P = T @ self.P @ T.T
            self.P[1:, 1:] += (1 - rho ** 2) * self.prior_sd ** 2 * np.eye(n - 1)
        else:
            q2 = self.process_sd_per_day ** 2 * dt
            self.P[1:, 1:] += q2 * np.eye(n - 1)
            self.P[0, 0] += (self.intercept_process_ratio * self.process_sd_per_day) ** 2 * dt
        self.last_time = when

    def _H(self, home: str, away: str, neutral: bool) -> np.ndarray:
        ih, ia = self._ensure(home), self._ensure(away)
        H = np.zeros(len(self.x))
        if self.mode == "margin":
            H[0] = 0.0 if neutral else 1.0
            H[ih] += 1.0
            H[ia] -= 1.0
        else:
            H[0] = 1.0
            H[ih] += 1.0
            H[ia] += 1.0
        return H

    # ------------------------------------------------------------ inference
    def predict(self, home: str, away: str, neutral: bool = False) -> tuple[float, float, float]:
        """Return (mean, parameter variance H P H', observation variance)."""
        H = self._H(home, away, neutral)
        return float(H @ self.x), float(H @ self.P @ H), self.obs_sd ** 2

    def sample_means(self, home: str, away: str, neutral: bool, n: int, rng: np.random.Generator) -> np.ndarray:
        mean, var, _ = self.predict(home, away, neutral)
        return rng.normal(mean, math.sqrt(max(var, 1e-12)), size=n)

    def update(self, home: str, away: str, y: float, neutral: bool = False,
               when: Optional[datetime] = None) -> float:
        """Assimilate one observed game. Returns the log predictive density."""
        if when is not None:
            self.advance(when)
        H = self._H(home, away, neutral)
        mu = float(H @ self.x)
        PHt = self.P @ H
        S = float(H @ PHt) + self.obs_sd ** 2
        innov = y - mu
        ll = -0.5 * (math.log(2 * math.pi * S) + innov * innov / S)
        c = self.clip_sds * math.sqrt(S)
        innov = max(-c, min(c, innov))
        K = PHt / S
        self.x = self.x + K * innov
        self.P = self.P - np.outer(K, PHt)
        self.P = 0.5 * (self.P + self.P.T)
        self.n_updates += 1
        self.loglik += ll
        return ll

    def rating(self, team: str) -> tuple[float, float]:
        if team not in self.index:
            return 0.0, self.prior_sd
        i = self.index[team]
        return float(self.x[i]), float(math.sqrt(self.P[i, i]))

    # ------------------------------------------------------------- tuning
    @classmethod
    def tune(cls, games: Iterable, template: "KalmanRatings", value=lambda g: g.margin,
             burn_in: int = 100) -> "KalmanRatings":
        """Maximum-likelihood estimate of (obs_sd, process_sd) by the prediction
        error decomposition. Returns a fresh, untrained filter with tuned values."""
        games = list(games)

        def nll(theta):
            obs_sd, q = np.exp(theta)
            kf = cls(obs_sd=obs_sd, prior_sd=template.prior_sd, process_sd_per_day=q,
                     intercept_prior=template.intercept_prior,
                     intercept_prior_sd=template.intercept_prior_sd,
                     season_carryover=template.season_carryover,
                     offseason_days=template.offseason_days, clip_sds=1e9, mode=template.mode)
            total = 0.0
            for i, g in enumerate(games):
                ll = kf.update(g.home, g.away, value(g), g.neutral, g.start_time)
                if i >= burn_in:
                    total += ll
            return -total

        x0 = np.log([template.obs_sd, template.process_sd_per_day])
        res = minimize(nll, x0, method="Nelder-Mead", options={"xatol": 1e-3, "fatol": 1e-2, "maxiter": 200})
        obs_sd, q = np.exp(res.x)
        return cls(obs_sd=float(obs_sd), prior_sd=template.prior_sd, process_sd_per_day=float(q),
                   intercept_prior=template.intercept_prior, intercept_prior_sd=template.intercept_prior_sd,
                   season_carryover=template.season_carryover, offseason_days=template.offseason_days,
                   clip_sds=template.clip_sds, mode=template.mode)
