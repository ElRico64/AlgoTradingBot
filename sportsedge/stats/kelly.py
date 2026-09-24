"""Bankroll sizing: Kelly criterion under parameter uncertainty.

* Single-bet Kelly:  f* = (p d - 1) / (d - 1)  for decimal odds d.
* Uncertainty shrinkage (in the spirit of Baker & McHale, 2013, "Optimal
  betting under parameter uncertainty"): when p is only known up to a
  posterior sd s, the expected-log-growth-optimal stake is shrunk by the
  signal-to-noise factor
        k = e^2 / (e^2 + d^2 s^2),   e = p d - 1,
  so bets whose edge is small relative to our uncertainty get tiny stakes.
* Fractional Kelly on top (default 1/4) because model error is never fully
  captured by s.
* Simultaneous Kelly for several independent concurrent bets: maximise
  E[log W] exactly over all 2^n outcome combinations (n <= 12).
"""
from __future__ import annotations

import itertools

import numpy as np
from scipy.optimize import minimize


def kelly_fraction(p: float, decimal_odds: float) -> float:
    b = decimal_odds - 1.0
    if b <= 0:
        return 0.0
    return max(0.0, (p * decimal_odds - 1.0) / b)


def uncertainty_shrinkage(p: float, p_sd: float, decimal_odds: float) -> float:
    e = p * decimal_odds - 1.0
    if e <= 0:
        return 0.0
    return e * e / (e * e + (decimal_odds * p_sd) ** 2)


def stake_fraction(p: float, p_sd: float, decimal_odds: float, kelly_multiplier: float = 0.25,
                   cap: float = 0.05) -> float:
    f = kelly_fraction(p, decimal_odds) * uncertainty_shrinkage(p, p_sd, decimal_odds)
    return float(min(cap, kelly_multiplier * f))


def simultaneous_kelly(probs, decimal_odds, max_total: float = 0.5) -> np.ndarray:
    probs = np.asarray(probs, float)
    d = np.asarray(decimal_odds, float)
    n = len(probs)
    if n == 0:
        return np.zeros(0)
    if n > 12:
        raise ValueError("simultaneous_kelly enumerates 2^n outcomes; use n <= 12")
    outcomes = np.array(list(itertools.product([0, 1], repeat=n)), float)
    po = np.prod(np.where(outcomes == 1, probs, 1 - probs), axis=1)

    def neg_growth(f):
        wealth = 1.0 - f.sum() + outcomes @ (f * d)
        if np.any(wealth <= 0):
            return 1e6
        return -float(po @ np.log(wealth))

    f0 = np.array([kelly_fraction(p, di) for p, di in zip(probs, d)]) / max(n, 1)
    cons = [{"type": "ineq", "fun": lambda f: max_total - f.sum()}]
    res = minimize(neg_growth, f0, bounds=[(0, 1)] * n, constraints=cons, method="SLSQP")
    return np.clip(res.x, 0, 1)
