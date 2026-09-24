"""Odds conversion and removal of the bookmaker margin ("de-vigging").

Methods implemented:
  * multiplicative (proportional normalisation)
  * additive (equal margin subtraction)
  * power (p_i = pi_i^k with k solved so that sum p_i = 1)
  * Shin (1993), which models the margin as the bookmaker's protection
    against a fraction z of insider money. Shin probabilities are
        p_i = [sqrt(z^2 + 4 (1-z) pi_i^2 / Pi) - z] / (2 (1-z))
    with z solved numerically so that sum p_i = 1. Shin corrects the
    favourite-longshot bias better than proportional normalisation and is the
    default here.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy.optimize import brentq


def american_to_decimal(american: float) -> float:
    if american is None:
        raise ValueError("odds is None")
    a = float(american)
    if a >= 100:
        return 1.0 + a / 100.0
    if a <= -100:
        return 1.0 + 100.0 / (-a)
    raise ValueError(f"invalid American odds {american}")


def decimal_to_american(decimal: float) -> float:
    if decimal <= 1.0:
        raise ValueError("decimal odds must exceed 1")
    if decimal >= 2.0:
        return (decimal - 1.0) * 100.0
    return -100.0 / (decimal - 1.0)


def american_to_implied(american: float) -> float:
    return 1.0 / american_to_decimal(american)


def prob_to_american(p: float) -> float:
    """Fair (no-vig) American price for probability p."""
    p = min(max(p, 1e-9), 1 - 1e-9)
    return decimal_to_american(1.0 / p)


def breakeven_probability(american: float) -> float:
    return american_to_implied(american)


def overround(prices_american: Sequence[float]) -> float:
    return sum(american_to_implied(a) for a in prices_american) - 1.0


def devig(prices_american: Sequence[float], method: str = "shin") -> np.ndarray:
    pi = np.array([american_to_implied(a) for a in prices_american], dtype=float)
    return devig_implied(pi, method)


def devig_implied(pi: np.ndarray, method: str = "shin") -> np.ndarray:
    pi = np.asarray(pi, dtype=float)
    big_pi = pi.sum()
    n = len(pi)
    if big_pi <= 1.0 + 1e-12:
        # No margin (or an arbitrage): proportional normalisation is all we can do.
        return pi / big_pi
    if method == "multiplicative":
        return pi / big_pi
    if method == "additive":
        p = pi - (big_pi - 1.0) / n
        if np.any(p <= 0):
            return pi / big_pi
        return p
    if method == "power":
        f = lambda k: np.sum(pi ** k) - 1.0
        k = brentq(f, 1.0, 50.0)
        return pi ** k
    if method == "shin":
        def shin_p(z: float) -> np.ndarray:
            return (np.sqrt(z * z + 4.0 * (1.0 - z) * pi * pi / big_pi) - z) / (2.0 * (1.0 - z))

        g = lambda z: shin_p(z).sum() - 1.0
        try:
            z = brentq(g, 0.0, 0.4)
        except ValueError:
            return pi / big_pi
        p = shin_p(z)
        return p / p.sum()
    raise ValueError(f"unknown devig method {method!r}")


def consensus_probability(book_prices: Sequence[tuple[float, float]], weights: Sequence[float] | None = None,
                          method: str = "shin") -> float:
    """Combine several books' two-way prices into one fair probability for side A.

    Probabilities are pooled in log-odds space (a logarithmic opinion pool),
    which is the natural geometric average for independent probability
    forecasts. `weights` lets sharper books (e.g. Pinnacle, Circa) count more.
    """
    if not book_prices:
        raise ValueError("no prices")
    ps = np.array([devig([a, b], method)[0] for a, b in book_prices])
    w = np.ones(len(ps)) if weights is None else np.asarray(weights, float)
    w = w / w.sum()
    ps = np.clip(ps, 1e-6, 1 - 1e-6)
    lo = np.sum(w * np.log(ps / (1 - ps)))
    return 1.0 / (1.0 + math.exp(-lo))
