"""Market check: how much of the model's disagreement with the sportsbooks holds up.

For a market (moneyline, spread or total) with de-vigged market probability q
and model probability p, the checked probability keeps a share s of the
model's disagreement on the log-odds scale:

    logit p* = logit q + s · (logit p − logit q)

s = 0 trusts the books, s = 1 trusts the model. s is learned per league and
market from every graded game the board priced, wins and losses alike
(learning from losses only would be selection on the outcome). It is a
maximum a posteriori estimate under a hierarchical prior:

    s[league, market] = mu[market] + d[league, market]
    d ~ N(0, TAU²)             leagues differ, but not wildly
    mu[market] ~ N(MU0, SD0²)  a priori the model and the books are equally informative

A league with few graded games borrows strength from the others in the same
market (partial pooling). As the record grows, its own data take over. The
transform is symmetric (checking the away side gives 1 − the checked home
side) and monotone, so it maps the 80% band too.
"""
from __future__ import annotations

import math
from typing import Iterable, Optional

import numpy as np
from scipy.optimize import minimize

MARKETS = ("moneyline", "spread", "total")
MU0, SD0, TAU = 0.5, 0.3, 0.2
MIN_ROWS_FOR_VALUE = 100  # graded games in a market before its checked edges are trusted for Best bets


def _logit(x):
    x = np.clip(x, 1e-6, 1 - 1e-6)
    return np.log(x / (1 - x))


def check(p: float, q: float, s: float) -> float:
    """Checked probability: keep share s of the model's disagreement with the market."""
    z = float(_logit(q) + s * (_logit(p) - _logit(q)))
    return 1.0 / (1.0 + math.exp(-z))


def fit_shares(rows: Iterable[tuple], leagues: Iterable[str] = ()) -> dict:
    """rows: (league, market, p_model, q_market, won 0/1), one side per game and market.
    Returns {(league, market): {"share": s, "n": graded games, "market_n": games in that market}}."""
    rows = [r for r in rows if r[1] in MARKETS and r[3] is not None and 0 < r[3] < 1]
    leagues = sorted(set(leagues) | {r[0] for r in rows})
    pairs = [(lg, mk) for mk in MARKETS for lg in leagues]
    idx = {pr: i for i, pr in enumerate(pairs)}
    nm = len(MARKETS)
    if rows:
        lp = _logit(np.array([r[2] for r in rows], float))
        lq = _logit(np.array([r[3] for r in rows], float))
        y = np.array([float(r[4]) for r in rows])
        d = lp - lq
        mk_i = np.array([MARKETS.index(r[1]) for r in rows])
        pr_i = np.array([idx[(r[0], r[1])] for r in rows])
    else:
        lp = lq = y = d = np.zeros(0)
        mk_i = pr_i = np.zeros(0, int)

    def objective(theta):
        mu, dev = theta[:nm], theta[nm:]
        s = mu[mk_i] + dev[pr_i]
        z = lq + s * d
        # numerically stable Bernoulli log-likelihood
        nll = np.sum(np.logaddexp(0.0, z) - y * z)
        r = 1.0 / (1.0 + np.exp(-z)) - y
        g_s = r * d
        grad_mu = np.bincount(mk_i, weights=g_s, minlength=nm) + (mu - MU0) / SD0 ** 2
        grad_dev = np.bincount(pr_i, weights=g_s, minlength=len(pairs)) + dev / TAU ** 2
        val = nll + 0.5 * np.sum((mu - MU0) ** 2) / SD0 ** 2 + 0.5 * np.sum(dev ** 2) / TAU ** 2
        return val, np.concatenate([grad_mu, grad_dev])

    theta0 = np.concatenate([np.full(nm, MU0), np.zeros(len(pairs))])
    theta = minimize(objective, theta0, jac=True, method="L-BFGS-B").x if rows else theta0
    counts = {pr: 0 for pr in pairs}
    for r in rows:
        counts[(r[0], r[1])] += 1
    per_market = {mk: sum(c for (lg, m), c in counts.items() if m == mk) for mk in MARKETS}
    out = {}
    for (lg, mk), i in idx.items():
        s = float(theta[MARKETS.index(mk)] + theta[nm + i])
        out[(lg, mk)] = {"share": round(min(max(s, 0.0), 1.0), 3), "n": counts[(lg, mk)],
                         "market_n": per_market[mk]}
    return out


def apply_to_prediction(pred, league: str, shares: Optional[dict]) -> None:
    """Check every market of a prediction that has a market probability, in place.
    Keeps the engine's own number as `raw_probability`."""
    if not shares:
        return
    for m in pred.markets:
        sh = shares.get((league, m.market))
        if sh is None or m.fair_market_prob is None:
            continue
        s, q = sh["share"], m.fair_market_prob
        m.raw_probability = m.probability
        m.probability = check(m.probability, q, s)
        m.lower = check(m.lower, q, s)
        m.upper = check(m.upper, q, s)
        m.check_share = s
        if sh["market_n"] >= MIN_ROWS_FOR_VALUE:
            m.market_aware = True  # its edge over the price is now an honest estimate
    ml = next((m for m in pred.markets if m.market == "moneyline" and m.side == "home"), None)
    if ml is not None and getattr(ml, "raw_probability", None) is not None:
        pred.home_win_prob, pred.home_win_lower, pred.home_win_upper = ml.probability, ml.lower, ml.upper
