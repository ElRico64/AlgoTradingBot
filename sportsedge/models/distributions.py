"""Final-score distributions and market probabilities derived from them.

All functions are vectorised over Monte-Carlo draws (first axis = draw), so
the engine can propagate parameter and news uncertainty through to every
market probability.

Two families:
  * Integer margin distributions for NBA/NFL: a Normal discretised to the
    integers, re-weighted at football "key numbers" (3, 7, 10, ...) whose
    weights can be estimated from historical margins, with OT tie handling.
  * Joint score matrices for NHL/MLB: independent Poisson or negative
    binomial goals/runs with the Dixon & Coles (1997) low-score dependence
    correction, plus overtime / extra-innings resolution.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from typing import Optional
from scipy import stats
from scipy.special import ndtr


@dataclass
class MarginDist:
    """Distribution over the *final* home margin for each draw."""

    support: np.ndarray  # (M,) integer margins
    pmf: np.ndarray  # (N, M)

    def win(self) -> np.ndarray:
        return self.pmf[:, self.support > 0].sum(axis=1)

    def loss(self) -> np.ndarray:
        return self.pmf[:, self.support < 0].sum(axis=1)

    def tie(self) -> np.ndarray:
        return self.pmf[:, self.support == 0].sum(axis=1)

    def cover(self, home_line: float) -> tuple[np.ndarray, np.ndarray]:
        """P(home covers home_line) and P(push). home_line=-3.5 => home -3.5."""
        adj = self.support + home_line
        return self.pmf[:, adj > 0].sum(axis=1), self.pmf[:, np.isclose(adj, 0)].sum(axis=1)

    def mean(self) -> np.ndarray:
        return self.pmf @ self.support


@dataclass
class TotalDist:
    support: np.ndarray
    pmf: np.ndarray

    def over(self, line: float) -> tuple[np.ndarray, np.ndarray]:
        return self.pmf[:, self.support > line].sum(axis=1), self.pmf[:, np.isclose(self.support, line)].sum(axis=1)

    def mean(self) -> np.ndarray:
        return self.pmf @ self.support


def discretized_normal(mu: np.ndarray, sd: float, lo: int, hi: int) -> tuple[np.ndarray, np.ndarray]:
    support = np.arange(lo, hi + 1)
    mu = np.atleast_1d(mu)[:, None]
    edges = np.arange(lo, hi + 2) - 0.5
    cdf = ndtr((edges - mu) / sd)
    pmf = np.diff(cdf, axis=1)
    pmf /= pmf.sum(axis=1, keepdims=True)
    return support, pmf


def skew_t_logkernel(z: np.ndarray, a: float, b: float) -> np.ndarray:
    """Jones & Faddy (2003) skew-t, unnormalised log density of the standard form:
    (1 + z/sqrt(a+b+z^2))^(a+1/2) (1 - z/sqrt(a+b+z^2))^(b+1/2). a = b is Student-t
    with 2a degrees of freedom; a > b skews right. Heavy tails for small a, b."""
    r = z / np.sqrt(a + b + z * z)
    return (a + 0.5) * np.log1p(r) + (b + 0.5) * np.log1p(-r)


def skew_t_logpdf(x: np.ndarray, a: float, b: float, loc: float, scale: float) -> np.ndarray:
    from scipy.special import betaln
    z = (np.asarray(x, float) - loc) / scale
    c = (a + b - 1) * np.log(2.0) + betaln(a, b) + 0.5 * np.log(a + b)
    return skew_t_logkernel(z, a, b) - c - np.log(scale)


def fit_skew_t(resid: np.ndarray) -> Optional[dict]:
    """Maximum-likelihood Jones-Faddy skew-t for margin residuals, kept only if it
    beats a normal on the most recent quarter of residuals (fitted on the rest)."""
    from scipy.optimize import minimize
    from scipy.stats import norm
    x = np.asarray(resid, float)
    if len(x) < 400:
        return None

    def fit(v):
        # bounded: a, b in [1, 60] spans heavy tails (Student-t with 2 dof) to nearly normal
        # (120 dof); unbounded fits on near-normal data drift to huge, equivalent values
        s0, m0 = float(np.std(v)), float(np.mean(v))
        f = lambda th: -np.sum(skew_t_logpdf(v, np.exp(th[0]), np.exp(th[1]), th[2], np.exp(th[3])))
        th0 = np.array([np.log(8.0), np.log(8.0), m0, np.log(s0 * 0.95)])
        bounds = [(0.0, np.log(60.0)), (0.0, np.log(60.0)), (m0 - 0.5 * s0, m0 + 0.5 * s0),
                  (np.log(0.3 * s0), np.log(2.0 * s0))]
        th = minimize(f, th0, method="L-BFGS-B", bounds=bounds).x
        return {"a": float(np.exp(th[0])), "b": float(np.exp(th[1])), "loc": float(th[2]),
                "scale": float(np.exp(th[3]))}

    cut = int(len(x) * 0.75)
    tr, te = x[:cut], x[cut:]
    sh = fit(tr)
    ll_skt = np.mean(skew_t_logpdf(te, sh["a"], sh["b"], sh["loc"], sh["scale"]))
    ll_norm = np.mean(norm.logpdf(te, np.mean(tr), np.std(tr)))
    if ll_skt <= ll_norm:
        return None
    out = fit(x)
    out["holdout_gain"] = float(ll_skt - ll_norm)
    return out


def discretized_skew_t(mu: np.ndarray, shape: dict, lo: int, hi: int) -> tuple[np.ndarray, np.ndarray]:
    support = np.arange(lo, hi + 1)
    mu = np.atleast_1d(mu)[:, None]
    z = (support[None, :] - mu - shape["loc"]) / shape["scale"]
    lk = skew_t_logkernel(z, shape["a"], shape["b"])
    pmf = np.exp(lk - lk.max(axis=1, keepdims=True))  # stable at any parameter values
    pmf /= pmf.sum(axis=1, keepdims=True)
    return support, pmf


def gaussian_margin_dist(mu: np.ndarray, sd: float, key_weights: dict[int, float] | None = None,
                         tie_mass_factor: float = 1.0, ot_home_prob: np.ndarray | float = 0.5,
                         ties_allowed: bool = False, span: int = 70, shape: Optional[dict] = None) -> MarginDist:
    """Margin distribution around mu: a discretised normal, or the learned skew-t `shape`."""
    if shape is not None:
        support, pmf = discretized_skew_t(mu, shape, -span, span)
    else:
        support, pmf = discretized_normal(mu, sd, -span, span)
    if key_weights:
        w = np.ones_like(support, dtype=float)
        for k, m in key_weights.items():
            w[np.abs(support) == k] *= m
        mass_before = pmf.sum(axis=1, keepdims=True)
        pmf = pmf * w
        pmf *= mass_before / pmf.sum(axis=1, keepdims=True)
    zero = support == 0
    ot = np.broadcast_to(np.asarray(ot_home_prob, float), (pmf.shape[0],))
    tie_mass = pmf[:, zero].sum(axis=1)
    keep = tie_mass * (tie_mass_factor if ties_allowed else 0.0)
    moved = tie_mass - keep
    # Games level at the end of regulation are settled in OT; in football the
    # typical OT winning margin is a field goal, in basketball it is small.
    step = 3 if key_weights else 1
    pmf[:, zero] = keep[:, None]
    pmf[:, support == step] += (moved * ot)[:, None]
    pmf[:, support == -step] += (moved * (1 - ot))[:, None]
    return MarginDist(support, pmf)


def gaussian_total_dist(mu: np.ndarray, sd: float) -> TotalDist:
    mu = np.atleast_1d(mu)
    lo = max(0, int(np.floor(mu.min() - 6 * sd)))
    hi = int(np.ceil(mu.max() + 6 * sd))
    support, pmf = discretized_normal(mu, sd, lo, hi)
    return TotalDist(support, pmf)


# --------------------------------------------------------------------------
# Count models (NHL / MLB)
# --------------------------------------------------------------------------
def count_pmf(lam: np.ndarray, max_k: int, dispersion: float | None) -> np.ndarray:
    k = np.arange(max_k + 1)
    lam = np.atleast_1d(lam)[:, None]
    if dispersion is None:
        pmf = stats.poisson.pmf(k, lam)
    else:
        r = dispersion
        pmf = stats.nbinom.pmf(k, r, r / (r + lam))
    # fold the tail into the last cell so every row sums to one
    pmf[:, -1] += np.clip(1.0 - pmf.sum(axis=1), 0, None)
    return pmf


def dixon_coles_tau(lh: np.ndarray, la: np.ndarray, rho: float) -> np.ndarray:
    """(N, 2, 2) multiplicative correction for scores (0,0),(0,1),(1,0),(1,1)."""
    tau = np.ones((len(lh), 2, 2))
    tau[:, 0, 0] = 1 - lh * la * rho
    tau[:, 0, 1] = 1 + lh * rho
    tau[:, 1, 0] = 1 + la * rho
    tau[:, 1, 1] = 1 - rho
    return np.clip(tau, 1e-6, None)


def joint_score_matrix(lh: np.ndarray, la: np.ndarray, max_k: int, dispersion: float | None,
                       rho: float = 0.0) -> np.ndarray:
    ph = count_pmf(lh, max_k, dispersion)
    pa = count_pmf(la, max_k, dispersion)
    J = ph[:, :, None] * pa[:, None, :]
    if rho:
        J[:, :2, :2] *= dixon_coles_tau(np.atleast_1d(lh), np.atleast_1d(la), rho)
        J /= J.sum(axis=(1, 2), keepdims=True)
    return J


def _projection(K: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """0/1 matrices mapping flattened (home, away) cells to margin / total bins,
    plus the flat indices of tied cells."""
    h = np.repeat(np.arange(K), K)
    a = np.tile(np.arange(K), K)
    Pm = np.zeros((K * K, 2 * K - 1))
    Pm[np.arange(K * K), h - a + K - 1] = 1.0
    Pt = np.zeros((K * K, 2 * K))
    Pt[np.arange(K * K), h + a] = 1.0
    return Pm, Pt, np.flatnonzero(h == a)


_PROJ_CACHE: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


def count_final_distributions(J: np.ndarray, extra_time_home: np.ndarray) -> tuple[MarginDist, TotalDist]:
    """Resolve regulation ties with an extra period won by one goal/run."""
    N, K, _ = J.shape
    if K not in _PROJ_CACHE:
        _PROJ_CACHE[K] = _projection(K)
    Pm, Pt, ties = _PROJ_CACHE[K]
    flat = J.reshape(N, -1)
    tie_cells = flat[:, ties]  # (N, K): P(k-k) for k = 0..K-1
    tie_mass = tie_cells.sum(axis=1)
    m_pmf = flat @ Pm
    t_pmf = flat @ Pt
    et = np.broadcast_to(np.asarray(extra_time_home, float), (N,))
    m_pmf[:, K - 1] = 0.0  # the margin-0 bin holds exactly the regulation ties
    m_pmf[:, K] += tie_mass * et  # home wins in extra time by 1
    m_pmf[:, K - 2] += tie_mass * (1 - et)
    # a k-k tie ends (k+1)-k: its total moves from 2k to 2k+1
    t_pmf[:, 2 * np.arange(K)] -= tie_cells
    t_pmf[:, 2 * np.arange(K) + 1] += tie_cells
    return MarginDist(np.arange(-(K - 1), K), m_pmf), TotalDist(np.arange(0, 2 * K), t_pmf)
