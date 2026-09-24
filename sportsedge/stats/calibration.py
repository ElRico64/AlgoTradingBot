"""Probability calibration, stacking and forecast-evaluation statistics.

A pick is only as good as the *calibration* of the probability behind it: a
"70% pick" must win ~70% of the time out-of-sample. This module provides

  * L2-regularised logistic regression solved by Newton-Raphson (IRLS)
  * Platt scaling          p = sigma(a * logit(q) + b)
  * Beta calibration       p = sigma(a ln q - b ln(1-q) + c)   (Kull et al., 2017)
  * Isotonic regression    pool-adjacent-violators (Barlow et al., 1972)
  * A stacking calibrator  that learns log-linear pooling weights over several
    component models *and* the de-vigged market price
  * Proper scoring rules (log loss, Brier + Murphy decomposition), expected
    calibration error, reliability tables
  * Wilson score intervals, exact binomial tests and a Poisson-binomial
    calibration z-test for the selected picks.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from scipy import stats

EPS = 1e-6


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(x):
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-x))


# --------------------------------------------------------------------------
# Logistic regression (IRLS with ridge penalty; intercept unpenalised)
# --------------------------------------------------------------------------
@dataclass
class LogisticRegression:
    l2: float = 1.0
    fit_intercept: bool = True
    max_iter: int = 100
    tol: float = 1e-9
    prior_mean: Optional[np.ndarray] = None  # shrink coefficients toward this
    nonneg: bool = False  # constrain all non-intercept coefficients to be >= 0
    coef_: Optional[np.ndarray] = None
    cov_: Optional[np.ndarray] = None  # Laplace posterior covariance

    def _design(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if self.fit_intercept:
            X = np.column_stack([np.ones(len(X)), X])
        return X

    def fit(self, X, y, sample_weight=None):
        Xd = self._design(X)
        y = np.asarray(y, dtype=float)
        n, d = Xd.shape
        w = np.ones(n) if sample_weight is None else np.asarray(sample_weight, float)
        pen = np.full(d, self.l2)
        if self.fit_intercept:
            pen[0] = 1e-8
        m0 = np.zeros(d)
        if self.prior_mean is not None:
            m0[-len(self.prior_mean):] = self.prior_mean
        beta = m0.copy()
        if self.nonneg:
            return self._fit_nonneg(Xd, y, w, pen, m0)
        for _ in range(self.max_iter):
            eta = Xd @ beta
            p = sigmoid(eta)
            g = Xd.T @ (w * (y - p)) - pen * (beta - m0)
            W = w * p * (1 - p)
            H = (Xd.T * W) @ Xd + np.diag(pen)
            step = np.linalg.solve(H, g)
            beta = beta + step
            if np.max(np.abs(step)) < self.tol:
                break
        self.coef_ = beta
        p = sigmoid(Xd @ beta)
        H = (Xd.T * (w * p * (1 - p))) @ Xd + np.diag(pen)
        self.cov_ = np.linalg.inv(H)
        return self

    def _fit_nonneg(self, Xd, y, w, pen, m0):
        """Penalised MLE with coefficient >= 0 bounds (L-BFGS-B)."""
        from scipy.optimize import minimize

        def f(b):
            eta = Xd @ b
            # log(1 + e^eta) computed stably
            nll = np.sum(w * (np.logaddexp(0.0, eta) - y * eta)) + 0.5 * np.sum(pen * (b - m0) ** 2)
            g = Xd.T @ (w * (sigmoid(eta) - y)) + pen * (b - m0)
            return nll, g

        d = Xd.shape[1]
        lo = 1 if self.fit_intercept else 0
        bounds = [(None, None)] * lo + [(0.0, None)] * (d - lo)
        x0 = np.clip(m0, [b[0] if b[0] is not None else -np.inf for b in bounds], None)
        res = minimize(f, x0, jac=True, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 500, "gtol": 1e-8})
        beta = res.x
        self.coef_ = beta
        p = sigmoid(Xd @ beta)
        H = (Xd.T * (w * p * (1 - p))) @ Xd + np.diag(pen)
        self.cov_ = np.linalg.inv(H)
        return self

    def decision_function(self, X):
        return self._design(X) @ self.coef_

    def predict_proba(self, X):
        return sigmoid(self.decision_function(X))


# --------------------------------------------------------------------------
# Calibrators
# --------------------------------------------------------------------------
class Calibrator:
    def fit(self, p, y, sample_weight=None):  # pragma: no cover - interface
        raise NotImplementedError

    def transform(self, p):  # pragma: no cover - interface
        raise NotImplementedError


class IdentityCalibrator(Calibrator):
    def fit(self, p, y, sample_weight=None):
        return self

    def transform(self, p):
        return np.clip(np.asarray(p, float), EPS, 1 - EPS)


class PlattCalibrator(Calibrator):
    def __init__(self, l2: float = 1.0):
        # Shrink slope toward 1 (identity map) rather than toward 0.
        self.lr = LogisticRegression(l2=l2, prior_mean=np.array([1.0]))

    def fit(self, p, y, sample_weight=None):
        self.lr.fit(logit(np.atleast_1d(p))[:, None], y, sample_weight)
        return self

    def transform(self, p):
        return self.lr.predict_proba(logit(np.atleast_1d(p))[:, None])


class BetaCalibrator(Calibrator):
    """Kull, Silva Filho & Flach (2017): three-parameter beta calibration."""

    def __init__(self, l2: float = 1.0):
        self.lr = LogisticRegression(l2=l2, prior_mean=np.array([1.0, 1.0]))

    @staticmethod
    def _feats(p):
        p = np.clip(np.atleast_1d(np.asarray(p, float)), EPS, 1 - EPS)
        return np.column_stack([np.log(p), -np.log(1 - p)])

    def fit(self, p, y, sample_weight=None):
        self.lr.fit(self._feats(p), y, sample_weight)
        return self

    def transform(self, p):
        return self.lr.predict_proba(self._feats(p))


class IsotonicCalibrator(Calibrator):
    """Pool-adjacent-violators; monotone, non-parametric. Needs lots of data."""

    def __init__(self):
        self.x_: Optional[np.ndarray] = None
        self.y_: Optional[np.ndarray] = None

    def fit(self, p, y, sample_weight=None):
        p = np.asarray(p, float)
        y = np.asarray(y, float)
        w = np.ones_like(p) if sample_weight is None else np.asarray(sample_weight, float)
        order = np.argsort(p)
        xs, ys, ws = p[order], y[order], w[order]
        # blocks: [value, weight, x_min, x_max]
        vals, wts, lo, hi = [], [], [], []
        for xi, yi, wi in zip(xs, ys, ws):
            vals.append(yi)
            wts.append(wi)
            lo.append(xi)
            hi.append(xi)
            while len(vals) > 1 and vals[-2] > vals[-1]:
                v = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / (wts[-2] + wts[-1])
                wsum = wts[-2] + wts[-1]
                l0 = lo[-2]
                h1 = hi[-1]
                for arr in (vals, wts, lo, hi):
                    arr.pop()
                vals[-1], wts[-1], lo[-1], hi[-1] = v, wsum, l0, h1
        # knots at block centres for linear interpolation
        self.x_ = np.array([(a + b) / 2 for a, b in zip(lo, hi)])
        self.y_ = np.clip(np.array(vals), EPS, 1 - EPS)
        return self

    def transform(self, p):
        return np.interp(np.asarray(p, float), self.x_, self.y_)


def make_calibrator(method: str, l2: float = 1.0) -> Calibrator:
    return {
        "identity": IdentityCalibrator,
        "platt": lambda: PlattCalibrator(l2),
        "beta": lambda: BetaCalibrator(l2),
        "isotonic": IsotonicCalibrator,
    }[method]()


# --------------------------------------------------------------------------
# Stacking: learned log-linear opinion pool over components (+ market)
# --------------------------------------------------------------------------
@dataclass
class StackingCalibrator:
    """logit p = b0 + sum_k w_k f(p_k)   [+ w_m logit(p_market)]

    Fitting this by penalised maximum likelihood on out-of-sample forecasts is
    both an ensemble (learns how much to trust each model and the market) and
    a calibration step. Weights are constrained to be non-negative (Breiman,
    1996): the components are highly collinear, and an unconstrained fit can
    give one of them a negative weight, which would make news that favours a
    team move the final probability *against* it. Two sub-models are kept: one for games with a market
    price and one without.

    features="logit": f(p) = logit p (one slope per component).
    features="beta":  f(p) = [ln p, -ln(1-p)] (Kull et al. 2017), which lets
    the calibrator bend the two tails independently — useful where a model is
    over-confident only at the extremes.
    """

    components: list[str]
    l2: float = 2.0
    features: str = "logit"
    with_market: Optional[LogisticRegression] = None
    without_market: Optional[LogisticRegression] = None
    n_fit_: int = 0

    def _feats(self, P: np.ndarray) -> np.ndarray:
        P = np.clip(np.atleast_2d(P), EPS, 1 - EPS)
        if self.features == "beta":
            return np.column_stack([f for j in range(P.shape[1])
                                    for f in (np.log(P[:, j]), -np.log(1 - P[:, j]))])
        return np.log(P / (1 - P))

    def _prior(self, with_market: bool) -> np.ndarray:
        k = len(self.components)
        per = 2 if self.features == "beta" else 1
        prior = np.full(k * per, 1.0 / k)
        return np.concatenate([prior * 0.5, [0.5]]) if with_market else prior

    def fit(self, comp_probs: Sequence[dict[str, float]], market: Sequence[Optional[float]], y):
        y = np.asarray(y, float)
        P = np.array([[c[k] for k in self.components] for c in comp_probs], float)
        self.without_market = LogisticRegression(l2=self.l2, prior_mean=self._prior(False),
                                                 nonneg=True).fit(self._feats(P), y)
        has_m = np.array([m is not None for m in market])
        if has_m.sum() >= 50:
            mk = np.array([m for m in market if m is not None], float)
            X = np.column_stack([self._feats(P[has_m]), logit(mk)])
            self.with_market = LogisticRegression(l2=self.l2, prior_mean=self._prior(True),
                                                  nonneg=True).fit(X, y[has_m])
        else:
            self.with_market = None
        self.n_fit_ = len(y)
        return self

    @property
    def fitted(self) -> bool:
        return self.without_market is not None

    def weights(self) -> dict[str, float]:
        names = [f"{c}{suf}" for c in self.components
                 for suf in ((":lnp", ":ln1mp") if self.features == "beta" else ("",))]
        out = {}
        if self.with_market is not None:
            out.update({f"mkt_model:{n}": float(c) for n, c in
                        zip(["intercept", *names, "market"], self.with_market.coef_)})
        if self.without_market is not None:
            out.update({f"model_only:{n}": float(c) for n, c in
                        zip(["intercept", *names], self.without_market.coef_)})
        return out

    def predict_many(self, P: np.ndarray, market: Optional[float]) -> np.ndarray:
        """P: (n_draws, n_components) probabilities -> stacked probabilities (n_draws,)."""
        P = np.atleast_2d(P)
        if self.without_market is None:
            return sigmoid(logit(P).mean(axis=1))
        F = self._feats(P)
        if market is not None and self.with_market is not None:
            return self.with_market.predict_proba(np.column_stack([F, np.full(len(F), logit(market))]))
        return self.without_market.predict_proba(F)

    def transform(self, comp: dict[str, float], market: Optional[float]) -> float:
        return float(self.predict_many(np.array([[comp[k] for k in self.components]]), market)[0])


# --------------------------------------------------------------------------
# Scoring & evaluation
# --------------------------------------------------------------------------
def log_loss(p, y) -> float:
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(p, y) -> float:
    p = np.asarray(p, float)
    y = np.asarray(y, float)
    return float(np.mean((p - y) ** 2))


def reliability_table(p, y, n_bins: int = 10) -> list[dict]:
    p = np.asarray(p, float)
    y = np.asarray(y, float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        n = int(m.sum())
        hits = int(y[m].sum())
        lo, hi = wilson_interval(hits, n)
        rows.append({
            "bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}",
            "n": n,
            "mean_pred": float(p[m].mean()),
            "observed": hits / n,
            "ci_low": lo,
            "ci_high": hi,
        })
    return rows


def expected_calibration_error(p, y, n_bins: int = 10) -> float:
    rows = reliability_table(p, y, n_bins)
    n = sum(r["n"] for r in rows)
    return float(sum(r["n"] / n * abs(r["mean_pred"] - r["observed"]) for r in rows)) if n else float("nan")


def brier_decomposition(p, y, n_bins: int = 10) -> dict[str, float]:
    """Murphy (1973): Brier = reliability - resolution + uncertainty."""
    p = np.asarray(p, float)
    y = np.asarray(y, float)
    base = y.mean()
    rows = reliability_table(p, y, n_bins)
    n = len(p)
    rel = sum(r["n"] * (r["mean_pred"] - r["observed"]) ** 2 for r in rows) / n
    res = sum(r["n"] * (r["observed"] - base) ** 2 for r in rows) / n
    return {"reliability": rel, "resolution": res, "uncertainty": base * (1 - base)}


def wilson_interval(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    z = stats.norm.ppf(0.5 + conf / 2)
    ph = k / n
    denom = 1 + z * z / n
    centre = (ph + z * z / (2 * n)) / denom
    half = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def binomial_test_greater(k: int, n: int, p0: float) -> float:
    """One-sided exact p-value for H0: true hit rate <= p0."""
    if n == 0:
        return 1.0
    return float(stats.binom.sf(k - 1, n, p0))


def poisson_binomial_z(p, y) -> float:
    """z-score of (hits - sum p) / sqrt(sum p(1-p)); |z| > 2 flags miscalibration."""
    p = np.asarray(p, float)
    y = np.asarray(y, float)
    var = np.sum(p * (1 - p))
    return float((y.sum() - p.sum()) / math.sqrt(var)) if var > 0 else 0.0
