"""Histogram gradient-boosted decision trees for binary outcomes (NumPy only).

This is the second-order ("Newton") boosting algorithm used by XGBoost and
LightGBM (Chen & Guestrin, 2016; Ke et al., 2017), written out so the engine
needs no heavy ML dependency:

  * features are bucketed into quantile bins, so a split search is a cumulative
    sum over a small histogram of gradients g = p - y and hessians h = p(1-p);
  * split gain  = G_L^2/(H_L+lambda) + G_R^2/(H_R+lambda) - G^2/(H+lambda) - gamma
  * leaf value  = -G / (H + lambda)
  * shrinkage (learning rate), row subsampling, a minimum leaf size and early
    stopping on the most recent slice of data (time-ordered, never shuffled).

**Boosting from an offset.** Every prediction starts from a given logit (the
structural model's forecast), so the trees only learn *corrections*: patterns
the structural models systematically miss. With nothing to learn, the model
stops after a few trees and returns the offset unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _logloss(F, y):
    return float(np.mean(np.logaddexp(0.0, F) - y * F))


@dataclass
class _Tree:
    feature: list[int] = field(default_factory=list)
    threshold: list[int] = field(default_factory=list)  # go left if bin <= threshold
    left: list[int] = field(default_factory=list)
    right: list[int] = field(default_factory=list)
    value: list[float] = field(default_factory=list)

    def add(self, value: float) -> int:
        self.feature.append(-1)
        self.threshold.append(0)
        self.left.append(-1)
        self.right.append(-1)
        self.value.append(value)
        return len(self.value) - 1

    def predict(self, Xb: np.ndarray) -> np.ndarray:
        feature = np.array(self.feature)
        threshold = np.array(self.threshold)
        left, right, value = np.array(self.left), np.array(self.right), np.array(self.value)
        node = np.zeros(len(Xb), dtype=int)
        active = feature[node] >= 0
        while active.any():
            idx = np.flatnonzero(active)
            f = feature[node[idx]]
            go_left = Xb[idx, f] <= threshold[node[idx]]
            node[idx] = np.where(go_left, left[node[idx]], right[node[idx]])
            active = feature[node] >= 0
        return value[node]


@dataclass
class HistGradientBoosting:
    n_estimators: int = 400
    learning_rate: float = 0.05
    max_depth: int = 3
    min_leaf: int = 40
    l2: float = 10.0
    gamma: float = 0.0
    subsample: float = 0.7
    n_bins: int = 32
    early_stopping: int = 40
    val_fraction: float = 0.2
    seed: int = 0

    edges_: list[np.ndarray] = field(default_factory=list)
    trees_: list[_Tree] = field(default_factory=list)
    importance_: Optional[np.ndarray] = None
    val_gain_: float = 0.0  # validation log-loss improvement over the offset alone

    # ------------------------------------------------------------------ binning
    def _fit_bins(self, X: np.ndarray) -> None:
        qs = np.linspace(0, 1, self.n_bins + 1)[1:-1]
        self.edges_ = [np.unique(np.quantile(X[:, j], qs)) for j in range(X.shape[1])]

    def _bin(self, X: np.ndarray) -> np.ndarray:
        Xb = np.empty(X.shape, dtype=np.int16)
        for j, e in enumerate(self.edges_):
            Xb[:, j] = np.searchsorted(e, X[:, j], side="right")
        return Xb

    # ---------------------------------------------------------------- building
    def _build(self, Xb, g, h, rng) -> _Tree:
        tree = _Tree()
        nb = self.n_bins
        root = tree.add(0.0)
        stack = [(root, np.arange(len(g)), 0)]
        while stack:
            node, idx, depth = stack.pop()
            G, H = g[idx].sum(), h[idx].sum()
            tree.value[node] = -G / (H + self.l2)
            if depth >= self.max_depth or len(idx) < 2 * self.min_leaf:
                continue
            parent = G * G / (H + self.l2)
            best = (self.gamma, -1, -1)
            for j in range(Xb.shape[1]):
                col = Xb[idx, j]
                gh = np.bincount(col, weights=g[idx], minlength=nb)
                hh = np.bincount(col, weights=h[idx], minlength=nb)
                nn = np.bincount(col, minlength=nb)
                GL, HL, NL = np.cumsum(gh)[:-1], np.cumsum(hh)[:-1], np.cumsum(nn)[:-1]
                GR, HR, NR = G - GL, H - HL, len(idx) - NL
                ok = (NL >= self.min_leaf) & (NR >= self.min_leaf)
                if not ok.any():
                    continue
                gain = GL ** 2 / (HL + self.l2) + GR ** 2 / (HR + self.l2) - parent
                gain[~ok] = -np.inf
                b = int(np.argmax(gain))
                if gain[b] > best[0]:
                    best = (float(gain[b]), j, b)
            gain, j, b = best
            if j < 0:
                continue
            self.importance_[j] += gain
            go_left = Xb[idx, j] <= b
            li, ri = tree.add(0.0), tree.add(0.0)
            tree.feature[node], tree.threshold[node], tree.left[node], tree.right[node] = j, b, li, ri
            stack.append((li, idx[go_left], depth + 1))
            stack.append((ri, idx[~go_left], depth + 1))
        return tree

    def fit(self, X: np.ndarray, y: np.ndarray, offset: np.ndarray) -> "HistGradientBoosting":
        """Rows must be in time order; the last `val_fraction` is the early-stopping set."""
        X, y, offset = np.asarray(X, float), np.asarray(y, float), np.asarray(offset, float)
        rng = np.random.default_rng(self.seed)
        n = len(y)
        n_val = max(1, int(n * self.val_fraction))
        tr, va = slice(0, n - n_val), slice(n - n_val, n)
        self._fit_bins(X[tr])
        Xb = self._bin(X)
        self.importance_ = np.zeros(X.shape[1])
        F_tr, F_va = offset[tr].copy(), offset[va].copy()
        base = best = _logloss(F_va, y[va])
        best_m, trees = 0, []
        ytr = y[tr]
        for m in range(self.n_estimators):
            p = _sigmoid(F_tr)
            g, h = p - ytr, p * (1 - p)
            idx = np.flatnonzero(rng.random(len(g)) < self.subsample)
            tree = self._build(Xb[tr][idx], g[idx], h[idx], rng)
            trees.append(tree)
            F_tr += self.learning_rate * tree.predict(Xb[tr])
            F_va += self.learning_rate * tree.predict(Xb[va])
            loss = _logloss(F_va, y[va])
            if loss < best - 1e-7:
                best, best_m = loss, m + 1
            elif m + 1 - best_m >= self.early_stopping:
                break
        self.trees_ = trees[:best_m]
        self.val_gain_ = base - best
        return self

    def fit_path(self, X, y, offset, Xv, yv, offv) -> np.ndarray:
        """Train all n_estimators trees on (X, y) and return the validation log loss
        after each tree (index 0 = offset only). Used for time-series CV."""
        X, y, offset = np.asarray(X, float), np.asarray(y, float), np.asarray(offset, float)
        rng = np.random.default_rng(self.seed)
        self._fit_bins(X)
        Xb, Xvb = self._bin(X), self._bin(np.asarray(Xv, float))
        self.importance_ = np.zeros(X.shape[1])
        F, Fv = offset.copy(), np.asarray(offv, float).copy()
        path = [_logloss(Fv, yv)]
        for _ in range(self.n_estimators):
            p = _sigmoid(F)
            g, h = p - y, p * (1 - p)
            idx = np.flatnonzero(rng.random(len(g)) < self.subsample)
            tree = self._build(Xb[idx], g[idx], h[idx], rng)
            F += self.learning_rate * tree.predict(Xb)
            Fv += self.learning_rate * tree.predict(Xvb)
            path.append(_logloss(Fv, yv))
        return np.array(path)

    def fit_fixed(self, X, y, offset, n_trees: int) -> "HistGradientBoosting":
        """Train exactly n_trees on all rows (tree count chosen elsewhere, e.g. by CV)."""
        X, y, offset = np.asarray(X, float), np.asarray(y, float), np.asarray(offset, float)
        rng = np.random.default_rng(self.seed)
        self._fit_bins(X)
        Xb = self._bin(X)
        self.importance_ = np.zeros(X.shape[1])
        F = offset.copy()
        self.trees_ = []
        for _ in range(n_trees):
            p = _sigmoid(F)
            g, h = p - y, p * (1 - p)
            idx = np.flatnonzero(rng.random(len(g)) < self.subsample)
            tree = self._build(Xb[idx], g[idx], h[idx], rng)
            self.trees_.append(tree)
            F += self.learning_rate * tree.predict(Xb)
        return self

    def correction(self, X: np.ndarray) -> np.ndarray:
        """Learned logit correction to add to the offset."""
        X = np.atleast_2d(np.asarray(X, float))
        if not self.trees_:
            return np.zeros(len(X))
        Xb = self._bin(X)
        return self.learning_rate * np.sum([t.predict(Xb) for t in self.trees_], axis=0)

    def predict_proba(self, X: np.ndarray, offset: np.ndarray) -> np.ndarray:
        return _sigmoid(np.asarray(offset, float) + self.correction(X))
