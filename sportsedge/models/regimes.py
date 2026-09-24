"""Hidden Markov regime detection on rating surprises ("form").

The Kalman ratings assume strength drifts smoothly. Real teams also jump
between regimes: a star returns, a rotation changes, a goalie gets hot. We
model each team's standardised surprise z_t (actual minus expected margin,
divided by its predictive sd, from the team's perspective) as emitted by a
3-state hidden Markov chain {cold, normal, hot}:

    s_t | s_{t-1} ~ A[s_{t-1}, .]          (sticky transitions)
    z_t | s_t     ~ N(mu_s, sigma^2)

The forward filter gives, before every game, the predictive regime
probabilities P(s_{t+1} | z_1..z_t). All parameters (means, shared variance,
transition matrix, initial distribution) are estimated by pooled Baum-Welch
EM over every team's sequence (Rabiner, 1989), with scaled forward-backward
recursions for numerical stability. States stay ordered cold < normal < hot.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

K = 3


@dataclass
class RegimeHMM:
    means: np.ndarray = field(default_factory=lambda: np.array([-0.6, 0.0, 0.6]))
    sd: float = 1.0
    A: np.ndarray = field(default_factory=lambda: np.array([[0.90, 0.10, 0.00],
                                                            [0.04, 0.92, 0.04],
                                                            [0.00, 0.10, 0.90]]))
    pi: np.ndarray = field(default_factory=lambda: np.array([0.15, 0.70, 0.15]))
    # Sticky Dirichlet prior on each row of A (Fox et al., 2011): pseudo-counts
    # that favour staying in a regime. Without it, EM on noisy surprises
    # collapses into a memoryless mixture (self-transition ~0.5) that carries
    # no information about the next game.
    sticky_strength: float = 300.0
    sticky_stay: float = 0.95
    fitted: bool = False

    def _emit(self, z) -> np.ndarray:
        z = np.atleast_1d(z)[:, None]
        return np.exp(-0.5 * ((z - self.means) / self.sd) ** 2) / self.sd + 1e-300

    def step(self, predictive: np.ndarray, z: float) -> np.ndarray:
        """Filter one observation; returns the predictive distribution for the next game."""
        post = predictive * self._emit(z)[0]
        post /= post.sum()
        nxt = post @ self.A
        return nxt / nxt.sum()

    def stationary(self) -> np.ndarray:
        w, v = np.linalg.eig(self.A.T)
        s = np.real(v[:, np.argmin(np.abs(w - 1))])
        s = np.abs(s)
        return s / s.sum()

    def fit(self, sequences: list[np.ndarray], n_iter: int = 30, tol: float = 1e-5) -> "RegimeHMM":
        seqs = [np.asarray(s, float) for s in sequences if len(s) >= 5]
        if sum(len(s) for s in seqs) < 200:
            return self
        prev = -np.inf
        for _ in range(n_iter):
            A_num = np.zeros((K, K))
            g_sum = np.zeros(K)
            gz = np.zeros(K)
            gz2 = np.zeros(K)
            pi_acc = np.zeros(K)
            ll = 0.0
            for z in seqs:
                B = self._emit(z)
                T = len(z)
                alpha = np.zeros((T, K))
                c = np.zeros(T)
                a = self.pi * B[0]
                c[0] = a.sum()
                alpha[0] = a / c[0]
                for t in range(1, T):
                    a = (alpha[t - 1] @ self.A) * B[t]
                    c[t] = a.sum()
                    alpha[t] = a / c[t]
                beta = np.ones((T, K))
                for t in range(T - 2, -1, -1):
                    beta[t] = (self.A @ (B[t + 1] * beta[t + 1])) / c[t + 1]
                gamma = alpha * beta
                gamma /= gamma.sum(axis=1, keepdims=True)
                for t in range(T - 1):
                    xi = alpha[t][:, None] * self.A * (B[t + 1] * beta[t + 1])[None, :] / c[t + 1]
                    A_num += xi
                pi_acc += gamma[0]
                g_sum += gamma.sum(axis=0)
                gz += gamma.T @ z
                gz2 += gamma.T @ (z * z)
                ll += np.log(c).sum()
            means = gz / np.maximum(g_sum, 1e-9)
            var = float(np.sum(gz2 - 2 * means * gz + means ** 2 * g_sum) / np.sum(g_sum))
            order = np.argsort(means)
            self.means = np.clip(means[order], -2.5, 2.5)
            self.sd = float(np.sqrt(np.clip(var, 0.25, 4.0)))
            prior = np.full((K, K), (1 - self.sticky_stay) / (K - 1)) + np.eye(K) * (self.sticky_stay - (1 - self.sticky_stay) / (K - 1))
            A_num = A_num + self.sticky_strength * prior
            A = A_num / np.maximum(A_num.sum(axis=1, keepdims=True), 1e-9)
            self.A = A[np.ix_(order, order)]
            self.A = 0.99 * self.A + 0.01 / K  # keep every transition possible
            self.pi = (pi_acc / pi_acc.sum())[order]
            if ll - prev < tol * max(1.0, abs(ll)):
                break
            prev = ll
        self.fitted = True
        return self

    def form(self, predictive: np.ndarray) -> tuple[float, float]:
        """(expected surprise next game, P(hot) - P(cold))."""
        return float(predictive @ self.means), float(predictive[2] - predictive[0])
