"""Player stat projections for NBA props, learned walk-forward from box scores.

For player i, stat s (points, rebounds, assists, made threes) in a game:

    mean   mu = minutes_hat × rate_is × opponent_s × pace
    count  X ~ NegBin(mean mu, dispersion phi_s)      Var X = mu + mu² / phi_s

* minutes_hat: exponentially weighted minutes in recent games played.
* rate_is: empirical-Bayes per-minute rate. The player's exponentially weighted
  stat and minutes are combined with K pseudo-minutes at his position group's
  league rate, (S + K·prior) / (M + K). Few minutes means close to the
  position average; a long record speaks for itself.
* opponent_s: how much of stat s the opponent allows per player-minute,
  relative to the league, shrunk the same way.
* pace: expected game total from both teams' recent scoring and allowing,
  over the league average.
* phi_s: maximum likelihood on the walk-forward residuals.

Every prediction in `fit` is made before that game updates the states. The
over/under probabilities are then recalibrated per stat with a beta calibration
map (logit p* = c + a ln p - b ln(1-p), which bends each tail separately), fitted
on those out-of-sample predictions against lines across the whole range books
offer: at the player's usual level, and well below or above it where alternate
lines and "X+" milestones sit. A stat is published only with enough
out-of-sample games behind its calibration.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import gammaln
from scipy.stats import nbinom

from ..stats.calibration import IsotonicCalibrator

PROP_STATS = ("pts", "reb", "ast", "fg3m")
STAT_NAMES = {"pts": "points", "reb": "rebounds", "ast": "assists", "fg3m": "made threes"}
HALF_LIFE_GAMES = 20.0
K_PLAYER = 250.0      # pseudo-minutes at the position rate
K_OPP = 4000.0        # pseudo player-minutes at the league rate
MIN_GAMES = 5         # games played before a player gets projections
MIN_CAL_ROWS = 2000   # out-of-sample rows per stat before it can be published


def _group(pos: str) -> str:
    p = (pos or "").upper()
    return "C" if p.startswith("C") else "G" if p.startswith(("G", "PG", "SG")) else "F"


@dataclass
class _Player:
    team: str = ""
    group: str = "F"
    S: dict = field(default_factory=lambda: defaultdict(float))
    M: float = 0.0
    min_ewma: Optional[float] = None
    avg: dict = field(default_factory=lambda: defaultdict(float))  # EW per-game average (for line placement)
    games: int = 0
    last: Optional[datetime] = None
    name: str = ""


@dataclass
class PropModel:
    players: dict = field(default_factory=dict)
    league_S: dict = field(default_factory=lambda: defaultdict(float))     # by (group, stat)
    league_M: dict = field(default_factory=lambda: defaultdict(float))     # by group
    opp_S: dict = field(default_factory=lambda: defaultdict(float))        # by (team, stat): allowed
    opp_M: dict = field(default_factory=lambda: defaultdict(float))        # by team: player-minutes faced
    team_pts: dict = field(default_factory=dict)                          # team -> [for_ewma, against_ewma]
    phi: dict = field(default_factory=dict)
    cal: dict = field(default_factory=dict)                               # stat -> beta map (c, a, b)
    iso: dict = field(default_factory=dict)                               # stat -> isotonic map on top
    health: dict = field(default_factory=dict)                            # stat -> forward calibration check
    oos: dict = field(default_factory=lambda: defaultdict(list))          # stat -> [(mu, line, actual)]
    reliability: dict = field(default_factory=dict)
    n_games: int = 0

    # ------------------------------------------------------------ helpers
    def _decay(self) -> float:
        return 0.5 ** (1.0 / HALF_LIFE_GAMES)

    def _league_rate(self, group: str, s: str) -> float:
        m = self.league_M.get(group, 0.0)
        return self.league_S.get((group, s), 0.0) / m if m > 0 else 0.0

    def _all_rate(self, s: str) -> float:
        m = sum(self.league_M.values())
        return sum(v for (g, st), v in self.league_S.items() if st == s) / m if m > 0 else 0.0

    def _pace(self, team: str, opp: str) -> float:
        tp = [v for v in self.team_pts.values() if v[0] is not None]
        if not tp or team not in self.team_pts or opp not in self.team_pts:
            return 1.0
        avg = float(np.mean([v[0] for v in tp]) * 2)
        a, b = self.team_pts[team], self.team_pts[opp]
        exp_total = (a[0] + a[1] + b[0] + b[1]) / 2
        return float(np.clip(exp_total / max(avg, 1e-9), 0.85, 1.15))

    def mean(self, pid: str, s: str, opp: str, minutes: Optional[float] = None) -> Optional[float]:
        p = self.players.get(pid)
        if p is None or p.games < MIN_GAMES or p.min_ewma is None:
            return None
        prior = self._league_rate(p.group, s)
        rate = (p.S[s] + K_PLAYER * prior) / (p.M + K_PLAYER)
        L = self._all_rate(s)
        opp_mult = 1.0
        if L > 0:
            opp_mult = ((self.opp_S.get((opp, s), 0.0) + K_OPP * L) / (self.opp_M.get(opp, 0.0) + K_OPP)) / L
        m = p.min_ewma if minutes is None else minutes
        return max(0.05, m * rate * opp_mult * self._pace(p.team, opp))

    def line_for(self, pid: str, s: str) -> Optional[float]:
        """Where a book would hang the line: half a unit off the player's recent average."""
        p = self.players.get(pid)
        if p is None or p.games < MIN_GAMES:
            return None
        return math.floor(p.avg[s]) + 0.5

    # ------------------------------------------------------------ probabilities
    def p_over_raw(self, s: str, mu: float, line: float) -> float:
        phi = self.phi.get(s, 20.0)
        k = math.floor(line)
        return float(nbinom.sf(k, phi, phi / (phi + mu)))

    def p_over(self, s: str, mu: float, line: float) -> float:
        p = min(max(self.p_over_raw(s, mu, line), 1e-6), 1 - 1e-6)
        if s not in self.cal:
            return p
        c = self.cal[s]
        if len(c) == 2:  # fitted by an earlier version: logistic map
            z = c[0] + c[1] * math.log(p / (1 - p))
        else:
            z = c[0] + c[1] * math.log(p) - c[2] * math.log(1 - p)
        q = 1 / (1 + math.exp(-z))
        iso = getattr(self, "iso", {}).get(s)
        if iso is not None:
            q = float(np.clip(iso.transform(np.array([q]))[0], 0.005, 0.995))
        return q

    def calibrated(self, s: str) -> bool:
        return s in self.cal and len(self.oos.get(s, [])) >= MIN_CAL_ROWS

    def trusted(self, s: str) -> bool:
        """Calibrated, and its 60%+ predictions held up on the latest games it never saw."""
        return self.calibrated(s) and bool(getattr(self, "health", {}).get(s, {}).get("ok"))

    # ------------------------------------------------------------ learning
    def _update(self, r: dict, when: datetime) -> None:
        pid = r["player_id"]
        p = self.players.get(pid)
        if p is None:
            p = self.players[pid] = _Player()
        if p.last is not None and (when - p.last).days > 60:  # offseason: older games count half
            for s in PROP_STATS:
                p.S[s] *= 0.5
            p.M *= 0.5
        dec = self._decay()
        mins = r["min"] or 0.0
        p.team, p.group, p.name = r["team"], _group(r["pos"]), r["player"]
        for s in PROP_STATS:
            v = r.get(s) or 0.0
            p.S[s] = p.S[s] * dec + v
            p.avg[s] = v if p.games == 0 else 0.85 * p.avg[s] + 0.15 * v
            self.league_S[(p.group, s)] = self.league_S[(p.group, s)] * 0.9995 + v
            self.opp_S[(r["opp"], s)] = self.opp_S[(r["opp"], s)] * 0.999 + v
        p.M = p.M * dec + mins
        self.league_M[p.group] = self.league_M[p.group] * 0.9995 + mins
        self.opp_M[r["opp"]] = self.opp_M[r["opp"]] * 0.999 + mins
        p.min_ewma = mins if p.min_ewma is None else 0.7 * p.min_ewma + 0.3 * mins
        p.games += 1
        p.last = when

    def _team_scores(self, day_rows: list[dict]) -> None:
        tot = defaultdict(float)
        for r in day_rows:
            tot[(r["game_id"], r["team"], r["opp"])] += r.get("pts") or 0.0
        for (gid, team, opp), pts in tot.items():
            allowed = tot.get((gid, opp, team))
            if allowed is None:
                continue
            cur = self.team_pts.get(team)
            self.team_pts[team] = [pts, allowed] if cur is None else [0.9 * cur[0] + 0.1 * pts,
                                                                     0.9 * cur[1] + 0.1 * allowed]

    def fit(self, rows: list[dict]) -> "PropModel":
        """Walk forward through box-score rows (chronological): predict, then learn, day by day."""
        by_day = defaultdict(list)
        for r in rows:
            by_day[r["date"][:10]].append(r)
        for day in sorted(by_day):
            day_rows = by_day[day]
            for r in day_rows:
                pid = r["player_id"]
                for s in PROP_STATS:
                    mu = self.mean(pid, s, r["opp"])
                    line = self.line_for(pid, s)
                    if mu is not None and line is not None and r.get(s) is not None:
                        self.oos[s].append((mu, line, r[s]))
            when = datetime.fromisoformat(day_rows[0]["date"])
            for r in day_rows:
                self._update(r, when)
            self._team_scores(day_rows)
            self.n_games += len({r["game_id"] for r in day_rows})
        self._fit_distributions()
        return self

    def _fit_distributions(self) -> None:
        for s in PROP_STATS:
            rows = self.oos.get(s, [])[-30000:]
            if len(rows) < 300:
                continue
            mu = np.array([r[0] for r in rows]); x = np.array([r[2] for r in rows])

            def nll(lphi):
                phi = math.exp(lphi)
                return -np.sum(gammaln(x + phi) - gammaln(phi) - gammaln(x + 1)
                               + phi * np.log(phi / (phi + mu)) + x * np.log(mu / (phi + mu)))
            self.phi[s] = float(math.exp(minimize_scalar(nll, bounds=(math.log(0.5), math.log(500)),
                                                         method="bounded").x))
            # calibrate across the whole range of lines books offer: at the player's usual level and
            # well below / above it (alternate lines and "X+" milestones), not just near the median
            base = np.array([r[1] for r in rows])
            d = np.maximum(1.0, np.round(0.35 * base))
            L = np.concatenate([base + k * d for k in (-2, -1, 0, 1, 2)])
            M = np.tile(mu, 5); Xs = np.tile(x, 5)
            keep = L >= 0.5
            L, M, Xs = L[keep], M[keep], Xs[keep]
            y = (Xs > L).astype(float)
            phi = self.phi[s]
            praw = np.clip(nbinom.sf(np.floor(L), phi, phi / (phi + M)), 1e-6, 1 - 1e-6)
            # beta calibration (Kull et al. 2017): logit p* = c + a ln p - b ln(1-p); bends each tail.
            # With enough rows, isotonic regression on top corrects any remaining monotone distortion
            # (points: minutes swings make the count distribution's shape wrong at the high end).
            rec = np.tile(np.arange(len(mu)), 5)[keep]
            cut = int(0.8 * len(mu))
            train = rec < cut  # reliability is reported on the latest 20%, never seen
            # recent games count more: roles and minutes change, especially across a summer
            w_tr = 0.5 ** ((cut - rec[train]) / max(0.25 * cut, 1.0))
            self.cal[s] = self._fit_beta(praw[train], y[train], w_tr)
            iso = None
            if train.sum() >= 15000:
                iso = IsotonicCalibrator().fit(self._beta(self.cal[s], praw[train]), y[train], w_tr)
            pc = self._beta(self.cal[s], praw[~train])
            if iso is not None:
                pc = np.clip(iso.transform(pc), 0.005, 0.995)
            yh = y[~train]
            bins = []
            for lo, hi in ((0.0, 0.2), (0.2, 0.35), (0.35, 0.5), (0.5, 0.65), (0.65, 0.8), (0.8, 0.9), (0.9, 1.01)):
                m = (pc >= lo) & (pc < hi)
                if m.sum() >= 30:
                    bins.append({"range": f"{lo:.2f}-{min(hi, 1):.2f}", "n": int(m.sum()),
                                 "said": round(float(pc[m].mean()), 4), "hit": round(float(yh[m].mean()), 4)})
            # health gate: legs need 60%+ predictions that hit within 3 points on the unseen latest games
            gaps = [b["said"] - b["hit"] for b in bins if b["said"] >= 0.6 and b["n"] >= 100]
            worst = max([abs(g) for g in gaps], default=None)
            self.health[s] = {"ok": worst is not None and worst <= 0.03,
                              "worst_gap_60plus": None if worst is None else round(worst, 4)}
            # final maps on everything, recent games weighted up
            w = 0.5 ** ((len(mu) - rec) / max(0.25 * len(mu), 1.0))
            self.cal[s] = self._fit_beta(praw, y, w)
            if iso is not None:
                self.iso[s] = IsotonicCalibrator().fit(self._beta(self.cal[s], praw), y, w)
            self.reliability[s] = bins

    @staticmethod
    def _fit_beta(praw: np.ndarray, y: np.ndarray, sw: Optional[np.ndarray] = None) -> tuple:
        F = np.column_stack([np.ones_like(praw), np.log(praw), -np.log(1 - praw)])
        sw = np.ones_like(y) if sw is None else sw
        prior = np.array([0.0, 1.0, 1.0])
        f = lambda w: np.sum(sw * (np.logaddexp(0, F @ w) - y * (F @ w))) + 0.5 * np.sum((w - prior)[1:] ** 2)
        g = lambda w: F.T @ (sw * (1 / (1 + np.exp(-(F @ w))) - y)) + np.concatenate([[0.0], (w - prior)[1:]])
        return tuple(float(v) for v in minimize(f, prior.copy(), jac=g, method="BFGS").x)

    @staticmethod
    def _beta(c: tuple, p: np.ndarray) -> np.ndarray:
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return 1 / (1 + np.exp(-(c[0] + c[1] * np.log(p) - c[2] * np.log(1 - p))))

    def observe(self, rows: list[dict]) -> None:
        """Learn new box scores (already-seen rows are the caller's to filter)."""
        self.fit(rows)

    def status(self) -> dict:
        return {"games": self.n_games, "players": len(self.players),
                "stats": {s: {"oos_rows": len(self.oos.get(s, [])), "dispersion": round(self.phi.get(s, 0), 2),
                              "calibrated": self.calibrated(s), "trusted": self.trusted(s),
                              "health": getattr(self, "health", {}).get(s),
                              "reliability": self.reliability.get(s, [])}
                          for s in PROP_STATS}}
