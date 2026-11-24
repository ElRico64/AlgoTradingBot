"""NBA player values from box scores, and what a missing player costs.

Value model (a box-score-informed regularised adjusted plus-minus):

    margin_g = a + sum_i s_gi (m_gi / 48) v_i + e_g,     s = +1 home, -1 away

    v_i = z_i' gamma + u_i,   u_i ~ N(0, sigma^2 / lambda)

* m_gi: minutes player i played in game g; v_i: his net points per 48 minutes
  against an average player.
* z_i: his box-score rates per 36 minutes (points, rebounds, assists, steals,
  blocks, turnovers, threes), shrunk toward the league rate by his minutes.
* gamma: learned jointly (hierarchical prior). A player with few games is
  valued from what his box score says such players are worth; a long record of
  team results with and without him moves him away from that.
* Games are weighted by recency (half-life HALF_LIFE_DAYS). The penalised normal
  equations are solved exactly.

Team strength for a set of players with expected minutes m_i, rescaled to the
240 minutes a game has:

    S = sum_i (240 m_i / sum m) / 48 * v_i

A player's cost when out is S(with him) - S(without him): his minutes go to
teammates in proportion to theirs. Part of that is already in the team's
rating when he has missed recent games, so the cost is scaled by the share of
the team's recent games he played.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable, Optional

import numpy as np

BOX_STATS = ("pts", "reb", "ast", "stl", "blk", "tov", "fg3m")
HALF_LIFE_DAYS = 365.0
LAMBDA = 30.0          # ridge on the player-specific part (u), in units of game-weight
LAMBDA_BOX = 1.0       # light ridge on the box-score weights (gamma)
K_MINUTES = 300.0      # pseudo-minutes at the league rate for a player's box-score rates
MIN_EWMA = 0.3         # weight of the newest game in expected minutes
DEFAULT_MINUTES = 12.0


@dataclass
class LineupGame:
    game_id: str
    when: datetime
    home: str
    away: str
    margin: float
    minutes: dict  # team -> {player_id: minutes}


def build_games(rows: Iterable[dict]) -> list[LineupGame]:
    """Box-score rows -> games with each team's players and minutes; margin from summed points."""
    g: dict = {}
    for r in rows:
        x = g.get(r["game_id"])
        if x is None:
            x = g[r["game_id"]] = {"when": r["date"], "teams": {}, "home": None, "away": None, "pts": defaultdict(float)}
        t = r["team"]
        x["teams"].setdefault(t, {})[r["player_id"]] = float(r.get("min") or 0.0)
        x["pts"][t] += float(r.get("pts") or 0.0)
        if int(r.get("home") or 0):
            x["home"], x["away"] = t, r["opp"]
        else:
            x["home"], x["away"] = r["opp"], t
    out = []
    for gid, x in g.items():
        h, a = x["home"], x["away"]
        if h not in x["teams"] or a not in x["teams"]:
            continue
        when = x["when"] if isinstance(x["when"], datetime) else datetime.fromisoformat(str(x["when"]).replace("Z", ""))
        out.append(LineupGame(gid, when, h, a, x["pts"][h] - x["pts"][a], x["teams"]))
    out.sort(key=lambda z: (z.when, z.game_id))
    return out


@dataclass
class PlayerValues:
    values: dict = field(default_factory=dict)   # player_id -> net points per 48 vs average
    gamma: Optional[np.ndarray] = None
    home_adv: float = 0.0
    fitted_through: Optional[datetime] = None
    n_games: int = 0

    def get(self, pid: str) -> float:
        return self.values.get(pid, getattr(self, "_default", 0.0))

    @staticmethod
    def _box_features(rows: list[dict], before: Optional[datetime]) -> tuple[dict, np.ndarray]:
        S = defaultdict(lambda: np.zeros(len(BOX_STATS)))
        M = defaultdict(float)
        for r in rows:
            if before is not None and r["_when"] >= before:
                continue
            m = float(r.get("min") or 0.0)
            if m <= 0:
                continue
            S[r["player_id"]] += np.array([float(r.get(s) or 0.0) for s in BOX_STATS])
            M[r["player_id"]] += m
        tot_m = sum(M.values()) or 1.0
        league = sum(S.values(), np.zeros(len(BOX_STATS))) / tot_m
        z = {pid: 36.0 * (S[pid] + K_MINUTES * league) / (M[pid] + K_MINUTES) for pid in S}
        return z, 36.0 * league

    def fit(self, games: list[LineupGame], rows: list[dict], before: Optional[datetime] = None,
            lam: float = LAMBDA, half_life_days: float = HALF_LIFE_DAYS) -> "PlayerValues":
        """Fit on games strictly before `before` (all games when None)."""
        for r in rows:
            if "_when" not in r:
                d = r["date"]
                r["_when"] = d if isinstance(d, datetime) else datetime.fromisoformat(str(d).replace("Z", ""))
        gs = [g for g in games if before is None or g.when < before]
        if len(gs) < 100:
            return self
        z, zbar = self._box_features(rows, before)
        pids = sorted({p for g in gs for t in (g.home, g.away) for p in g.minutes[t]})
        idx = {p: k for k, p in enumerate(pids)}
        P, K = len(pids), len(BOX_STATS)
        Z = np.array([(z.get(p, zbar) - zbar) / np.maximum(zbar, 1e-6) for p in pids])  # relative rates
        last = max(g.when for g in gs)
        w = np.array([0.5 ** ((last - g.when).days / half_life_days) for g in gs])
        X = np.zeros((len(gs), P))
        for r, g in enumerate(gs):
            for t, s in ((g.home, 1.0), (g.away, -1.0)):
                for p, m in g.minutes[t].items():
                    X[r, idx[p]] += s * m / 48.0
        y = np.array([g.margin for g in gs])
        # unknowns: [a, gamma (K), u (P)];  v = Z gamma + u
        D = np.hstack([np.ones((len(gs), 1)), X @ Z, X])
        pen = np.concatenate([[1e-6], np.full(K, LAMBDA_BOX), np.full(P, lam)])
        A = (D * w[:, None]).T @ D + np.diag(pen)
        b = (D * w[:, None]).T @ y
        theta = np.linalg.solve(A, b)
        a, gamma, u = theta[0], theta[1:1 + K], theta[1 + K:]
        v = Z @ gamma + u
        self.values = {p: float(v[idx[p]]) for p in pids}
        self.gamma, self.home_adv, self.fitted_through, self.n_games = gamma, float(a), last, len(gs)
        # a player never seen: valued like a fringe player (the 25th percentile)
        self._default = float(np.percentile(v, 25)) if len(v) else 0.0
        return self


def strength(values: PlayerValues, minutes: dict) -> float:
    """Team strength (points per game vs an average team) of players with expected minutes."""
    tot = sum(minutes.values())
    if tot <= 0:
        return 0.0
    return sum((240.0 * m / tot) / 48.0 * values.get(p) for p, m in minutes.items())


COST_MODE = "share"  # "bench" was tested on real games and predicted worse (see METHODOLOGY 8e)
BENCH_MINUTES = 24.0


def cost_when_out(values: PlayerValues, minutes: dict, pid: str, mode: Optional[str] = None) -> float:
    """The margin a team loses when `pid` sits.

    "share" (default): his minutes are shared by all teammates in proportion to theirs.
    "bench": his minutes go to the players behind him in the rotation (expected minutes under
    BENCH_MINUTES): (v_i - v_bench) * m_i / 48. Tested on 1,585 real games: it predicted worse
    (log-loss 0.598 vs 0.592), so it is kept only for comparison."""
    if pid not in minutes:
        return 0.0
    rest = {p: m for p, m in minutes.items() if p != pid}
    if (mode or COST_MODE) == "bench":
        bench = {p: m for p, m in rest.items() if m < BENCH_MINUTES}
        tot = sum(bench.values())
        if tot > 0:
            v_bench = sum(m * values.get(p) for p, m in bench.items()) / tot
            return (values.get(pid) - v_bench) * minutes[pid] / 48.0
    return strength(values, minutes) - strength(values, rest)


class MinutesTracker:
    """Expected minutes and recent availability per player, walked forward game by game."""

    def __init__(self, half_life_games: float = 8.0):
        self.min_ewma: dict = {}
        self.team: dict = {}
        self.name: dict = {}
        self.played_share: dict = {}      # EW share of the team's games he played
        self.team_games: dict = defaultdict(int)
        self.d = 0.5 ** (1.0 / half_life_games)

    def expected(self, pid: str) -> float:
        return self.min_ewma.get(pid, DEFAULT_MINUTES)

    def roster(self, team: str, min_share: float = 0.25) -> dict:
        """Players currently with `team` who play: {player_id: expected minutes}."""
        return {p: self.min_ewma[p] for p, t in self.team.items()
                if t == team and self.played_share.get(p, 0.0) >= min_share and self.min_ewma.get(p, 0) > 0}

    def update(self, g: LineupGame, names: Optional[dict] = None) -> None:
        for t in (g.home, g.away):
            played = g.minutes[t]
            for p, m in played.items():
                if self.team.get(p) != t:  # new team: start over
                    self.min_ewma[p], self.played_share[p] = m, 1.0
                else:
                    self.min_ewma[p] = (1 - MIN_EWMA) * self.min_ewma[p] + MIN_EWMA * m
                    self.played_share[p] = self.d * self.played_share[p] + (1 - self.d)
                self.team[p] = t
                if names and p in names:
                    self.name[p] = names[p]
            for p, tm in self.team.items():
                if tm == t and p not in played:
                    self.played_share[p] = self.d * self.played_share.get(p, 0.0)
            self.team_games[t] += 1


def absence_features(rows: list[dict], refit_days: int = 14) -> dict:
    """Walk-forward absence cost per past game, home perspective (positive favours home):

        x_g = sum_away cost_out(i) * share(i)  -  sum_home cost_out(i) * share(i)

    over each team's current regulars who did not play in game g. Player values are refitted
    every `refit_days` on earlier games only; expected minutes and recent availability are taken
    before the game. This is what a pregame injury report tells (who sits), priced by value."""
    from datetime import timedelta
    from itertools import groupby

    games = build_games(rows)
    mt = MinutesTracker()
    vals, next_fit, out = PlayerValues(), None, {}
    for day, it in groupby(games, key=lambda g: g.when.date()):
        day_games = list(it)
        when = day_games[0].when.replace(hour=0, minute=0, second=0, microsecond=0)
        if next_fit is None or when >= next_fit:
            vals = PlayerValues().fit(games, rows, before=when)
            next_fit = when + timedelta(days=refit_days)
        if vals.n_games:
            for g in day_games:
                x = 0.0
                for t, sign in ((g.home, -1.0), (g.away, 1.0)):
                    roster = mt.roster(t)
                    x += sign * sum(cost_when_out(vals, roster, p) * mt.played_share.get(p, 0.0)
                                    for p in roster if p not in g.minutes[t])
                out[g.game_id] = x
        for g in day_games:
            mt.update(g)
    return out


def player_impacts(rows: list[dict], scale: float = 1.0, as_of: Optional[datetime] = None) -> list[dict]:
    """What each current player's absence costs his team (margin points), for the injury model.

    impact = scale * cost_when_out * played_share: a player who has been missing for weeks is
    already priced into the team's rating, so only the part not yet reflected counts."""
    games = build_games(rows)
    if as_of is not None:
        games = [g for g in games if g.when < as_of]
    vals = PlayerValues().fit(games, rows)
    names = {r["player_id"]: r["player"] for r in rows}
    mt = MinutesTracker()
    for g in games:
        mt.update(g, names)
    out = []
    for team in sorted(set(mt.team.values())):
        roster = mt.roster(team)
        for p in roster:
            c = cost_when_out(vals, roster, p) * mt.played_share.get(p, 0.0) * scale
            if c > 0.05:
                out.append({"league": "NBA", "team": team, "player": mt.name.get(p, p), "player_id": p,
                            "impact": round(c, 2), "impact_sd": round(max(0.3, 0.35 * c), 2),
                            "value_per48": round(vals.get(p), 2), "minutes": round(roster[p], 1)})
    return sorted(out, key=lambda x: -x["impact"])
