"""NFL player values from box scores: quarterbacks first, then running backs and receivers.

NFL teams play 17 games, so a game-by-game plus-minus like the NBA's cannot separate 53
players. What box scores do measure well is the quarterback, the one player whose
absence moves a line by a touchdown, and the production of the ball carriers and
receivers.

* Quarterback: adjusted net yards per dropback,
      ANY/A = (pass yards + 20 TD - 45 INT - sack yards) / (attempts + sacks),
  shrunk toward a below-average prior (QB_PRIOR_DROPBACKS dropbacks at the league
  rate minus 1.0): a passer with a short record is assumed to be a backup until shown
  otherwise. His rushing yards per game count too, shrunk the same way.
  Cost of the usual starter not playing:
      POINTS_PER_YARD * [(ANY/A_starter - ANY/A_backup) * dropbacks per game
                         + (rush yards/game_starter - rush yards/game_backup)]
  The backup is the team's next quarterback by recent starts, or a replacement-level
  passer when none has played.
* Running backs and receivers: production per touch, (yards + 20 TD) / (carries +
  targets), shrunk toward the league rate. A replacement gets the same touches at
  REPLACEMENT of the league's production per touch:
      cost = POINTS_PER_YARD * (per-touch_i - REPLACEMENT * league per-touch) * touches per game
* Linemen and defenders do not show in box scores; the injury model keeps its
  position defaults for them.

Costs are in points of margin. How much of them shows up in results is learned by the
engine from its own residuals (see engine.lineup_scale).
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from itertools import groupby
from typing import Optional

POINTS_PER_YARD = 0.065
QB_PRIOR_DROPBACKS = 250.0
QB_RUSH_PRIOR_GAMES = 4.0
QB_DECAY = 0.97          # per game he plays: older games count less
SKILL_PRIOR_TOUCHES = 60.0
SKILL_DECAY = 0.95
REPLACEMENT = 0.85
START_DECAY = 0.7        # per team game: who starts at quarterback now
APPEAR_DECAY = 0.75      # per team game: share of games a player appears in
MIN_DROPBACKS_START = 10
COVERED_ROLES = ("QB", "RB", "FB", "WR", "TE")


def _f(r: dict, k: str) -> float:
    v = r.get(k)
    return float(v) if v not in (None, "") else 0.0


def _games(rows: list[dict]) -> list[dict]:
    g: dict = {}
    for r in rows:
        x = g.setdefault(r["game_id"], {"id": r["game_id"], "date": r["date"], "teams": defaultdict(dict),
                                        "home": None, "away": None})
        x["teams"][r["team"]][r["player_id"]] = r
        if int(r.get("home") or 0):
            x["home"], x["away"] = r["team"], r["opp"]
        else:
            x["home"], x["away"] = r["opp"], r["team"]
    out = [x for x in g.values() if x["home"] in x["teams"] and x["away"] in x["teams"]]
    for x in out:
        d = x["date"]
        x["when"] = d if isinstance(d, datetime) else datetime.fromisoformat(str(d).replace("Z", ""))
    return sorted(out, key=lambda x: (x["when"], x["id"]))


def _is_qb_line(r: dict) -> bool:
    return r.get("pos") == "QB" or _f(r, "att") >= 5


class NFLTracker:
    def __init__(self):
        self.qb = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])      # net pass yards, dropbacks, rush yards, games
        self.skill = defaultdict(lambda: [0.0, 0.0, 0.0])        # production, touches, games
        self.lg_qb = [0.0, 0.0]
        self.lg_skill = [0.0, 0.0]
        self.start = defaultdict(dict)                            # team -> {qb: EW start share}
        self.appear = defaultdict(dict)                           # team -> {player: EW appearance share}
        self.dropbacks = {}                                       # team -> EW dropbacks per game
        self.name: dict = {}

    # ---- rates
    def lg_anya(self) -> float:
        return self.lg_qb[0] / self.lg_qb[1] if self.lg_qb[1] > 500 else 6.0

    def anya(self, pid: Optional[str]) -> float:
        prior = self.lg_anya() - 1.0
        if pid is None:
            return self.lg_anya() - 1.5  # replacement level
        s = self.qb[pid]
        return (s[0] + QB_PRIOR_DROPBACKS * prior) / (s[1] + QB_PRIOR_DROPBACKS)

    def qb_rush(self, pid: Optional[str]) -> float:
        if pid is None:
            return 8.0
        s = self.qb[pid]
        return (s[2] + QB_RUSH_PRIOR_GAMES * 10.0) / (s[3] + QB_RUSH_PRIOR_GAMES)

    def lg_ppt(self) -> float:
        return self.lg_skill[0] / self.lg_skill[1] if self.lg_skill[1] > 500 else 6.5

    def skill_value(self, pid: str) -> float:
        s = self.skill[pid]
        if s[2] <= 0:
            return 0.0
        ppt = (s[0] + SKILL_PRIOR_TOUCHES * self.lg_ppt()) / (s[1] + SKILL_PRIOR_TOUCHES)
        touches_pg = s[1] / s[2]
        return max(0.0, POINTS_PER_YARD * (ppt - REPLACEMENT * self.lg_ppt()) * touches_pg)

    # ---- people
    def usual_qb(self, team: str, min_share: float = 0.5) -> Optional[str]:
        sh = self.start.get(team) or {}
        if not sh:
            return None
        q = max(sh, key=sh.get)
        return q if sh[q] >= min_share else None

    def backup_qb(self, team: str) -> Optional[str]:
        sh = self.start.get(team) or {}
        u = self.usual_qb(team, 0.0)
        rest = {q: v for q, v in sh.items() if q != u and v > 0.02}
        return max(rest, key=rest.get) if rest else None

    def qb_cost(self, team: str, usual: Optional[str], other: Optional[str]) -> float:
        if usual is None:
            return 0.0
        db = self.dropbacks.get(team, 36.0)
        return POINTS_PER_YARD * ((self.anya(usual) - self.anya(other)) * db + (self.qb_rush(usual) - self.qb_rush(other)))

    def regulars(self, team: str) -> dict:
        """Skill players who appear in most of the team's games with real volume: {player: share}."""
        out = {}
        for p, sh in (self.appear.get(team) or {}).items():
            s = self.skill.get(p)
            if sh >= 0.5 and s is not None and s[2] > 0 and s[1] / s[2] >= 6:
                out[p] = sh
        return out

    @staticmethod
    def starter(lines: dict) -> Optional[str]:
        qbs = {p: _f(r, "att") + _f(r, "sacks") for p, r in lines.items() if _is_qb_line(r)}
        if not qbs:
            return None
        q = max(qbs, key=qbs.get)
        return q if qbs[q] >= MIN_DROPBACKS_START else None

    # ---- learning
    def update(self, team: str, lines: dict) -> None:
        st = self.starter(lines)
        sh = self.start[team]
        for q in list(sh):
            sh[q] *= START_DECAY
        if st is not None:
            sh[st] = sh.get(st, 0.0) + (1 - START_DECAY)
        ap = self.appear[team]
        for p in list(ap):
            ap[p] *= APPEAR_DECAY
        db_total = 0.0
        for p, r in lines.items():
            self.name[p] = r.get("player") or p
            if _is_qb_line(r):
                net = _f(r, "pass_yds") + 20 * _f(r, "pass_td") - 45 * _f(r, "int") - _f(r, "sack_yds")
                db = _f(r, "att") + _f(r, "sacks")
                s = self.qb[p]
                s[0], s[1] = s[0] * QB_DECAY + net, s[1] * QB_DECAY + db
                s[2], s[3] = s[2] * QB_DECAY + _f(r, "rush_yds"), s[3] * QB_DECAY + 1.0
                self.lg_qb[0] += net
                self.lg_qb[1] += db
                db_total += db
                continue
            touches = _f(r, "rush_att") + (_f(r, "tgt") or _f(r, "rec"))
            if touches <= 0:
                continue
            prod = _f(r, "rush_yds") + _f(r, "rec_yds") + 20 * (_f(r, "rush_td") + _f(r, "rec_td"))
            s = self.skill[p]
            s[0], s[1], s[2] = s[0] * SKILL_DECAY + prod, s[1] * SKILL_DECAY + touches, s[2] * SKILL_DECAY + 1.0
            self.lg_skill[0] += prod
            self.lg_skill[1] += touches
            ap[p] = ap.get(p, 0.0) + (1 - APPEAR_DECAY)
            for other, osh in self.appear.items():  # a player who changed teams leaves his old one
                if other != team and p in osh:
                    osh.pop(p)
        if db_total > 0:
            self.dropbacks[team] = db_total if team not in self.dropbacks else 0.8 * self.dropbacks[team] + 0.2 * db_total


def absence_features(rows: list[dict]) -> dict:
    """Walk-forward cost of who did not play, per past NFL game, home perspective (positive favours home):
    the usual quarterback replaced by the one who started, and regular backs / receivers missing."""
    tr = NFLTracker()
    out = {}
    games = _games(rows)
    for _, it in groupby(games, key=lambda g: g["when"].date()):
        day = list(it)
        for g in day:
            x = 0.0
            for t, sign in ((g["home"], -1.0), (g["away"], 1.0)):
                lines = g["teams"][t]
                u, st = tr.usual_qb(t), tr.starter(lines)
                cost = tr.qb_cost(t, u, st) if (u is not None and st is not None and st != u) else 0.0
                cost += sum(tr.skill_value(p) * sh for p, sh in tr.regulars(t).items() if p not in lines)
                x += sign * cost
            out[g["id"]] = x
        for g in day:
            for t in (g["home"], g["away"]):
                tr.update(t, g["teams"][t])
    return out


def player_impacts(rows: list[dict], scale: float = 1.0, as_of: Optional[datetime] = None) -> list[dict]:
    """Today's cost of each team's starting quarterback and regular backs / receivers being out."""
    tr = NFLTracker()
    for g in _games(rows):
        if as_of is not None and g["when"] >= as_of:
            continue
        for t in (g["home"], g["away"]):
            tr.update(t, g["teams"][t])
    out = []
    for team in sorted(tr.start):
        u = tr.usual_qb(team, 0.3)
        if u is not None:
            c = max(0.0, tr.qb_cost(team, u, tr.backup_qb(team))) * tr.start[team][u] * scale
            if c > 0.1:
                out.append({"league": "NFL", "team": team, "player": tr.name.get(u, u), "player_id": u,
                            "impact": round(c, 2), "impact_sd": round(max(0.5, 0.35 * c), 2),
                            "value_per48": round(tr.anya(u), 2), "minutes": round(tr.dropbacks.get(team, 36.0), 1),
                            "role": "QB"})
        for p, sh in tr.regulars(team).items():
            c = tr.skill_value(p) * sh * scale
            if c > 0.1:
                out.append({"league": "NFL", "team": team, "player": tr.name.get(p, p), "player_id": p,
                            "impact": round(c, 2), "impact_sd": round(max(0.3, 0.4 * c), 2),
                            "value_per48": round(c / max(scale, 1e-9), 2), "minutes": 0.0, "role": ""})
    return sorted(out, key=lambda x: -x["impact"])
