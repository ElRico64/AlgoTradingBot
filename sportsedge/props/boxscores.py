"""Player box scores (NBA, NHL, NFL) from ESPN's free game pages, kept as growing local datasets.

Each finished game's page (`summary?event=<id>`) lists every player's line. Rows are
stored in data/players/<LEAGUE>.csv; data/players/<LEAGUE>_events.json remembers which
games are done and which days were scanned for game ids. Downloading is time-budgeted
and runs newest-first, so the current season is ready first and older seasons fill in
over the following updates.
"""
from __future__ import annotations

import csv
import json
import logging
import os
import time
from datetime import date, datetime, timedelta
from typing import Optional

from ..data.official import canonical_id

log = logging.getLogger(__name__)

STATS = ("min", "pts", "reb", "ast", "fg3m", "stl", "blk", "tov")
FIELDS = ("game_id", "espn_id", "date", "team", "opp", "home", "player_id", "player", "pos", "starter") + STATS
# ESPN box-score columns (labels and keys vary a little between seasons)
_LABELS = {"MIN": "min", "PTS": "pts", "REB": "reb", "AST": "ast", "3PT": "fg3m", "STL": "stl", "BLK": "blk",
           "TO": "tov"}
_KEYS = {"minutes": "min", "points": "pts", "rebounds": "reb", "totalRebounds": "reb", "assists": "ast",
         "threePointFieldGoalsMade-threePointFieldGoalsAttempted": "fg3m", "steals": "stl", "blocks": "blk",
         "turnovers": "tov"}


def _num(v) -> Optional[float]:
    if v is None:
        return None
    t = str(v).strip()
    if not t or t in ("--", "-", "DNP"):
        return None
    if "-" in t[1:]:          # "4-9" made-attempted: keep made
        t = t.split("-")[0]
    if ":" in t:              # "35:12" minutes
        m, sec = t.split(":")[:2]
        try:
            return float(m) + float(sec) / 60
        except ValueError:
            return None
    try:
        return float(t.replace("+", ""))
    except ValueError:
        return None


# ---- NHL: skaters (time on ice, goals, assists, shots, +/-, ...) and goalies (shots, goals against)
NHL_STATS = ("toi", "g", "a", "s", "pm", "bs", "ht", "tk", "gv", "sa", "ga", "sv")
_NHL_LABELS = {"TOI": "toi", "G": "g", "A": "a", "S": "s", "SOG": "s", "+/-": "pm", "BS": "bs", "HT": "ht",
               "TK": "tk", "GV": "gv", "SA": "sa", "GA": "ga", "SV": "sv", "SAVES": "sv"}
_NHL_KEYS = {"timeOnIce": "toi", "goals": "g", "assists": "a", "shotsTotal": "s", "shots": "s", "plusMinus": "pm",
             "blockedShots": "bs", "hits": "ht", "takeaways": "tk", "giveaways": "gv", "shotsAgainst": "sa",
             "goalsAgainst": "ga", "saves": "sv"}
# ---- NFL: passing, rushing, receiving lines merged per player
NFL_STATS = ("cmp", "att", "pass_yds", "pass_td", "int", "sacks", "sack_yds", "rush_att", "rush_yds", "rush_td",
             "rec", "rec_yds", "rec_td", "tgt")
_NFL_GROUPS = {
    "passing": {"C/ATT": ("cmp", "att"), "YDS": "pass_yds", "TD": "pass_td", "INT": "int", "SACKS": ("sacks", "sack_yds")},
    "rushing": {"CAR": "rush_att", "YDS": "rush_yds", "TD": "rush_td"},
    "receiving": {"REC": "rec", "YDS": "rec_yds", "TD": "rec_td", "TGTS": "tgt"},
}
LEAGUE_STATS = {"NBA": STATS, "NHL": NHL_STATS, "NFL": NFL_STATS}


def fields(league: str = "NBA") -> tuple:
    return ("game_id", "espn_id", "date", "team", "opp", "home", "player_id", "player", "pos", "starter") + \
        LEAGUE_STATS[league]


def _pair(v) -> tuple[Optional[float], Optional[float]]:
    """"22/31" or "2-14" -> (22, 31) / (2, 14)."""
    t = str(v or "").strip()
    for sep in ("/", "-"):
        if sep in t[1:] and not t.startswith("-"):
            a, b = t.split(sep, 1)
            try:
                return float(a), float(b)
            except ValueError:
                return None, None
    return _num(t), None


def _header(data: dict, league: str, aliases: dict):
    comp = ((data.get("header") or {}).get("competitions") or [{}])[0]
    teams = {c.get("homeAway"): aliases.get((c.get("team") or {}).get("abbreviation"),
                                            (c.get("team") or {}).get("abbreviation"))
             for c in comp.get("competitors") or []}
    if "home" not in teams or "away" not in teams or not comp.get("date"):
        return None
    try:
        start = datetime.fromisoformat(comp["date"].replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None
    status = ((comp.get("status") or {}).get("type") or {})
    if status and not status.get("completed", False):
        return None
    gid = canonical_id(league, start, teams["home"], teams["away"])
    eid = str((data.get("header") or {}).get("id") or comp.get("id") or "")
    return teams, start, gid, eid


def _row(gid, eid, start, team, teams, ath: dict, pos: str, starter: bool, stats: dict, league: str) -> dict:
    opp = teams["away"] if team == teams["home"] else teams["home"]
    return {"game_id": gid, "espn_id": eid, "date": start.isoformat(), "team": team, "opp": opp,
            "home": int(team == teams["home"]), "player_id": str(ath.get("id") or ""),
            "player": ath.get("displayName") or ath.get("shortName") or "", "pos": pos, "starter": int(starter),
            **{s: stats.get(s) for s in LEAGUE_STATS[league]}}


def _parse_nhl(data: dict, aliases: dict) -> list[dict]:
    h = _header(data, "NHL", aliases)
    if h is None:
        return []
    teams, start, gid, eid = h
    rows = []
    for block in (data.get("boxscore") or {}).get("players") or []:
        team = aliases.get((block.get("team") or {}).get("abbreviation"), (block.get("team") or {}).get("abbreviation"))
        if team not in (teams["home"], teams["away"]):
            continue
        for st in block.get("statistics") or []:
            keys, labels = st.get("keys") or [], st.get("labels") or st.get("names") or []
            cols = [(_NHL_KEYS.get(keys[i]) if i < len(keys) else None) or
                    (_NHL_LABELS.get(str(labels[i]).upper()) if i < len(labels) else None)
                    for i in range(max(len(keys), len(labels)))]
            goalie = "goal" in str(st.get("name") or st.get("type") or "").lower() or ("sa" in cols and "g" not in cols)
            if "toi" not in cols:
                continue
            for a in st.get("athletes") or []:
                vals = a.get("stats") or []
                rec = {c: _num(vals[i]) for i, c in enumerate(cols) if c and i < len(vals)}
                if goalie and rec.get("sv") is None and rec.get("sa") is not None and rec.get("ga") is not None:
                    rec["sv"] = rec["sa"] - rec["ga"]
                if goalie and rec.get("sa") is None and rec.get("sv") is not None and rec.get("ga") is not None:
                    rec["sa"] = rec["sv"] + rec["ga"]
                if not rec.get("toi"):
                    continue
                ath = a.get("athlete") or {}
                pos = "G" if goalie else ((ath.get("position") or {}).get("abbreviation") or "")
                rows.append(_row(gid, eid, start, team, teams, ath, pos, bool(a.get("starter")), rec, "NHL"))
    return [r for r in rows if r["player_id"]]


def _parse_nfl(data: dict, aliases: dict) -> list[dict]:
    h = _header(data, "NFL", aliases)
    if h is None:
        return []
    teams, start, gid, eid = h
    merged: dict = {}
    for block in (data.get("boxscore") or {}).get("players") or []:
        team = aliases.get((block.get("team") or {}).get("abbreviation"), (block.get("team") or {}).get("abbreviation"))
        if team not in (teams["home"], teams["away"]):
            continue
        for st in block.get("statistics") or []:
            spec = _NFL_GROUPS.get(str(st.get("name") or "").lower())
            if not spec:
                continue
            labels = [str(x).upper() for x in (st.get("labels") or [])]
            for a in st.get("athletes") or []:
                ath = a.get("athlete") or {}
                pid = str(ath.get("id") or "")
                if not pid:
                    continue
                vals = a.get("stats") or []
                rec = merged.setdefault((team, pid), {"ath": ath, "stats": {}})["stats"]
                for i, lab in enumerate(labels):
                    tgt = spec.get(lab)
                    if tgt is None or i >= len(vals):
                        continue
                    if isinstance(tgt, tuple):
                        x, y = _pair(vals[i])
                        rec[tgt[0]], rec[tgt[1]] = x, y
                    else:
                        rec[tgt] = _num(vals[i]) if "/" not in str(vals[i]) else _pair(vals[i])[0]
    rows = []
    for (team, pid), v in merged.items():
        st = v["stats"]
        pos = ((v["ath"].get("position") or {}).get("abbreviation") or "")
        if not pos:
            pos = "QB" if (st.get("att") or 0) >= 5 else ""
        rows.append(_row(gid, eid, start, team, teams, v["ath"], pos, False, st, "NFL"))
    return rows


def parse_boxscore(data: dict, aliases: Optional[dict] = None, league: str = "NBA") -> list[dict]:
    """Player rows from an ESPN game summary. Players who did not play are skipped."""
    if league == "NHL":
        return _parse_nhl(data, aliases or {})
    if league == "NFL":
        return _parse_nfl(data, aliases or {})
    aliases = aliases or {}
    comp = ((data.get("header") or {}).get("competitions") or [{}])[0]
    teams = {c.get("homeAway"): aliases.get((c.get("team") or {}).get("abbreviation"),
                                            (c.get("team") or {}).get("abbreviation"))
             for c in comp.get("competitors") or []}
    if "home" not in teams or "away" not in teams or not comp.get("date"):
        return []
    try:
        start = datetime.fromisoformat(comp["date"].replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return []
    status = ((comp.get("status") or {}).get("type") or {})
    if status and not status.get("completed", False):
        return []
    gid = canonical_id("NBA", start, teams["home"], teams["away"])
    eid = str((data.get("header") or {}).get("id") or comp.get("id") or "")
    rows = []
    for block in (data.get("boxscore") or {}).get("players") or []:
        team = aliases.get((block.get("team") or {}).get("abbreviation"), (block.get("team") or {}).get("abbreviation"))
        if team not in (teams["home"], teams["away"]):
            continue
        opp = teams["away"] if team == teams["home"] else teams["home"]
        for st in block.get("statistics") or []:
            cols = []
            keys, labels = st.get("keys") or [], st.get("labels") or st.get("names") or []
            for i in range(max(len(keys), len(labels))):
                k = _KEYS.get(keys[i]) if i < len(keys) else None
                k = k or (_LABELS.get(str(labels[i]).upper()) if i < len(labels) else None)
                cols.append(k)
            if "pts" not in cols or "min" not in cols:
                continue
            for a in st.get("athletes") or []:
                if a.get("didNotPlay"):
                    continue
                vals = a.get("stats") or []
                rec = {c: _num(vals[i]) for i, c in enumerate(cols) if c and i < len(vals)}
                if not rec.get("min"):
                    continue
                ath = a.get("athlete") or {}
                rows.append({"game_id": gid, "espn_id": eid, "date": start.isoformat(), "team": team, "opp": opp,
                             "home": int(team == teams["home"]), "player_id": str(ath.get("id") or ""),
                             "player": ath.get("displayName") or ath.get("shortName") or "",
                             "pos": ((ath.get("position") or {}).get("abbreviation") or ""),
                             "starter": int(bool(a.get("starter"))),
                             **{s: rec.get(s) for s in STATS}})
    return [r for r in rows if r["player_id"]]


# ------------------------------------------------------------------ storage
def _paths(data_dir: str, league: str = "NBA") -> tuple[str, str]:
    d = os.path.join(data_dir, "players")
    return os.path.join(d, f"{league}.csv"), os.path.join(d, f"{league}_events.json")


def load_rows(data_dir: str, league: str = "NBA") -> list[dict]:
    path, _ = _paths(data_dir, league)
    if not os.path.exists(path):
        return []
    out = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            for s in LEAGUE_STATS[league]:
                r[s] = float(r[s]) if r.get(s) not in (None, "") else None
            r["home"], r["starter"] = int(r["home"]), int(r["starter"])
            out.append(r)
    out.sort(key=lambda r: r["date"])
    return out


def _load_registry(data_dir: str, league: str = "NBA") -> dict:
    _, p = _paths(data_dir, league)
    try:
        with open(p) as f:
            reg = json.load(f)
    except (OSError, ValueError):
        reg = {}
    reg.setdefault("done", {})
    reg.setdefault("days", {})
    return reg


def _save(data_dir: str, new_rows: list[dict], reg: dict, league: str = "NBA") -> None:
    path, rp = _paths(data_dir, league)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    exists = os.path.exists(path)
    cols = fields(league)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if not exists:
            w.writeheader()
        w.writerows({k: ("" if r.get(k) is None else r[k]) for k in cols} for r in new_rows)
    tmp = rp + ".tmp"
    with open(tmp, "w") as f:
        json.dump(reg, f, sort_keys=True)
    os.replace(tmp, rp)


def _save_sample(data_dir: str, league: str, data: dict) -> None:
    try:
        d = os.path.join(data_dir, "players", "samples")
        os.makedirs(d, exist_ok=True)
        keep = {"header": data.get("header"), "boxscore": {"players": (data.get("boxscore") or {}).get("players")}}
        with open(os.path.join(d, f"{league}_summary.json"), "w") as f:
            json.dump(keep, f, indent=0)
    except (OSError, TypeError, ValueError):
        pass


# ------------------------------------------------------------------ downloading
def update_boxscores(data_dir: str, game_days: list[date], today: date, budget_seconds: float = 150.0,
                     fetch_day=None, fetch_summary=None, league: str = "NBA") -> dict:
    """Download box scores for finished games, newest day first, within a time budget.
    `game_days`: days with games (from the results history). Returns a status dict."""
    from ..data import espn
    from ..news.feeds import ESPN_BASE, SPORT_PATH, _get_json

    def _day(d):
        games = espn.fetch_day(league, d)
        return None if espn.last_day_failed else games
    fetch_day = fetch_day or _day
    fetch_summary = fetch_summary or (lambda eid: _get_json(f"{ESPN_BASE}/{SPORT_PATH[league]}/summary?event={eid}"))
    aliases = espn.ABBR_ALIASES.get(league, {})
    reg = _load_registry(data_dir, league)
    sample_saved = False
    t0, new_rows, games_done, errors = time.time(), [], 0, 0
    days = sorted({d for d in game_days if d < today} | {today - timedelta(days=k) for k in (1, 2)}, reverse=True)
    for d in days:
        if time.time() - t0 > budget_seconds:
            break
        key = d.isoformat()
        ids = reg["days"].get(key)
        if ids is None or (today - d).days <= 2:  # recent days: rescan, games may have just finished
            games = fetch_day(d)
            if games is None:
                errors += 1
                continue
            ids = [str((g.extras or {}).get("espn_id")) for g in games
                   if (g.extras or {}).get("espn_id") and ((g.extras or {}).get("status") or {}).get("state", "post") == "post"]
            reg["days"][key] = ids
        for eid in ids:
            if eid in reg["done"] or time.time() - t0 > budget_seconds:
                continue
            data = fetch_summary(eid)
            if not data:
                errors += 1
                continue
            if league != "NBA" and not sample_saved:  # one raw page per update, to check the format by eye
                _save_sample(data_dir, league, data)
                sample_saved = True
            rows = parse_boxscore(data, aliases, league)
            if rows:
                new_rows.extend(rows)
                reg["done"][eid] = rows[0]["game_id"]
                games_done += 1
    if new_rows or reg:
        _save(data_dir, new_rows, reg, league)
    return {"new_games": games_done, "new_rows": len(new_rows), "errors": errors,
            "games_stored": len(reg["done"]), "days_scanned": len(reg["days"]),
            "seconds": round(time.time() - t0, 1)}
