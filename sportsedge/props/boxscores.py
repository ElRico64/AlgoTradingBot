"""NBA player box scores from ESPN's free game pages, kept as a growing local dataset.

Each finished game's page (`summary?event=<id>`) lists every player's line. Rows are
stored in data/players/NBA.csv; data/players/NBA_events.json remembers which games
are done and which days were scanned for game ids. Downloading is time-budgeted and
runs newest-first, so the current season is ready first and older seasons fill in
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


def parse_boxscore(data: dict, aliases: Optional[dict] = None) -> list[dict]:
    """Player rows from an ESPN NBA game summary. Players who did not play are skipped."""
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
def _paths(data_dir: str) -> tuple[str, str]:
    d = os.path.join(data_dir, "players")
    return os.path.join(d, "NBA.csv"), os.path.join(d, "NBA_events.json")


def load_rows(data_dir: str) -> list[dict]:
    path, _ = _paths(data_dir)
    if not os.path.exists(path):
        return []
    out = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            for s in STATS:
                r[s] = float(r[s]) if r.get(s) not in (None, "") else None
            r["home"], r["starter"] = int(r["home"]), int(r["starter"])
            out.append(r)
    out.sort(key=lambda r: r["date"])
    return out


def _load_registry(data_dir: str) -> dict:
    _, p = _paths(data_dir)
    try:
        with open(p) as f:
            reg = json.load(f)
    except (OSError, ValueError):
        reg = {}
    reg.setdefault("done", {})
    reg.setdefault("days", {})
    return reg


def _save(data_dir: str, new_rows: list[dict], reg: dict) -> None:
    path, rp = _paths(data_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if not exists:
            w.writeheader()
        w.writerows({k: ("" if r.get(k) is None else r[k]) for k in FIELDS} for r in new_rows)
    tmp = rp + ".tmp"
    with open(tmp, "w") as f:
        json.dump(reg, f, sort_keys=True)
    os.replace(tmp, rp)


# ------------------------------------------------------------------ downloading
def update_boxscores(data_dir: str, game_days: list[date], today: date, budget_seconds: float = 150.0,
                     fetch_day=None, fetch_summary=None) -> dict:
    """Download box scores for finished games, newest day first, within a time budget.
    `game_days`: days with NBA games (from the results history). Returns a status dict."""
    from ..data import espn
    from ..news.feeds import ESPN_BASE, SPORT_PATH, _get_json

    def _day(d):
        games = espn.fetch_day("NBA", d)
        return None if espn.last_day_failed else games
    fetch_day = fetch_day or _day
    fetch_summary = fetch_summary or (lambda eid: _get_json(f"{ESPN_BASE}/{SPORT_PATH['NBA']}/summary?event={eid}"))
    aliases = espn.ABBR_ALIASES.get("NBA", {})
    reg = _load_registry(data_dir)
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
            rows = parse_boxscore(data, aliases)
            if rows:
                new_rows.extend(rows)
                reg["done"][eid] = rows[0]["game_id"]
                games_done += 1
    if new_rows or reg:
        _save(data_dir, new_rows, reg)
    return {"new_games": games_done, "new_rows": len(new_rows), "errors": errors,
            "games_stored": len(reg["done"]), "days_scanned": len(reg["days"]),
            "seconds": round(time.time() - t0, 1)}
