"""Prop prices. Never invented: a prop parlay is only built from posted prices.

* The Odds API (free key, ODDS_API_KEY): player points / rebounds / assists /
  threes over-unders from US books. Each game costs one credit per market, so
  the calls are budgeted with the game-odds calls (at most one props refresh a
  day).
* ESPN: whether ESPN's free API lists player props is checked on each update,
  and a small sample of what it returns is saved to data/props/samples/, so a
  free source can be added once its format is known.
"""
from __future__ import annotations

import json
import logging
import os
import re
import unicodedata
from datetime import datetime, timezone
from typing import Optional

log = logging.getLogger(__name__)

MARKETS = {"player_points": "pts", "player_rebounds": "reb", "player_assists": "ast", "player_threes": "fg3m",
           # alternate lines ("LeBron over 5.5 assists"): usually over-only
           "player_points_alternate": "pts", "player_rebounds_alternate": "reb",
           "player_assists_alternate": "ast", "player_threes_alternate": "fg3m"}
DEFAULT_MARKETS = "player_points,player_rebounds,player_assists,player_threes"
CORE = "https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba"


def norm_name(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z ]", " ", s)
    parts = [p for p in s.split() if p not in ("jr", "sr", "ii", "iii", "iv", "v")]
    return " ".join(parts)


def _american_to_prob(a: float) -> float:
    return 100 / (a + 100) if a >= 100 else -a / (-a + 100)


def parse_event_props(data: dict) -> list[dict]:
    """The Odds API event-odds response -> one row per (player, stat, line) with the
    best over and under prices across books and the books' consensus no-vig P(over)."""
    groups: dict = {}
    for bk in data.get("bookmakers") or []:
        for mk in bk.get("markets") or []:
            stat = MARKETS.get(mk.get("key"))
            if not stat:
                continue
            pairs: dict = {}
            for o in mk.get("outcomes") or []:
                name, side, price, point = o.get("description"), (o.get("name") or "").lower(), o.get("price"), o.get("point")
                if not name or side not in ("over", "under") or price is None or point is None:
                    continue
                pairs.setdefault((norm_name(name), float(point)), {"name": name})[side] = float(price)
            for (nn, point), v in pairs.items():
                g = groups.setdefault((nn, stat, point), {"player": v["name"], "stat": stat, "line": point,
                                                          "over": None, "under": None, "fair": [], "books": 0})
                for side in ("over", "under"):
                    if side in v and (g[side] is None or _american_to_prob(v[side]) < _american_to_prob(g[side])):
                        g[side] = v[side]  # best price for that side
                if "over" in v and "under" in v:
                    po, pu = _american_to_prob(v["over"]), _american_to_prob(v["under"])
                    g["fair"].append(po / (po + pu))
                    g["books"] += 1
    out = []
    for g in groups.values():
        if g["over"] is None:
            continue
        if g["fair"]:
            g["fair"] = sum(g["fair"]) / len(g["fair"])
        else:
            # over-only (alternate line): the price with its margin left in is the cautious "fair"
            # chance; it overstates the books' view, so it can only understate the model's edge
            g["fair"] = _american_to_prob(g["over"])
        out.append(g)
    return out


def fetch_oddsapi_props(api_key: str, games: list[dict], budget, data_dir: str) -> tuple[dict, str]:
    """Props for today's NBA games from The Odds API. Returns ({game_id: [rows]}, message)."""
    import requests

    from ..teams import resolve_team

    markets = os.environ.get("PROPS_MARKETS", DEFAULT_MARKETS)
    n_mk = len(markets.split(",")) * len(budget.regions.split(","))
    if not budget.allow_amount("NBA-props", n_mk, min_hours=20):
        return {}, "prop prices: today's share of the free odds credits is used (next refresh tomorrow)"
    try:
        r = requests.get("https://api.the-odds-api.com/v4/sports/basketball_nba/events",
                         params={"apiKey": api_key}, timeout=20)
        r.raise_for_status()
        events = r.json()
    except Exception as e:
        return {}, f"prop prices: The Odds API did not answer ({type(e).__name__})"
    want = {(g["home"]["abbr"], g["away"]["abbr"]): g["id"] for g in games}
    out, spent = {}, 0
    for ev in events:
        key = (resolve_team("NBA", ev.get("home_team", "")), resolve_team("NBA", ev.get("away_team", "")))
        gid = want.get(key)
        if gid is None:
            continue
        if not budget.allow_amount("NBA-props", n_mk, min_hours=0):
            break
        meta = {}
        try:
            r = requests.get(f"https://api.the-odds-api.com/v4/sports/basketball_nba/events/{ev['id']}/odds",
                             params={"apiKey": api_key, "regions": budget.regions, "markets": markets,
                                     "oddsFormat": "american"}, timeout=20)
            for k in ("x-requests-remaining", "x-requests-last"):
                if k in r.headers:
                    meta[k] = float(r.headers[k])
            r.raise_for_status()
            out[gid] = parse_event_props(r.json())
        except Exception as e:
            log.warning("props fetch failed for %s: %s", gid, e)
        budget.spend("NBA-props", meta or {"x-requests-last": n_mk})
        spent += 1
    return out, f"prop prices: The Odds API, {sum(len(v) for v in out.values())} props on {len(out)} games"


def probe_espn_props(espn_ids: list[str], data_dir: str) -> str:
    """Check whether ESPN's free API lists player props for an upcoming game; save a sample."""
    from ..data.http import get_json

    if not espn_ids:
        return "ESPN props: no upcoming game to check"
    eid = espn_ids[0]
    found = []
    providers = get_json(f"{CORE}/events/{eid}/competitions/{eid}/odds?lang=en&region=us") or {}
    sample = {"checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "event": eid,
              "providers": [], "propBets": {}}
    for item in (providers.get("items") or [])[:6]:
        ref = item.get("$ref") or ""
        m = re.search(r"/odds/(\d+)", ref)
        pid = m.group(1) if m else (item.get("provider") or {}).get("id")
        if not pid:
            continue
        sample["providers"].append(pid)
        pb = get_json(f"{CORE}/events/{eid}/competitions/{eid}/odds/{pid}/propBets?lang=en&region=us&limit=50")
        if pb and (pb.get("items") or pb.get("count")):
            found.append(pid)
            sample["propBets"][pid] = {"count": pb.get("count"), "items": (pb.get("items") or [])[:3]}
    d = os.path.join(data_dir, "props", "samples")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "espn_props_probe.json"), "w") as f:
        json.dump(sample, f, indent=1)
    if found:
        return f"ESPN props: listed by provider(s) {', '.join(found)} (sample saved; parser to be added)"
    return "ESPN props: none listed on ESPN's free API for the game checked"
