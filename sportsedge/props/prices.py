"""Prop prices. Never invented: a prop parlay is only built from posted prices.

* The Odds API (free key, ODDS_API_KEY): player points / rebounds / assists /
  threes over-unders from US books. Each game costs one credit per market, so
  the calls are budgeted with the game-odds calls (at most one props refresh a
  day).
* ESPN (free, no key): player props from the sportsbook ESPN lists. Each
  over/under comes as two prices with no side label; which one is the over is
  learned from the data (board.espn_orientation) before any is used.
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


# ------------------------------------------------------------------ ESPN (free)
# ESPN's core API lists each over/under as two consecutive items at the same line with no side
# label. Which one is the over is learned from data (see board.espn_orientation), never assumed.
def espn_stat(type_name: str) -> Optional[str]:
    n = (type_name or "").lower()
    if any(c in n for c in (",", "+", " and ", "&", "double", "first", "quarter", "half", "1st", "2nd")):
        return None  # combos, milestones and partial-game props are not modelled
    if "three" in n or "3-point" in n or "3 point" in n or "3pt" in n:
        return "fg3m"
    if "rebound" in n:
        return "reb"
    if "assist" in n:
        return "ast"
    if "point" in n:
        return "pts"
    return None


def parse_espn_propbets(items: list[dict]) -> tuple[list[dict], dict]:
    """Group ESPN prop items into pairs: (athlete id, stat, line) -> prices in listed order.
    Returns (pairs, counts of type names seen)."""
    groups: dict = {}
    names: dict = {}
    for it in items:
        tname = ((it.get("type") or {}).get("name")) or ""
        names[tname] = names.get(tname, 0) + 1
        stat = espn_stat(tname)
        m = re.search(r"/athletes/(\d+)", ((it.get("athlete") or {}).get("$ref") or ""))
        odds = it.get("odds") or {}
        line = ((it.get("current") or {}).get("target") or {}).get("value")
        if line is None:
            line = (odds.get("total") or {}).get("value")
        price = (odds.get("american") or {}).get("value")
        if not stat or not m or line is None or price in (None, ""):
            continue
        try:
            line, price = float(line), float(str(price).replace("+", ""))
        except ValueError:
            continue
        groups.setdefault((m.group(1), stat, line), []).append(price)
    pairs = [{"player_id": pid, "stat": stat, "line": line, "prices": prices}
             for (pid, stat, line), prices in groups.items() if len(prices) == 2]
    return pairs, names


def fetch_espn_props(games: list[dict]) -> tuple[dict, dict]:
    """Today's ESPN prop pairs per game id (first provider that lists props). Free, no key."""
    from ..data.http import get_json

    out, names = {}, {}
    for g in games:
        eid = g.get("espn_id")
        if not eid:
            continue
        providers = get_json(f"{CORE}/events/{eid}/competitions/{eid}/odds?lang=en&region=us") or {}
        for item in providers.get("items") or []:
            m = re.search(r"/odds/(\d+)", item.get("$ref") or "")
            pid = m.group(1) if m else (item.get("provider") or {}).get("id")
            if not pid:
                continue
            items, page, pages = [], 1, 1
            while page <= pages and page <= 5:
                pb = get_json(f"{CORE}/events/{eid}/competitions/{eid}/odds/{pid}/propBets"
                              f"?lang=en&region=us&limit=1000&page={page}") or {}
                items.extend(pb.get("items") or [])
                pages = int(pb.get("pageCount") or 1)
                page += 1
            if items:
                pairs, nm = parse_espn_propbets(items)
                for k, v in nm.items():
                    names[k] = names.get(k, 0) + v
                if pairs:
                    out[g["id"]] = pairs
                    break
    return out, names


def priced_from_pairs(pairs_by_game: dict, first_is_over: bool) -> dict:
    """Pairs -> priced props (over, under, no-vig P(over)) once the side order is known."""
    out = {}
    for gid, pairs in pairs_by_game.items():
        rows = []
        for pr in pairs:
            over, under = pr["prices"] if first_is_over else pr["prices"][::-1]
            po, pu = _american_to_prob(over), _american_to_prob(under)
            rows.append({"player_id": pr["player_id"], "player": "", "stat": pr["stat"], "line": pr["line"],
                         "over": over, "under": under, "fair": po / (po + pu), "books": 1})
        out[gid] = rows
    return out
