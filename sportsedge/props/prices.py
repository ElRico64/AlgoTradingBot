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
_ESPN_STATS = {"points": "pts", "rebounds": "reb", "assists": "ast", "3-point field goals": "fg3m",
               "three point field goals": "fg3m", "3-pointers made": "fg3m", "threes": "fg3m"}


def espn_stat(type_name: str) -> tuple[Optional[str], Optional[str]]:
    """(stat, kind) for an ESPN prop type: kind "pair" for full-game over/unders
    ("Total Points"), "milestone" for over-only thresholds ("Points Milestones", i.e. 25+).
    Combinations, partial games and specials are not modelled."""
    n = " ".join((type_name or "").lower().split())
    if n.startswith("total ") and n[6:] in _ESPN_STATS:
        return _ESPN_STATS[n[6:]], "pair"
    if n.endswith(" milestones") and n[:-11] in _ESPN_STATS:
        return _ESPN_STATS[n[:-11]], "milestone"
    return None, None


LADDER_TOLERANCE = 0.10  # how far a milestone's implied chance may sit on the wrong side of the main line


def _ladders(seq: list[tuple[float, float]]) -> list[list[tuple[float, float]]]:
    """Split one player's milestones (target, price), in listed order, into ladders: a new ladder
    starts whenever the target stops rising. ESPN can list a second ladder for the same player and
    stat under the same type (seen: a part-game "2+ … 8+ rebounds" next to the full game's "6+ … 20+")."""
    out: list = []
    for t, p in seq:
        if not out or t <= out[-1][-1][0]:
            out.append([])
        out[-1].append((t, p))
    return out


def _ladder_fits(ladder: list[tuple[float, float]], anchors: list[tuple[float, list[float]]]) -> bool:
    """A full-game ladder agrees with the player's full-game over/under: chances fall as the target
    rises, targets below the main line are likelier than the line's sides, targets above it less likely.
    Checked without knowing which of the pair's prices is the over (only their range is used)."""
    probs = [_american_to_prob(p) for _, p in ladder]
    if any(b > a + 0.03 for a, b in zip(probs, probs[1:])):
        return False
    for line, prices in anchors:
        q = [_american_to_prob(x) for x in prices]
        lo, hi = min(q) / sum(q), max(q) / sum(q)
        for (t, _), pr in zip(ladder, probs):
            over_of = t - 0.5
            if over_of < line and pr < lo - LADDER_TOLERANCE:
                return False
            if over_of > line and pr > hi + LADDER_TOLERANCE:
                return False
            if over_of == line and not (lo - LADDER_TOLERANCE <= pr <= hi + LADDER_TOLERANCE):
                return False
    return True


def parse_espn_propbets(items: list[dict]) -> tuple[list[dict], list[dict], dict]:
    """ESPN prop items -> (over/under pairs with prices in listed order, milestones, type counts).
    A milestone "25+" is the over of a 24.5 line. Milestones are kept only when they are
    unambiguous: each ladder must agree with the player's full-game over/under for that stat,
    and agreeing ladders must not price the same target differently."""
    groups: dict = {}
    seqs: dict = {}
    names: dict = {}
    for it in items:
        tname = ((it.get("type") or {}).get("name")) or ""
        names[tname] = names.get(tname, 0) + 1
        stat, kind = espn_stat(tname)
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
        if kind == "pair":
            groups.setdefault((m.group(1), stat, line), []).append(price)
        elif float(line).is_integer():
            seqs.setdefault((m.group(1), stat), []).append((line, price))
    pairs = [{"player_id": pid, "stat": stat, "line": line, "prices": prices}
             for (pid, stat, line), prices in groups.items() if len(prices) == 2]
    anchors: dict = {}
    for pr in pairs:
        anchors.setdefault((pr["player_id"], pr["stat"]), []).append((pr["line"], pr["prices"]))
    milestones, dropped = [], 0
    for (pid, stat), seq in seqs.items():
        ladders = _ladders(seq)
        good = [ld for ld in ladders if anchors.get((pid, stat)) and _ladder_fits(ld, anchors[(pid, stat)])]
        price_at: dict = {}
        clash = False
        for ld in good:
            for t, p in ld:
                if t in price_at and abs(_american_to_prob(price_at[t]) - _american_to_prob(p)) > 0.03:
                    clash = True
                price_at[t] = p
        if clash:
            good, price_at = [], {}
        dropped += len(ladders) - len(good)
        milestones += [{"player_id": pid, "stat": stat, "line": t - 0.5, "price": p}
                       for t, p in sorted(price_at.items())]
    if dropped:
        names["(milestone ladders left out: no matching full-game line, or ambiguous)"] = dropped
    return pairs, milestones, names


def fetch_espn_props(games: list[dict]) -> tuple[dict, dict, dict]:
    """Today's ESPN props per game id: (over/under pairs, milestones, type counts), from the
    first provider that lists props. Free, no key."""
    from ..data.http import get_json

    out, mout, names = {}, {}, {}
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
                _save_milestone_sample(items)
                pairs, miles, nm = parse_espn_propbets(items)
                for k, v in nm.items():
                    names[k] = names.get(k, 0) + v
                if pairs or miles:
                    out[g["id"]], mout[g["id"]] = pairs, miles
                    break
    return out, mout, names


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


SAMPLE_DIR: Optional[str] = None  # set by the board: where to keep a raw sample for inspection


def _save_milestone_sample(items: list[dict]) -> None:
    """Keep the raw items of a few players (all prop types) so the format can be checked by eye."""
    if not SAMPLE_DIR:
        return
    keep, seen = [], []
    for it in items:
        ref = ((it.get("athlete") or {}).get("$ref") or "")
        if ref not in seen and len(seen) < 3:
            seen.append(ref)
        if ref in seen:
            keep.append({k: v for k, v in it.items() if k not in ("competition", "provider")})
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    with open(os.path.join(SAMPLE_DIR, "espn_props_raw.json"), "w") as f:
        json.dump(keep[:400], f, indent=0)


def priced_from_milestones(miles_by_game: dict) -> dict:
    """Milestones ("25+ points") as over-only props. Their price with the margin left in is the
    cautious fair chance: it overstates the books' view, so it can only understate the model's edge."""
    return {gid: [{"player_id": m["player_id"], "player": "", "stat": m["stat"], "line": m["line"],
                   "over": m["price"], "under": None, "fair": _american_to_prob(m["price"]), "books": 1,
                   "alt": True} for m in ms] for gid, ms in miles_by_game.items()}
