"""The Odds API (https://the-odds-api.com) v4 client: multi-book odds.

Two things matter for a betting model and both are done here:
  * a *consensus fair price* — every book is de-vigged (Shin) and the results
    are pooled in log-odds space with sharp books weighted more heavily;
  * *line shopping* — the best available price for each side is what we bet,
    so the EV gate is evaluated against a price we can actually get.
Requires an API key (env ODDS_API_KEY or --odds-api-key).
"""
from __future__ import annotations

import logging
from collections import Counter
from datetime import datetime
from typing import Optional

import requests

from ..stats.odds import american_to_decimal, consensus_probability
from ..teams import resolve_team
from ..types import MarketOdds

log = logging.getLogger(__name__)

SPORT_KEYS = {"NFL": "americanfootball_nfl", "NBA": "basketball_nba", "MLB": "baseball_mlb", "NHL": "icehockey_nhl"}
SHARP_WEIGHTS = {"pinnacle": 3.0, "circasports": 2.5, "betonlineag": 1.5, "lowvig": 1.5, "bookmaker": 1.5}


def _best(prices: list[float]) -> Optional[float]:
    return max(prices, key=american_to_decimal) if prices else None


def parse_event(league: str, ev: dict) -> tuple[Optional[str], Optional[str], Optional[datetime], MarketOdds]:
    home = resolve_team(league, ev.get("home_team", ""))
    away = resolve_team(league, ev.get("away_team", ""))
    try:
        start = datetime.fromisoformat(ev["commence_time"].replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        start = None
    h2h: list[tuple[float, float]] = []
    h2h_w: list[float] = []
    spreads: dict[float, list[tuple[float, float, float]]] = {}
    totals: dict[float, list[tuple[float, float, float]]] = {}
    for bk in ev.get("bookmakers", []) or []:
        w = SHARP_WEIGHTS.get(bk.get("key", ""), 1.0)
        for mk in bk.get("markets", []) or []:
            oc = {o.get("name"): o for o in mk.get("outcomes", []) or []}
            if mk.get("key") == "h2h" and ev.get("home_team") in oc and ev.get("away_team") in oc:
                h2h.append((oc[ev["home_team"]]["price"], oc[ev["away_team"]]["price"]))
                h2h_w.append(w)
            elif mk.get("key") == "spreads" and ev.get("home_team") in oc and ev.get("away_team") in oc:
                hp = oc[ev["home_team"]]
                spreads.setdefault(float(hp["point"]), []).append((hp["price"], oc[ev["away_team"]]["price"], w))
            elif mk.get("key") == "totals" and "Over" in oc and "Under" in oc:
                totals.setdefault(float(oc["Over"]["point"]), []).append(
                    (oc["Over"]["price"], oc["Under"]["price"], w))
    mo = MarketOdds(source="the-odds-api")
    if h2h:
        mo.home_ml = _best([a for a, _ in h2h])
        mo.away_ml = _best([b for _, b in h2h])
        mo.fair_home_prob = consensus_probability(h2h, h2h_w)
    if spreads:
        line = Counter({k: len(v) for k, v in spreads.items()}).most_common(1)[0][0]
        rows = spreads[line]
        mo.spread = line
        mo.home_spread_price = _best([r[0] for r in rows])
        mo.away_spread_price = _best([r[1] for r in rows])
        mo.fair_home_cover_prob = consensus_probability([(r[0], r[1]) for r in rows], [r[2] for r in rows])
    if totals:
        line = Counter({k: len(v) for k, v in totals.items()}).most_common(1)[0][0]
        rows = totals[line]
        mo.total = line
        mo.over_price = _best([r[0] for r in rows])
        mo.under_price = _best([r[1] for r in rows])
        mo.fair_over_prob = consensus_probability([(r[0], r[1]) for r in rows], [r[2] for r in rows])
    return home, away, start, mo


def fetch_odds(league: str, api_key: str, regions: str = "us,eu") -> dict[tuple[str, str], MarketOdds]:
    url = f"https://api.the-odds-api.com/v4/sports/{SPORT_KEYS[league]}/odds"
    params = {"apiKey": api_key, "regions": regions, "markets": "h2h,spreads,totals", "oddsFormat": "american"}
    try:
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        events = r.json()
    except Exception as e:
        log.warning("odds fetch failed: %s", e)
        return {}
    out = {}
    for ev in events:
        home, away, _, mo = parse_event(league, ev)
        if home and away:
            out[(home, away)] = mo
    return out
