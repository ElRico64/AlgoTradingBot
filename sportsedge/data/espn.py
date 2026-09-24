"""ESPN public scoreboard: schedules, results, probable pitchers and posted odds.

Unofficial, undocumented endpoint — parsed defensively. Use `fetch_history`
to build a results CSV and `fetch_slate` for today's games.
"""
from __future__ import annotations

import logging
import re
import time
from datetime import date, datetime, timedelta
from typing import Optional

from ..news.feeds import ESPN_BASE, SPORT_PATH, _get_json
from ..teams import REGISTRY
from ..types import Game, GameResult, MarketOdds

log = logging.getLogger(__name__)


def _num(v) -> Optional[float]:
    try:
        return float(str(v).replace("+", "")) if v not in (None, "", "EVEN") else (100.0 if v == "EVEN" else None)
    except ValueError:
        return None


def _parse_odds(comp: dict, home_abbr: str) -> Optional[MarketOdds]:
    odds_list = comp.get("odds") or []
    if not odds_list:
        return None
    o = odds_list[0]
    ho, ao = o.get("homeTeamOdds") or {}, o.get("awayTeamOdds") or {}
    spread = None
    details = o.get("details") or ""
    m = re.match(r"^\s*([A-Z]{2,4})\s+([+-]?\d+(\.\d+)?)\s*$", details)
    if m:
        v = float(m.group(2))
        spread = v if m.group(1) == home_abbr else -v
    elif details.strip().upper() in ("EVEN", "PK", "PICK"):
        spread = 0.0
    mo = MarketOdds(
        home_ml=_num(ho.get("moneyLine")), away_ml=_num(ao.get("moneyLine")), spread=spread,
        home_spread_price=_num(ho.get("spreadOdds")) or -110.0, away_spread_price=_num(ao.get("spreadOdds")) or -110.0,
        total=_num(o.get("overUnder")), over_price=_num(o.get("overOdds")) or -110.0,
        under_price=_num(o.get("underOdds")) or -110.0, source=(o.get("provider") or {}).get("name", "espn"))
    return mo if (mo.has_moneyline() or mo.spread is not None or mo.total is not None) else None


def _probable(c: dict) -> Optional[dict]:
    for p in c.get("probables") or []:
        ath = p.get("athlete") or {}
        out = {"name": ath.get("displayName") or ath.get("fullName")}
        for s in p.get("statistics") or []:
            key = (s.get("abbreviation") or s.get("name") or "").upper()
            if key == "ERA":
                out["era"] = _num(s.get("displayValue"))
            elif key in ("IP", "INNINGSPITCHED"):
                out["ip"] = _num(s.get("displayValue"))
        return out
    return None


def _team_display(c: dict) -> dict:
    t = c.get("team") or {}
    color = (t.get("color") or "").strip("#")
    alt = (t.get("alternateColor") or "").strip("#")
    return {"name": t.get("displayName") or t.get("name") or t.get("abbreviation"),
            "color": f"#{color}" if len(color) == 6 else None, "alt": f"#{alt}" if len(alt) == 6 else None,
            "record": ((c.get("records") or [{}])[0] or {}).get("summary")}


def parse_scoreboard(league: str, data: dict) -> list[Game]:
    games: list[Game] = []
    for ev in data.get("events", []) or []:
        comps = ev.get("competitions") or [{}]
        comp = comps[0]
        teams = {c.get("homeAway"): c for c in comp.get("competitors") or []}
        if "home" not in teams or "away" not in teams:
            continue
        h, a = teams["home"], teams["away"]
        habbr = (h.get("team") or {}).get("abbreviation")
        aabbr = (a.get("team") or {}).get("abbreviation")
        season = ev.get("season") or {}
        if season.get("type") == 1 or season.get("slug") == "preseason":
            continue  # preseason / spring training says little about real strength
        known = REGISTRY.get(league, {})
        if known and (habbr not in known or aabbr not in known):
            continue  # All-Star and exhibition games
        try:
            start = datetime.fromisoformat((ev.get("date") or comp.get("date")).replace("Z", "+00:00")).replace(tzinfo=None)
        except Exception:
            continue
        extras: dict = {}
        if league == "MLB":
            hp, ap = _probable(h), _probable(a)
            if hp:
                extras["home_pitcher"] = hp
            if ap:
                extras["away_pitcher"] = ap
        wx = ev.get("weather") or comp.get("weather")
        if wx and wx.get("temperature") is not None:
            extras["weather"] = {"temp_f": _num(wx.get("temperature"))}
        venue = comp.get("venue") or {}
        if venue.get("indoor") is True:
            extras["indoor"] = True
        status = ((ev.get("status") or comp.get("status") or {}).get("type") or {})
        # display-only context for the dashboard (never used by the model)
        extras["status"] = {"state": status.get("state") or ("post" if status.get("completed") else "pre"),
                            "detail": status.get("shortDetail") or status.get("detail") or ""}
        extras["teams"] = {side: _team_display(c) for side, c in (("home", h), ("away", a))}
        if extras["status"]["state"] in ("in", "post"):
            try:
                extras["live"] = {"home": int(float(h.get("score"))), "away": int(float(a.get("score")))}
            except (TypeError, ValueError):
                pass
        common = dict(game_id=str(ev.get("id")), league=league, start_time=start, home=habbr, away=aabbr,
                      neutral=bool(comp.get("neutralSite")), extras=extras)
        odds = _parse_odds(comp, habbr)
        if status.get("completed"):
            try:
                games.append(GameResult(**common, home_score=int(float(h.get("score"))),
                                        away_score=int(float(a.get("score"))), odds=odds))
            except (TypeError, ValueError):
                continue
        else:
            g = Game(**common)
            g.extras["odds"] = odds
            games.append(g)
    return games


def fetch_scoreboard(league: str, day: date) -> list[Game]:
    data = _get_json(f"{ESPN_BASE}/{SPORT_PATH[league]}/scoreboard?dates={day:%Y%m%d}&limit=300")
    return parse_scoreboard(league, data) if data else []


def fetch_day(league: str, day: date) -> list[Game]:
    """Every game on `day` (US date): upcoming, live and final."""
    return fetch_scoreboard(league, day)


def fetch_slate(league: str, day: Optional[date] = None) -> list[Game]:
    day = day or date.today()
    return [g for g in fetch_scoreboard(league, day) if not isinstance(g, GameResult)]


def fetch_history(league: str, start: date, end: date, pause: float = 0.05, workers: int = 6,
                  progress=None) -> list[GameResult]:
    """Download every completed game between two dates (a few parallel requests, politely paced)."""
    from concurrent.futures import ThreadPoolExecutor

    days = [start + timedelta(days=k) for k in range((end - start).days + 1)]

    def one(d: date) -> list[GameResult]:
        time.sleep(pause)
        return [g for g in fetch_scoreboard(league, d) if isinstance(g, GameResult)]

    out: list[GameResult] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for k, games in enumerate(ex.map(one, days), 1):
            out.extend(games)
            if progress and (k % 60 == 0 or k == len(days)):
                progress(f"{league}: downloaded {k}/{len(days)} days, {len(out)} games")
    out.sort(key=lambda g: g.start_time)
    return out
