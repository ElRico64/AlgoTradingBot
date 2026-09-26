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
from .official import canonical_id
from ..types import Game, GameResult, MarketOdds

log = logging.getLogger(__name__)


def _num(v) -> Optional[float]:
    """Parse ESPN numbers: 8.5, "-150", "+130", "EVEN", "o8.5", "u8.5"."""
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    t = str(v).strip().upper()
    if t in ("EVEN", "EV"):
        return 100.0
    if t in ("PK", "PICK"):
        return 0.0
    t = t.lstrip("OU").replace("+", "").replace(",", "")
    try:
        return float(t)
    except ValueError:
        return None


def _price(v) -> Optional[float]:
    """An American price; anything between -100 and +100 is not one."""
    x = _num(v)
    return x if x is not None and abs(x) >= 100 else None


def _quote(node, key: str) -> Optional[float]:
    """Newer ESPN layout: {"close": {...}, "current": {...}, "open": {...}} -> latest value of `key`."""
    if not isinstance(node, dict):
        return None
    for when in ("close", "current", "open"):
        leaf = node.get(when)
        if isinstance(leaf, dict) and leaf.get(key) not in (None, ""):
            v = _num(leaf.get(key))
            if v is not None:
                return v
    return None


def _parse_one(o: dict, league: str, home_abbr: str, away_abbr: str) -> Optional[MarketOdds]:
    ho, ao = o.get("homeTeamOdds") or {}, o.get("awayTeamOdds") or {}
    ml, ps, tot = o.get("moneyline") or {}, o.get("pointSpread") or {}, o.get("total")
    tot = tot if isinstance(tot, dict) else {}

    # moneyline: legacy homeTeamOdds.moneyLine, then moneyline.home.close.odds
    home_ml = _price(ho.get("moneyLine")) or _price(_quote(ml.get("home"), "odds"))
    away_ml = _price(ao.get("moneyLine")) or _price(_quote(ml.get("away"), "odds"))

    # spread, quoted from the home side
    spread = _quote(ps.get("home"), "line")
    if spread is None:
        away_line = _quote(ps.get("away"), "line")
        spread = -away_line if away_line is not None else None
    details = str(o.get("details") or "").strip()
    m = re.match(r"^([A-Z]{2,4})\s+([+-]?\d+(\.\d+)?)$", details.upper())
    if m:
        team = ABBR_ALIASES.get(league, {}).get(m.group(1), m.group(1))
        v = float(m.group(2))
        # MLB/NHL boards often show the favourite's moneyline here ("NYY -155"), not a spread
        if abs(v) < 60 and spread is None and team in (home_abbr, away_abbr):
            spread = v if team == home_abbr else -v
    elif spread is None and details.upper() in ("EVEN", "PK", "PICK"):
        spread = 0.0
    if spread is not None and abs(spread) > 60:
        spread = None  # a price, not a line

    home_sp = _price(ho.get("spreadOdds")) or _price(_quote(ps.get("home"), "odds"))
    away_sp = _price(ao.get("spreadOdds")) or _price(_quote(ps.get("away"), "odds"))

    total = _num(o.get("overUnder"))
    if total is None:
        total = _quote(tot.get("over"), "line") or _quote(tot.get("under"), "line")
    if total is not None and not (0 < total < 400):
        total = None
    over_p = _price(o.get("overOdds")) or _price(_quote(tot.get("over"), "odds"))
    under_p = _price(o.get("underOdds")) or _price(_quote(tot.get("under"), "odds"))

    mo = MarketOdds(
        home_ml=home_ml, away_ml=away_ml, spread=spread,
        home_spread_price=home_sp or -110.0, away_spread_price=away_sp or -110.0,
        total=total, over_price=over_p or -110.0, under_price=under_p or -110.0,
        source=(o.get("provider") or {}).get("name") or "espn")
    return mo if (mo.has_moneyline() or mo.spread is not None or mo.total is not None) else None


def parse_odds_list(odds_list, league: str, home_abbr: str, away_abbr: str) -> Optional[MarketOdds]:
    """First usable entry of an ESPN odds / pickcenter list, highest-priority provider first."""
    if not isinstance(odds_list, list):
        return None
    entries = [o for o in odds_list if isinstance(o, dict)]
    entries.sort(key=lambda o: (o.get("provider") or {}).get("priority", 99) or 99)
    best = None
    for o in entries:
        mo = _parse_one(o, league, home_abbr, away_abbr)
        if mo is None:
            continue
        if mo.has_moneyline():
            return mo
        best = best or mo
    return best


def _parse_odds(comp: dict, league: str, home_abbr: str, away_abbr: str) -> Optional[MarketOdds]:
    return parse_odds_list(comp.get("odds") or [], league, home_abbr, away_abbr)


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
        alias = ABBR_ALIASES.get(league, {})
        habbr, aabbr = alias.get(habbr, habbr), alias.get(aabbr, aabbr)
        known = REGISTRY.get(league, {})
        if known and (habbr not in known or aabbr not in known):
            continue  # All-Star and exhibition games
        try:
            start = datetime.fromisoformat((ev.get("date") or comp.get("date")).replace("Z", "+00:00")).replace(tzinfo=None)
        except Exception:
            continue
        extras: dict = {}
        if ev.get("id"):
            extras["espn_id"] = str(ev["id"])
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
        common = dict(game_id=canonical_id(league, start, habbr, aabbr), league=league, start_time=start, home=habbr, away=aabbr,
                      neutral=bool(comp.get("neutralSite")), extras=extras)
        odds = _parse_odds(comp, league, habbr, aabbr)
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


# alternate / historical codes -> the codes this project uses (e.g. the Athletics were "OAK" before 2025)
ABBR_ALIASES = {
    "MLB": {"OAK": "ATH", "AZ": "ARI", "CWS": "CHW", "WAS": "WSH", "SDP": "SD", "SFG": "SF", "TBR": "TB", "KCR": "KC"},
    "NBA": {"GSW": "GS", "NOP": "NO", "NYK": "NY", "SAS": "SA", "UTA": "UTAH", "WAS": "WSH", "PHO": "PHX", "BRK": "BKN"},
    "NHL": {"LAK": "LA", "NJD": "NJ", "SJS": "SJ", "TBL": "TB", "UTA": "UTAH", "WAS": "WSH"},
    "NFL": {"WAS": "WSH", "LA": "LAR", "JAC": "JAX"},
}

last_day_failed = False  # set by fetch_day: True when ESPN did not answer


def fetch_scoreboard_raw(league: str, day: date) -> Optional[list[Game]]:
    """Games on `day`, or None when ESPN could not be reached (vs. [] for no games)."""
    data = _get_json(f"{ESPN_BASE}/{SPORT_PATH[league]}/scoreboard?dates={day:%Y%m%d}")
    return None if data is None else parse_scoreboard(league, data)


def fetch_scoreboard(league: str, day: date) -> list[Game]:
    return fetch_scoreboard_raw(league, day) or []


def fetch_day(league: str, day: date) -> list[Game]:
    """Every game on `day` (US date): upcoming, live and final."""
    global last_day_failed
    games = fetch_scoreboard_raw(league, day)
    last_day_failed = games is None
    return games or []


def fetch_game_odds(league: str, espn_id: str, home: str, away: str) -> Optional[MarketOdds]:
    """One game's lines from its ESPN game page ("pickcenter"), which often has
    prices when the scoreboard shows none."""
    data = _get_json(f"{ESPN_BASE}/{SPORT_PATH[league]}/summary?event={espn_id}")
    if not data:
        return None
    return (parse_odds_list(data.get("pickcenter") or [], league, home, away)
            or parse_odds_list(data.get("odds") or [], league, home, away))


def fetch_slate(league: str, day: Optional[date] = None) -> list[Game]:
    day = day or date.today()
    return [g for g in fetch_scoreboard(league, day) if not isinstance(g, GameResult)]


last_history_failed_share = 0.0  # set by fetch_history
# months with regular-season or playoff games; other days are skipped to spare requests
SEASON_MONTHS = {"MLB": {3, 4, 5, 6, 7, 8, 9, 10, 11}, "NHL": {10, 11, 12, 1, 2, 3, 4, 5, 6},
                 "NBA": {10, 11, 12, 1, 2, 3, 4, 5, 6}, "NFL": {9, 10, 11, 12, 1, 2}}


def fetch_history(league: str, start: date, end: date, pause: float = 0.0, workers: int = 3,
                  progress=None) -> list[GameResult]:
    """Download every completed game between two dates. Requests are paced by the
    shared HTTP client; a refusing ESPN makes this return early with what it has."""
    global last_history_failed_share
    from concurrent.futures import ThreadPoolExecutor

    from .http import is_blocked

    months = SEASON_MONTHS.get(league)
    days = [d for d in (start + timedelta(days=k) for k in range((end - start).days + 1))
            if not months or d.month in months]
    failed = 0

    def one(d: date) -> Optional[list[Game]]:
        if pause:
            time.sleep(pause)
        if is_blocked("site.api.espn.com") and is_blocked("site.web.api.espn.com"):
            return None
        return fetch_scoreboard_raw(league, d)

    out: list[GameResult] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for k, games in enumerate(ex.map(one, days), 1):
            if games is None:
                failed += 1
            else:
                out.extend(g for g in games if isinstance(g, GameResult))
            if progress and (k % 60 == 0 or k == len(days)):
                progress(f"{league} (ESPN): {k}/{len(days)} days, {len(out)} games" +
                         (f", {failed} days refused" if failed else ""))
    last_history_failed_share = failed / max(1, len(days))
    out.sort(key=lambda g: g.start_time)
    return out
