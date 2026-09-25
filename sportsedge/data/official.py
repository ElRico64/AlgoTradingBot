"""Official league feeds used as backups when ESPN refuses requests.

* MLB: statsapi.mlb.com, the public Stats API behind MLB.com. A single
  request covers a whole date range, so a season of results takes a few calls.
* NHL: api-web.nhle.com, the public API behind NHL.com.

Neither feed carries betting odds. With these sources the board still
predicts every game and publishes 70%+ picks; value bets need prices from
ESPN or The Odds API. Both feeds are free, keyless and undocumented, so
parsing is defensive.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Optional

from ..teams import REGISTRY, resolve_team
from ..types import Game, GameResult
from .http import get_json

log = logging.getLogger(__name__)

try:
    from zoneinfo import ZoneInfo

    US_EASTERN = ZoneInfo("America/New_York")
except Exception:  # pragma: no cover
    US_EASTERN = timezone(timedelta(hours=-5))

MLB_BASE = "https://statsapi.mlb.com/api/v1/schedule"
NHL_BASE = "https://api-web.nhle.com/v1/score"
MLB_ABBR = {"AZ": "ARI", "CWS": "CHW", "OAK": "ATH", "WSN": "WSH", "SDP": "SD", "SFG": "SF", "TBR": "TB",
            "KCR": "KC"}
NHL_ABBR = {"LAK": "LA", "NJD": "NJ", "SJS": "SJ", "TBL": "TB", "UTA": "UTAH", "WAS": "WSH"}
MLB_GAME_TYPES = "R,F,D,L,W"  # regular season + postseason rounds (no spring training / exhibitions)


def canonical_id(league: str, start: datetime, home: str, away: str) -> str:
    """Source-independent game id: league, US Eastern date and hour, teams."""
    et = start.replace(tzinfo=timezone.utc).astimezone(US_EASTERN)
    return f"{league}-{et:%Y%m%d}-{away}-{home}-{et:%H}"


def _utc(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc).replace(tzinfo=None)
    except ValueError:
        return None


# ------------------------------------------------------------------------ MLB
def _mlb_team(t: dict) -> Optional[str]:
    team = t.get("team") or {}
    abbr = MLB_ABBR.get(team.get("abbreviation", ""), team.get("abbreviation"))
    if abbr in REGISTRY["MLB"]:
        return abbr
    return resolve_team("MLB", team.get("name") or team.get("teamName") or "")


def parse_mlb_schedule(data: dict) -> list[Game]:
    out: list[Game] = []
    for d in (data or {}).get("dates", []) or []:
        for g in d.get("games", []) or []:
            if g.get("gameType") not in MLB_GAME_TYPES.split(","):
                continue
            teams = g.get("teams") or {}
            h, a = teams.get("home") or {}, teams.get("away") or {}
            home, away = _mlb_team(h), _mlb_team(a)
            start = _utc(g.get("gameDate"))
            if not home or not away or start is None:
                continue
            st = g.get("status") or {}
            abstract = (st.get("abstractGameState") or "").lower()
            detailed = st.get("detailedState") or ""
            if detailed in ("Postponed", "Cancelled", "Suspended"):
                continue
            state = {"final": "post", "live": "in"}.get(abstract, "pre")
            extras: dict = {"status": {"state": state, "detail": detailed if state != "pre" else ""},
                            "teams": {s: {"name": ((t.get("team") or {}).get("name")), "color": None, "alt": None,
                                          "record": _mlb_record(t)} for s, t in (("home", h), ("away", a))},
                            "source": "statsapi.mlb.com"}
            for side, t in (("home", h), ("away", a)):
                pp = t.get("probablePitcher") or {}
                if pp.get("fullName"):
                    extras[f"{side}_pitcher"] = {"name": pp["fullName"]}
            common = dict(game_id=canonical_id("MLB", start, home, away), league="MLB", start_time=start,
                          home=home, away=away, neutral=False, extras=extras)
            if state == "post" and h.get("score") is not None and a.get("score") is not None:
                out.append(GameResult(**common, home_score=int(h["score"]), away_score=int(a["score"])))
            else:
                if state == "in" and h.get("score") is not None:
                    extras["live"] = {"home": int(h.get("score") or 0), "away": int(a.get("score") or 0)}
                out.append(Game(**common))
    return out


def _mlb_record(t: dict) -> Optional[str]:
    rec = t.get("leagueRecord") or {}
    if "wins" in rec and "losses" in rec:
        return f"{rec['wins']}-{rec['losses']}"
    return None


def mlb_range(start: date, end: date, progress=None) -> list[Game]:
    out: list[Game] = []
    d = start
    while d <= end:
        stop = min(end, d + timedelta(days=45))
        data = get_json(MLB_BASE, params={"sportId": 1, "startDate": d.isoformat(), "endDate": stop.isoformat(),
                                          "gameType": MLB_GAME_TYPES, "hydrate": "team,probablePitcher"})
        out.extend(parse_mlb_schedule(data or {}))
        if progress:
            progress(f"MLB (statsapi.mlb.com): through {stop.isoformat()}, {len(out)} games")
        d = stop + timedelta(days=1)
    return out


def mlb_day(day: date) -> list[Game]:
    return parse_mlb_schedule(get_json(MLB_BASE, params={
        "sportId": 1, "date": day.isoformat(), "gameType": MLB_GAME_TYPES, "hydrate": "team,probablePitcher"}) or {})


# ------------------------------------------------------------------------ NHL
def _nhl_team(t: dict) -> Optional[str]:
    abbr = t.get("abbrev") or ""
    abbr = NHL_ABBR.get(abbr, abbr)
    if abbr in REGISTRY["NHL"]:
        return abbr
    name = (t.get("name") or {}).get("default") if isinstance(t.get("name"), dict) else t.get("name")
    return resolve_team("NHL", name or "")


def parse_nhl_score(data: dict) -> list[Game]:
    out: list[Game] = []
    for g in (data or {}).get("games", []) or []:
        if g.get("gameType") not in (2, 3):
            continue  # preseason / all-star
        h, a = g.get("homeTeam") or {}, g.get("awayTeam") or {}
        home, away = _nhl_team(h), _nhl_team(a)
        start = _utc(g.get("startTimeUTC"))
        if not home or not away or start is None:
            continue
        gs = (g.get("gameState") or "").upper()
        state = "post" if gs in ("OFF", "FINAL") else "in" if gs in ("LIVE", "CRIT") else "pre"
        pd = g.get("periodDescriptor") or {}
        detail = ""
        if state == "post":
            detail = "Final" + ("/" + pd["periodType"] if pd.get("periodType") in ("OT", "SO") else "")
        elif state == "in":
            detail = f"P{pd.get('number', '')} {(g.get('clock') or {}).get('timeRemaining', '')}".strip()
        extras: dict = {"status": {"state": state, "detail": detail},
                        "teams": {s: {"name": None, "color": None, "alt": None, "record": t.get("record")}
                                  for s, t in (("home", h), ("away", a))},
                        "source": "api-web.nhle.com"}
        common = dict(game_id=canonical_id("NHL", start, home, away), league="NHL", start_time=start,
                      home=home, away=away, neutral=bool(g.get("neutralSite")), extras=extras)
        if state == "post" and h.get("score") is not None and a.get("score") is not None:
            out.append(GameResult(**common, home_score=int(h["score"]), away_score=int(a["score"])))
        else:
            if state == "in" and h.get("score") is not None:
                extras["live"] = {"home": int(h.get("score") or 0), "away": int(a.get("score") or 0)}
            out.append(Game(**common))
    return out


def nhl_day(day: date) -> list[Game]:
    return parse_nhl_score(get_json(f"{NHL_BASE}/{day.isoformat()}") or {})


def nhl_range(start: date, end: date, progress=None) -> list[Game]:
    from concurrent.futures import ThreadPoolExecutor

    days = [start + timedelta(days=k) for k in range((end - start).days + 1)]
    out: list[Game] = []
    with ThreadPoolExecutor(max_workers=2) as ex:
        for k, games in enumerate(ex.map(nhl_day, days), 1):
            out.extend(games)
            if progress and (k % 60 == 0 or k == len(days)):
                progress(f"NHL (api-web.nhle.com): {k}/{len(days)} days, {len(out)} games")
    return out


OFFICIAL_RANGE = {"MLB": mlb_range, "NHL": nhl_range}
OFFICIAL_DAY = {"MLB": mlb_day, "NHL": nhl_day}
