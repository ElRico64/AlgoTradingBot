"""Live news ingestion: ESPN injury reports, ESPN headlines and any RSS feed.

ESPN's site API is public but unofficial and undocumented; its JSON shape can
change without notice, so every parser here is defensive and failures are
logged rather than raised. Check each provider's terms of use before
commercial use.
"""
from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import Iterable, Optional

import requests

from ..teams import resolve_team
from ..types import NewsEvent
from .parser import normalize_status, parse_text

log = logging.getLogger(__name__)

SPORT_PATH = {"NFL": "football/nfl", "NBA": "basketball/nba", "MLB": "baseball/mlb", "NHL": "hockey/nhl"}
ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports"
UA = {"User-Agent": "sportsedge/0.1 (+research)"}

POSITION_ROLE = {"QB": "QB", "G": "G", "SP": "SP", "K": "K"}


def _get_json(url: str, timeout: float = 15.0) -> Optional[dict]:
    try:
        r = requests.get(url, headers=UA, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:  # network / JSON errors are non-fatal
        log.warning("fetch failed %s: %s", url, e)
        return None


def _parse_time(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        try:
            return parsedate_to_datetime(s)
        except Exception:
            return None


def parse_espn_injuries(league: str, data: dict) -> list[NewsEvent]:
    events = []
    for team_block in data.get("injuries", []) or []:
        team_name = team_block.get("displayName") or (team_block.get("team") or {}).get("displayName", "")
        team = resolve_team(league, team_name) or (team_block.get("team") or {}).get("abbreviation")
        for inj in team_block.get("injuries", []) or []:
            ath = inj.get("athlete") or {}
            name = ath.get("displayName") or ath.get("fullName")
            if not name or not team:
                continue
            pos = ((ath.get("position") or {}).get("abbreviation") or "").upper()
            status = normalize_status(inj.get("status") or (inj.get("type") or {}).get("description", ""))
            comment = inj.get("shortComment") or inj.get("longComment") or ""
            if status == "unknown":
                parsed = parse_text(league, comment, team_hint=team, player_hint=name)
                status = parsed[0].status if parsed else "unknown"
            events.append(NewsEvent(
                league=league, team=team, player=name, event_type="injury", status=status,
                published=_parse_time(inj.get("date")), source="espn-injuries", reliability=0.95,
                role=POSITION_ROLE.get(pos), text=comment))
    return events


def fetch_espn_injuries(league: str) -> list[NewsEvent]:
    data = _get_json(f"{ESPN_BASE}/{SPORT_PATH[league]}/injuries")
    return parse_espn_injuries(league, data) if data else []


def parse_espn_news(league: str, data: dict) -> list[tuple[str, Optional[datetime], Optional[str]]]:
    """Return (text, published, team_abbr_hint) tuples for downstream parsing."""
    out = []
    for art in data.get("articles", []) or []:
        text = ". ".join(x for x in (art.get("headline"), art.get("description")) if x)
        team_hint = None
        for c in art.get("categories", []) or []:
            if c.get("type") == "team":
                team_hint = resolve_team(league, c.get("description") or (c.get("team") or {}).get("description", ""))
                if team_hint:
                    break
        out.append((text, _parse_time(art.get("published")), team_hint))
    return out


def fetch_espn_news(league: str, limit: int = 50) -> list[tuple[str, Optional[datetime], Optional[str]]]:
    data = _get_json(f"{ESPN_BASE}/{SPORT_PATH[league]}/news?limit={limit}")
    return parse_espn_news(league, data) if data else []


def fetch_rss(url: str) -> list[tuple[str, Optional[datetime], Optional[str]]]:
    try:
        r = requests.get(url, headers=UA, timeout=15)
        r.raise_for_status()
        root = ET.fromstring(r.content)
    except Exception as e:
        log.warning("rss failed %s: %s", url, e)
        return []
    out = []
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        desc = (item.findtext("description") or "").strip()
        out.append((f"{title}. {desc}".strip(". "), _parse_time(item.findtext("pubDate")), None))
    return out


def gather_news(league: str, rss_urls: Iterable[str] = (), use_espn: bool = True, analyzer=None) -> list[NewsEvent]:
    """Collect structured availability events from all configured sources.

    `analyzer` may be a `ClaudeNewsAnalyzer`; if given, free-text items are
    sent to it, otherwise the rule-based parser is used.
    """
    events: list[NewsEvent] = []
    texts: list[tuple[str, Optional[datetime], Optional[str]]] = []
    if use_espn:
        events.extend(fetch_espn_injuries(league))
        texts.extend(fetch_espn_news(league))
    for u in rss_urls:
        texts.extend(fetch_rss(u))
    if analyzer is not None and texts:
        events.extend(analyzer.analyze(league, texts))
    else:
        for text, when, team in texts:
            events.extend(parse_text(league, text, when, "news", team_hint=team))
    return events
