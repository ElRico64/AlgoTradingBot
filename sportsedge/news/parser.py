"""Rule-based extraction of injury / availability news from free text.

This is the dependency-free baseline. `news/llm.py` offers a Claude-powered
extractor that handles messier prose; both emit the same `NewsEvent`s.
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Optional

from ..teams import REGISTRY
from ..types import NewsEvent

# (pattern, status, event_type) — checked in order, first match wins
STATUS_RULES: list[tuple[re.Pattern, str, str]] = [
    (re.compile(r"\b(activated|reinstated)\b", re.I), "available", "lineup"),
    (re.compile(r"\bsuspend(ed|s|ing)?\b", re.I), "out", "suspension"),
    (re.compile(r"\b(season[- ]ending|out for (the )?(rest of the )?season|out indefinitely)\b", re.I), "ir", "injury"),
    (re.compile(r"\b(placed on|landed on|to) (the )?(injured reserve|IR|(10|15|60)-day (IL|injured list)|injured list|IL|long-term injured reserve|LTIR)\b", re.I), "ir", "injury"),
    (re.compile(r"\b(long-term injured reserve|LTIR|injured reserve|(7|10|15|60)-day (IL|injured list))\b", re.I), "ir", "injury"),
    (re.compile(r"\b(ruled out|will not play|won't play|will miss|won't suit up|sidelined|scratched|out (tonight|for (the )?game|vs\.?|against|with))\b", re.I), "out", "injury"),
    (re.compile(r"\b(will rest|resting|load management|rest day|given the night off)\b", re.I), "out", "rest"),
    (re.compile(r"\bdoubtful\b", re.I), "doubtful", "injury"),
    (re.compile(r"\bgame[- ]time decision\b", re.I), "game-time decision", "injury"),
    (re.compile(r"\bquestionable\b", re.I), "questionable", "injury"),
    (re.compile(r"\bday[- ]to[- ]day\b", re.I), "day-to-day", "injury"),
    (re.compile(r"\bprobable\b", re.I), "probable", "injury"),
    (re.compile(r"\b(cleared to (play|return)|will (play|start|return|suit up)|activated|returns? (tonight|to the lineup)|(is|was) available)\b", re.I), "available", "lineup"),
    (re.compile(r"\btraded\b", re.I), "out", "trade"),
    (re.compile(r"\((\w+[\s/-]?){1,3}\)\s+out\b|\bout\b(?!\s+of\b)(?!side)", re.I), "out", "injury"),
]

RUMOR = re.compile(r"\b(reportedly|sources?|rumou?r|expected to|likely|could|may|might|hopes? to|trending)\b", re.I)
ROLE_RULES = [
    (re.compile(r"\b(QB|quarterback)\b"), "QB"),
    (re.compile(r"\b(goalie|goaltender|netminder)\b", re.I), "G"),
    (re.compile(r"\b(starting pitcher|right-hander|left-hander|righty|lefty|SP)\b"), "SP"),
    (re.compile(r"\b(All-Star|MVP|superstar|star)\b", re.I), "star"),
    (re.compile(r"\b(kicker)\b", re.I), "K"),
]
NAME = re.compile(r"\b([A-Z][a-zA-Z'\.\-]+(?:\s+(?:[A-Z][a-zA-Z'\.\-]+|de|van|von|St\.))+(?:\s+(?:Jr\.|Sr\.|II|III|IV))?)")
STOP_NAMES = {
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    "January", "February", "March", "April", "May", "June", "July", "August",
    "September", "October", "November", "December", "Game", "Week", "Injury Report",
    "Head Coach", "General Manager", "Associated Press", "Eastern Conference", "Western Conference",
}


def _team_spans(league: str, text: str) -> list[tuple[int, str, int]]:
    out = []
    for abbr, t in REGISTRY.get(league, {}).items():
        for alias in {t.name, t.nickname}:
            for m in re.finditer(rf"\b{re.escape(alias)}\b", text):
                out.append((m.start(), abbr, m.end()))
    return sorted(out)


POSITION_TOKENS = re.compile(
    r"\b(QB|RB|WR|TE|OL|OT|DE|DT|LB|CB|FS|SS|K|P|PG|SG|SF|PF|C|F|G|SP|RP|RHP|LHP|OF|IF|1B|2B|3B|D|LW|RW)\b")


def _find_player(text: str, team_spans) -> Optional[str]:
    # blank out team names and position tokens so they cannot glue onto a name
    chars = list(text)
    for s, _, e in team_spans:
        chars[s:e] = "|" * (e - s)
    masked = POSITION_TOKENS.sub(lambda m: "|" * len(m.group(0)), "".join(chars))
    for m in NAME.finditer(masked):
        name = m.group(1).strip()
        if name in STOP_NAMES or any(w in STOP_NAMES for w in name.split()):
            continue
        return name.rstrip(".")
    return None


def parse_text(league: str, text: str, published: Optional[datetime] = None, source: str = "text",
               team_hint: Optional[str] = None, player_hint: Optional[str] = None) -> list[NewsEvent]:
    """Extract at most one availability event from a headline/blurb."""
    if not text:
        return []
    status = event_type = None
    for pat, st, et in STATUS_RULES:
        if pat.search(text):
            status, event_type = st, et
            break
    if status is None:
        return []
    spans = _team_spans(league, text)
    team = team_hint or (spans[0][1] if spans else None)
    player = player_hint or _find_player(text, spans)
    if team is None or player is None:
        return []
    role = next((r for pat, r in ROLE_RULES if pat.search(text)), None)
    reliability = 0.6 if RUMOR.search(text) else 0.9
    return [NewsEvent(league=league, team=team, player=player, event_type=event_type, status=status,
                      published=published, source=source, reliability=reliability, role=role, text=text)]


def normalize_status(raw: str) -> str:
    s = (raw or "").strip().lower()
    mapping = {
        "out": "out", "o": "out", "inactive": "out", "injured reserve": "ir", "ir": "ir",
        "10-day il": "ir", "15-day il": "ir", "60-day il": "ir", "7-day il": "ir", "ltir": "ir",
        "suspension": "suspended", "suspended": "suspended",
        "doubtful": "doubtful", "d": "doubtful", "questionable": "questionable", "q": "questionable",
        "day-to-day": "day-to-day", "dtd": "day-to-day", "probable": "probable", "p": "probable",
        "active": "available", "available": "available",
    }
    if s in mapping:
        return mapping[s]
    for k, v in mapping.items():
        if len(k) > 2 and k in s:
            return v
    return "unknown"
