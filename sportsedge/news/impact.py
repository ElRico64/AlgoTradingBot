"""Turn news into probabilistic availability factors.

Each uncertain player becomes a Bernoulli mixture component:

    P(win) = sum over availability scenarios s of P(s) * P(win | s)

where P(plays) comes from the reported status (league-specific priors in
config.py), shrunk toward "plays" when the report is unreliable or stale:

    q = 1 - r * (1 - q_status),   r = reliability * exp(-max(0, age_h - 24) / 96)

and the player's value (margin units) is itself uncertain,
impact ~ N(mu, sd^2), truncated at 0. The engine integrates over all of this
by Monte Carlo, so a "questionable" star *widens* the probability band
instead of just nudging the point estimate — and a wide band can push a pick
below the confidence gate. That is the intended behaviour.

Long-term absences (IR / season-ending) are down-weighted because a team's
recent results — and therefore its rating — already reflect them.
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Optional

from ..config import LeagueConfig
from ..types import NewsEvent

DEFENSIVE_ROLES = {"G", "SP"}
LONG_TERM_DISCOUNT = 0.3


@dataclass
class AvailabilityFactor:
    team: str
    player: str
    play_prob: float
    impact: float
    impact_sd: float
    defensive: bool
    label: str


class PlayerImpactRegistry:
    """Optional table of player values. CSV columns:
    league,team,player,impact,impact_sd[,role]
    `impact` is the margin cost (points / goals / runs per game) of losing
    the player versus his replacement (e.g. from RAPM/EPM/WAR/QB EPA models)."""

    def __init__(self):
        self._d: dict[tuple[str, str], tuple[float, float, Optional[str]]] = {}

    def add(self, league: str, player: str, impact: float, impact_sd: float, role: Optional[str] = None):
        self._d[(league.upper(), player.lower())] = (impact, impact_sd, role)

    def get(self, league: str, player: str):
        return self._d.get((league.upper(), (player or "").lower()))

    @classmethod
    def from_csv(cls, path: str) -> "PlayerImpactRegistry":
        reg = cls()
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                reg.add(row["league"], row["player"], float(row["impact"]),
                        float(row.get("impact_sd") or float(row["impact"]) * 0.4), row.get("role") or None)
        return reg


def _aware(d: Optional[datetime]) -> datetime:
    if d is None:
        return datetime.min.replace(tzinfo=timezone.utc)
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def _age_hours(published: Optional[datetime], as_of: datetime) -> float:
    if published is None:
        return 0.0
    return max(0.0, (_aware(as_of) - _aware(published)).total_seconds() / 3600.0)


def build_factors(events: Iterable[NewsEvent], cfg: LeagueConfig, as_of: datetime,
                  registry: Optional[PlayerImpactRegistry] = None) -> list[AvailabilityFactor]:
    latest: dict[tuple[str, str], NewsEvent] = {}
    for ev in events:
        if not ev.player or ev.league.upper() != cfg.league:
            continue
        key = (ev.team, ev.player.lower())
        cur = latest.get(key)
        if cur is None or _aware(ev.published) >= _aware(cur.published):
            latest[key] = ev
    factors = []
    for ev in latest.values():
        q_status = cfg.status_play_prob.get(ev.status, cfg.status_play_prob.get("unknown", 0.85))
        r = max(0.0, min(1.0, ev.reliability)) * math.exp(-max(0.0, _age_hours(ev.published, as_of) - 24) / 96)
        q = 1.0 - r * (1.0 - q_status)
        if q >= 0.999:
            continue
        reg = registry.get(cfg.league, ev.player) if registry else None
        role = ev.role or (reg[2] if reg else None)
        if reg:
            imp, sd = reg[0], reg[1]
        elif ev.impact is not None:
            imp, sd = ev.impact, ev.impact_sd if ev.impact_sd is not None else abs(ev.impact) * 0.5
        elif role and role in cfg.role_impacts:
            imp, sd = cfg.role_impacts[role]
        else:
            imp, sd = cfg.default_player_impact, cfg.default_player_impact_sd
        if ev.status == "ir" or ev.event_type == "trade":
            imp *= LONG_TERM_DISCOUNT
            sd *= LONG_TERM_DISCOUNT
        if imp <= 0:
            continue
        factors.append(AvailabilityFactor(
            team=ev.team, player=ev.player, play_prob=q, impact=imp, impact_sd=sd,
            defensive=role in DEFENSIVE_ROLES,
            label=f"{ev.player} ({ev.team}, {ev.status}{', ' + role if role else ''}) P(plays)={q:.0%} "
                  f"impact≈{imp:.2f}±{sd:.2f}"))
    return factors
