"""Situational adjustments: rest, travel, time zones, altitude, starting
pitchers (MLB), starting goalies (NHL) and weather.

Quality stats for pitchers/goalies are shrunk toward the league mean with
empirical-Bayes (Beta/Normal-Normal) shrinkage, because small samples of ERA
or save % are mostly noise:

    x_shrunk = (n * x_obs + k * x_league) / (n + k)

where k is the stabilisation sample size (innings or shots) at which the
observed stat is ~50% signal.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from ..config import LeagueConfig
from ..teams import get_team, haversine_miles
from ..types import Game

LEAGUE_RA9 = 4.35
PITCHER_FIP_K = 80.0  # innings
PITCHER_ERA_K = 150.0
STARTER_SHARE = 0.55  # share of innings a starter pitches, hedged for team-level double counting
LEAGUE_SV = 0.903
GOALIE_SV_K = 2000.0  # shots
GOALIE_SHARE = 0.6


@dataclass
class Adjustment:
    margin: float = 0.0  # home-perspective margin units
    total: float = 0.0  # additive to expected total (gaussian leagues)
    log_rate_home: float = 0.0  # count leagues
    log_rate_away: float = 0.0
    parts: dict[str, float] = field(default_factory=dict)

    def add_margin(self, name: str, v: float) -> None:
        if v:
            self.margin += v
            self.parts[name] = self.parts.get(name, 0.0) + v


@dataclass
class ScheduleTracker:
    last_time: dict[str, datetime] = field(default_factory=dict)
    last_venue: dict[str, str] = field(default_factory=dict)

    def record(self, g: Game) -> None:
        for t in (g.home, g.away):
            self.last_time[t] = g.start_time
            self.last_venue[t] = g.home

    def rest_days(self, team: str, when: datetime) -> Optional[float]:
        t = self.last_time.get(team)
        if t is None:
            return None
        return (when - t).total_seconds() / 86400.0

    def travel(self, league: str, team: str, venue: str) -> tuple[float, float]:
        prev = self.last_venue.get(team)
        a, b = get_team(league, prev) if prev else None, get_team(league, venue)
        if a is None or b is None:
            return 0.0, 0.0
        return haversine_miles(a.lat, a.lon, b.lat, b.lon), abs(a.lon - b.lon) / 15.0


def _shrink(obs: float, n: float, league: float, k: float) -> float:
    return (n * obs + k * league) / (n + k)


def pitcher_log_factor(p: dict | None) -> tuple[float, str]:
    if not p:
        return 0.0, ""
    # Unknown sample size: assume a moderate one rather than full shrinkage.
    ip = 60.0 if p.get("ip") is None else float(p["ip"])
    if p.get("fip") is not None:
        ra = _shrink(float(p["fip"]), ip, LEAGUE_RA9, PITCHER_FIP_K)
    elif p.get("era") is not None:
        ra = _shrink(float(p["era"]), ip, LEAGUE_RA9, PITCHER_ERA_K)
    else:
        return 0.0, ""
    return STARTER_SHARE * math.log(max(ra, 1.0) / LEAGUE_RA9), f"{p.get('name', 'SP')} RA9~{ra:.2f}"


def goalie_log_factor(gk: dict | None) -> tuple[float, str]:
    if not gk or gk.get("sv_pct") is None:
        return 0.0, ""
    sa = float(gk.get("shots_against") or 0.0)
    sv = _shrink(float(gk["sv_pct"]), sa, LEAGUE_SV, GOALIE_SV_K)
    return GOALIE_SHARE * math.log((1 - sv) / (1 - LEAGUE_SV)), f"{gk.get('name', 'G')} sv%~{sv:.3f}"


def compute_adjustment(cfg: LeagueConfig, game: Game, sched: ScheduleTracker) -> Adjustment:
    adj = Adjustment()
    lg = cfg.league
    # --- rest -------------------------------------------------------------
    rh = sched.rest_days(game.home, game.start_time)
    ra = sched.rest_days(game.away, game.start_time)
    if rh is not None and ra is not None:
        if cfg.back_to_back_penalty:
            adj.add_margin("back_to_back", -cfg.back_to_back_penalty * (rh < 1.5) + cfg.back_to_back_penalty * (ra < 1.5))
        diff = max(-3.0, min(3.0, min(rh, 10) - min(ra, 10)))
        adj.add_margin("rest", cfg.rest_advantage_per_day * diff)
    # --- travel / time zones ---------------------------------------------------
    if not game.neutral:
        da, tza = sched.travel(lg, game.away, game.home)
        dh, tzh = sched.travel(lg, game.home, game.home)
        adj.add_margin("travel", cfg.travel_penalty_per_1000mi * (da - dh) / 1000.0)
        adj.add_margin("timezones", cfg.timezone_penalty_per_hour * (round(tza) - round(tzh)))
        th, ta = get_team(lg, game.home), get_team(lg, game.away)
        if th and th.altitude and not (ta and ta.altitude):
            adj.add_margin("altitude", cfg.altitude_bonus)
    # --- starters ------------------------------------------------------------------
    ex = game.extras or {}
    if lg == "MLB":
        f_home_sp, d1 = pitcher_log_factor(ex.get("home_pitcher"))
        f_away_sp, d2 = pitcher_log_factor(ex.get("away_pitcher"))
        adj.log_rate_away += f_home_sp  # home SP suppresses away runs
        adj.log_rate_home += f_away_sp
        if f_home_sp:
            adj.parts["home_SP_log_runs_allowed"] = f_home_sp
        if f_away_sp:
            adj.parts["away_SP_log_runs_allowed"] = f_away_sp
    if lg == "NHL":
        f_hg, _ = goalie_log_factor(ex.get("home_goalie"))
        f_ag, _ = goalie_log_factor(ex.get("away_goalie"))
        adj.log_rate_away += f_hg
        adj.log_rate_home += f_ag
        if f_hg:
            adj.parts["home_G_log_goals_allowed"] = f_hg
        if f_ag:
            adj.parts["away_G_log_goals_allowed"] = f_ag
    # --- weather ----------------------------------------------------------------------
    w = ex.get("weather") or {}
    if w and not ex.get("indoor"):
        temp = w.get("temp_f")
        wind = w.get("wind_mph") or 0.0
        if lg == "NFL":
            t_adj = -0.3 * max(0.0, wind - 12.0) - (1.0 if w.get("precip") else 0.0) - \
                    (1.0 if temp is not None and temp < 25 else 0.0)
            if t_adj:
                adj.total += t_adj
                adj.parts["weather_total"] = t_adj
        if lg == "MLB":
            f = 0.0
            if temp is not None:
                f += 0.002 * (temp - 70.0)
            f += 0.01 * float(w.get("wind_out_mph") or 0.0)
            if f:
                adj.log_rate_home += f
                adj.log_rate_away += f
                adj.parts["weather_log_runs"] = f
    return adj
