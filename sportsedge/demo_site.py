"""Demo dashboard built from synthetic leagues.

Teams and players are fictional on purpose: a simulated track record must
never be mistakable for real picks on real teams. Everything else is the real
pipeline: a walk-forward backtest for the record, a trained engine for the
day's board, the real news mixture model and the real pick gates.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
from datetime import date, datetime, time, timedelta, timezone
from typing import Optional

from .backtest import walk_forward
from .config import get_config
from .daily import (ALL_LEAGUES, US_EASTERN, _code, build_site_data, pick_id, pick_label, predict_games,
                    today_eastern)
from .engine import EngineSettings, LeagueEngine
from .news.impact import build_factors
from .picks import PickPolicy, confidence_policy, value_line
from .synthetic import generate
from .teams import REGISTRY
from .types import Game, NewsEvent

CITIES = [
    ("Harborview", "HBV"), ("Red Mesa", "RMS"), ("Northgate", "NGT"), ("Silver Falls", "SVF"),
    ("Cedar Ridge", "CDR"), ("Port Alder", "PTA"), ("Lakemont", "LKM"), ("Granite Bay", "GRB"),
    ("Ironwood", "IRW"), ("Stonebridge", "STB"), ("Fairhaven", "FHV"), ("Bluewater", "BLW"),
    ("Pine Hollow", "PNH"), ("Eastmoor", "EMR"), ("Copper Flats", "CPF"), ("Kestrel Bay", "KSB"),
    ("Willow Creek", "WLC"), ("Summit Hill", "SMH"), ("Brightwater", "BRW"), ("Alder Coast", "ALC"),
    ("Falcon Point", "FCP"), ("Marlow", "MRL"), ("Thornbury", "THB"), ("Wrenfield", "WRF"),
    ("Ember Valley", "EMV"), ("Glass Harbor", "GLH"), ("Highcastle", "HCS"), ("Juniper Vale", "JNV"),
    ("Larkspur", "LKS"), ("Maple Crossing", "MPC"), ("Obsidian Bay", "OBB"), ("Quarry Park", "QYP"),
]
MASCOTS = {
    "MLB": ["Comets", "Foxes", "Anchors", "Pilots", "Owls", "Rustlers", "Lanterns", "Herons"],
    "NHL": ["Glaciers", "Lynx", "Blizzard", "Otters", "Frost", "Moose", "Pikes", "Icebreakers"],
    "NBA": ["Dunes", "Voltage", "Cyclones", "Pioneers", "Cobras", "Aviators", "Monarchs", "Tempest"],
    "NFL": ["Ironclads", "Stallions", "Sentinels", "Bison", "Marauders", "Storm", "Badgers", "Harriers"],
}
OFFSET = {"MLB": 0, "NHL": 7, "NBA": 15, "NFL": 23}
# (days from today, ET start) slots, cycled over each league's demo slate
SLOTS = {
    "MLB": [(0, time(13, 5)), (0, time(16, 10)), (0, time(18, 40)), (0, time(19, 5)), (0, time(19, 10)),
            (0, time(20, 10)), (0, time(21, 40)), (0, time(22, 10))],
    "NHL": [(0, time(19, 0)), (0, time(19, 30)), (0, time(20, 0)), (0, time(22, 0))],
    "NBA": [(0, time(19, 0)), (0, time(19, 30)), (0, time(20, 0)), (0, time(21, 0)), (0, time(22, 0))],
    "NFL": [(0, time(20, 15))] + [(3, time(13, 0))] * 9 + [(3, time(16, 5)), (3, time(16, 25)),
                                                         (3, time(16, 25)), (3, time(20, 20)), (4, time(20, 15))],
}
PLAYERS = ["Rowan Vale", "Idris Moreau", "Tomas Keel", "Dario Fenwick", "Caspian Holt", "Emeric Dunn",
           "Soren Pike", "Lucan Marsh", "Arlo Brenner", "Felix Oyelaran", "Kade Winslow", "Ansel Rourke"]
BLURBS = [
    ("questionable", "star", "{p} ({inj}) is listed as questionable and will be a game-time decision"),
    ("out", "starter", "{p} has been ruled out with a {inj} injury, the team announced"),
    ("doubtful", "star", "{p} is doubtful after missing practice with {inj} soreness"),
    ("probable", "starter", "{p} ({inj}) is probable and expected to play"),
]
INJURIES = ["ankle", "hamstring", "knee", "illness", "back", "wrist"]


def demo_names(league: str) -> dict[str, tuple[str, str]]:
    abbrs = sorted(REGISTRY[league])
    out = {}
    for i, a in enumerate(abbrs):
        city, code = CITIES[(i + OFFSET[league]) % len(CITIES)]
        out[a] = (code, f"{city} {MASCOTS[league][i % 8]}")
    return out


def _utc(day: date, t: time) -> datetime:
    return datetime.combine(day, t, US_EASTERN).astimezone(timezone.utc).replace(tzinfo=None)


def build_demo(site_dir: str = "site", leagues=ALL_LEAGUES, today: Optional[date] = None,
               policy: Optional[PickPolicy] = None, seed: int = 3, record_days: int = 75,
               progress=None, parallel: bool = True) -> dict:
    from concurrent.futures import ProcessPoolExecutor

    from .site import write_site

    today = today or today_eastern()
    now = datetime.now(timezone.utc)
    policy = policy or PickPolicy()
    jobs = [(li, lg, today, now, policy, seed, record_days) for li, lg in enumerate(leagues)]
    results = []
    if parallel and len(jobs) > 1:
        try:
            with ProcessPoolExecutor(max_workers=min(4, len(jobs))) as ex:
                for lg, res in zip(leagues, ex.map(_demo_league, jobs)):
                    results.append(res)
                    if progress:
                        progress(f"{lg}: simulated season, backtest and today's board done")
        except (OSError, RuntimeError):  # no multiprocessing available: fall back to one at a time
            results = []
    if not results:
        for job in jobs:
            results.append(_demo_league(job))
            if progress:
                progress(f"{job[1]}: simulated season, backtest and today's board done")
    runs = [r for r, _ in results]
    ledger = [e for _, entries in results for e in entries]
    data = build_site_data(runs, ledger, policy, today, mode="demo")
    write_site(data, site_dir)
    return data


def _demo_league(job):
    li, league, today, now, policy, seed, record_days = job
    names = demo_names(league)
    ledger: list[dict] = []
    games = generate(league, seasons=4 if league == "NFL" else 2, seed=seed, patterns=True)
    bt = walk_forward(league, games, policy, n_draws=200, extra_tiers={"confidence": confidence_policy(policy)})
    dates = sorted({g.start_time.date() for g in games})
    value_counts = Counter(p["date"] for p in bt.picks if p["tier"] == "value")
    counts = Counter(p["date"] for p in bt.picks)
    # "today" = a recent slate with qualifying picks, so the demo shows the features
    d0 = max(dates[-40:], key=lambda d: (min(value_counts.get(d.isoformat(), 0), 2),
                                         counts.get(d.isoformat(), 0), d))
    delta = datetime.combine(today, time()) - datetime.combine(d0, time())
    history = [replace(g, start_time=g.start_time + delta) for g in games if g.start_time.date() < d0]
    src = [g for g in games if g.start_time.date() == d0]
    slate = []
    for i, g in enumerate(src):
        off, t = SLOTS[league][i % len(SLOTS[league])]
        slate.append(Game(g.game_id, league, _utc(today + timedelta(days=off), t), g.home, g.away, g.neutral,
                          {"odds": g.odds, "status": {"state": "pre", "detail": ""}}))
    engine = LeagueEngine(league, EngineSettings(n_draws=1500, seed=seed)).fit(history)
    events = []
    for k, g in enumerate(slate[:3]):
        status, role, text = BLURBS[(k + li) % len(BLURBS)]
        player = PLAYERS[(3 * li + k) % len(PLAYERS)]
        team = g.home if k % 2 == 0 else g.away
        events.append(NewsEvent(league, team, player, "injury", status,
                                published=now - timedelta(minutes=25 + 70 * k + 13 * li),
                                source="Team injury report" if k % 2 else "Beat reporter", reliability=0.9,
                                role=role, text=text.format(p=player, inj=INJURIES[(k + li) % 6])))
    factors = build_factors(events, get_config(league), now)
    run = predict_games(engine, slate, {}, events, factors, policy, today.isoformat(), ledger, names)
    _demo_states(run, src, names, now)
    by_id = {g.game_id: g for g in games}
    for p in bt.picks:
        d = date.fromisoformat(p["date"])
        if not (d0 - timedelta(days=record_days) <= d < d0):
            continue
        g = by_id[p["game_id"]]
        home, away = _code(g.home, names), _code(g.away, names)
        status = "push" if p["profit"] == 0 else ("won" if p["profit"] > 0 else "lost")
        day = (d + (today - d0)).isoformat()
        tier = p["tier"]
        ledger.append({
            "id": pick_id(tier, day, g.game_id, p["market"], p["side"]), "tier": tier, "date": day,
            "league": league, "game_id": g.game_id, "start": (g.start_time + delta).isoformat(),
            "home": home, "away": away, "label": pick_label(home, away, p["market"], p["side"], p["line"]),
            "market": p["market"], "side": p["side"], "line": p["line"], "price": p["price"],
            "p": round(p["p"], 4), "lower": round(p["lower"], 4), "fair": round(p["fair_market"], 4),
            "edge": round(p["p"] - p["fair_market"], 4), "value_line": round(value_line(p["p"], policy.min_ev)),
            "stake": round(p["stake"], 4) if tier == "value" else 0.0, "status": status,
            "profit": round(p["profit"], 4), "final": f"{away} {g.away_score} – {home} {g.home_score}",
            "reasons": [],
        })
    return run, ledger


def _demo_states(run, src, names, now) -> None:
    """Show the live/final states: the earliest games without a pick become
    'final' and 'in progress' using their simulated scores."""
    results = {g.game_id: g for g in src}
    started = [g for g in run.games if not any(g["picks"].values())
               and datetime.fromisoformat(g["start"]).replace(tzinfo=timezone.utc) < now][:2]
    for gj, state in zip(started, ("post", "in")):
        r = results[gj["id"]]
        gj["frozen"] = True
        gj["state"] = state
        if state == "post":
            gj["detail"], gj["live"] = "Final", {"home": r.home_score, "away": r.away_score}
        else:
            gj["detail"] = {"MLB": "Bot 6th", "NHL": "2nd 08:41", "NBA": "3rd 5:12", "NFL": "3rd 9:04"}[run.league]
            gj["live"] = {"home": int(r.home_score * 0.6), "away": int(r.away_score * 0.6)}
