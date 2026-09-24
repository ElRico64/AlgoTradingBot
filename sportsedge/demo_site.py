"""Demo dashboard built from synthetic leagues.

Teams are fictional on purpose: a simulated track record must never be
mistakable for real picks on real teams. Everything else is the real
pipeline: walk-forward backtest for the record, a trained engine for today's
board, the real pick gate.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
from datetime import date, datetime, timedelta
from typing import Optional

from .backtest import walk_forward
from .config import get_config
from .daily import ALL_LEAGUES, build_site_data, pick_label, predict_slate, today_eastern
from .engine import EngineSettings, LeagueEngine
from .news.impact import build_factors
from .picks import PickPolicy
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


def demo_names(league: str) -> dict[str, tuple[str, str]]:
    abbrs = sorted(REGISTRY[league])
    out = {}
    for i, a in enumerate(abbrs):
        city, code = CITIES[(i + OFFSET[league]) % len(CITIES)]
        out[a] = (code, f"{city} {MASCOTS[league][i % 8]}")
    return out


def _shift(g, delta: timedelta):
    return replace(g, start_time=g.start_time + delta)


def build_demo(site_dir: str = "site", leagues=ALL_LEAGUES, today: Optional[date] = None,
               policy: Optional[PickPolicy] = None, seed: int = 3, record_days: int = 75) -> dict:
    from .site import write_site

    today = today or today_eastern()
    policy = policy or PickPolicy()
    runs, ledger = [], []
    for league in leagues:
        names = demo_names(league)
        games = generate(league, seasons=4 if league == "NFL" else 2, seed=seed)
        bt = walk_forward(league, games, policy, n_draws=200)
        dates = sorted({g.start_time.date() for g in games})
        counts = Counter(p["date"] for p in bt.picks)
        # "today" = the recent slate with the most qualifying picks, so the demo shows the features
        d0 = max(dates[-40:], key=lambda d: (counts.get(d.isoformat(), 0), d))
        delta = datetime.combine(today, datetime.min.time()) - datetime.combine(d0, datetime.min.time())
        history = [_shift(g, delta) for g in games if g.start_time.date() < d0]
        slate_src = [_shift(g, delta) for g in games if g.start_time.date() == d0]
        slate = [Game(g.game_id, league, g.start_time, g.home, g.away, g.neutral, {"odds": g.odds})
                 for g in slate_src]
        engine = LeagueEngine(league, EngineSettings(n_draws=1500, seed=seed)).fit(history)
        factors = []
        if slate:
            g0 = slate[0]
            ev = NewsEvent(league, g0.home, "Demo Player", "injury", "questionable", reliability=0.9, role="star",
                           text="Fictional example: listed as questionable")
            factors = build_factors([ev], get_config(league), datetime.combine(today, datetime.min.time()))
        run = predict_slate(engine, slate, {}, factors, policy, today.isoformat(), names)
        runs.append(run)
        by_id = {g.game_id: g for g in games}
        for p in bt.picks:
            d = date.fromisoformat(p["date"])
            if not (d0 - timedelta(days=record_days) <= d < d0):
                continue
            g = by_id[p["game_id"]]
            home, away = names[g.home][0], names[g.away][0]
            status = "push" if p["profit"] == 0 else ("won" if p["profit"] > 0 else "lost")
            day = (d + (today - d0)).isoformat()
            ledger.append({
                "id": f"{day}:{g.game_id}:{p['market']}:{p['side']}", "date": day, "league": league,
                "game_id": g.game_id, "start": (g.start_time + delta).isoformat(), "home": home, "away": away,
                "label": pick_label(home, away, p["market"], p["side"], p["line"]), "market": p["market"],
                "side": p["side"], "line": p["line"], "price": p["price"], "p": round(p["p"], 4),
                "lower": round(p["lower"], 4), "fair": round(p["fair_market"], 4),
                "edge": round(p["p"] - p["fair_market"], 4), "stake": round(p["stake"], 4),
                "status": status, "profit": round(p["profit"], 4),
                "final": f"{away} {g.away_score} – {home} {g.home_score}", "reasons": [],
            })
        ledger.extend(run.new_picks)
    data = build_site_data(runs, ledger, policy, today, mode="demo")
    write_site(data, site_dir)
    return data
