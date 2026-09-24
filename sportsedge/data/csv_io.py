"""CSV persistence for historical results (the input to training/backtests).

Required columns: date, home, away, home_score, away_score
Optional:        game_id, league, neutral, home_ml, away_ml, spread,
                 home_spread_price, away_spread_price, total, over_price,
                 under_price, extras (JSON)
`date` may be YYYY-MM-DD or a full ISO-8601 timestamp. `spread` is the home
line (negative = home favoured).
"""
from __future__ import annotations

import csv
import json
from datetime import datetime
from typing import Iterable

from ..types import Game, GameResult, MarketOdds

FIELDS = ["game_id", "league", "date", "home", "away", "home_score", "away_score", "neutral",
          "home_ml", "away_ml", "spread", "home_spread_price", "away_spread_price", "total",
          "over_price", "under_price", "extras"]


def _f(v):
    return None if v in (None, "", "NA", "nan", "None") else float(v)


def _dt(s: str) -> datetime:
    s = s.strip()
    if len(s) == 10:
        return datetime.fromisoformat(s + "T00:00:00")
    return datetime.fromisoformat(s.replace("Z", "+00:00")).replace(tzinfo=None)


def _parse_row(row: dict, i: int, league: str, with_scores: bool = True) -> GameResult:
    odds = None
    if any(_f(row.get(k)) is not None for k in ("home_ml", "spread", "total")):
        odds = MarketOdds(
            home_ml=_f(row.get("home_ml")), away_ml=_f(row.get("away_ml")),
            spread=_f(row.get("spread")),
            home_spread_price=_f(row.get("home_spread_price")) or -110.0,
            away_spread_price=_f(row.get("away_spread_price")) or -110.0,
            total=_f(row.get("total")),
            over_price=_f(row.get("over_price")) or -110.0,
            under_price=_f(row.get("under_price")) or -110.0, source="csv")
    extras = json.loads(row["extras"]) if row.get("extras") else {}
    return GameResult(
        game_id=row.get("game_id") or f"{league}-{i}", league=(row.get("league") or league).upper(),
        start_time=_dt(row["date"]), home=row["home"].strip(), away=row["away"].strip(),
        neutral=str(row.get("neutral", "")).strip().lower() in ("1", "true", "yes"),
        extras=extras,
        home_score=int(float(row["home_score"])) if with_scores else 0,
        away_score=int(float(row["away_score"])) if with_scores else 0,
        odds=odds)


def read_results(path: str, league: str) -> list[GameResult]:
    with open(path, newline="") as f:
        out = [_parse_row(row, i, league) for i, row in enumerate(csv.DictReader(f))]
    out.sort(key=lambda g: g.start_time)
    return out


def read_slate(path: str, league: str) -> list[Game]:
    """Upcoming games: same columns as results with scores omitted. Odds, if
    present, are attached as `extras['odds']`."""
    games = []
    with open(path, newline="") as f:
        for i, row in enumerate(csv.DictReader(f)):
            r = _parse_row(row, i, league, with_scores=False)
            games.append(Game(r.game_id, r.league, r.start_time, r.home, r.away, r.neutral,
                              {**r.extras, "odds": r.odds}))
    games.sort(key=lambda g: g.start_time)
    return games


DISPLAY_ONLY = ("status", "teams", "live", "odds")


def _model_extras(extras: dict) -> dict:
    return {k: v for k, v in (extras or {}).items() if k not in DISPLAY_ONLY}


def write_results(path: str, games: Iterable[GameResult]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for g in games:
            o = g.odds or MarketOdds()
            w.writerow({
                "game_id": g.game_id, "league": g.league, "date": g.start_time.isoformat(),
                "home": g.home, "away": g.away, "home_score": g.home_score, "away_score": g.away_score,
                "neutral": int(g.neutral), "home_ml": o.home_ml, "away_ml": o.away_ml, "spread": o.spread,
                "home_spread_price": o.home_spread_price if g.odds else None,
                "away_spread_price": o.away_spread_price if g.odds else None, "total": o.total,
                "over_price": o.over_price if g.odds else None, "under_price": o.under_price if g.odds else None,
                "extras": json.dumps(_model_extras(g.extras)) if _model_extras(g.extras) else "",
            })
