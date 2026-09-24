"""The daily run behind the dashboard.

For every league, each morning:
  1. append yesterday's results (ESPN) to data/history/<league>.csv
  2. grade any pending picks in the ledger against those results
  3. retrain the engine on the full history (calibrators fit out-of-sample)
  4. fetch today's slate, odds (The Odds API if a key is set, else ESPN) and news
  5. predict every game, apply the pick gate, add new picks to the ledger
  6. write site data (picks, full board with near misses, track record)

The ledger (data/ledger.json) is append-only apart from grading, so the track
record on the dashboard is exactly what was published before each game.
"""
from __future__ import annotations

import json
import logging
import os
import traceback
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Optional

from .config import get_config
from .engine import LeagueEngine
from .picks import PickPolicy
from .stats.odds import american_to_decimal, breakeven_probability, prob_to_american
from .teams import REGISTRY
from .types import Game, GameResult, Pick, Prediction

log = logging.getLogger(__name__)

try:
    from zoneinfo import ZoneInfo

    US_EASTERN = ZoneInfo("America/New_York")
except Exception:  # pragma: no cover
    US_EASTERN = timezone(timedelta(hours=-5))

ALL_LEAGUES = ("MLB", "NHL", "NBA", "NFL")
VOID_AFTER_DAYS = 4


def today_eastern() -> date:
    return datetime.now(US_EASTERN).date()


# --------------------------------------------------------------------------- ledger
def pick_id(day: str, game_id: str, market: str, side: str) -> str:
    return f"{day}:{game_id}:{market}:{side}"


def pick_label(home: str, away: str, market: str, side: str, line: Optional[float]) -> str:
    if market == "moneyline":
        return f"{home if side == 'home' else away} ML"
    if market == "spread":
        team = home if side == "home" else away
        ln = line if side == "home" else -line
        return f"{team} {ln:+g}"
    return f"{'Over' if side == 'over' else 'Under'} {line:g}"


def _code(abbr: str, names: dict | None) -> str:
    return names[abbr][0] if names and abbr in names else abbr


def grade(entry: dict, result: GameResult) -> dict:
    m, side, line = entry["market"], entry["side"], entry.get("line")
    if m == "moneyline":
        edge = result.margin if side == "home" else -result.margin
    elif m == "spread":
        edge = (result.margin + line) if side == "home" else -(result.margin + line)
    else:
        edge = (result.total - line) if side == "over" else (line - result.total)
    status = "push" if edge == 0 else ("won" if edge > 0 else "lost")
    profit = 0.0 if status == "push" else (american_to_decimal(entry["price"]) - 1 if status == "won" else -1.0)
    entry.update(status=status, profit=round(profit, 4),
                 final=f"{result.away} {result.away_score} – {result.home} {result.home_score}",
                 graded_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    return entry


def settle_ledger(ledger: list[dict], results: dict[str, GameResult], today: date) -> int:
    n = 0
    for e in ledger:
        if e.get("status") != "pending":
            continue
        r = results.get(e["game_id"])
        if r is not None:
            grade(e, r)
            n += 1
        elif (today - date.fromisoformat(e["date"])).days > VOID_AFTER_DAYS:
            e.update(status="void", profit=0.0)
    return n


def ledger_entry(day: str, pk: Pick, names: dict | None = None) -> dict:
    m, g = pk.market, pk.game
    home, away = _code(g.home, names), _code(g.away, names)
    return {
        "id": pick_id(day, g.game_id, m.market, m.side), "date": day, "league": g.league,
        "game_id": g.game_id, "start": g.start_time.isoformat(), "home": home, "away": away,
        "label": pick_label(home, away, m.market, m.side, m.line), "market": m.market, "side": m.side, "line": m.line,
        "price": m.price, "p": round(m.probability, 4), "lower": round(m.lower, 4), "upper": round(m.upper, 4),
        "fair": None if m.fair_market_prob is None else round(m.fair_market_prob, 4),
        "edge": None if m.edge is None else round(m.edge, 4),
        "ev": None if m.expected_value is None else round(m.expected_value, 4),
        "stake": round(pk.stake_fraction, 4), "reasons": pk.reasons, "status": "pending", "profit": None,
    }


# ----------------------------------------------------------------------- serialise
def _team(league: str, abbr: str, names: dict | None) -> dict:
    if names and abbr in names:
        code, name = names[abbr]
        return {"abbr": code, "name": name}
    t = REGISTRY.get(league, {}).get(abbr)
    return {"abbr": abbr, "name": t.name if t else abbr}


def _relabel_note(note: str, names: dict | None) -> str:
    if not names:
        return note
    for abbr, (code, _) in names.items():
        note = note.replace(f"({abbr},", f"({code},")
    return note


def market_json(m, game: Game, policy: PickPolicy, names: dict | None = None) -> dict:
    label = pick_label(_code(game.home, names), _code(game.away, names), m.market, m.side, m.line)
    return {
        "market": m.market, "side": m.side, "line": m.line, "label": label,
        "price": m.price, "p": round(m.probability, 4), "lo": round(m.lower, 4), "hi": round(m.upper, 4),
        "fair": None if m.fair_market_prob is None else round(m.fair_market_prob, 4),
        "edge": None if m.edge is None else round(m.edge, 4),
        "ev": None if m.expected_value is None else round(m.expected_value, 4),
        "breakeven": None if m.price is None else round(breakeven_probability(m.price), 4),
        "fair_price": round(prob_to_american(m.probability)), "push": round(m.push_probability, 4),
        "fails": policy.failures(m),
    }


def game_json(pred: Prediction, policy: PickPolicy, picks: list[Pick], names: dict | None = None) -> dict:
    g = pred.game
    markets = [market_json(m, g, policy, names) for m in pred.markets]
    best = max(markets, key=lambda x: x["p"]) if markets else None
    pk = next((p for p in picks if p.game.game_id == g.game_id), None)
    return {
        "id": g.game_id, "league": g.league, "start": g.start_time.isoformat(),
        "home": _team(g.league, g.home, names), "away": _team(g.league, g.away, names),
        "p_home": round(pred.home_win_prob, 4), "lo": round(pred.home_win_lower, 4),
        "hi": round(pred.home_win_upper, 4), "exp_margin": round(pred.expected_margin, 2),
        "exp_total": round(pred.expected_total, 2),
        "components": {k: round(v, 4) for k, v in pred.components.items()},
        "adjustments": {k: round(v, 3) for k, v in pred.adjustments.items()},
        "notes": [_relabel_note(n, names) for n in pred.notes], "markets": markets, "best": best,
        "pick": None if pk is None else pick_label(_code(g.home, names), _code(g.away, names),
                                                   pk.market.market, pk.market.side, pk.market.line),
    }


# ---------------------------------------------------------------------- the run
@dataclass
class LeagueRun:
    league: str
    status: str = "ok"
    message: str = ""
    games: list[dict] = field(default_factory=list)
    new_picks: list[dict] = field(default_factory=list)
    trained_games: int = 0
    calibrated: bool = False
    market_weight: Optional[float] = None


def predict_slate(engine: LeagueEngine, slate: list[Game], odds_map: dict, factors, policy: PickPolicy,
                  day: str, names: dict | None = None) -> LeagueRun:
    run = LeagueRun(engine.league, trained_games=len(engine.history), calibrated=engine.calibrated)
    w = engine.stacker.weights()
    run.market_weight = w.get("mkt_model:market")
    preds = []
    for g in slate:
        odds = odds_map.get((g.home, g.away)) or g.extras.pop("odds", None)
        preds.append(engine.predict(g, odds, factors))
    picks = policy.select(preds)
    run.games = [game_json(p, policy, picks, names) for p in preds]
    run.new_picks = [ledger_entry(day, pk, names) for pk in picks]
    if not slate:
        run.status, run.message = "idle", "No games scheduled today."
    return run


def run_league(league: str, data_dir: str, day: date, policy: PickPolicy, ledger: list[dict],
               odds_key: Optional[str], analyzer, bootstrap_days: int) -> LeagueRun:
    from .data.csv_io import read_results, write_results
    from .data.espn import fetch_history, fetch_slate
    from .news.feeds import gather_news
    from .news.impact import PlayerImpactRegistry, build_factors

    hist_path = os.path.join(data_dir, "history", f"{league}.csv")
    history = read_results(hist_path, league) if os.path.exists(hist_path) else []
    start = (history[-1].start_time.date() - timedelta(days=2)) if history else day - timedelta(days=bootstrap_days)
    fresh = fetch_history(league, start, day - timedelta(days=1))
    merged = {g.game_id: g for g in history}
    merged.update({g.game_id: g for g in fresh})
    history = sorted(merged.values(), key=lambda g: g.start_time)
    if not history:
        return LeagueRun(league, "error", "No historical results available (ESPN unreachable?).")
    os.makedirs(os.path.dirname(hist_path), exist_ok=True)
    write_results(hist_path, history)
    graded = settle_ledger([e for e in ledger if e["league"] == league], merged, day)
    log.info("%s: %d history games, graded %d picks", league, len(history), graded)

    engine = LeagueEngine(league).fit(history)
    slate = fetch_slate(league, day)
    odds_map = {}
    if odds_key and slate:
        from .data.oddsapi import fetch_odds

        odds_map = fetch_odds(league, odds_key, regions=os.environ.get("ODDS_API_REGIONS", "us"))
    factors = []
    if slate:
        events = gather_news(league, analyzer=analyzer)
        imp_path = os.path.join(data_dir, "player_impacts.csv")
        reg = PlayerImpactRegistry.from_csv(imp_path) if os.path.exists(imp_path) else None
        factors = build_factors(events, get_config(league), datetime.now(timezone.utc), reg)
    return predict_slate(engine, slate, odds_map, factors, policy, day.isoformat())


def build_site_data(runs: list[LeagueRun], ledger: list[dict], policy: PickPolicy, day: date,
                    mode: str = "live") -> dict:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "date": day.isoformat(), "mode": mode,
        "gate": {"min_probability": policy.min_probability, "min_lower_bound": policy.min_lower_bound,
                 "require_positive_ev": policy.require_positive_ev, "min_edge": policy.min_edge,
                 "min_ev": policy.min_ev,
                 "kelly_multiplier": policy.kelly_multiplier},
        "leagues": [{"league": r.league, "status": r.status, "message": r.message, "games": len(r.games),
                     "picks": len(r.new_picks), "trained_games": r.trained_games, "calibrated": r.calibrated,
                     "market_weight": r.market_weight} for r in runs],
        "games": [g for r in runs for g in r.games],
        "picks": [p for r in runs for p in r.new_picks],
        "ledger": sorted(ledger, key=lambda e: (e["date"], e["id"]), reverse=True),
    }


def run_daily(leagues=ALL_LEAGUES, data_dir: str = "data", site_dir: str = "site",
              day: Optional[date] = None, policy: Optional[PickPolicy] = None,
              odds_key: Optional[str] = None, use_llm: bool = False, bootstrap_days: int = 730) -> dict:
    from .site import write_site

    day = day or today_eastern()
    policy = policy or PickPolicy()
    ledger_path = os.path.join(data_dir, "ledger.json")
    ledger = json.load(open(ledger_path)) if os.path.exists(ledger_path) else []
    analyzer = None
    if use_llm:
        from .news.llm import ClaudeNewsAnalyzer

        analyzer = ClaudeNewsAnalyzer()
    runs = []
    for lg in leagues:
        try:
            run = run_league(lg, data_dir, day, policy, ledger, odds_key, analyzer, bootstrap_days)
        except Exception as e:  # one league failing must not take down the others
            log.error("%s failed: %s\n%s", lg, e, traceback.format_exc())
            run = LeagueRun(lg, "error", f"{type(e).__name__}: {e}")
        known = {e["id"] for e in ledger}
        ledger.extend(p for p in run.new_picks if p["id"] not in known)
        runs.append(run)
    os.makedirs(data_dir, exist_ok=True)
    with open(ledger_path, "w") as f:
        json.dump(ledger, f, indent=1)
    data = build_site_data(runs, ledger, policy, day, "live")
    write_site(data, site_dir)
    return data
