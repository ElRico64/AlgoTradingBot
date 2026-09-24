"""The run behind the dashboard. Safe to run every 30 minutes.

First run of the day (per league):
  * append yesterday's results (ESPN) to data/history/<league>.csv
  * retrain the engine on the full history and cache it for the day
Every run:
  * pull the day's scoreboard (upcoming, live, final), grade finished picks
  * pull fresh odds (The Odds API if a key is set, else ESPN's posted lines)
    and fresh news/injuries
  * re-predict every game that has not started; games that have started keep
    the prediction they had at first pitch / puck drop / tip-off / kickoff
  * publish new picks in two tiers:
      value       70%+ AND +EV at the listed price (needs a price)
      confidence  70%+ win probability, with the worst price still worth taking
  * withdraw a pending pick if news before the start pushes it below the gate
  * write the dashboard

The ledger (data/ledger.json) records every published pick with the time and
price it was published at; grading never rewrites those fields.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import traceback
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Optional

from .config import get_config
from .engine import LeagueEngine
from .news.impact import AvailabilityFactor
from .picks import PickPolicy, confidence_policy, value_line
from .stats.odds import american_to_decimal, breakeven_probability, prob_to_american
from .teams import REGISTRY
from .types import Game, GameResult, NewsEvent, Pick, Prediction

log = logging.getLogger(__name__)

try:
    from zoneinfo import ZoneInfo

    US_EASTERN = ZoneInfo("America/New_York")
except Exception:  # pragma: no cover
    US_EASTERN = timezone(timedelta(hours=-5))

ALL_LEAGUES = ("MLB", "NHL", "NBA", "NFL")
TIERS = ("value", "confidence")
VOID_AFTER_DAYS = 4
LOOKAHEAD_DAYS = {"NFL": 6}  # NFL plays weekly: show the whole upcoming week
TEAM_COLORS = ["#1D428A", "#C8102E", "#00471B", "#F58426", "#552583", "#007A78", "#B4975A", "#222222",
               "#0C2340", "#862633", "#005A9C", "#E03A3E", "#4F2683", "#006BB6", "#CE1141", "#00338D"]


def today_eastern() -> date:
    return datetime.now(US_EASTERN).date()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def game_day(g: Game) -> str:
    """The game's US Eastern calendar date (start times are stored as naive UTC)."""
    return g.start_time.replace(tzinfo=timezone.utc).astimezone(US_EASTERN).date().isoformat()


def _state(g: Game) -> str:
    if isinstance(g, GameResult):
        return "post"
    return ((g.extras or {}).get("status") or {}).get("state", "pre")


# --------------------------------------------------------------------------- ledger
def pick_id(tier: str, day: str, game_id: str, market: str, side: str) -> str:
    return f"{tier}:{day}:{game_id}:{market}:{side}"


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
    profit = None
    if entry.get("price") is not None:
        profit = 0.0 if status == "push" else (american_to_decimal(entry["price"]) - 1 if status == "won" else -1.0)
        profit = round(profit, 4)
    entry.update(status=status, profit=profit, graded_at=_now(),
                 final=f"{entry.get('away', result.away)} {result.away_score} – "
                       f"{entry.get('home', result.home)} {result.home_score}")
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
            e.update(status="void", profit=0.0 if e.get("price") is not None else None)
    return n


def ledger_entry(pk: Pick, tier: str, names: dict | None, min_ev: float) -> dict:
    m, g = pk.market, pk.game
    day = game_day(g)
    home, away = _code(g.home, names), _code(g.away, names)
    return {
        "id": pick_id(tier, day, g.game_id, m.market, m.side), "tier": tier, "date": day, "league": g.league,
        "game_id": g.game_id, "start": g.start_time.isoformat(), "home": home, "away": away,
        "label": pick_label(home, away, m.market, m.side, m.line), "market": m.market, "side": m.side,
        "line": m.line, "price": m.price, "p": round(m.probability, 4), "lower": round(m.lower, 4),
        "upper": round(m.upper, 4), "fair": None if m.fair_market_prob is None else round(m.fair_market_prob, 4),
        "edge": None if m.edge is None else round(m.edge, 4),
        "ev": None if m.expected_value is None else round(m.expected_value, 4),
        "value_line": round(value_line(m.probability, min_ev)),
        "stake": round(pk.stake_fraction, 4) if tier == "value" else 0.0,
        "reasons": [_relabel_note(r, names) for r in pk.reasons],
        "published_at": _now(), "status": "pending", "profit": None,
    }


# ----------------------------------------------------------------------- serialise
def _color(abbr: str) -> str:
    return TEAM_COLORS[int(hashlib.md5(abbr.encode()).hexdigest(), 16) % len(TEAM_COLORS)]


def _team(league: str, abbr: str, side: str, game: Game, names: dict | None) -> dict:
    disp = ((game.extras or {}).get("teams") or {}).get(side) or {}
    if names and abbr in names:
        code, name = names[abbr]
        return {"abbr": code, "name": name, "color": _color(code), "record": None}
    t = REGISTRY.get(league, {}).get(abbr)
    return {"abbr": abbr, "name": disp.get("name") or (t.name if t else abbr),
            "color": disp.get("color") or _color(abbr), "record": disp.get("record")}


def _relabel_note(note: str, names: dict | None) -> str:
    if not names:
        return note
    for abbr, (code, _) in names.items():
        note = note.replace(f"({abbr},", f"({code},")
    return note


def market_json(m, game: Game, policies: dict[str, PickPolicy], names: dict | None) -> dict:
    label = pick_label(_code(game.home, names), _code(game.away, names), m.market, m.side, m.line)
    return {
        "market": m.market, "side": m.side, "line": m.line, "label": label,
        "price": m.price, "p": round(m.probability, 4), "lo": round(m.lower, 4), "hi": round(m.upper, 4),
        "fair": None if m.fair_market_prob is None else round(m.fair_market_prob, 4),
        "edge": None if m.edge is None else round(m.edge, 4),
        "ev": None if m.expected_value is None else round(m.expected_value, 4),
        "breakeven": None if m.price is None else round(breakeven_probability(m.price), 4),
        "fair_price": round(prob_to_american(m.probability)),
        "value_line": round(value_line(m.probability, policies["value"].min_ev)),
        "push": round(m.push_probability, 4),
        "fails": policies["value"].failures(m), "conf_fails": policies["confidence"].failures(m),
    }


def game_json(pred: Prediction, policies: dict[str, PickPolicy], picks: dict[str, list[Pick]],
              names: dict | None, news: list[dict]) -> dict:
    g = pred.game
    markets = [market_json(m, g, policies, names) for m in pred.markets]
    best = max(markets, key=lambda x: x["p"]) if markets else None
    tier_labels = {}
    for tier, pks in picks.items():
        pk = next((p for p in pks if p.game.game_id == g.game_id), None)
        tier_labels[tier] = None if pk is None else pick_label(_code(g.home, names), _code(g.away, names),
                                                               pk.market.market, pk.market.side, pk.market.line)
    return {
        "id": g.game_id, "league": g.league, "start": g.start_time.isoformat(),
        "home": _team(g.league, g.home, "home", g, names), "away": _team(g.league, g.away, "away", g, names),
        "p_home": round(pred.home_win_prob, 4), "lo": round(pred.home_win_lower, 4),
        "hi": round(pred.home_win_upper, 4), "exp_margin": round(pred.expected_margin, 2),
        "exp_total": round(pred.expected_total, 2),
        "components": {k: round(v, 4) for k, v in pred.components.items()},
        "adjustments": {k: round(v, 3) for k, v in pred.adjustments.items()},
        "notes": [_relabel_note(n, names) for n in pred.notes], "news": news,
        "markets": markets, "best": best, "picks": tier_labels, "predicted_at": _now(), "frozen": False,
    }


def news_json(ev: NewsEvent, names: dict | None) -> dict:
    return {"league": ev.league, "team": _code(ev.team, names), "player": ev.player, "status": ev.status,
            "event": ev.event_type, "role": ev.role, "text": ev.text[:280], "source": ev.source,
            "published": ev.published.isoformat() if ev.published else None}


# ---------------------------------------------------------------------- the run
@dataclass
class LeagueRun:
    league: str
    status: str = "ok"
    message: str = ""
    games: list[dict] = field(default_factory=list)
    picks: list[dict] = field(default_factory=list)  # entries published or kept today
    news: list[dict] = field(default_factory=list)
    trained_games: int = 0
    calibrated: bool = False
    market_weight: Optional[float] = None
    odds_source: str = "none"


def _policies(policy: PickPolicy) -> dict[str, PickPolicy]:
    return {"value": policy, "confidence": confidence_policy(policy)}


def predict_games(engine: LeagueEngine, games: list[Game], odds_map: dict, events: list[NewsEvent],
                  factors: list[AvailabilityFactor], policy: PickPolicy, day: str, ledger: list[dict],
                  names: dict | None = None, previous: dict | None = None) -> LeagueRun:
    """Predict upcoming games, keep frozen predictions for started ones, and
    reconcile the ledger (new picks, withdrawals, reinstatements)."""
    policies = _policies(policy)
    run = LeagueRun(engine.league, trained_games=len(engine.history), calibrated=engine.calibrated)
    run.market_weight = engine.stacker.weights().get("mkt_model:market")
    previous = previous or {}
    by_id = {e["id"]: e for e in ledger}
    fresh: list[Prediction] = []
    frozen_json: list[dict] = []
    for g in games:
        state = _state(g)
        prev = previous.get(g.game_id)
        if state != "pre" and prev is not None:
            gj = dict(prev)
            gj["frozen"] = True
            frozen_json.append(_with_status(gj, g))
            continue
        odds = odds_map.get((g.home, g.away)) or (g.extras or {}).get("odds") or getattr(g, "odds", None)
        if odds is not None and run.odds_source == "none":
            run.odds_source = odds.source
        pred = engine.predict(g, odds, factors)
        fresh.append(pred)
    upcoming = [p for p in fresh if _state(p.game) == "pre"]
    picks = {t: pol.select(upcoming) for t, pol in policies.items()}
    teams_today = {g.home for g in games} | {g.away for g in games}
    news = [news_json(ev, names) for ev in events if ev.team in teams_today]
    run.news = sorted(news, key=lambda n: n["published"] or "", reverse=True)[:60]
    for pred in fresh:
        gnews = [n for n in run.news if n["team"] in (_code(pred.game.home, names), _code(pred.game.away, names))]
        gj = game_json(pred, policies, picks, names, gnews[:6])
        gj["frozen"] = _state(pred.game) != "pre"
        run.games.append(_with_status(gj, pred.game))
    run.games.extend(frozen_json)
    run.games.sort(key=lambda x: x["start"])

    # --- reconcile the ledger for upcoming games ---------------------------------
    preds_by_game = {p.game.game_id: p for p in upcoming}
    for e in ledger:
        if e.get("league") != engine.league or e.get("status") not in ("pending", "withdrawn"):
            continue
        pred = preds_by_game.get(e["game_id"])
        if pred is None:
            continue  # started, finished or not on today's board: leave it alone
        m = next((x for x in pred.markets if x.market == e["market"] and x.side == e["side"]
                  and x.line == e["line"]), None)
        if m is None:
            continue  # the line moved; the pick stands at the line it was published at
        pol = policies[e.get("tier", "value")]
        still = [f for f in pol.failures(m) if f.startswith(("probability", "too uncertain"))]
        if still and e["status"] == "pending":
            e.update(status="withdrawn", withdrawn_at=_now(), note="Withdrawn before start: " + "; ".join(still))
        elif not still and e["status"] == "withdrawn":
            e.update(status="pending", note="Reinstated after new information", p=round(m.probability, 4))
    for tier, pks in picks.items():
        for pk in pks:
            entry = ledger_entry(pk, tier, names, policy.min_ev)
            if entry["id"] not in by_id:
                ledger.append(entry)
                by_id[entry["id"]] = entry
    ids_today = {g["id"] for g in run.games}
    run.picks = [e for e in ledger if e.get("league") == engine.league and e["game_id"] in ids_today]
    if not games:
        run.status, run.message = "idle", "No games scheduled."
    return run


def _with_status(gj: dict, g: Game) -> dict:
    ex = g.extras or {}
    st = ex.get("status") or {}
    gj["state"] = "post" if isinstance(g, GameResult) else st.get("state", "pre")
    gj["detail"] = st.get("detail") or ("Final" if gj["state"] == "post" else "")
    live = ex.get("live")
    if isinstance(g, GameResult):
        live = {"home": g.home_score, "away": g.away_score}
    gj["live"] = live
    return gj


def _load_engine(league: str, data_dir: str, day: date, bootstrap_days: int) -> tuple[LeagueEngine, list[GameResult]]:
    from .data.csv_io import read_results, write_results
    from .data.espn import fetch_history

    cache = os.path.join(data_dir, ".cache", f"{league}-{day.isoformat()}.pkl")
    if os.path.exists(cache):
        with open(cache, "rb") as f:
            eng = pickle.load(f)
        return eng, eng.history
    hist_path = os.path.join(data_dir, "history", f"{league}.csv")
    history = read_results(hist_path, league) if os.path.exists(hist_path) else []
    start = (history[-1].start_time.date() - timedelta(days=2)) if history else day - timedelta(days=bootstrap_days)
    merged = {g.game_id: g for g in history}
    merged.update({g.game_id: g for g in fetch_history(league, start, day - timedelta(days=1))})
    history = sorted(merged.values(), key=lambda g: g.start_time)
    if not history:
        raise RuntimeError("No historical results available (is ESPN reachable?)")
    os.makedirs(os.path.dirname(hist_path), exist_ok=True)
    write_results(hist_path, history)
    eng = LeagueEngine(league).fit(history)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    for old in os.listdir(os.path.dirname(cache)):
        if old.startswith(f"{league}-") and old != os.path.basename(cache):
            os.remove(os.path.join(os.path.dirname(cache), old))
    with open(cache, "wb") as f:
        pickle.dump(eng, f)
    return eng, history


def run_league(league: str, data_dir: str, day: date, policy: PickPolicy, ledger: list[dict],
               odds_key: Optional[str], analyzer, bootstrap_days: int) -> LeagueRun:
    from .data.espn import fetch_day
    from .news.feeds import gather_news
    from .news.impact import PlayerImpactRegistry, build_factors

    engine, history = _load_engine(league, data_dir, day, bootstrap_days)
    games: list[Game] = []
    for k in range(LOOKAHEAD_DAYS.get(league, 0) + 1):
        games.extend(fetch_day(league, day + timedelta(days=k)))
    finals = {g.game_id: g for g in history}
    finals.update({g.game_id: g for g in fetch_day(league, day - timedelta(days=1)) if isinstance(g, GameResult)})
    finals.update({g.game_id: g for g in games if isinstance(g, GameResult)})
    settle_ledger([e for e in ledger if e["league"] == league], finals, day)

    upcoming = [g for g in games if _state(g) == "pre"]
    odds_map, events, factors = {}, [], []
    if upcoming:
        if odds_key:
            from .data.oddsapi import fetch_odds

            odds_map = fetch_odds(league, odds_key, regions=os.environ.get("ODDS_API_REGIONS", "us"))
        events = gather_news(league, analyzer=analyzer)
        imp_path = os.path.join(data_dir, "player_impacts.csv")
        reg = PlayerImpactRegistry.from_csv(imp_path) if os.path.exists(imp_path) else None
        factors = build_factors(events, get_config(league), datetime.now(timezone.utc), reg)
    board_path = os.path.join(data_dir, ".cache", f"board-{league}.json")
    previous = {}
    if os.path.exists(board_path):
        with open(board_path) as f:
            saved = json.load(f)
        previous = {g["id"]: g for g in saved.get("games", [])}
    run = predict_games(engine, games, odds_map, events, factors, policy, day.isoformat(), ledger,
                        previous=previous)
    if odds_map:
        run.odds_source = "the-odds-api"
    os.makedirs(os.path.dirname(board_path), exist_ok=True)
    with open(board_path, "w") as f:
        json.dump({"date": day.isoformat(), "games": run.games}, f)
    return run


def build_site_data(runs: list[LeagueRun], ledger: list[dict], policy: PickPolicy, day: date,
                    mode: str = "live", refresh_minutes: int = 30) -> dict:
    return {
        "generated_at": _now(), "date": day.isoformat(), "mode": mode, "refresh_minutes": refresh_minutes,
        "gate": {"min_probability": policy.min_probability, "min_lower_bound": policy.min_lower_bound,
                 "require_positive_ev": policy.require_positive_ev, "min_edge": policy.min_edge,
                 "min_ev": policy.min_ev, "kelly_multiplier": policy.kelly_multiplier},
        "leagues": [{"league": r.league, "status": r.status, "message": r.message, "games": len(r.games),
                     "picks": len([p for p in r.picks if p["status"] != "withdrawn"]),
                     "trained_games": r.trained_games, "calibrated": r.calibrated,
                     "market_weight": r.market_weight, "odds_source": r.odds_source} for r in runs],
        "games": [g for r in runs for g in r.games],
        "news": sorted([n for r in runs for n in r.news], key=lambda n: n["published"] or "", reverse=True)[:80],
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
        runs.append(run)
    os.makedirs(data_dir, exist_ok=True)
    with open(ledger_path, "w") as f:
        json.dump(ledger, f, indent=1)
    data = build_site_data(runs, ledger, policy, day, "live", int(os.environ.get("REFRESH_MINUTES", "30")))
    write_site(data, site_dir)
    return data
