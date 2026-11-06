"""NBA player-props parlays for the board.

Rule (as requested): a props parlay pays at least 2.0x and the model gives it at
least a 65% chance of hitting. Legs come from different games (results of
separate games are independent; teammates' and opponents' stats in one game are
not), each from a posted price. Each leg's chance is the calibrated player
model, market-checked against the books' no-vig price with a share learned from
graded props. Payouts are never invented: without posted prices there are no
props parlays, only the model's projections.
"""
from __future__ import annotations

import json
import logging
import math
import os
import pickle
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import numpy as np

from ..parlays import ParlayPolicy, et_date, parlay_band, parlay_entry, scan, settle_parlay
from ..stats.market_check import check, fit_shares
from .boxscores import load_rows, update_boxscores
from .model import PROP_STATS, STAT_NAMES, PropModel
from .prices import fetch_espn_props, fetch_oddsapi_props, norm_name, priced_from_pairs

log = logging.getLogger(__name__)

PROP_POLICY = ParlayPolicy(min_decimal=2.0, min_probability=0.65, min_ev=0.0, min_leg_ev=-1.0,
                           min_leg_probability=0.55, max_legs=4, max_per_day=2)
MIN_ROWS = 3000          # box-score rows before the model projects anything
MIN_MINUTES = 20.0       # projected minutes for a leg
MIN_PLAYER_GAMES = 15    # games in the data before a player can be a leg (thin records disagree with books out of ignorance)
INJURED = ("out", "doubtful", "questionable", "ir", "suspended")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _props_dir(data_dir: str) -> str:
    d = os.path.join(data_dir, "props")
    os.makedirs(d, exist_ok=True)
    return d


def _load_json(path: str, default):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def _save_json(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=0, sort_keys=True)
    os.replace(tmp, path)


def train_model(data_dir: str, day: date, rows: list[dict]) -> PropModel:
    cache = os.path.join(data_dir, ".cache", f"props-model-{day.isoformat()}-{len(rows)}.pkl")
    if os.path.exists(cache):
        try:
            with open(cache, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass
    model = PropModel().fit(rows)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    for old in os.listdir(os.path.dirname(cache)):
        if old.startswith("props-model-"):
            os.remove(os.path.join(os.path.dirname(cache), old))
    with open(cache + ".tmp", "wb") as f:
        pickle.dump(model, f)
    os.replace(cache + ".tmp", cache)
    return model


def _leg_outcome(leg: dict, by_game: dict) -> Optional[str]:
    game = by_game.get(leg["game_id"])
    if game is None:
        return None                       # box score not downloaded yet
    row = game.get(leg["player_id"])
    if row is None or row.get(leg["stat"]) is None:
        return "void"                     # did not play: books void the leg
    v = row[leg["stat"]]
    if v == leg["line"]:
        return "push"
    won = v > leg["line"] if leg["side"] == "over" else v < leg["line"]
    return "won" if won else "lost"


def grade_props(ledger: list[dict], rows: list[dict], today: date) -> int:
    by_game: dict = {}
    for r in rows:
        by_game.setdefault(r["game_id"], {})[r["player_id"]] = r
    n = 0
    for e in ledger:
        if e.get("tier") != "props" or e.get("status") != "pending":
            continue
        for leg in e["legs"]:
            if leg.get("status") == "pending":
                out = _leg_outcome(leg, by_game)
                if out is None and (today - date.fromisoformat(leg["date"])).days > 4:
                    out = "void"
                if out is not None:
                    leg["status"] = out
                    v = (by_game.get(leg["game_id"]) or {}).get(leg["player_id"], {}).get(leg["stat"])
                    leg["final"] = None if v is None else f"{v:g}"
        before = e["status"]
        settle_parlay(e)
        n += e["status"] != before
    return n


def prop_shares(data_dir: str, rows: list[dict]) -> dict:
    """Market check for props: share of the model's disagreement with the books that held up,
    from every priced prop whose box score is in (wins and losses alike)."""
    learn = _load_json(os.path.join(_props_dir(data_dir), "learn.json"), {})
    by_game: dict = {}
    for r in rows:
        by_game.setdefault(r["game_id"], {})[r["player_id"]] = r
    out = []
    for k, (gid, pid, stat, line, p_raw, fair) in learn.items():
        row = (by_game.get(gid) or {}).get(pid)
        if row is None or row.get(stat) is None or row[stat] == line:
            continue
        out.append(("NBA", "prop", p_raw, fair, float(row[stat] > line)))
    return fit_shares(out, leagues=("NBA",))


def _injured(news: list[dict]) -> set:
    return {norm_name(n.get("player") or "") for n in news if (n.get("status") or "") in INJURED and n.get("player")}


def _player_index(model: PropModel, day: date) -> dict:
    """(normalized name, team) -> player id, for players active in the last 20 days."""
    idx = {}
    cutoff = datetime.combine(day, datetime.min.time()) - timedelta(days=20)
    for pid, p in model.players.items():
        if p.last is not None and p.last >= cutoff:
            idx[(norm_name(p.name), p.team)] = pid
    return idx


def build_legs(model: PropModel, games: list[dict], priced: dict, news: list[dict], share: float,
               day: date, learn: dict) -> list[dict]:
    idx, hurt, legs = _player_index(model, day), _injured(news), []
    active = {(pid, t) for (_, t), pid in idx.items()}
    for g in games:
        home, away = g["home"]["abbr"], g["away"]["abbr"]
        for pr in priced.get(g["id"], []):
            pid = pr.get("player_id")
            if pid:  # ESPN: the same athlete ids as the box scores
                pl0 = model.players.get(pid)
                if pl0 is None or pl0.team not in (home, away) or (pid, pl0.team) not in active:
                    continue
                nn = norm_name(pl0.name)
            else:
                nn = norm_name(pr["player"])
                pid = idx.get((nn, home)) or idx.get((nn, away))
            if pid is None or nn in hurt or not model.calibrated(pr["stat"]):
                continue
            pl = model.players[pid]
            if (pl.min_ewma or 0) < MIN_MINUTES or pl.games < MIN_PLAYER_GAMES:
                continue
            opp = away if pl.team == home else home
            mu = model.mean(pid, pr["stat"], opp)
            if mu is None:
                continue
            p_over = model.p_over(pr["stat"], mu, pr["line"])
            learn[f"{g['id']}|{pid}|{pr['stat']}|{pr['line']}"] = [g["id"], pid, pr["stat"], pr["line"],
                                                                    round(p_over, 4), round(pr["fair"], 4)]
            for side in ("over", "under"):
                if pr.get(side) is None:
                    continue  # alternate lines are over-only
                p_raw = p_over if side == "over" else 1 - p_over
                fair = pr["fair"] if side == "over" else 1 - pr["fair"]
                p = check(p_raw, fair, share)
                label = f"{pl.name} {side} {pr['line']:g} {STAT_NAMES[pr['stat']]}"
                legs.append({"league": "NBA", "game_id": g["id"], "date": day.isoformat(), "start": g["start"],
                             "home": home, "away": away, "market": f"prop:{pr['stat']}", "side": side,
                             "line": pr["line"], "label": label, "price": pr[side], "p": round(p, 4),
                             "p_model": round(p_raw, 4), "lo": max(0.0, p - 0.04), "hi": min(1.0, p + 0.04),
                             "fair": round(fair, 4), "status": "pending", "player_id": pid, "player": pl.name,
                             "team": pl.team, "stat": pr["stat"], "projection": round(mu, 1)})
    return legs


def _shortlist(legs: list[dict]) -> list[dict]:
    """Per game: the most likely leg and the best-value leg (search size stays small)."""
    from ..parlays import american_to_decimal

    by_game: dict = {}
    for x in legs:
        if x["p"] >= PROP_POLICY.min_leg_probability:
            by_game.setdefault(x["game_id"], []).append(x)
    out = []
    for xs in by_game.values():
        best_p = max(xs, key=lambda x: x["p"])
        best_v = max(xs, key=lambda x: x["p"] * american_to_decimal(x["price"]))
        out.extend([best_p] + ([best_v] if best_v is not best_p else []))
    return out


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def espn_orientation(data_dir: str, model: PropModel, pairs_today: dict, rows: list[dict], day: date,
                     games: list[dict] = ()) -> dict:
    """Which of ESPN's two prices is the over? Learned, never assumed, two ways:
    * model check (now): if the first price is the over, the books' no-vig chance of the
      first item rises with the model's P(over) across hundreds of props;
    * results check (after games): the first item's chance predicts "stat > line".
    Known only when the evidence is decisive and the two checks do not disagree."""
    from .prices import _american_to_prob

    path = os.path.join(_props_dir(data_dir), "espn_orientation.json")
    snap_dir = os.path.join(_props_dir(data_dir), "espn")
    os.makedirs(snap_dir, exist_ok=True)
    if pairs_today:
        _save_json(os.path.join(snap_dir, f"{day.isoformat()}.json"), pairs_today)
    for f in os.listdir(snap_dir):  # keep three weeks of snapshots
        if f.endswith(".json") and f[:10] < (day - timedelta(days=21)).isoformat():
            os.remove(os.path.join(snap_dir, f))

    def q1(pr):
        a, b = (_american_to_prob(x) for x in pr["prices"])
        return a / (a + b)

    teams = {g["id"]: (g["home"]["abbr"], g["away"]["abbr"]) for g in games}
    xs, ms = [], []
    for gid, pairs in pairs_today.items():
        home, away = teams.get(gid, ("", ""))
        for pr in pairs:
            p = model.players.get(pr["player_id"])
            if p is None or not model.calibrated(pr["stat"]):
                continue
            mu = model.mean(pr["player_id"], pr["stat"], away if p.team == home else home)
            if mu is None:
                continue
            xs.append(_logit(q1(pr)))
            ms.append(_logit(model.p_over(pr["stat"], mu, pr["line"])))
    r_model = float(np.corrcoef(xs, ms)[0, 1]) if len(xs) >= 30 and np.std(xs) > 0 and np.std(ms) > 0 else None
    by_game: dict = {}
    for r in rows:
        by_game.setdefault(r["game_id"], {})[r["player_id"]] = r
    ll, n_res = 0.0, 0
    for f in sorted(os.listdir(snap_dir)):
        if not f.endswith(".json") or f[:10] >= day.isoformat():
            continue
        for gid, pairs in _load_json(os.path.join(snap_dir, f), {}).items():
            for pr in pairs:
                row = (by_game.get(gid) or {}).get(pr["player_id"])
                if row is None or row.get(pr["stat"]) is None or row[pr["stat"]] == pr["line"]:
                    continue
                ll += (1 if row[pr["stat"]] > pr["line"] else -1) * _logit(q1(pr))
                n_res += 1
    vote_model = 0 if r_model is None or len(xs) < 100 or abs(r_model) < 0.2 else (1 if r_model > 0 else -1)
    vote_res = 0 if n_res < 200 or abs(ll) < 10 else (1 if ll > 0 else -1)
    if vote_model and vote_res and vote_model != vote_res:
        verdict = 0
    else:
        verdict = vote_model or vote_res
    out = {"first_is_over": None if verdict == 0 else verdict > 0, "model_corr": r_model, "model_n": len(xs),
           "results_loglik_diff": round(ll, 2), "results_n": n_res, "checked_at": _now()}
    _save_json(path, out)
    return out


def projections(model: PropModel, games: list[dict], news: list[dict], day: date, top: int = 15) -> list[dict]:
    hurt, cutoff, out = _injured(news), datetime.combine(day, datetime.min.time()) - timedelta(days=20), []
    for g in games:
        home, away = g["home"]["abbr"], g["away"]["abbr"]
        for pid, p in model.players.items():
            if p.team not in (home, away) or p.last is None or p.last < cutoff or (p.min_ewma or 0) < MIN_MINUTES:
                continue
            if norm_name(p.name) in hurt:
                continue
            opp = away if p.team == home else home
            proj = {s: model.mean(pid, s, opp) for s in PROP_STATS}
            if any(v is None for v in proj.values()):
                continue
            out.append({"player": p.name, "team": p.team, "opp": opp, "game_id": g["id"], "start": g["start"],
                        "min": round(p.min_ewma, 1), **{s: round(v, 1) for s, v in proj.items()}})
    out.sort(key=lambda r: -r["pts"])
    return out[:top]


def run_props(data_dir: str, day: date, games_json: list[dict], news: list[dict], odds_key: Optional[str],
              ledger: list[dict], budget=None, fetch_seconds: Optional[float] = None) -> dict:
    d = _props_dir(data_dir)
    hist_path = os.path.join(data_dir, "history", "NBA.csv")
    game_days = []
    if os.path.exists(hist_path):
        import csv
        with open(hist_path, newline="") as f:
            game_days = sorted({datetime.fromisoformat(r["date"]).date() for r in csv.DictReader(f)})
    secs = float(os.environ.get("PROPS_FETCH_SECONDS", "150")) if fetch_seconds is None else fetch_seconds
    box = update_boxscores(data_dir, game_days, day, budget_seconds=secs)
    rows = load_rows(data_dir)
    graded = grade_props(ledger, rows, day)
    status = {"updated_at": _now(), "boxscores": box, "rows": len(rows), "graded_now": graded}
    section = {"ready": False, "status": status, "projections": [], "closest": None, "prices": "", "espn": ""}
    if len(rows) < MIN_ROWS:
        section["message"] = (f"Collecting player box scores: {box['games_stored']} games so far. Projections "
                              f"start at about {MIN_ROWS // 25} games.")
        _save_json(os.path.join(d, "status.json"), {**status, "message": section["message"]})
        return section
    model = train_model(data_dir, day, rows)
    status["model"] = model.status()
    today = [g for g in games_json if g.get("state") == "pre" and not g.get("early") and et_date(g["start"]) == day.isoformat()]
    section["projections"] = projections(model, today, news, day)
    shares = prop_shares(data_dir, rows)
    share = shares[("NBA", "prop")]
    status["market_check"] = share
    # prices
    priced, msg = {}, "prop prices: none (add a free ODDS_API_KEY to price props; nothing is invented)"
    cache = os.path.join(data_dir, ".cache", f"props-prices-{day.isoformat()}.json")
    if today and odds_key:
        cached = _load_json(cache, None)
        if cached is not None:
            priced, msg = cached, "prop prices: The Odds API (fetched earlier today)"
        elif budget is not None:
            priced, msg = fetch_oddsapi_props(odds_key, today, budget, data_dir)
            if priced:
                os.makedirs(os.path.dirname(cache), exist_ok=True)
                _save_json(cache, priced)
    espn_msg = ""
    if today and not priced:
        try:
            pairs, names = fetch_espn_props(today)
            orient = espn_orientation(data_dir, model, pairs, rows, day, today)
            status["espn_types"] = dict(sorted(names.items(), key=lambda kv: -kv[1])[:25])
            status["espn_orientation"] = orient
            n_pairs = sum(len(v) for v in pairs.values())
            if not pairs:
                espn_msg = "ESPN props: none listed for tonight's games yet"
            elif orient["first_is_over"] is None:
                espn_msg = (f"ESPN props: {n_pairs} over/unders found; still confirming which price is the over "
                            "(no props parlays until it is certain)")
            else:
                priced = priced_from_pairs(pairs, orient["first_is_over"])
                msg = f"prop prices: ESPN (free), {n_pairs} over/unders on {len(pairs)} games"
                espn_msg = "ESPN props: side order confirmed by the data"
        except Exception as e:  # prices must never break the update
            espn_msg = f"ESPN props: check failed ({type(e).__name__})"
    section["prices"], section["espn"] = msg, espn_msg
    status["prices"], status["espn"] = msg, espn_msg
    # legs, re-check of today's props parlays, new parlays
    learn_path = os.path.join(d, "learn.json")
    learn = _load_json(learn_path, {})
    legs = build_legs(model, today, priced, news, share["share"], day, learn)
    cutoff = (day - timedelta(days=180)).strftime("%Y%m%d")
    learn = {k: v for k, v in learn.items() if v[0].split("-")[1] >= cutoff}
    _save_json(learn_path, learn)
    status["candidate_legs"] = len(legs)
    cur = {(x["game_id"], x["player_id"], x["stat"], x["line"], x["side"]): x for x in legs}
    todays = [e for e in ledger if e.get("tier") == "props" and e.get("date") == day.isoformat()]
    for e in todays:
        if e["status"] not in ("pending", "withdrawn"):
            continue
        now = [cur.get((x["game_id"], x["player_id"], x["stat"], x["line"], x["side"])) for x in e["legs"]]
        if any(c is None for c in now):
            continue
        p, _, _ = parlay_band(now, PROP_POLICY.draws)
        ok = p >= PROP_POLICY.min_probability
        if not ok and e["status"] == "pending":
            e.update(status="withdrawn", withdrawn_at=_now(), note=f"Withdrawn before start: chance now {p:.1%}")
        elif ok and e["status"] == "withdrawn":
            e.update(status="pending", note="Reinstated after new information")
    used = {x["game_id"] for e in todays for x in e["legs"]}
    chosen, near = scan(_shortlist(legs), PROP_POLICY, used)
    ids = {e["id"] for e in ledger}
    for combo in chosen[:max(PROP_POLICY.max_per_day - len(todays), 0)]:
        entry = parlay_entry(combo, day.isoformat(), PROP_POLICY, _now())
        entry.update(id=entry["id"].replace("parlay:", "props:", 1), tier="props", league="PROPS",
                     leagues=["NBA"], label=f"{len(combo)}-leg props parlay")
        entry["reasons"] = [r.replace("legs from different games", "players from different games")
                            for r in entry["reasons"]]
        if entry["id"] not in ids:
            ledger.append(entry)
            ids.add(entry["id"])
    section.update(ready=True, closest=near, candidates=len(legs),
                   calibrated={s: model.calibrated(s) for s in PROP_STATS},
                   policy={"min_decimal": PROP_POLICY.min_decimal, "min_probability": PROP_POLICY.min_probability})
    _save_json(os.path.join(d, "status.json"), status)
    return section
