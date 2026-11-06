"""NBA player props: box-score parsing and storage, player model, prop prices, props parlays, grading."""
import json
from datetime import date, datetime, timedelta

import numpy as np
import pytest

from sportsedge.props import board
from sportsedge.props.boxscores import load_rows, parse_boxscore, update_boxscores
from sportsedge.props.model import PROP_STATS, PropModel
from sportsedge.props.prices import norm_name, parse_event_props


def _summary(eid="401700001", date_iso="2026-11-04T00:30Z", home="LAL", away="PHI", completed=True):
    labels = ["MIN", "FG", "3PT", "FT", "OREB", "DREB", "REB", "AST", "STL", "BLK", "TO", "PF", "+/-", "PTS"]
    def ath(i, name, stats, starter=True, dnp=False):
        return {"athlete": {"id": str(i), "displayName": name, "position": {"abbreviation": "F"}},
                "starter": starter, "didNotPlay": dnp, "stats": stats}
    return {"header": {"id": eid, "competitions": [{"date": date_iso, "status": {"type": {"completed": completed}},
                                                    "competitors": [{"homeAway": "home", "team": {"abbreviation": home}},
                                                                    {"homeAway": "away", "team": {"abbreviation": away}}]}]},
            "boxscore": {"players": [
                {"team": {"abbreviation": home}, "statistics": [{"labels": labels, "athletes": [
                    ath(1966, "LeBron James", ["36", "10-19", "2-6", "4-5", "1", "7", "8", "9", "1", "0", "3", "2", "+5", "26"]),
                    ath(2, "Bench Guy", [], starter=False, dnp=True)]}]},
                {"team": {"abbreviation": away}, "statistics": [{"labels": labels, "athletes": [
                    ath(4431678, "Tyrese Maxey", ["38:20", "9-21", "4-10", "3-3", "0", "3", "3", "6", "2", "0", "2", "1", "-5", "25"])]}]}]}}


def test_parse_boxscore_reads_espn_game_page():
    rows = parse_boxscore(_summary())
    assert len(rows) == 2  # did-not-play skipped
    lbj = next(r for r in rows if r["player"] == "LeBron James")
    assert (lbj["pts"], lbj["reb"], lbj["ast"], lbj["fg3m"], lbj["min"]) == (26, 8, 9, 2, 36)
    assert lbj["team"] == "LAL" and lbj["opp"] == "PHI" and lbj["home"] == 1
    maxey = next(r for r in rows if r["player"] == "Tyrese Maxey")
    assert maxey["min"] == pytest.approx(38 + 20 / 60) and maxey["fg3m"] == 4
    assert lbj["game_id"].startswith("NBA-20261103-PHI-LAL")
    assert parse_boxscore(_summary(completed=False)) == []  # not final yet


def test_box_scores_download_once_and_newest_first(tmp_path):
    class G:  # scoreboard stub
        def __init__(self, eid): self.extras = {"espn_id": eid, "status": {"state": "post"}}
    days = [date(2026, 11, 3), date(2026, 11, 4)]
    calls = []
    fd = lambda d: [G("401700001")] if d == date(2026, 11, 3) else []
    fs = lambda eid: calls.append(eid) or _summary(eid)
    st = update_boxscores(str(tmp_path), days, date(2026, 11, 5), fetch_day=fd, fetch_summary=fs)
    assert st["new_games"] == 1 and len(load_rows(str(tmp_path))) == 2
    st = update_boxscores(str(tmp_path), days, date(2026, 11, 5), fetch_day=fd, fetch_summary=fs)
    assert st["new_games"] == 0 and calls == ["401700001"]  # never downloaded twice
    assert len(load_rows(str(tmp_path))) == 2


def _name(pid):  # letters only, like real names
    t, j = pid[1:].split("-")
    return "Player " + "ABCDEFGHIJKLMNOP"[int(t)] + "abcdefghijkl"[int(j)]


def synth_rows(n_teams=10, per_team=8, days=120, seed=0):
    """Simulated league: each player has true per-minute rates; stats are negative binomial."""
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(n_teams)]
    truth = {}
    for t in teams:
        for j in range(per_team):
            pid = f"{t}-{j}"
            truth[pid] = {"team": t, "min": rng.uniform(18, 36),
                          "rate": {"pts": rng.uniform(0.3, 0.9), "reb": rng.uniform(0.1, 0.35),
                                   "ast": rng.uniform(0.05, 0.3), "fg3m": rng.uniform(0.01, 0.1)}}
    rows, start = [], datetime(2025, 10, 22, 23)
    for d in range(days):
        order = rng.permutation(teams)
        for k in range(0, n_teams, 2):
            home, away = order[k], order[k + 1]
            when = start + timedelta(days=d)
            gid = f"NBA-{when:%Y%m%d}-{away}-{home}-19"
            for t, opp in ((home, away), (away, home)):
                for j in range(per_team):
                    pid = f"{t}-{j}"; tr = truth[pid]
                    mins = float(np.clip(rng.normal(tr["min"], 4), 5, 44))
                    rec = {"game_id": gid, "espn_id": gid, "date": when.isoformat(), "team": t, "opp": opp,
                           "home": int(t == home), "player_id": pid, "player": _name(pid), "pos": "F",
                           "starter": int(j < 5), "min": mins, "stl": 1.0, "blk": 0.0, "tov": 1.0}
                    for s in PROP_STATS:
                        mu = mins * tr["rate"][s]
                        phi = 8.0
                        rec[s] = float(rng.negative_binomial(phi, phi / (phi + mu)))
                    rows.append(rec)
    return rows, truth


@pytest.fixture(scope="module")
def fitted():
    rows, truth = synth_rows()
    return PropModel().fit(rows), rows, truth


def test_model_projects_near_the_truth(fitted):
    model, rows, truth = fitted
    errs = []
    for pid, tr in truth.items():
        mu = model.mean(pid, "pts", "T0")
        errs.append(abs(mu - model.players[pid].min_ewma * tr["rate"]["pts"]) / max(tr["min"] * tr["rate"]["pts"], 1))
    assert np.median(errs) < 0.12


def test_model_probabilities_are_calibrated_out_of_sample(fitted):
    model, _, _ = fitted
    for s in ("pts", "reb", "ast"):
        assert model.calibrated(s), s
        for b in model.reliability[s]:  # within sampling error of each bucket
            se = (b["said"] * (1 - b["said"]) / b["n"]) ** 0.5
            assert abs(b["said"] - b["hit"]) < 3 * se + 0.015, (s, b)


def test_parse_odds_api_props_best_price_and_consensus():
    data = {"bookmakers": [
        {"key": "a", "markets": [{"key": "player_assists", "outcomes": [
            {"name": "Over", "description": "LeBron James", "price": -120, "point": 7.5},
            {"name": "Under", "description": "LeBron James", "price": -110, "point": 7.5}]}]},
        {"key": "b", "markets": [{"key": "player_assists", "outcomes": [
            {"name": "Over", "description": "LeBron James", "price": -105, "point": 7.5},
            {"name": "Under", "description": "LeBron James", "price": -125, "point": 7.5}]}]}]}
    (row,) = parse_event_props(data)
    assert row["stat"] == "ast" and row["line"] == 7.5 and row["over"] == -105 and row["under"] == -110
    assert 0.49 < row["fair"] < 0.53 and row["books"] == 2
    alt = {"bookmakers": [{"key": "a", "markets": [{"key": "player_assists_alternate", "outcomes": [
        {"name": "Over", "description": "LeBron James", "price": -300, "point": 5.5}]}]}]}
    (a,) = parse_event_props(alt)
    assert a["under"] is None and a["fair"] == pytest.approx(0.75)  # over-only: cautious fair
    assert norm_name("Jaren Jackson Jr.") == "jaren jackson" and norm_name("Nikola Jokić") == "nikola jokic"


def test_props_parlay_rule_and_grading(fitted, tmp_path):
    model, rows, truth = fitted
    # two games tonight; give the model's favourite props generous prices so the rule can be met
    games, priced = [], {}
    pairs = [("T0", "T1"), ("T2", "T3")]
    for i, (home, away) in enumerate(pairs):
        gid = f"NBA-20260219-{away}-{home}-19"
        games.append({"id": gid, "start": "2026-02-20T00:00:00", "state": "pre", "early": False,
                      "home": {"abbr": home}, "away": {"abbr": away}})
        pid = f"{home}-0"
        mu = model.mean(pid, "pts", away)
        line = max(0.5, float(np.floor(mu * 0.35)) + 0.5)  # an easy over
        priced[gid] = [{"player": _name(pid), "stat": "pts", "line": line, "over": 120.0, "under": -150.0,
                        "fair": 0.5, "books": 3}]
    day = date(2026, 2, 19)
    learn: dict = {}
    legs = board.build_legs(model, games, priced, [], 1.0, day, learn)
    assert legs and len(learn) == 2
    from sportsedge.parlays import scan
    chosen, near = scan(board._shortlist(legs), board.PROP_POLICY)
    assert chosen, near
    combo = chosen[0]
    dec = np.prod([1 + x["price"] / 100 if x["price"] > 0 else 1 + 100 / -x["price"] for x in combo])
    assert dec >= 2.0 and np.prod([x["p"] for x in combo]) >= 0.65
    assert len({x["game_id"] for x in combo}) == len(combo)
    # injured players are never legs
    hurt = [{"player": _name(x["player_id"]), "status": "questionable"} for x in combo]
    assert not [x for x in board.build_legs(model, games, priced, hurt, 1.0, day, {}) if x["player_id"] in {c["player_id"] for c in combo}]
    # grading from box scores: over hits, a missing player voids the leg
    from sportsedge.parlays import parlay_entry
    e = parlay_entry(combo, day.isoformat(), board.PROP_POLICY, "now")
    e["tier"] = "props"
    box = [{"game_id": combo[0]["game_id"], "player_id": combo[0]["player_id"], combo[0]["stat"]: combo[0]["line"] + 5}]
    if len(combo) > 1:
        box.append({"game_id": combo[1]["game_id"], "player_id": "someone-else", "pts": 10})
    board.grade_props([e], box, day + timedelta(days=1))
    assert e["legs"][0]["status"] == "won"
    if len(combo) > 1:
        assert e["legs"][1]["status"] == "void" and e["status"] == "won"


def test_run_props_waits_for_data_and_never_invents_prices(tmp_path, monkeypatch):
    import sportsedge.data.espn as espn
    monkeypatch.setattr(espn, "fetch_day", lambda league, d: [])
    monkeypatch.setattr(espn, "last_day_failed", False)
    sec = board.run_props(str(tmp_path), date(2026, 11, 5), [], [], None, [], fetch_seconds=1)
    assert not sec["ready"] and "Collecting" in sec["message"]
    # with data but no odds key: projections, no parlays
    rows, _ = synth_rows(days=60)
    from sportsedge.props.boxscores import FIELDS
    import csv, os
    os.makedirs(tmp_path / "players", exist_ok=True)
    with open(tmp_path / "players" / "NBA.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows({k: r.get(k, "") for k in FIELDS} for r in rows)
    games = [{"id": "NBA-20251221-T1-T0-19", "start": "2025-12-22T00:00:00", "state": "pre", "early": False,
              "home": {"abbr": "T0"}, "away": {"abbr": "T1"}}]
    ledger: list = []
    sec = board.run_props(str(tmp_path), date(2025, 12, 21), games, [], None, ledger, fetch_seconds=1)
    assert sec["ready"] and sec["projections"] and not ledger
    assert "nothing is invented" in sec["prices"]
    assert json.load(open(tmp_path / "props" / "status.json"))["rows"] == len(rows)


def _espn_item(aid, tname, line, american):
    return {"athlete": {"$ref": f"http://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/seasons/2027/athletes/{aid}?lang=en"},
            "type": {"id": "1", "name": tname}, "odds": {"american": {"value": str(american)}, "total": {"value": str(line)}},
            "current": {"target": {"value": line}}}


def test_parse_espn_propbets_pairs_and_skips():
    from sportsedge.props.prices import espn_stat, parse_espn_propbets, priced_from_pairs
    items = [_espn_item(3136193, "Total Points", 24.5, -124), _espn_item(3136193, "Total Points", 24.5, -103),
             _espn_item(3136193, "Total Points, Rebounds and Assists", 36.5, -110),
             _espn_item(3136193, "Total Points, Rebounds and Assists", 36.5, -110),
             _espn_item(3155526, "Total Rebounds", 5.5, 130)]  # single item: not a pair
    pairs, names = parse_espn_propbets(items)
    assert pairs == [{"player_id": "3136193", "stat": "pts", "line": 24.5, "prices": [-124.0, -103.0]}]
    assert names["Total Points"] == 2
    assert [espn_stat(n) for n in ("Total Assists", "Total 3-Point Field Goals Made", "Double Double", "1st Quarter Points")] == ["ast", "fg3m", None, None]
    (row,) = priced_from_pairs({"g": pairs}, first_is_over=False)["g"]
    assert (row["over"], row["under"]) == (-103.0, -124.0) and row["player_id"] == "3136193" and 0.45 < row["fair"] < 0.5


def test_espn_side_order_is_learned_from_the_model(fitted, tmp_path):
    from sportsedge.props.prices import _american_to_prob
    model, rows, truth = fitted
    rng = np.random.default_rng(2)

    def am(p):  # fair probability -> American price with ~4.5% margin
        q = min(max(p * 1.045 / 1.0, 0.02), 0.97)
        return -100 * q / (1 - q) if q >= 0.5 else 100 * (1 - q) / q
    games, pairs_over_first = [], {}
    for k, (home, away) in enumerate((("T0", "T1"), ("T2", "T3"), ("T4", "T5"), ("T6", "T7"), ("T8", "T9"))):
        gid = f"NBA-20260219-{away}-{home}-19"
        games.append({"id": gid, "home": {"abbr": home}, "away": {"abbr": away}})
        ps = []
        for t in (home, away):
            for j in range(8):
                pid = f"{t}-{j}"
                for s in ("pts", "reb", "ast"):
                    mu = model.mean(pid, s, away if t == home else home)
                    line = float(np.floor(mu + rng.normal(0, 1.5))) + 0.5
                    po = model.p_over(s, mu, line)
                    ps.append({"player_id": pid, "stat": s, "line": line, "prices": [am(po), am(1 - po)]})
        pairs_over_first[gid] = ps
    day = date(2026, 2, 19)
    o = board.espn_orientation(str(tmp_path), model, pairs_over_first, [], day, games)
    assert o["first_is_over"] is True and o["model_corr"] > 0.5
    flipped = {g: [{**p, "prices": p["prices"][::-1]} for p in ps] for g, ps in pairs_over_first.items()}
    assert board.espn_orientation(str(tmp_path / "b"), model, flipped, [], day, games)["first_is_over"] is False
    noise = {g: [{**p, "prices": [-110.0 - rng.integers(0, 20), -110.0 - rng.integers(0, 20)]} for p in ps]
             for g, ps in pairs_over_first.items()}
    assert board.espn_orientation(str(tmp_path / "c"), model, noise, [], day, games)["first_is_over"] is None
