import json
from datetime import date, datetime, timedelta

import pytest

import sportsedge.data.espn as espn
import sportsedge.news.feeds as feeds
from sportsedge.daily import grade, pick_label, run_daily, settle_ledger
from sportsedge.site import render_fragment
from sportsedge.synthetic import generate
from sportsedge.types import Game, GameResult


def _entry(market, side, line, price=-150):
    return {"id": "x", "date": "2026-01-01", "league": "NBA", "game_id": "g1", "market": market, "side": side,
            "line": line, "price": price, "p": 0.72, "status": "pending", "profit": None}


def _result(hs, as_):
    return GameResult("g1", "NBA", datetime(2026, 1, 1, 19), "BOS", "NY", home_score=hs, away_score=as_)


@pytest.mark.parametrize("market,side,line,hs,as_,status", [
    ("moneyline", "home", None, 101, 99, "won"),
    ("moneyline", "away", None, 101, 99, "lost"),
    ("spread", "home", -3.0, 103, 100, "push"),
    ("spread", "away", -3.5, 103, 100, "won"),  # stored home line -3.5 = away +3.5; loses by 3 -> covers
    ("total", "over", 200.5, 101, 99, "lost"),
    ("total", "under", 200.5, 101, 99, "won"),
])
def test_grading(market, side, line, hs, as_, status):
    e = grade(_entry(market, side, line), _result(hs, as_))
    assert e["status"] == status
    assert e["profit"] == {"won": pytest.approx(100 / 150, abs=1e-4), "lost": -1.0, "push": 0.0}[status]


def test_settle_voids_stale_pending():
    ledger = [_entry("moneyline", "home", None), dict(_entry("moneyline", "home", None), game_id="gone")]
    n = settle_ledger(ledger, {"g1": _result(1, 0)}, date(2026, 1, 10))
    assert n == 1 and ledger[0]["status"] == "won" and ledger[1]["status"] == "void"


def test_labels():
    assert pick_label("BOS", "NY", "spread", "away", -5.5) == "NY +5.5"
    assert pick_label("BOS", "NY", "total", "over", 221.5) == "Over 221.5"


def test_embedded_data_cannot_break_out_of_script():
    html = render_fragment({"games": [], "picks": [], "ledger": [], "note": "</script><script>alert(1)</script>"})
    assert "</script><script>alert(1)" not in html
    assert "__SPORTSEDGE_DATA__" not in html


def test_full_daily_run_offline(tmp_path, monkeypatch):
    games = generate("NBA", seasons=1, seed=9)
    cut = games[-1].start_time.date()
    history = [g for g in games if g.start_time.date() < cut]
    today_games = [g for g in games if g.start_time.date() == cut]
    day = cut

    def fake_history(league, start, end, pause=0.0):
        return [g for g in history if start <= g.start_time.date() <= end]

    def fake_slate(league, d=None):
        return [Game(g.game_id, league, g.start_time, g.home, g.away, g.neutral, {"odds": g.odds}) for g in today_games]

    monkeypatch.setattr(espn, "fetch_history", fake_history)
    monkeypatch.setattr(espn, "fetch_slate", fake_slate)
    monkeypatch.setattr(feeds, "gather_news", lambda league, **kw: [])
    data_dir, site_dir = tmp_path / "data", tmp_path / "site"
    out = run_daily(["NBA"], str(data_dir), str(site_dir), day=day, bootstrap_days=400)
    assert out["leagues"][0]["status"] == "ok"
    assert len(out["games"]) == len(today_games)
    assert (site_dir / "index.html").exists() and (data_dir / "history" / "NBA.csv").exists()
    for g in out["games"]:
        assert 0 < g["p_home"] < 1 and g["markets"]
    # second day: yesterday's picks get graded against the new results
    ledger = json.loads((data_dir / "ledger.json").read_text())
    history.extend(today_games)
    monkeypatch.setattr(espn, "fetch_slate", lambda league, d=None: [])
    out2 = run_daily(["NBA"], str(data_dir), str(site_dir), day=day + timedelta(days=1))
    graded = {e["id"]: e for e in out2["ledger"]}
    for e in ledger:
        assert graded[e["id"]]["status"] in ("won", "lost", "push")
    assert out2["leagues"][0]["status"] == "idle"
