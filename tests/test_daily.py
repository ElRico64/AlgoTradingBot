import json
from datetime import date, datetime, timedelta

import pytest

import sportsedge.data.espn as espn
import sportsedge.news.feeds as feeds
from sportsedge.daily import grade, pick_label, run_daily, settle_ledger
from sportsedge.site import render_fragment
from sportsedge.synthetic import generate
from sportsedge.types import Game, GameResult, NewsEvent


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
    todays = [g for g in games if g.start_time.date() == cut]
    state = {"started": False}

    def fake_history(league, start, end, pause=0.0, **kw):
        return [g for g in history if start <= g.start_time.date() <= end]

    def fake_day(league, d):
        if d == cut and not state["started"]:
            # no posted odds: exercises the no-API-key path, where news drives the model directly
            return [Game(g.game_id, league, g.start_time, g.home, g.away, g.neutral,
                         {"odds": None, "status": {"state": "pre", "detail": ""}}) for g in todays]
        if d == cut:
            return list(todays)  # finals
        return [g for g in history if g.start_time.date() == d]

    news = {"events": []}
    monkeypatch.setattr(espn, "fetch_history", fake_history)
    monkeypatch.setattr(espn, "fetch_day", fake_day)
    monkeypatch.setattr(feeds, "gather_news", lambda league, **kw: news["events"])
    data_dir, site_dir = tmp_path / "data", tmp_path / "site"

    out = run_daily(["NBA"], str(data_dir), str(site_dir), day=cut, bootstrap_days=400)
    assert out["leagues"][0]["status"] == "ok" and len(out["games"]) == len(todays)
    assert (site_dir / "index.html").exists() and (data_dir / "history" / "NBA.csv").exists()
    assert list((data_dir / ".cache").iterdir())  # engine cached for the day's refreshes
    conf = [e for e in out["ledger"] if e["tier"] == "confidence"]
    assert conf, "a synthetic NBA slate should have at least one 70%+ pick"
    assert not [e for e in out["ledger"] if e["tier"] == "value"]  # no price, no value bets
    for e in out["ledger"]:
        assert e["status"] == "pending" and e["value_line"] <= -100 and e["price"] is None

    # refresh: star player of a picked team ruled out -> the pick is withdrawn before the start
    target = conf[0]
    team = next(g for g in todays if g.game_id == target["game_id"])
    picked = team.home if target["side"] == "home" else team.away
    news["events"] = [NewsEvent("NBA", picked, "Franchise Star", "injury", "out", reliability=1.0, role="star",
                                impact=12.0, impact_sd=0.5)]
    out = run_daily(["NBA"], str(data_dir), str(site_dir), day=cut)
    after = {e["id"]: e for e in out["ledger"]}[target["id"]]
    assert after["status"] == "withdrawn" and "Withdrawn before start" in after["note"]
    assert out["news"] and out["news"][0]["player"] == "Franchise Star"

    # games finish: predictions stay frozen, pending picks are graded, withdrawn ones are not
    news["events"] = []
    state["started"] = True
    out = run_daily(["NBA"], str(data_dir), str(site_dir), day=cut)
    assert all(g["frozen"] and g["state"] == "post" for g in out["games"])
    for e in out["ledger"]:
        assert e["status"] in ("won", "lost", "push", "withdrawn")
        if e["status"] != "withdrawn":
            assert e["final"] and e["profit"] is None  # graded W/L; no price, so no units


def test_next_day_update_matches_full_retrain(tmp_path, monkeypatch):
    """A new day loads yesterday's saved engine and learns only the new results;
    the forecast must equal training from scratch. Stale versions are retrained."""
    import pickle

    from sportsedge.daily import _load_engine
    from sportsedge.engine import LeagueEngine

    games = generate("NBA", seasons=1, seed=3)
    days = sorted({g.start_time.date() for g in games})

    def fake_history(league, start, end, progress=None, **kw):
        espn.last_history_failed_share = 0.0
        return [g for g in games if start <= g.start_time.date() <= end]

    monkeypatch.setattr(espn, "fetch_history", fake_history)
    _load_engine("NBA", str(tmp_path), days[-6], 400)
    eng, _ = _load_engine("NBA", str(tmp_path), days[-1], 400)
    full = LeagueEngine("NBA").fit([g for g in games if g.start_time.date() < days[-1]])
    g = next(x for x in games if x.start_time.date() == days[-1])
    assert eng.predict(g, g.odds, n_draws=300).home_win_prob == pytest.approx(
        full.predict(g, g.odds, n_draws=300).home_win_prob, abs=1e-9)
    cache = tmp_path / ".cache"
    assert [p.name for p in cache.iterdir()] == [f"NBA-{days[-1].isoformat()}.pkl"]  # old engine cleaned up
    # an engine saved by older code is ignored and rebuilt, never reused
    eng.version = -1
    with open(cache / f"NBA-{days[-1].isoformat()}.pkl", "wb") as f:
        pickle.dump(eng, f)
    rebuilt, _ = _load_engine("NBA", str(tmp_path), days[-1], 400)
    assert rebuilt.version != -1
