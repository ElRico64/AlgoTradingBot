from datetime import datetime

import pytest

from sportsedge.data.csv_io import read_results, read_slate, write_results
from sportsedge.data.espn import parse_scoreboard
from sportsedge.data.oddsapi import parse_event
from sportsedge.picks import PickPolicy
from sportsedge.types import Game, GameResult, MarketOdds, MarketProbability, Prediction

SCOREBOARD = {"events": [
    {"id": "401", "date": "2026-04-02T23:05Z", "status": {"type": {"completed": True}},
     "competitions": [{"neutralSite": False, "competitors": [
         {"homeAway": "home", "score": "5", "team": {"abbreviation": "NYY"}},
         {"homeAway": "away", "score": "3", "team": {"abbreviation": "BOS"}}],
         "odds": [{"details": "NYY -1.5", "overUnder": 8.5, "homeTeamOdds": {"moneyLine": -150},
                   "awayTeamOdds": {"moneyLine": 130}}]}]},
    {"id": "402", "date": "2026-04-03T23:05Z", "status": {"type": {"completed": False}},
     "competitions": [{"competitors": [
         {"homeAway": "home", "team": {"abbreviation": "LAD"},
          "probables": [{"athlete": {"displayName": "Ace Pitcher"},
                         "statistics": [{"abbreviation": "ERA", "displayValue": "2.10"}]}]},
         {"homeAway": "away", "team": {"abbreviation": "SD"}}],
         "odds": [{"details": "SD -1.5", "overUnder": 7.5, "homeTeamOdds": {"moneyLine": 120},
                   "awayTeamOdds": {"moneyLine": -140}}]}]},
]}


def test_parse_scoreboard():
    done, upcoming = parse_scoreboard("MLB", SCOREBOARD)
    assert isinstance(done, GameResult) and done.margin == 2 and done.odds.spread == -1.5
    assert not isinstance(upcoming, GameResult)
    assert upcoming.extras["odds"].spread == 1.5  # away favoured -> home +1.5
    assert upcoming.extras["home_pitcher"]["era"] == 2.10


def test_parse_odds_api_event_line_shops_and_pools():
    ev = {"home_team": "Boston Celtics", "away_team": "New York Knicks", "commence_time": "2026-01-10T00:30:00Z",
          "bookmakers": [
              {"key": "pinnacle", "markets": [
                  {"key": "h2h", "outcomes": [{"name": "Boston Celtics", "price": -200},
                                              {"name": "New York Knicks", "price": 180}]},
                  {"key": "spreads", "outcomes": [{"name": "Boston Celtics", "price": -108, "point": -5.5},
                                                  {"name": "New York Knicks", "price": -102, "point": 5.5}]},
                  {"key": "totals", "outcomes": [{"name": "Over", "price": -110, "point": 221.5},
                                                 {"name": "Under", "price": -110, "point": 221.5}]}]},
              {"key": "draftkings", "markets": [
                  {"key": "h2h", "outcomes": [{"name": "Boston Celtics", "price": -185},
                                              {"name": "New York Knicks", "price": 155}]}]}]}
    home, away, start, mo = parse_event("NBA", ev)
    assert (home, away) == ("BOS", "NY")
    assert mo.home_ml == -185 and mo.away_ml == 180  # best price per side
    assert 0.63 < mo.fair_home_prob < 0.67
    assert mo.spread == -5.5 and mo.total == 221.5


def test_csv_roundtrip(tmp_path):
    g = GameResult("1", "NBA", datetime(2025, 1, 2, 19), "BOS", "NY", home_score=100, away_score=90,
                   odds=MarketOdds(home_ml=-200, away_ml=170, spread=-5.5, total=220.5))
    p = tmp_path / "h.csv"
    write_results(str(p), [g])
    back = read_results(str(p), "NBA")[0]
    assert back.margin == 10 and back.odds.spread == -5.5 and back.odds.home_ml == -200
    slate = read_slate(str(p), "NBA")[0]
    assert isinstance(slate, Game) and slate.extras["odds"].total == 220.5


def _pred(p, lower, price, fair, calibrated=True):
    g = Game("1", "NBA", datetime(2026, 1, 1), "BOS", "NY")
    m = MarketProbability("moneyline", "home", None, p, lower, min(1, p + 0.05), price, fair, 0.0, calibrated)
    return Prediction(g, p, lower, p + 0.05, 5.0, 220.0, [m], {"kalman": p, "elo": p})


@pytest.mark.parametrize("p,lower,price,fair,calibrated,expected", [
    (0.76, 0.70, -200, 0.70, True, True),   # 76% at -200 (needs 66.7%) with +6% edge
    (0.68, 0.62, +100, 0.50, True, False),  # below 70%
    (0.76, 0.55, -200, 0.70, True, False),  # band too wide
    (0.72, 0.68, -300, 0.74, True, False),  # 72% but price needs 75% -> -EV
    (0.76, 0.70, -200, 0.70, False, False), # uncalibrated
])
def test_pick_gate(p, lower, price, fair, calibrated, expected):
    picks = PickPolicy().evaluate(_pred(p, lower, price, fair, calibrated))
    assert bool(picks) is expected
    if picks:
        assert 0 < picks[0].stake_fraction <= 0.03 and picks[0].reasons


def test_pick_gate_without_ev_requirement():
    picks = PickPolicy(require_positive_ev=False).evaluate(_pred(0.72, 0.68, -300, 0.74))
    assert picks
