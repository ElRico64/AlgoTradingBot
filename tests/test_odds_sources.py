"""ESPN odds in both layouts, the per-game fallback, and the pre-game odds store."""
from datetime import datetime

import sportsedge.data.espn as espn
from sportsedge.daily import attach_stored_odds, fill_espn_odds, load_odds_store, remember_odds, save_odds_store
from sportsedge.types import Game, GameResult, MarketOdds


def _event(odds, home="CHW", away="COL", eid="777", completed=False):
    return {"id": eid, "date": "2026-09-26T23:10Z",
            "status": {"type": {"completed": completed, "state": "post" if completed else "pre"}},
            "competitions": [{"competitors": [
                {"homeAway": "home", "score": "6", "team": {"abbreviation": home}},
                {"homeAway": "away", "score": "1", "team": {"abbreviation": away}}],
                "odds": odds}]}


# ESPN's newer layout: nested open/close quotes, string prices, "o8.5" totals
NEW_LAYOUT = [{
    "provider": {"name": "ESPN BET", "priority": 1},
    "details": "CHW -155", "overUnder": 8.5, "spread": -1.5,
    "homeTeamOdds": {"favorite": True}, "awayTeamOdds": {"favorite": False},
    "moneyline": {"home": {"open": {"odds": "-140"}, "close": {"odds": "-155"}},
                  "away": {"open": {"odds": "+120"}, "close": {"odds": "+135"}}},
    "pointSpread": {"home": {"close": {"line": "-1.5", "odds": "+125"}},
                    "away": {"close": {"line": "+1.5", "odds": "-145"}}},
    "total": {"over": {"close": {"line": "o8.5", "odds": "-110"}},
              "under": {"close": {"line": "u8.5", "odds": "EVEN"}}},
}]


def test_new_espn_layout():
    (g,) = espn.parse_scoreboard("MLB", {"events": [_event(NEW_LAYOUT, home="CWS")]})
    o = g.extras["odds"]
    assert g.home == "CHW" and g.extras["espn_id"] == "777"
    assert (o.home_ml, o.away_ml) == (-155.0, 135.0)
    assert (o.spread, o.home_spread_price, o.away_spread_price) == (-1.5, 125.0, -145.0)
    assert (o.total, o.over_price, o.under_price) == (8.5, -110.0, 100.0)
    assert o.source == "ESPN BET"


def test_moneyline_in_details_is_not_a_spread():
    odds = [{"details": "COL -160", "overUnder": 9.0}]
    (g,) = espn.parse_scoreboard("MLB", {"events": [_event(odds)]})
    o = g.extras["odds"]
    assert o.spread is None and o.total == 9.0 and not o.has_moneyline()


def test_details_spread_uses_team_aliases():
    odds = [{"details": "CWS -1.5"}]
    (g,) = espn.parse_scoreboard("MLB", {"events": [_event(odds, home="CWS")]})
    assert g.extras["odds"].spread == -1.5


def test_game_page_fills_missing_moneyline(monkeypatch):
    (g,) = espn.parse_scoreboard("MLB", {"events": [_event([{"overUnder": 8.0}])]})
    assert not g.extras["odds"].has_moneyline()
    pickcenter = {"pickcenter": [{"provider": {"name": "Book"}, "overUnder": 8.0,
                                  "homeTeamOdds": {"moneyLine": -130}, "awayTeamOdds": {"moneyLine": 110}}]}
    calls = []
    monkeypatch.setattr(espn, "_get_json", lambda url, timeout=15.0: calls.append(url) or pickcenter)
    fill_espn_odds("MLB", [g])
    assert "summary?event=777" in calls[0]
    assert (g.extras["odds"].home_ml, g.extras["odds"].away_ml) == (-130.0, 110.0)


def test_odds_store_round_trip_attaches_to_results(tmp_path):
    start = datetime(2026, 9, 26, 23, 10)
    pre = Game("MLB-x", "MLB", start, "CHW", "COL", extras={"status": {"state": "pre"},
                                                              "odds": MarketOdds(home_ml=-155, away_ml=135, total=8.5)})
    store = load_odds_store(str(tmp_path), "MLB")
    assert remember_odds(store, [pre], {}) == 1
    save_odds_store(str(tmp_path), "MLB", store)
    done = GameResult("MLB-x", "MLB", start, "CHW", "COL", home_score=6, away_score=1)
    assert attach_stored_odds([done], load_odds_store(str(tmp_path), "MLB")) == 1
    assert done.odds.home_ml == -155 and done.odds.total == 8.5
