"""Parlays: independence maths, uncertainty band, search rules, withdrawal and grading."""
from datetime import date, datetime

import numpy as np

from sportsedge.daily import settle_parlays, update_parlays
from sportsedge.parlays import (ParlayPolicy, american_to_decimal, beta_params, candidate_legs, parlay_band,
                                scan, settle_parlay)
from sportsedge.types import GameResult

DAY = "2026-10-04"


def _game(i, p_home, price, lo=None, hi=None, league="NFL", start="2026-10-04T17:00:00", total=None):
    lo = p_home - 0.04 if lo is None else lo
    hi = p_home + 0.04 if hi is None else hi
    markets = [{"market": "moneyline", "side": "home", "line": None, "label": f"H{i} ML", "price": price,
                "p": p_home, "lo": lo, "hi": hi, "fair": 0.62, "fails": [], "conf_fails": []},
               {"market": "moneyline", "side": "away", "line": None, "label": f"A{i} ML", "price": 150,
                "p": 1 - p_home, "lo": 1 - hi, "hi": 1 - lo, "fair": 0.38, "fails": [], "conf_fails": []}]
    if total:
        markets.append({"market": "total", "side": "over", "line": 44.5, "label": "Over 44.5", "price": -110,
                        "p": total, "lo": total - 0.04, "hi": total + 0.04, "fair": 0.5, "fails": [], "conf_fails": []})
    return {"id": f"G{i}", "league": league, "start": start, "state": "pre", "early": False, "frozen": False,
            "home": {"abbr": f"H{i}"}, "away": {"abbr": f"A{i}"}, "markets": markets}


def test_beta_matches_mean_and_band():
    a, b = beta_params(0.8, 0.74, 0.86)
    draws = np.random.default_rng(0).beta(a, b, 200_000)
    assert abs(draws.mean() - 0.8) < 0.002
    assert abs(np.quantile(draws, 0.1) - 0.74) < 0.01 and abs(np.quantile(draws, 0.9) - 0.86) < 0.01


def test_band_is_product_with_uncertainty():
    legs = [{"league": "NFL", "game_id": "a", "market": "moneyline", "side": "home", "p": 0.9, "lo": 0.85, "hi": 0.95},
            {"league": "NFL", "game_id": "b", "market": "moneyline", "side": "home", "p": 0.85, "lo": 0.8, "hi": 0.9}]
    p, lo, hi = parlay_band(legs)
    assert abs(p - 0.765) < 1e-9 and lo < 0.765 < hi and hi - lo < 0.2


def test_scan_finds_2x_70pct_parlay_and_respects_rules():
    pol = ParlayPolicy()
    games = [_game(1, 0.86, -125, total=0.9), _game(2, 0.85, -120), _game(3, 0.62, -200)]
    legs = candidate_legs(games, DAY, pol)
    chosen, near = scan(legs, pol)
    assert chosen, "0.86 x 0.85 = 73% at 1.8 x 1.83 = 3.3x should qualify"
    combo = chosen[0]
    d = np.prod([american_to_decimal(x["price"]) for x in combo])
    p = np.prod([x["p"] for x in combo])
    assert d >= 2.0 and p >= 0.70
    assert len({x["game_id"] for x in combo}) == len(combo), "never two legs from one game"


def test_no_parlay_when_the_bar_is_not_met():
    pol = ParlayPolicy()
    games = [_game(1, 0.80, -400), _game(2, 0.78, -350), _game(3, 0.75, -300)]  # short prices: 2x needs too many legs
    chosen, near = scan(candidate_legs(games, DAY, pol), pol)
    assert not chosen and near is not None and near["p"] < 0.70 and near["decimal"] >= 2.0


def test_uncertain_legs_are_vetoed_by_the_band():
    pol = ParlayPolicy()
    games = [_game(1, 0.86, -125, lo=0.55, hi=0.99), _game(2, 0.85, -120, lo=0.55, hi=0.99)]
    chosen, _ = scan(candidate_legs(games, DAY, pol), pol)
    assert not chosen


def test_update_publishes_then_withdraws_and_grades():
    ledger = []
    games = [_game(1, 0.86, -125), _game(2, 0.85, -120)]
    update_parlays(ledger, games, DAY)
    assert len(ledger) == 1 and ledger[0]["tier"] == "parlay" and ledger[0]["status"] == "pending"
    e = ledger[0]
    assert e["decimal"] >= 2.0 and e["p"] >= 0.70 and e["league"] == "PARLAY"
    # news drops leg 1 to 70%: the parlay (now ~60%) is withdrawn before the start
    games[0]["markets"][0].update(p=0.70, lo=0.66, hi=0.74)
    update_parlays(ledger, games, DAY)
    assert e["status"] == "withdrawn"
    games[0]["markets"][0].update(p=0.86, lo=0.82, hi=0.90)
    update_parlays(ledger, games, DAY)
    assert e["status"] == "pending" and len(ledger) == 1  # reinstated, not duplicated
    finals = {f"G{i}": GameResult(f"G{i}", "NFL", datetime(2026, 10, 4, 17), f"H{i}", f"A{i}", home_score=24, away_score=17)
              for i in (1, 2)}
    settle_parlays(ledger, finals, date(2026, 10, 5))
    assert e["status"] == "won" and abs(e["profit"] - (e["decimal"] - 1)) < 1e-3


def test_settle_rules():
    def mk(*st):
        return {"legs": [{"status": s, "price": -120, "label": "x"} for s in st], "status": "pending"}
    e = mk("won", "lost", "pending"); settle_parlay(e); assert e["status"] == "lost" and e["profit"] == -1.0
    e = mk("won", "pending"); settle_parlay(e); assert e["status"] == "pending"
    e = mk("won", "push"); settle_parlay(e); assert e["status"] == "won" and abs(e["profit"] - (1 + 100 / 120 - 1)) < 1e-3
    e = mk("push", "void"); settle_parlay(e); assert e["status"] == "push" and e["profit"] == 0.0
