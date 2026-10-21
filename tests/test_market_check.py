"""Market check: learned share of the model's disagreement with the market, partial pooling, application."""
import numpy as np
import pytest

from sportsedge.stats.market_check import MU0, apply_to_prediction, check, fit_shares
from sportsedge.types import Game, MarketProbability, Prediction
from datetime import datetime


def _rows(league, market, share, n, rng):
    out = []
    for _ in range(n):
        q = rng.uniform(0.3, 0.7)
        p = 1 / (1 + np.exp(-(np.log(q / (1 - q)) + rng.normal(0, 0.6))))
        truth = check(p, q, share)
        out.append((league, market, p, q, float(rng.random() < truth)))
    return out


def test_check_is_monotone_symmetric_and_bounded():
    assert check(0.8, 0.6, 0.0) == pytest.approx(0.6) and check(0.8, 0.6, 1.0) == pytest.approx(0.8)
    assert check(0.8, 0.6, 0.5) == pytest.approx(1 - check(0.2, 0.4, 0.5))
    assert check(0.7, 0.6, 0.5) < check(0.8, 0.6, 0.5)


def test_recovers_shares_per_market():
    rng = np.random.default_rng(3)
    rows = _rows("NHL", "moneyline", 0.7, 15000, rng) + _rows("NHL", "total", 0.0, 15000, rng)
    sh = fit_shares(rows, leagues=("NHL",))
    assert abs(sh[("NHL", "moneyline")]["share"] - 0.7) < 0.08
    assert sh[("NHL", "total")]["share"] < 0.08
    assert sh[("NHL", "spread")]["share"] == pytest.approx(MU0, abs=0.01)  # no data: the prior


def test_partial_pooling_lends_strength_to_a_league_without_games():
    rng = np.random.default_rng(4)
    rows = _rows("MLB", "total", 0.0, 8000, rng) + _rows("NHL", "total", 0.0, 8000, rng)
    sh = fit_shares(rows, leagues=("MLB", "NHL", "NBA"))
    assert sh[("NBA", "total")]["n"] == 0 and sh[("NBA", "total")]["share"] < 0.15  # borrowed from MLB/NHL


def test_apply_to_prediction_checks_both_sides_and_the_headline_probability():
    g = Game("g", "NHL", datetime(2026, 10, 20, 23), "CAR", "PHI")
    pred = Prediction(game=g, home_win_prob=0.80, home_win_lower=0.76, home_win_upper=0.84,
                      expected_margin=0.8, expected_total=6.0)
    pred.markets = [MarketProbability("moneyline", "home", None, 0.80, 0.76, 0.84, -290, 0.72),
                    MarketProbability("moneyline", "away", None, 0.20, 0.16, 0.24, 240, 0.28),
                    MarketProbability("total", "over", 6.5, 0.60, 0.55, 0.65, -110, None)]
    shares = {("NHL", "moneyline"): {"share": 0.5, "n": 200, "market_n": 400},
              ("NHL", "total"): {"share": 0.1, "n": 200, "market_n": 400}}
    apply_to_prediction(pred, "NHL", shares)
    home, away, tot = pred.markets
    assert home.raw_probability == 0.80 and 0.72 < home.probability < 0.80
    assert home.probability + away.probability == pytest.approx(1.0)
    assert home.lower < home.probability < home.upper and home.market_aware
    assert pred.home_win_prob == home.probability
    assert tot.raw_probability is None and tot.probability == 0.60  # no market probability: untouched


def test_market_check_runs_inside_the_daily_pipeline():
    from sportsedge.daily import predict_games, remember_model
    from sportsedge.engine import EngineSettings, LeagueEngine
    from sportsedge.picks import PickPolicy
    from sportsedge.synthetic import generate

    games = generate("NBA", seasons=1, seed=9)
    cut = games[-1].start_time.date()
    history = [g for g in games if g.start_time.date() < cut]
    todays = [g for g in games if g.start_time.date() == cut and g.odds is not None]
    engine = LeagueEngine("NBA", EngineSettings(n_draws=300)).fit(history)
    slate = [Game(g.game_id, "NBA", g.start_time, g.home, g.away, g.neutral,
                  {"odds": g.odds, "status": {"state": "pre", "detail": ""}}) for g in todays]
    shares = {("NBA", mk): {"share": 0.3, "n": 500, "market_n": 500} for mk in ("moneyline", "spread", "total")}
    ledger: list = []
    run = predict_games(engine, slate, {}, [], [], PickPolicy(), cut.isoformat(), ledger, shares=shares)
    checked = [m for g in run.games for m in g["markets"] if m["check"] is not None]
    assert checked, "markets with a market probability are checked"
    for m in checked:  # the checked chance sits between the model and the books
        lo_, hi_ = sorted((m["p_raw"], m["fair"]))
        assert lo_ - 1e-3 <= m["p"] <= hi_ + 1e-3
    store = {g["id"]: {"home_ml": -110} for g in run.games}
    remember_model(store, run.games)
    stored = [row for v in store.values() for row in v.get("model", [])]
    raw = {(g["id"], m["market"], m["side"]): m["p_raw"] for g in run.games for m in g["markets"]}
    assert stored and all(row[3] == raw[(gid, row[0], row[1])] for gid, v in store.items() for row in v["model"])
    for e in ledger:  # published picks say what the check did
        if e.get("p_model") is not None:
            assert any(r.startswith("market check:") for r in e["reasons"])
