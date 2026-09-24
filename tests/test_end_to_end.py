"""End-to-end statistical checks on synthetic leagues with known truth."""
import numpy as np
import pytest

from sportsedge.backtest import walk_forward
from sportsedge.picks import PickPolicy
from sportsedge.synthetic import generate


@pytest.mark.parametrize("league,seasons", [("NBA", 2), ("NHL", 2)])
def test_walk_forward_is_calibrated(league, seasons):
    games = generate(league, seasons=seasons, seed=21)
    # Relax the EV requirement so enough picks exist to test their calibration.
    res = walk_forward(league, games, PickPolicy(require_positive_ev=False, min_lower_bound=0.0),
                       n_draws=200)
    s = res.summary
    assert s["n_games"] > 300
    assert s["ece_model"] < 0.05
    assert s["logloss_model"] < 0.69  # informative
    assert s["logloss_model"] < s["logloss_model_only"] + 0.01  # market information helps
    p = s["picks"]
    assert p["n"] > 20
    assert p["mean_p"] >= 0.70
    assert abs(p["calibration_z"]) < 3.0  # 70%+ picks win about as often as claimed


def test_prediction_rolls_ratings_forward_to_game_date():
    from datetime import timedelta

    from sportsedge.engine import EngineSettings, LeagueEngine
    from sportsedge.types import Game

    games = generate("NBA", seasons=1, seed=8)
    eng = LeagueEngine("NBA", EngineSettings(n_draws=500)).fit(games)
    ratings = {t: eng.kf_margin.rating(t)[0] for t in eng.kf_margin.index}
    best, worst = max(ratings, key=ratings.get), min(ratings, key=ratings.get)
    last = games[-1].start_time
    soon = eng.predict(Game("a", "NBA", last + timedelta(days=1), best, worst))
    next_season = eng.predict(Game("b", "NBA", last + timedelta(days=150), best, worst))
    # between-season regression pulls the matchup toward even and widens the band
    assert 0.5 < next_season.home_win_prob < soon.home_win_prob
    assert (next_season.home_win_upper - next_season.home_win_lower) > (soon.home_win_upper - soon.home_win_lower)
    assert eng.kf_margin.last_time == games[-1].start_time  # learned state untouched
