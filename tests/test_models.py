import math
from datetime import datetime, timedelta

import numpy as np
import pytest
from scipy.stats import skellam

from sportsedge.config import get_config
from sportsedge.models.count_models import CountRatings
from sportsedge.models.distributions import (count_final_distributions, gaussian_margin_dist, gaussian_total_dist,
                                             joint_score_matrix)
from sportsedge.models.kalman import KalmanRatings
from sportsedge.types import GameResult

T0 = datetime(2024, 1, 1)


def _gaussian_games(n_games=3000, seed=0):
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(12)]
    strength = np.linspace(-6, 6, 12)
    out = []
    for i in range(n_games):
        h, a = rng.choice(12, 2, replace=False)
        m = strength[h] - strength[a] + 2.0 + rng.normal(0, 12)
        out.append(GameResult(str(i), "NBA", T0 + timedelta(hours=6 * i), teams[h], teams[a],
                              home_score=110 + int(round(m / 2)), away_score=110 - int(round(m / 2))))
    return teams, strength, out


def test_kalman_recovers_strengths_and_shrinks_variance():
    teams, strength, games = _gaussian_games()
    kf = KalmanRatings(obs_sd=12, prior_sd=5, process_sd_per_day=0.0, intercept_prior=0,
                       intercept_prior_sd=3, offseason_days=10_000)
    _, v0, _ = kf.predict("T0", "T11")
    for g in games:
        kf.update(g.home, g.away, g.margin, when=g.start_time)
    est = np.array([kf.rating(t)[0] for t in teams])
    assert np.corrcoef(est, strength)[0, 1] > 0.97
    assert kf.x[0] == pytest.approx(2.0, abs=0.6)  # home advantage
    _, v1, _ = kf.predict("T0", "T11")
    assert v1 < v0 / 10


def test_kalman_tune_finds_sensible_noise():
    _, _, games = _gaussian_games(1500)
    tmpl = KalmanRatings(obs_sd=8, prior_sd=5, process_sd_per_day=0.05, intercept_prior=0,
                         intercept_prior_sd=3, offseason_days=10_000)
    tuned = KalmanRatings.tune(games, tmpl)
    assert 10.5 < tuned.obs_sd < 13.5


def test_margin_distributions():
    md = gaussian_margin_dist(np.array([2.5, -1.0]), 13.5, get_config("NFL").key_number_weights, 0.08, 0.5,
                              ties_allowed=True, span=80)
    assert md.pmf.sum(axis=1) == pytest.approx([1, 1])
    i3, i2 = np.flatnonzero(md.support == 3)[0], np.flatnonzero(md.support == 2)[0]
    assert md.pmf[0, i3] > 1.8 * md.pmf[0, i2]  # key number
    assert md.tie()[0] < 0.01
    p, push = md.cover(-3.0)
    assert push[0] > 0.05  # landing exactly on 3
    nba = gaussian_margin_dist(np.array([0.0]), 12.0, None, 1.0, 0.5, ties_allowed=False, span=90)
    assert nba.tie()[0] == 0 and nba.win()[0] == pytest.approx(0.5, abs=1e-6)
    td = gaussian_total_dist(np.array([224.0]), 18.0)
    over, _ = td.over(224.5)
    assert over[0] == pytest.approx(0.49, abs=0.02)


def test_poisson_score_matrix_matches_skellam():
    lh, la = 3.2, 2.7
    J = joint_score_matrix(np.array([lh]), np.array([la]), 20, None)
    md, td = count_final_distributions(J, np.array([0.5]))
    p_reg_win = skellam.sf(0, lh, la)
    p_tie = skellam.pmf(0, lh, la)
    assert md.win()[0] == pytest.approx(p_reg_win + 0.5 * p_tie, abs=1e-6)
    assert md.tie()[0] == 0
    assert td.mean()[0] == pytest.approx(lh + la + p_tie, abs=1e-3)


def test_count_model_recovers_attack_and_dispersion():
    rng = np.random.default_rng(5)
    teams = [f"T{i}" for i in range(10)]
    att = np.linspace(-0.3, 0.3, 10)
    games = []
    r_true = 6.0
    for i in range(4000):
        h, a = rng.choice(10, 2, replace=False)
        lh = math.exp(math.log(4.4) + 0.05 + att[h])
        la = math.exp(math.log(4.4) + att[a])
        games.append(GameResult(str(i), "MLB", T0 + timedelta(hours=4 * i), teams[h], teams[a],
                                home_score=int(rng.negative_binomial(r_true, r_true / (r_true + lh))),
                                away_score=int(rng.negative_binomial(r_true, r_true / (r_true + la)))))
    cm = CountRatings(decay_per_day=0.0, ridge_sd=0.3, park_sd=0.01, window_days=100_000, dispersion=10.0)
    cm.fit(games, games[-1].start_time + timedelta(days=1))
    est = np.array([cm.beta[2 + cm.teams[t]] for t in teams])
    assert np.corrcoef(est, att)[0, 1] > 0.95
    assert 4.0 < cm.dispersion < 9.0
    m, S = cm.predict_log_rates("T9", "T0")
    assert m[0] > m[1] and S[0, 0] > 0
