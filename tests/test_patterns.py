import numpy as np
import pytest

from sportsedge.models.boosting import HistGradientBoosting, _logloss
from sportsedge.models.regimes import RegimeHMM


def _interaction_data(n=3000, seed=0, effect=0.6):
    rng = np.random.default_rng(seed)
    X = np.column_stack([rng.random(n) < 0.5, rng.random(n) * 3, rng.normal(size=(n, 5))]).astype(float)
    off = rng.normal(0, 1, n)
    eff = effect * ((X[:, 0] == 1) & (X[:, 1] > 1.0)) - 0.2 * (X[:, 0] == 1)
    y = (rng.random(n) < 1 / (1 + np.exp(-(off + eff)))).astype(float)
    return X, y, off, eff


def test_boosting_recovers_an_interaction():
    X, y, off, eff = _interaction_data()
    m = HistGradientBoosting().fit(X, y, off)
    oracle = _logloss(off[-600:], y[-600:]) - _logloss(off[-600:] + eff[-600:], y[-600:])
    assert m.trees_ and m.val_gain_ > 0.7 * oracle
    # the learned correction is large exactly where the planted effect is
    c = m.correction(X)
    hit = (X[:, 0] == 1) & (X[:, 1] > 1.0)
    assert c[hit].mean() - c[~hit].mean() > 0.15  # shrunk by early stopping, clearly separated


def test_boosting_leaves_offset_alone_without_signal():
    X, y, off, _ = _interaction_data(effect=0.0)
    rng = np.random.default_rng(3)
    y = (rng.random(len(off)) < 1 / (1 + np.exp(-off))).astype(float)  # outcome = offset only
    m = HistGradientBoosting().fit(X, y, off)
    assert np.abs(m.correction(X)).mean() < 0.05


def test_cv_path_and_fixed_fit_agree_in_shape():
    X, y, off, _ = _interaction_data(n=1500)
    path = HistGradientBoosting(n_estimators=30).fit_path(X[:1000], y[:1000], off[:1000], X[1000:], y[1000:], off[1000:])
    assert len(path) == 31 and path[0] > path.min()
    m = HistGradientBoosting().fit_fixed(X, y, off, int(np.argmin(path)))
    assert len(m.trees_) == int(np.argmin(path))


def test_sticky_hmm_recovers_regimes():
    rng = np.random.default_rng(1)
    means, stay = np.array([-0.8, 0.0, 0.8]), 0.95
    seqs, states = [], []
    for _ in range(30):
        s, z, st = 1, [], []
        for _ in range(120):
            if rng.random() > stay:
                s = rng.choice(3)
            z.append(rng.normal(means[s], 1.0))
            st.append(s)
        seqs.append(np.array(z))
        states.append(np.array(st))
    hmm = RegimeHMM().fit(seqs)
    assert hmm.means[0] < -0.4 and hmm.means[2] > 0.4
    assert np.all(np.diag(hmm.A) > 0.85)
    # filtered form is higher going into games played in the hot state
    hot_forms, cold_forms = [], []
    for z, st in zip(seqs, states):
        pred = hmm.stationary()
        for zt, s in zip(z, st):
            f = hmm.form(pred)[1]
            (hot_forms if s == 2 else cold_forms if s == 0 else []).append(f)
            pred = hmm.step(pred, zt)
    assert np.mean(hot_forms) > np.mean(cold_forms) + 0.3


def test_pattern_engine_is_leak_free_and_gated():
    from datetime import datetime, timedelta

    from sportsedge.models.patterns import FEATURES, PatternModel

    pm = PatternModel(offseason_days=90)
    t0 = datetime(2025, 1, 1)
    args = dict(neutral=False, structural_logit=0.2, rating_sd=0.1, rest=(2.0, 1.0), travel=(1.2, 1.0, 0.0), pace=0.0)
    x1 = pm.features("A", "B", t0, **args)
    x2 = pm.features("A", "B", t0, **args)
    assert np.array_equal(x1, x2)  # asking twice changes nothing
    pm.record("A", "B", t0, x1, 0.2, 1.0, z=2.5, margin=12)
    x3 = pm.features("A", "B", t0 + timedelta(days=2), **args)
    assert x3[FEATURES.index("recent_home")] > 0 and x3[FEATURES.index("recent_away")] < 0  # result used only afterwards
    assert pm.correction(x3) == 0.0 and not pm.active  # nothing learned yet -> no correction
