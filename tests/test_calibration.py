import numpy as np
import pytest

from sportsedge.stats.calibration import (BetaCalibrator, IsotonicCalibrator, LogisticRegression, PlattCalibrator,
                                          StackingCalibrator, binomial_test_greater, expected_calibration_error,
                                          log_loss, logit, poisson_binomial_z, sigmoid, wilson_interval)


def test_logistic_recovers_coefficients():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20000, 2))
    y = rng.random(20000) < sigmoid(0.3 + 1.5 * X[:, 0] - 0.7 * X[:, 1])
    lr = LogisticRegression(l2=1e-3).fit(X, y)
    assert lr.coef_ == pytest.approx([0.3, 1.5, -0.7], abs=0.08)


def _overconfident(n=20000, seed=1):
    rng = np.random.default_rng(seed)
    true = rng.uniform(0.2, 0.8, n)
    y = (rng.random(n) < true).astype(float)
    q = sigmoid(2.0 * logit(true))  # overconfident forecaster
    return q, y


@pytest.mark.parametrize("cal", [PlattCalibrator, BetaCalibrator, IsotonicCalibrator])
def test_calibrators_fix_overconfidence(cal):
    q, y = _overconfident()
    c = cal().fit(q[:10000], y[:10000])
    fixed = c.transform(q[10000:])
    assert log_loss(fixed, y[10000:]) < log_loss(q[10000:], y[10000:])
    assert expected_calibration_error(fixed, y[10000:]) < 0.02


def test_isotonic_is_monotone():
    q, y = _overconfident(3000)
    c = IsotonicCalibrator().fit(q, y)
    out = c.transform(np.linspace(0, 1, 101))
    assert np.all(np.diff(out) >= -1e-12)


def test_stacker_learns_to_trust_better_source():
    rng = np.random.default_rng(3)
    n = 8000
    truth = rng.normal(0, 1, n)
    y = (rng.random(n) < sigmoid(truth)).astype(float)
    good = sigmoid(truth + rng.normal(0, 0.2, n))
    bad = sigmoid(truth + rng.normal(0, 1.5, n))
    market = sigmoid(truth + rng.normal(0, 0.4, n))
    st = StackingCalibrator(["good", "bad"], l2=1.0)
    st.fit([{"good": g, "bad": b} for g, b in zip(good, bad)], list(market), y)
    w = st.weights()
    assert w["mkt_model:good"] > w["mkt_model:bad"]
    assert w["model_only:good"] > 0.6


def test_interval_and_tests():
    lo, hi = wilson_interval(70, 100)
    assert lo == pytest.approx(0.604, abs=0.002) and hi == pytest.approx(0.781, abs=0.002)
    assert binomial_test_greater(80, 100, 0.70) < 0.02
    assert binomial_test_greater(70, 100, 0.70) > 0.4
    p = np.full(1000, 0.7)
    assert abs(poisson_binomial_z(p, np.r_[np.ones(700), np.zeros(300)])) < 1e-9
