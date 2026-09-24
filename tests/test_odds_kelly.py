import numpy as np
import pytest

from sportsedge.stats.kelly import kelly_fraction, simultaneous_kelly, stake_fraction, uncertainty_shrinkage
from sportsedge.stats.odds import (american_to_decimal, consensus_probability, decimal_to_american, devig,
                                   prob_to_american)


def test_conversions_roundtrip():
    assert american_to_decimal(150) == pytest.approx(2.5)
    assert american_to_decimal(-200) == pytest.approx(1.5)
    for a in (-350, -110, 100, 145, 900):
        assert decimal_to_american(american_to_decimal(a)) == pytest.approx(a)
    assert prob_to_american(0.5) == pytest.approx(100)


@pytest.mark.parametrize("method", ["multiplicative", "additive", "power", "shin"])
def test_devig_sums_to_one(method):
    p = devig([-250, 205], method)
    assert p.sum() == pytest.approx(1.0)
    assert p[0] > 0.66


def test_devig_symmetric_market_is_even():
    for m in ("multiplicative", "shin", "power"):
        assert devig([-110, -110], m)[0] == pytest.approx(0.5)


def test_shin_shades_longshot_more_than_proportional():
    mult = devig([-400, 300], "multiplicative")
    shin = devig([-400, 300], "shin")
    assert shin[1] < mult[1]  # favourite-longshot bias correction


def test_consensus_weights_sharp_book():
    p_equal = consensus_probability([(-150, 130), (-200, 170)])
    p_sharp = consensus_probability([(-150, 130), (-200, 170)], weights=[1, 5])
    assert p_sharp > p_equal


def test_kelly_and_shrinkage():
    assert kelly_fraction(0.6, 2.0) == pytest.approx(0.2)
    assert kelly_fraction(0.4, 2.0) == 0.0
    assert uncertainty_shrinkage(0.6, 0.0, 2.0) == pytest.approx(1.0)
    assert 0 < uncertainty_shrinkage(0.6, 0.05, 2.0) < 1
    assert stake_fraction(0.6, 0.05, 2.0) < 0.25 * 0.2 + 1e-12


def test_simultaneous_kelly_single_bet_matches_kelly():
    f = simultaneous_kelly([0.6], [2.0], max_total=1.0)
    assert f[0] == pytest.approx(0.2, abs=1e-3)
    f2 = simultaneous_kelly([0.6, 0.6, 0.6], [2.0, 2.0, 2.0], max_total=1.0)
    assert np.all(f2 < 0.2) and np.all(f2 > 0.1)
