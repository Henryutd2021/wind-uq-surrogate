"""Predictive distributions, scores, pinball-loss calibration and PIT recalibration."""

import numpy as np
import pytest
from scipy import integrate, stats

from wind_uq_surrogate.calibration import (CRPS_SCALE_CONSTANT, PITRecalibrator, block_labels,
                                           crps_optimal_scale, optimal_scale)
from wind_uq_surrogate.probabilistic import (cdf, crps, crps_gaussian, crps_laplace, default_levels,
                                             interval_scores, pinball_loss, quantiles, reliability)

LEVELS = default_levels(0.05)


def test_levels_and_quantiles():
    assert LEVELS[0] == pytest.approx(0.05) and LEVELS[-1] == pytest.approx(0.95) and len(LEVELS) == 19
    q = quantiles([1.0, 2.0], [0.5, 2.0], [0.25, 0.5, 0.75], "laplace")
    assert q[1] == pytest.approx(stats.laplace.ppf([0.25, 0.5, 0.75], loc=2.0, scale=2.0))
    qg = quantiles([0.0], [3.0], [0.1, 0.9], "gaussian")
    assert qg[0] == pytest.approx(stats.norm.ppf([0.1, 0.9], scale=3.0))
    with pytest.raises(ValueError):
        quantiles([0.0], [0.0], [0.5], "gaussian")
    with pytest.raises(ValueError):
        quantiles([0.0], [1.0], [0.5, 0.4], "gaussian")
    with pytest.raises(ValueError):
        quantiles([0.0], [1.0], [0.5], "student")
    with pytest.raises(ValueError):
        default_levels(0.3)


def test_pinball_hand_computed():
    y = np.array([1.0, 3.0])
    q = np.array([[2.0, 2.0], [2.0, 2.0]])
    # tau=0.1: y=1 -> 0.9*1, y=3 -> 0.1*1; tau=0.9: y=1 -> 0.1, y=3 -> 0.9
    assert pinball_loss(y, q, [0.1, 0.9], average=False) == pytest.approx([0.5, 0.5])
    assert pinball_loss(y, q, [0.1, 0.9]) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        pinball_loss(y, q[:, :1], [0.1, 0.9])


@pytest.mark.parametrize("dist,family", [("gaussian", stats.norm), ("laplace", stats.laplace)])
def test_closed_form_crps_matches_integral(dist, family):
    for y, m, s in [(0.3, 0.0, 1.0), (2.5, 1.0, 0.4), (-1.0, 0.5, 2.0)]:
        f = lambda x: (family.cdf(x, loc=m, scale=s) - (y <= x)) ** 2
        num = integrate.quad(f, m - 40 * s, y, limit=200)[0] + integrate.quad(f, y, m + 40 * s, limit=200)[0]
        assert crps(np.array([y]), m, s, dist)[0] == pytest.approx(num, rel=1e-6)
    assert crps_gaussian(0.0, 0.0, 1.0) == pytest.approx((np.sqrt(2) - 1) / np.sqrt(np.pi))
    assert crps_laplace(0.0, 0.0, 1.0) == pytest.approx(0.25)


def test_reliability_and_intervals():
    rng = np.random.default_rng(8)
    y = rng.laplace(0.0, 2.0, 40000)
    q = quantiles(np.zeros_like(y), 2.0, LEVELS, "laplace")
    rel = reliability(y, q, LEVELS)
    assert rel["deviation"].abs().max() < 0.01
    iv = interval_scores(y, q, LEVELS, (0.5, 0.9))
    assert iv["coverage"].tolist() == pytest.approx([0.5, 0.9], abs=0.01)
    assert iv["mean_width"].iloc[1] == pytest.approx(2 * 2.0 * np.log(1 / 0.1))  # Laplace 5%-95% width
    with pytest.raises(ValueError):
        interval_scores(y, q, LEVELS, (0.8 + 1e-3,))
    assert cdf(0.0, 0.0, 1.0, "laplace") == pytest.approx(0.5)


def test_optimal_scale_per_sample_brute_force():
    rng = np.random.default_rng(9)
    m = rng.uniform(0, 1, 50)
    y = m + rng.laplace(0, 0.1, 50)
    s, loss = optimal_scale(y, m, LEVELS, "laplace", bounds=(1e-4, 2.0))
    grid = np.linspace(1e-4, 2.0, 40001)
    for i in range(0, 50, 7):
        q = m[i] + grid[:, None] * stats.laplace.ppf(LEVELS)[None, :]
        u = y[i] - q
        obj = np.maximum(LEVELS * u, (LEVELS - 1) * u).mean(axis=1)
        assert loss[i] <= obj.min() + 1e-12
        assert loss[i] == pytest.approx(pinball_loss(y[i:i + 1], quantiles(m[i:i + 1], s[i:i + 1], LEVELS,
                                                                          "laplace"), LEVELS))


def test_zero_error_gives_lower_bound_and_bounds_respected():
    s, _ = optimal_scale(np.array([1.0, 1.0]), np.array([1.0, 0.0]), LEVELS, "gaussian", bounds=(0.01, 0.3))
    assert s[0] == pytest.approx(0.01)
    assert 0.01 <= s[1] <= 0.3
    with pytest.raises(ValueError):
        optimal_scale([1.0], [0.0], LEVELS, bounds=(0.0, 1.0))
    with pytest.raises(ValueError):
        optimal_scale([np.nan], [0.0], LEVELS)


@pytest.mark.parametrize("dist", ["gaussian", "laplace"])
def test_dense_levels_approach_crps_optimum(dist):
    dense = np.arange(1, 1000) / 1000
    e = np.array([0.05, 0.3, 1.2])
    s, _ = optimal_scale(e, np.zeros(3), dense, dist)
    assert s == pytest.approx(crps_optimal_scale(e, np.zeros(3), dist), rel=0.02)
    c = CRPS_SCALE_CONSTANT[dist]
    b = 1.0 / c   # CRPS-optimal scale for |e| = 1 is a minimum of the closed-form CRPS
    grid = np.linspace(0.2, 3.0, 20001)
    assert grid[np.argmin(crps(np.ones_like(grid), 0.0, grid, dist))] == pytest.approx(b, rel=1e-3)


def test_group_scale_matches_brute_force_and_singletons():
    rng = np.random.default_rng(10)
    m = rng.uniform(0, 1, 60)
    y = m + rng.normal(0, 0.2, 60)
    groups = block_labels(60, 20)
    s, loss = optimal_scale(y, m, LEVELS, "gaussian", bounds=(1e-3, 3.0), groups=groups)
    assert len(np.unique(s)) <= 3
    grid = np.linspace(1e-3, 3.0, 30001)
    for g in range(3):
        sel = groups == g
        q = m[sel][None, :, None] + grid[:, None, None] * stats.norm.ppf(LEVELS)[None, None, :]
        u = y[sel][None, :, None] - q
        obj = np.maximum(LEVELS * u, (LEVELS - 1) * u).mean(axis=(1, 2))
        assert loss[sel][0] <= obj.min() + 1e-12
    s1, l1 = optimal_scale(y, m, LEVELS, "gaussian", bounds=(1e-3, 3.0), groups=np.arange(60))
    s2, l2 = optimal_scale(y, m, LEVELS, "gaussian", bounds=(1e-3, 3.0))
    assert np.allclose(l1, l2, atol=1e-12)
    assert np.allclose(s1, s2, rtol=1e-9)


def test_block_labels():
    assert block_labels(7, 3).tolist() == [0, 0, 0, 1, 1, 1, 2]
    with pytest.raises(ValueError):
        block_labels(5, 0)


def test_recalibration_fixes_overconfident_scale():
    rng = np.random.default_rng(11)
    n = 20000
    m = np.zeros(n)
    y = rng.laplace(0.0, 1.0, n)
    too_small = np.full(n, 0.4)     # overconfident predictive scale
    before = reliability(y[n // 2:], quantiles(m[n // 2:], too_small[n // 2:], LEVELS, "laplace"), LEVELS)
    rec = PITRecalibrator("laplace").fit(y[:n // 2], m[:n // 2], too_small[:n // 2])
    after = reliability(y[n // 2:], rec.quantiles(m[n // 2:], too_small[n // 2:], LEVELS), LEVELS)
    assert before["deviation"].abs().mean() > 0.1
    assert after["deviation"].abs().mean() < 0.01
    lv = rec.calibrated_levels(LEVELS)
    assert np.all(np.diff(lv) >= 0) and lv.min() > 0 and lv.max() < 1
    with pytest.raises(RuntimeError):
        PITRecalibrator("laplace").calibrated_levels(LEVELS)
