"""Single-series and agreement metrics against direct formulas and hand-computed cases."""

import numpy as np
import pandas as pd
import pytest

from wind_uq_surrogate.metrics_pair import (agreement_metrics, moving_correlation, moving_mape, moving_nmae,
                                            moving_nrmse)
from wind_uq_surrogate.metrics_single import (moving_spectral_entropy, moving_std, normalized_range,
                                              single_series_metrics, spectral_entropy, turbulence_intensity)
from wind_uq_surrogate.windows import rolling, trailing_windows


def test_trailing_alignment_and_warmup():
    x = np.arange(6.0)
    out = rolling(x, 3, lambda w: w[:, -1])
    assert np.isnan(out[:2]).all()
    assert np.array_equal(out[2:], x[2:])  # the window ending at t contains x[t]
    assert trailing_windows(x, 3).shape == (4, 3)


def test_no_look_ahead():
    rng = np.random.default_rng(3)
    x = rng.uniform(1, 2, 40)
    y = x.copy()
    y[25:] += 10.0  # change the future only
    for f in (moving_std, turbulence_intensity, normalized_range, moving_spectral_entropy):
        a, b = f(x, 8), f(y, 8)
        assert np.allclose(a[:25], b[:25], equal_nan=True)


def test_std_ti_range_direct():
    x = np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
    w = 4
    for t in range(w - 1, len(x)):
        win = x[t - w + 1:t + 1]
        assert moving_std(x, w)[t] == pytest.approx(np.std(win))
        assert turbulence_intensity(x, w)[t] == pytest.approx(np.std(win) / win.mean())
        assert normalized_range(x, w)[t] == pytest.approx((win.max() - win.min()) / win.mean())
    assert moving_std(x, 4, ddof=1)[3] == pytest.approx(np.std(x[:4], ddof=1))


def test_ratio_metrics_nan_for_nonpositive_mean_and_missing():
    x = np.array([0.0, 0.0, 0.0, 1.0, np.nan, 2.0])
    ti = turbulence_intensity(x, 3)
    assert np.isnan(ti[2])            # mean zero
    assert np.isfinite(ti[3])
    assert np.isnan(ti[4]) and np.isnan(ti[5])  # window with a missing value


def test_spectral_entropy_limits():
    n = 16
    t = np.arange(n)
    assert spectral_entropy(np.sin(2 * np.pi * 2 * t / n)) == pytest.approx(0.0, abs=1e-12)
    impulse = np.zeros(n)
    impulse[0] = 1.0                  # flat periodogram at every non-zero frequency
    assert spectral_entropy(impulse) == pytest.approx(1.0, abs=1e-12)
    assert np.isnan(spectral_entropy(np.full(n, 3.0)))
    noise = np.random.default_rng(0).standard_normal(256)
    assert 0.8 < spectral_entropy(noise) < 1.0
    with pytest.raises(ValueError):
        spectral_entropy([1.0, 2.0, 3.0])


def test_spectral_entropy_linear_detrend_removes_ramp():
    ramp = np.arange(12.0)
    assert np.isnan(spectral_entropy(ramp, detrend="linear"))
    assert np.isfinite(spectral_entropy(ramp, detrend="mean"))


def test_moving_spectral_entropy_matches_scalar():
    x = np.random.default_rng(1).uniform(3, 9, 30)
    out = moving_spectral_entropy(x, 8)
    for t in (7, 15, 29):
        assert out[t] == pytest.approx(spectral_entropy(x[t - 7:t + 1]))


def test_single_series_table_columns_and_index():
    idx = pd.date_range("2001-01-01", periods=10, freq="h")
    s = pd.Series(np.linspace(1, 3, 10), index=idx)
    tab = single_series_metrics(s, 4, prefix="f_")
    assert list(tab.columns) == ["f_std", "f_ti", "f_nrange", "f_sentropy"]
    assert tab.index.equals(idx)
    with pytest.raises(ValueError):
        single_series_metrics(s, 4, metrics=("nope",))


def test_window_validation():
    with pytest.raises(TypeError):
        moving_std(np.ones(5), 2.5)
    with pytest.raises(ValueError):
        moving_std(np.ones(5), 1)
    with pytest.raises(ValueError):
        moving_std(np.ones(5), 6)


def test_pair_metrics_direct():
    r = np.array([2.0, 4.0, 5.0, 3.0, 6.0, 8.0])
    s = np.array([2.5, 3.0, 5.5, 3.0, 7.0, 7.0])
    w = 4
    t = 5
    rw, sw = r[t - w + 1:t + 1], s[t - w + 1:t + 1]
    assert moving_correlation(r, s, w)[t] == pytest.approx(np.corrcoef(rw, sw)[0, 1])
    assert moving_mape(r, s, w)[t] == pytest.approx(100 * np.mean(np.abs(rw - sw) / rw))
    rmse = np.sqrt(np.mean((rw - sw) ** 2))
    mae = np.mean(np.abs(rw - sw))
    assert moving_nrmse(r, s, w, 10.0)[t] == pytest.approx(rmse / 10.0)
    assert moving_nrmse(r, s, w, "range")[t] == pytest.approx(rmse / (rw.max() - rw.min()))
    assert moving_nmae(r, s, w, "mean")[t] == pytest.approx(mae / rw.mean())


def test_pair_metric_edge_cases():
    r = np.array([0.0, 0.0, 1.0, 1.0, 1.0])
    s = np.array([0.5, 0.2, 1.0, 2.0, 1.0])
    mape = moving_mape(r, s, 2, min_reference=0.0)
    assert np.isnan(mape[1])                      # no reference above the threshold
    assert mape[2] == pytest.approx(0.0)          # only the step with r = 1 counts
    assert np.isnan(moving_nrmse(r, s, 2, "range")[4])  # constant reference: zero range
    assert np.isnan(moving_correlation(r, s, 3)[4])     # constant reference
    with pytest.raises(ValueError):
        moving_nrmse(r, s, 2, -1.0)
    with pytest.raises(ValueError):
        moving_nrmse(r, s, 2, "capacity")
    with pytest.raises(ValueError):
        moving_mape(r, s[:4], 2)


def test_agreement_table():
    r = np.linspace(1, 2, 12)
    s = r + 0.1 * np.sin(np.arange(12))
    tab = agreement_metrics(r, s, 5, normalizer=2.0, prefix="a_vs_b_")
    assert list(tab.columns) == ["a_vs_b_corr", "a_vs_b_mape", "a_vs_b_nrmse", "a_vs_b_nmae"]
    assert tab.iloc[:4].isna().all().all() and tab.iloc[4:].notna().all().all()
