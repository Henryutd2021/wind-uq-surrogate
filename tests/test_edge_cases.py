"""Leakage guards, ties, invariances and input validation."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from wind_uq_surrogate.calibration import PITRecalibrator, optimal_scale
from wind_uq_surrogate.features import FeatureSpec, PlantSeries, training_table
from wind_uq_surrogate.metrics_pair import moving_correlation
from wind_uq_surrogate.metrics_single import spectral_entropy
from wind_uq_surrogate.metrics_spatial import variogram_over_time
from wind_uq_surrogate.probabilistic import default_levels, pinball_loss, quantiles, reliability
from wind_uq_surrogate.surrogate import Surrogate
from wind_uq_surrogate.synthetic import synthetic_fleet
from wind_uq_surrogate.validation import (cross_validate, nested_feasible, nested_recalibrated_quantiles, plant_splits,
                                          time_split)

LEVELS = default_levels(0.05)
SPEC = FeatureSpec(window=8, pairs=(("ws_a", "ws_b"),), member_prefix="ens_")


def _with_actual(plant, new_actual):
    data = plant.data.copy()
    data["actual"] = new_actual
    return PlantSeries(plant.plant_id, data, plant.capacity)


def test_nested_recalibration_does_not_use_test_plant_labels():
    fl = synthetic_fleet(n_plants=5, n_unobserved=0, n_steps=150, n_members=4, seed=6)
    factory = lambda: Surrogate("ridge")   # noqa: E731
    tab = training_table(fl.observed, SPEC, LEVELS, target_window=9)
    splits = plant_splits(tab["plant_id"])
    base = nested_recalibrated_quantiles(tab, factory, splits, LEVELS, inner_n_splits=None)
    # change only the observations of plant P01 (the test plant of the first fold)
    changed = [(_with_actual(p, p.data["actual"][::-1].to_numpy()) if p.plant_id == "P01" else p)
               for p in fl.observed]
    tab2 = training_table(changed, SPEC, LEVELS, target_window=9)
    alt = nested_recalibrated_quantiles(tab2, factory, splits, LEVELS, inner_n_splits=None)
    qcols = [c for c in base.columns if c.startswith("q_")]
    sel = base["plant_id"] == "P01"
    assert np.allclose(base.loc[sel, qcols].to_numpy(), alt.loc[sel, qcols].to_numpy())
    assert not np.allclose(base.loc[~sel, qcols].to_numpy(), alt.loc[~sel, qcols].to_numpy())


def test_time_split_purges_rows_whose_targets_see_the_test_period():
    fl = synthetic_fleet(n_plants=2, n_unobserved=0, n_steps=130, n_members=3, seed=7)
    tab = training_table(fl.observed, SPEC, LEVELS, target_window=9)
    assert (tab["target_reach"] == 4).all()
    (tr, te), = time_split(tab, 0.25)
    changed = []
    for p in fl.observed:
        rows = tab.index[tab["plant_id"] == p.plant_id]
        first_test_time = tab.loc[np.intersect1d(rows, te), "time"].min()
        a = p.data["actual"].copy()
        a[a.index >= first_test_time] = 0.0          # change only test-period observations
        changed.append(_with_actual(p, a.to_numpy()))
    tab2 = training_table(changed, SPEC, LEVELS, target_window=9)
    assert np.allclose(tab["target_scale"].iloc[tr], tab2["target_scale"].iloc[tr])
    (tr0, _), = time_split(tab, 0.25, gap=0)
    assert len(tr0) == len(tr) + 2 * 4
    with pytest.raises(ValueError):
        time_split(tab, 0.25, gap=-1)
    assert (training_table(fl.observed, SPEC, LEVELS, block=6)["target_reach"] == 5).all()
    # the reach survives a CSV round trip; without it the gap must be given
    import io
    buf = io.StringIO()
    tab.to_csv(buf, index=False)
    back = pd.read_csv(io.StringIO(buf.getvalue()))
    (tr_back, te_back), = time_split(back, 0.25)
    assert np.array_equal(tr_back, tr) and np.array_equal(te_back, te)
    with pytest.raises(ValueError, match="give gap explicitly"):
        time_split(back.drop(columns="target_reach"), 0.25)


def test_plant_splits_with_groups():
    ids = np.array(["A", "A", "B", "C", "C", "D"])
    splits = plant_splits(ids, groups={"A": "north", "B": "north", "C": "south", "D": "south"})
    assert len(splits) == 2
    for tr, te in splits:
        assert {"A", "B"} <= set(ids[te]) or {"C", "D"} <= set(ids[te])
    with pytest.raises(ValueError):
        plant_splits(ids, groups={"A": 1})


def test_surrogate_uses_column_names():
    rng = np.random.default_rng(8)
    X = pd.DataFrame({"a": rng.uniform(0, 1, 60), "b": rng.uniform(0, 100, 60)})
    y = np.exp(0.5 * X["a"] + 0.01 * X["b"])
    model = Surrogate("ridge").fit(X, y)
    assert np.allclose(model.predict(X), model.predict(X[["b", "a"]]))
    with pytest.raises(ValueError):
        model.predict(X[["a"]])
    with pytest.raises(ValueError):
        model.predict(X.assign(c=1.0))
    with pytest.raises(ValueError):
        model.predict(X.to_numpy()[:, :1])
    arr_model = Surrogate("ridge").fit(X.to_numpy(), y)
    with pytest.raises(ValueError):
        arr_model.predict(X)


def test_recalibrator_ties_and_saturation():
    # every observation equals the forecast: all PIT values are 0.5, so every recalibrated quantile is the forecast
    rec = PITRecalibrator("gaussian").fit(np.zeros(100), np.zeros(100), np.ones(100))
    q = rec.quantiles(np.zeros(3), np.ones(3), [0.05, 0.5, 0.95])
    assert np.allclose(q, 0.0)
    # repeated PIT values: the generalized inverse picks the tied value
    y = np.array([0.0, 0.0, 0.0, 1.0])
    rec2 = PITRecalibrator("laplace", interpolation="inverted_cdf").fit(y, np.zeros(4), np.ones(4))
    lv = rec2.calibrated_levels([0.25, 0.5, 0.75, 0.9])
    assert np.allclose(lv[:3], 0.5) and lv[3] == pytest.approx(stats.laplace.cdf(1.0))
    # observations far below every prediction: levels saturate at eps, quantiles stay finite
    rec3 = PITRecalibrator("gaussian").fit(np.full(10, -50.0), np.zeros(10), np.ones(10))
    assert np.isfinite(rec3.quantiles([0.0], [1.0], LEVELS)).all()
    with pytest.raises(ValueError):
        PITRecalibrator("gaussian").fit(np.zeros(3), np.zeros(3), -np.ones(3))
    with pytest.raises(ValueError):
        rec.quantiles([0.0], [-1.0], LEVELS)
    with pytest.raises(ValueError):
        PITRecalibrator("gaussian", interpolation="nearest")


def test_group_scale_returns_smallest_minimizer_on_a_flat_optimum():
    # with levels 0.25 and 0.75 each error adds slope -c/2 before its breakpoint e/c and +c/2 after it, so for two
    # errors the summed loss is flat between 1/c and 2/c; the smallest minimizer is the left end
    e = np.array([1.0, 2.0])
    tau = np.array([0.25, 0.75])
    c = stats.norm.ppf(0.75)
    for scale in (1.0, 1e-3):
        s, loss = optimal_scale(scale * e, np.zeros(2), tau, "gaussian", bounds=(1e-6, 10.0), groups=np.zeros(2))
        assert s[0] == pytest.approx(scale / c)
        grid = np.linspace(scale / c, 2 * scale / c, 7)
        u = scale * e[None, :, None] - grid[:, None, None] * np.array([-c, c])[None, None, :]
        assert np.allclose(np.maximum(tau * u, (tau - 1) * u).mean(axis=(1, 2)), loss[0])


def test_median_only_levels_are_rejected():
    with pytest.raises(ValueError, match="not identifiable"):
        optimal_scale([1.0, 2.0], [0.0, 0.0], [0.5])
    with pytest.raises(ValueError, match="not identifiable"):
        optimal_scale([1.0, 2.0], [0.0, 0.0], [0.5], bounds=(0.1, 1.0), groups=[0, 0])


def test_entropy_and_correlation_are_shift_and_scale_invariant():
    x = np.arange(8.0) + np.array([0, 0.3, -0.2, 0.1, 0, 0.4, -0.1, 0.2])
    h = spectral_entropy(x)
    for t in (1e-8 * x, x + 1e7, 1e6 * x - 3.0):
        assert spectral_entropy(t) == pytest.approx(h, rel=1e-6)
    y = np.sin(np.arange(8.0))
    r = moving_correlation(x, y, 8)[-1]
    assert moving_correlation(1e-7 * x, 1e-7 * y, 8)[-1] == pytest.approx(r, rel=1e-9)
    assert moving_correlation(x + 1e6, y, 8)[-1] == pytest.approx(r, rel=1e-6)
    assert moving_correlation(1e-7 * x, 1e-7 * x, 8)[-1] == pytest.approx(1.0)
    assert np.isnan(moving_correlation(np.full(8, 1e7), y, 8)[-1])


def test_scores_reject_missing_values():
    q = quantiles(np.zeros(3), 1.0, LEVELS, "laplace")
    with pytest.raises(ValueError):
        pinball_loss(np.array([0.0, np.nan, 1.0]), q, LEVELS)
    q[1, 2] = np.nan
    with pytest.raises(ValueError):
        reliability(np.zeros(3), q, LEVELS)
    with pytest.raises(ValueError):
        quantiles([np.nan], [1.0], LEVELS, "laplace")
    with pytest.raises(ValueError):
        quantiles([0.0], [1.0], [0.1, np.nan], "laplace")


def test_variogram_configuration_errors_are_raised():
    field = np.random.default_rng(9).standard_normal((3, 10))
    coords = np.random.default_rng(10).uniform(0, 5, (10, 2))
    with pytest.raises(ValueError):
        variogram_over_time(field, coords, [0.0, 1.0, 2.0, 3.0], model="linear")
    with pytest.raises(ValueError):
        variogram_over_time(field, coords, [0.0, 1.0, 2.0, 3.0], weighting="robust")
    with pytest.raises(ValueError):
        variogram_over_time(field, coords, [2.0, 1.0])


def test_cached_outer_predictions_are_matched_by_key():
    fl = synthetic_fleet(n_plants=6, n_unobserved=0, n_steps=90, n_members=4, seed=6)
    factory = lambda: Surrogate("ridge")   # noqa: E731
    tab = training_table(fl.observed, SPEC, LEVELS, target_window=7)
    splits = plant_splits(tab["plant_id"], 2)
    cv = cross_validate(tab, {"m": factory}, splits)
    direct = nested_recalibrated_quantiles(tab, factory, splits, LEVELS)
    cached = nested_recalibrated_quantiles(tab, factory, splits, LEVELS, outer_predictions=cv)
    shuffled = nested_recalibrated_quantiles(tab, factory, splits, LEVELS, outer_predictions=cv.iloc[::-1])
    pd.testing.assert_frame_equal(direct, cached)
    pd.testing.assert_frame_equal(direct, shuffled)
    with pytest.raises(ValueError):
        nested_recalibrated_quantiles(tab, factory, splits, LEVELS, outer_predictions=cv.iloc[1:])
    with pytest.raises(ValueError):
        nested_recalibrated_quantiles(tab, factory, splits, LEVELS, outer_predictions=pd.concat([cv, cv.iloc[:1]]))
    assert nested_feasible(tab, splits)
    halves = {p: p in ("P01", "P02", "P03") for p in tab["plant_id"].unique()}
    gsplits = plant_splits(tab["plant_id"], groups=halves)
    assert nested_feasible(tab, gsplits) and not nested_feasible(tab, gsplits, halves)   # one group per training set


def test_pit_recalibrator_shapes_and_bins():
    with pytest.raises(ValueError):
        PITRecalibrator("gaussian").fit(np.array([0.0, 10.0]), np.array([0.0, 10.0])[:, None], np.ones(2))
    with pytest.raises(ValueError):
        PITRecalibrator("gaussian").fit(np.array([0.0, 10.0]), np.zeros(3), 1.0)
    rec = PITRecalibrator("gaussian").fit(np.array([0.0, 10.0]), np.array([0.0, 10.0]), 1.0)
    assert np.allclose(rec.calibrated_levels([0.1, 0.9]), 0.5)
    field = np.random.default_rng(11).standard_normal((2, 12))
    coords = np.random.default_rng(12).uniform(0, 5, (12, 2))
    with pytest.raises(ValueError):
        variogram_over_time(field, coords, [0.0, 1.0, 2.0, np.nan, 5.0])


def test_per_sample_optimum_is_not_disturbed_by_a_huge_upper_bound():
    e = np.array([1.0])
    s1, l1 = optimal_scale(e, np.zeros(1), [0.1, 0.9], "gaussian", bounds=(1e-6, 1e14))
    s2, l2 = optimal_scale(e, np.zeros(1), [0.1, 0.9], "gaussian", bounds=(1e-6, 1e14), groups=[0])
    assert s1[0] == pytest.approx(1.0 / stats.norm.ppf(0.9)) and s1[0] == pytest.approx(s2[0])
    assert l1[0] == pytest.approx(l2[0])


def test_column_names_must_stay_unique_as_text():
    X = pd.DataFrame([[1.0, 2.0], [2.0, 1.0], [3.0, 0.5]], columns=[1, "1"])
    with pytest.raises(ValueError, match="duplicated"):
        Surrogate("ridge").fit(X, np.ones(3))
