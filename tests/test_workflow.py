"""Features, surrogates, validation and the end-to-end workflow on small synthetic fleets."""

import numpy as np
import pandas as pd
import pytest

from wind_uq_surrogate.features import (FeatureSpec, PlantSeries, SpatialSpec, build_features, calibration_targets,
                                        feature_columns, prediction_table, training_table)
from wind_uq_surrogate.probabilistic import default_levels
from wind_uq_surrogate.surrogate import Surrogate, gpr_kernel, select_gpr_kernel
from wind_uq_surrogate.synthetic import power_curve, synthetic_fleet
from wind_uq_surrogate.validation import cross_validate, plant_splits, score_predictive, score_scale, time_split
from wind_uq_surrogate.workflow import WorkflowConfig, run_workflow, save_result

LEVELS = default_levels(0.05)
SPEC = FeatureSpec(window=18, pairs=(("ws_a", "ws_b"),), member_prefix="ens_", pi_history=36)


@pytest.fixture(scope="module")
def fleet():
    return synthetic_fleet(n_plants=6, n_unobserved=1, n_steps=24 * 25, n_members=8, seed=2)


def test_synthetic_fleet_structure(fleet):
    assert len(fleet.observed) == 5 and len(fleet.unobserved) == 1
    p = fleet.plants[0]
    assert {"forecast", "ws_a", "ws_b", "actual"} <= set(p.data.columns)
    assert sum(c.startswith("ens_") for c in p.data.columns) == 8
    assert "actual" not in fleet.unobserved[0].data.columns
    assert (p.data["actual"] >= 0).all() and (p.data["actual"] <= p.capacity).all()
    assert power_curve(0.0) == pytest.approx(0.0) and power_curve(30.0) == pytest.approx(1.0, abs=1e-6)
    again = synthetic_fleet(n_plants=6, n_unobserved=1, n_steps=24 * 25, n_members=8, seed=2)
    assert again.plants[3].data.equals(fleet.plants[3].data)


def test_plant_series_checks():
    idx = pd.RangeIndex(5)
    with pytest.raises(ValueError):
        PlantSeries("x", pd.DataFrame({"actual": np.ones(5)}, index=idx), 10.0)
    with pytest.raises(ValueError):
        PlantSeries("x", pd.DataFrame({"forecast": np.ones(5)}, index=idx[::-1]), 10.0)
    with pytest.raises(ValueError):
        PlantSeries("x", pd.DataFrame({"forecast": [1, 2, np.nan, 4, 5.0]}, index=idx), 10.0)
    with pytest.raises(ValueError):
        PlantSeries("x", pd.DataFrame({"forecast": np.ones(5)}, index=idx), 0.0)


def test_features_never_use_actual():
    with pytest.raises(ValueError):
        FeatureSpec(window=7, single_series=("actual",))
    with pytest.raises(ValueError):
        FeatureSpec(window=7, pairs=(("ws_a", "actual"),))
    with pytest.raises(ValueError):
        FeatureSpec(window=7, member_prefix="act")


def test_features_independent_of_actual_and_future(fleet):
    p = fleet.plants[0]
    base = build_features(p, SPEC)
    changed = p.data.copy()
    changed["actual"] = changed["actual"][::-1].to_numpy()       # observations do not enter the features
    changed.iloc[400:, changed.columns.get_loc("ws_a")] += 5.0   # nor do future covariate values
    alt = build_features(PlantSeries(p.plant_id, changed, p.capacity), SPEC)
    pd.testing.assert_frame_equal(base.iloc[:400], alt.iloc[:400])
    assert base.iloc[:17].isna().any(axis=1).all()


def test_spatial_features():
    fl = synthetic_fleet(n_plants=2, n_unobserved=0, n_steps=60, n_members=0, n_turbines=16, seed=3)
    spec = FeatureSpec(window=10, spatial=SpatialSpec(bin_edges=(0.0, 1.0, 2.0, 3.0, 4.5, 6.0)))
    feats = build_features(fl.plants[0], spec)
    assert {"vg_nugget", "vg_sill", "vg_range"} <= set(feats.columns)
    assert feats["vg_sill"].notna().mean() > 0.9
    with pytest.raises(ValueError):
        build_features(synthetic_fleet(n_plants=2, n_unobserved=0, n_steps=60, seed=3).plants[0], spec)


def test_targets_and_tables(fleet):
    tgt = calibration_targets(fleet.plants[1], LEVELS)
    assert (tgt["target_scale"] > 0).all()
    blk = calibration_targets(fleet.plants[1], LEVELS, block=24)
    assert blk["target_scale"].iloc[:24].nunique() == 1
    with pytest.raises(ValueError):
        calibration_targets(fleet.unobserved[0], LEVELS)
    tab = training_table(fleet.observed, SPEC, LEVELS)
    assert tab["plant_id"].nunique() == 5 and not tab.isna().any().any()
    feats = feature_columns(tab)
    assert "target_scale" not in feats and "actual_pu" not in feats and "level" in feats
    ptab = prediction_table(fleet.unobserved, SPEC)
    assert set(feature_columns(ptab)) == set(feats)


def test_surrogates_fit_predict(fleet):
    tab = training_table(fleet.observed, SPEC, LEVELS)
    X, y = tab[feature_columns(tab)], tab["target_scale"]
    for name in ("gpr", "svr", "random_forest", "ridge", "constant"):
        model = Surrogate(name, max_train_samples=300).fit(X, y)
        pred = model.predict(X.iloc[:50])
        assert pred.shape == (50,) and np.all(pred > 0)
    gp = Surrogate("gpr", max_train_samples=300).fit(X, y)
    assert np.all(gp.predict_log_std(X.iloc[:10]) > 0)
    assert np.isfinite(gp.log_marginal_likelihood_)
    with pytest.raises(ValueError):
        Surrogate("ridge").fit(X, y).predict_log_std(X)
    with pytest.raises(ValueError):
        Surrogate("nope")
    with pytest.raises(ValueError):
        Surrogate("ridge").fit(X, -y)
    with pytest.raises(RuntimeError):
        Surrogate("ridge").predict(X)
    best, ranking = select_gpr_kernel(X.iloc[:200], y.iloc[:200], kinds=("rbf", "matern52"))
    assert best in ("rbf", "matern52") and len(ranking) == 2
    with pytest.raises(ValueError):
        gpr_kernel(3, "cosine")


def test_splits(fleet):
    tab = training_table(fleet.observed, SPEC, LEVELS)
    loo = plant_splits(tab["plant_id"])
    assert len(loo) == 5
    for tr, te in loo:
        assert set(tab["plant_id"].iloc[tr]).isdisjoint(set(tab["plant_id"].iloc[te]))
    assert len(plant_splits(tab["plant_id"], 2)) == 2
    with pytest.raises(ValueError):
        plant_splits(tab["plant_id"], 9)
    (tr, te), = time_split(tab, 0.25)
    last_train = tab.iloc[tr].groupby("plant_id")["time"].max()
    first_test = tab.iloc[te].groupby("plant_id")["time"].min()
    assert (last_train < first_test).all()


def _cv_pinball(fleet_, models):
    tab = training_table(fleet_.observed, SPEC, LEVELS)
    factories = {m: (lambda m=m: Surrogate(m, max_train_samples=400)) for m in models}
    cv = cross_validate(tab, factories, plant_splits(tab["plant_id"]))
    scores, _ = score_predictive(cv, LEVELS)
    return scores.set_index("model")["pinball"], score_scale(cv).set_index("model")


def test_surrogate_beats_constant_when_signal_present():
    fl = synthetic_fleet(n_plants=6, n_unobserved=0, n_steps=24 * 30, n_members=8, signal=1.5, seed=4)
    pin, sc = _cv_pinball(fl, ("gpr", "random_forest", "constant"))
    assert pin["gpr"] < pin["constant"] and pin["random_forest"] < pin["constant"]
    assert sc.loc["gpr", "log_scale_rmse"] < sc.loc["constant", "log_scale_rmse"]


def test_negative_control_no_gain_without_signal():
    fl = synthetic_fleet(n_plants=6, n_unobserved=0, n_steps=24 * 30, n_members=8, signal=0.0, clip=False, seed=4)
    pin, _ = _cv_pinball(fl, ("ridge", "random_forest", "constant"))
    # nothing to learn: no surrogate improves materially on the constant baseline
    assert pin["ridge"] > 0.98 * pin["constant"] and pin["random_forest"] > 0.98 * pin["constant"]


def test_run_workflow_end_to_end(fleet, tmp_path):
    cfg = WorkflowConfig(features=SPEC, models=("gpr", "ridge", "constant"), gpr_max_train=300)
    res = run_workflow(fleet.observed, fleet.unobserved, cfg)
    assert set(res.predictive_scores["model"]) == {"gpr", "ridge", "constant"}
    assert res.chosen_model in ("gpr", "ridge", "constant")
    pred = res.predictions
    qcols = [c for c in pred.columns if c.startswith("q")]
    assert len(qcols) == len(LEVELS) and pred["plant_id"].unique().tolist() == [fleet.unobserved[0].plant_id]
    assert np.all(np.diff(pred[qcols].to_numpy(), axis=1) >= -1e-12)   # quantiles do not cross
    oracle = res.oracle["pinball"].iloc[0]
    assert (res.predictive_scores["pinball"] >= oracle - 1e-12).all()   # per-step targets are a lower bound
    assert res.recalibrated_scores is not None
    files = save_result(res, tmp_path, cfg)
    assert (tmp_path / "summary.json").exists() and len(files) >= 8
    with pytest.raises(ValueError):
        run_workflow(fleet.observed[:1], [], cfg)
    with pytest.raises(ValueError):
        WorkflowConfig(features=SPEC, models=("gpr", "lasso"))
    with pytest.raises(ValueError):
        WorkflowConfig(features=SPEC, models=("ridge",), final_model="gpr")
