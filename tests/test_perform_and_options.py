"""Tests of the v1.1 additions: absolute spread indices, per-unit features, per-surrogate caps and parameters, and
the PERFORM reader (on small files written here in the PERFORM layout; needs h5py)."""

import numpy as np
import pandas as pd
import pytest

import wind_uq_surrogate as w
from wind_uq_surrogate.metrics_ensemble import ensemble_metrics, spread_index, total_spread_index


# ---------------------------------------------------------------- absolute spread indices

def test_absolute_spread_is_the_width():
    rng = np.random.default_rng(1)
    m = rng.random((30, 11))
    q = np.percentile(m, [25, 75], axis=1)
    assert np.allclose(spread_index(m, 75, 25, relative=False), q[1] - q[0])
    assert np.allclose(total_spread_index(m, relative=False), m.max(axis=1) - m.min(axis=1))
    rel = spread_index(m, 75, 25)
    assert np.allclose(rel, 100 * (q[1] - q[0]) / m.mean(axis=1))


def test_absolute_spread_is_defined_at_zero_mean():
    m = np.zeros((5, 4))
    m[1, 0] = 0.2
    assert np.isnan(spread_index(m, 75, 25)).sum() == 4  # relative: undefined where the mean is zero
    absolute = spread_index(m, 75, 25, relative=False)
    assert np.isfinite(absolute).all() and np.all(absolute >= 0)
    tab = ensemble_metrics(m, history=2, relative=False)
    assert np.isfinite(tab["si_75_25"]).all() and np.isfinite(tab["si_total"]).all()
    assert np.isnan(tab["pi_75_25"].iloc[:2]).all() and np.isfinite(tab["pi_75_25"].iloc[2:]).all()


# ---------------------------------------------------------------- per-unit features

def _plant(capacity=50.0, n=60, seed=0):
    rng = np.random.default_rng(seed)
    t = pd.date_range("2020-01-01", periods=n, freq="h")
    f = capacity * rng.random(n)
    data = pd.DataFrame({"forecast": f, "fcst_2da": f + capacity * 0.1 * rng.standard_normal(n),
                         "actual": np.clip(f + capacity * 0.1 * rng.standard_normal(n), 0, capacity)}, index=t)
    for j in range(5):
        data[f"ens_{j:02d}"] = np.clip(f + capacity * 0.05 * rng.standard_normal(n), 0, capacity)
    return w.PlantSeries("p", data, capacity)


def test_per_unit_features_do_not_depend_on_capacity():
    spec = w.FeatureSpec(window=6, pairs=(("forecast", "fcst_2da"),), pair_metrics=("nmae", "nrmse"),
                         pair_normalizer=1.0, per_unit=("forecast", "fcst_2da"), member_prefix="ens_",
                         members_per_unit=True, si_relative=False, pi_history=4)
    small = _plant(capacity=50.0)
    big = w.PlantSeries("p", small.data * 4.0, 200.0)
    fa, fb = w.build_features(small, spec), w.build_features(big, spec)
    pd.testing.assert_frame_equal(fa, fb)
    # per-unit pair metric with normalizer 1: the mean absolute per-unit difference
    d = (small.data["forecast"] - small.data["fcst_2da"]).abs() / 50.0
    assert np.allclose(fa["forecast_vs_fcst_2da_nmae"].iloc[5:], d.rolling(6).mean().iloc[5:])


def test_default_pairs_and_members_unchanged():
    spec = w.FeatureSpec(window=6, pairs=(("forecast", "fcst_2da"),), pair_normalizer=1.0, member_prefix="ens_")
    p = _plant()
    f = w.build_features(p, spec)
    d = (p.data["forecast"] - p.data["fcst_2da"]).abs()  # in MW: the forecast is not per unit in pairs by default
    assert np.allclose(f["forecast_vs_fcst_2da_nmae"].iloc[5:], d.rolling(6).mean().iloc[5:])
    cols = [c for c in p.data if c.startswith("ens_")]
    assert np.allclose(f["ens_si_75_25"], spread_index(p.data[cols].to_numpy(), 75, 25), equal_nan=True)


def test_per_unit_rejects_actual_and_unknown_columns():
    with pytest.raises(ValueError):
        w.FeatureSpec(window=6, per_unit=("actual",))
    spec = w.FeatureSpec(window=6, per_unit=("missing",))
    with pytest.raises(ValueError):
        w.build_features(_plant(), spec)


# ---------------------------------------------------------------- per-surrogate caps and parameters

def test_workflow_caps_and_params():
    spec = w.FeatureSpec(window=6)
    cfg = w.WorkflowConfig(features=spec, models=("gpr", "random_forest", "constant"),
                           max_train={"random_forest": 50, "gpr": 40},
                           model_params={"random_forest": {"n_estimators": 7}})
    fac = cfg.factories()
    assert fac["gpr"]().max_train_samples == 40
    rf = fac["random_forest"]()
    assert rf.max_train_samples == 50 and rf.params == {"n_estimators": 7}
    assert fac["constant"]().max_train_samples is None
    assert w.WorkflowConfig(features=spec).factories()["gpr"]().max_train_samples == 600
    with pytest.raises(ValueError):
        w.WorkflowConfig(features=spec, models=("ridge",), max_train={"svr": 10})
    with pytest.raises(ValueError):
        w.WorkflowConfig(features=spec, max_train={"svr": 1})
    with pytest.raises(ValueError):
        w.WorkflowConfig(features=spec, model_params={"lasso": {}})


# ---------------------------------------------------------------- PERFORM reader

h5py = pytest.importorskip("h5py")
from wind_uq_surrogate import perform  # noqa: E402


def _times(index):
    return np.array([t.strftime("%Y-%m-%d %H:%M:%S+00:00").encode() for t in index])


def _write_actuals(path, values, start="2018-01-01", capacity=100.0, region="North"):
    t = pd.date_range(start, periods=len(values), freq="5min")
    text = h5py.string_dtype()  # variable-length strings, as in the PERFORM actuals files
    meta = np.array([(2018.0, b"site_a", capacity, region.encode(), b"Existing", 30.0, -100.0)],
                    dtype=[("year", "<f8"), ("site", text), ("Capacity", "<f8"), ("Region", text), ("Group", text),
                           ("latitude", "<f8"), ("longitude", "<f8")])
    with h5py.File(path, "w") as f:
        f["actuals"] = np.asarray(values, dtype=float)
        f["time_index"] = np.array([x.decode() for x in _times(t)], dtype=object).astype(text)
        f["meta"] = meta


def _write_forecasts(path, values, start="2018-01-01", name=b"Site A"):
    t = pd.date_range(start, periods=values.shape[0], freq="h")
    meta = np.array([(name, 100.0)], dtype=[("Facility.Name", "S16"), ("Capacity", "<f8")])
    with h5py.File(path, "w") as f:
        f["forecasts"] = values
        f["forecast_time"] = _times(t)
        f["issue_time"] = _times(t.floor("D") - pd.Timedelta(hours=6))
        f["meta"] = meta


def test_file_names_and_keys():
    a = perform.parse_site_file("Site_Aguayo_Wind_wind_actuals_2018.h5")
    b = perform.parse_site_file("Site_aguayo_wind_wind_day-ahead_fcst_2018.h5")
    c = perform.parse_site_file("Site_Aguayo_Wind_wind_2day-ahead_fcst_ecmwf_2018.h5")
    assert a == {"key": "aguayo_wind", "kind": "actuals", "ecmwf": False, "year": 2018}
    assert b["key"] == c["key"] == "aguayo_wind" and not b["ecmwf"] and c["ecmwf"] and c["kind"] == "2day-ahead_fcst"
    assert perform.parse_site_file("BA_wind_actuals_2018.h5") is None
    assert perform.site_key("Adams Wind Generations, LLC") == perform.site_key("adams_wind_generations_llc")
    assert perform.site_key("Logan's Gap Wind") == perform.site_key("logans_gap_wind") == "logans_gap_wind"
    with pytest.raises(ValueError):
        perform.site_key("--")


def test_hourly_conventions():
    t = pd.date_range("2018-01-01", periods=36, freq="5min")
    s = pd.Series(np.arange(36, dtype=float), index=t)
    beg = perform.hourly(s, "beginning")
    assert beg.loc["2018-01-01 00:00"] == np.mean(np.arange(12)) and beg.loc["2018-01-01 02:00"] == np.mean(
        np.arange(24, 36))
    end = perform.hourly(s, "ending")
    assert end.loc["2018-01-01 01:00"] == np.mean(np.arange(1, 13))
    assert np.isnan(end.loc["2018-01-01 00:00"])  # only one value in (23:00, 00:00]
    inst = perform.hourly(s, "instant")
    assert list(inst) == [0.0, 12.0, 24.0]
    with pytest.raises(ValueError):
        perform.hourly(s, "centre")


def test_alignment_errors_find_the_convention():
    rng = np.random.default_rng(3)
    t = pd.date_range("2018-01-01", periods=12 * 48, freq="5min")
    s = pd.Series(np.cumsum(rng.standard_normal(len(t))), index=t)
    target = perform.hourly(s, "ending").dropna()
    errs = perform.alignment_errors(s, target)
    assert errs["ending"] == pytest.approx(0.0, abs=1e-12)
    assert min(errs, key=errs.get) == "ending"


def test_plant_from_files_and_percentiles(tmp_path):
    hours = 48
    rng = np.random.default_rng(5)
    act = 100 * rng.random(12 * hours)
    _write_actuals(tmp_path / "a.h5", act)
    ens = 100 * rng.random((hours, 51))
    ens2 = 100 * rng.random((hours, 51))
    pct = np.sort(100 * rng.random((hours, 99)), axis=1)
    _write_forecasts(tmp_path / "e.h5", ens)
    _write_forecasts(tmp_path / "e2.h5", ens2)
    _write_forecasts(tmp_path / "p.h5", pct)
    plant, meta = perform.plant_from_files("w001", tmp_path / "a.h5", tmp_path / "e.h5", tmp_path / "e2.h5")
    assert meta["Region"] == "North" and plant.capacity == 100.0 and plant.observed
    assert list(plant.data.columns[:2]) == ["forecast", "fcst_2da"]
    assert sum(c.startswith("ens_") for c in plant.data.columns) == 51
    assert np.allclose(plant.data["forecast"], ens.mean(axis=1))
    assert np.allclose(plant.data["fcst_2da"], ens2.mean(axis=1))
    assert np.allclose(plant.data["actual"], act.reshape(hours, 12).mean(axis=1))
    unobs, _ = perform.plant_from_files("w001", tmp_path / "a.h5", tmp_path / "e.h5", with_actual=False)
    assert not unobs.observed and "fcst_2da" not in unobs.data
    q = perform.percentile_forecasts(tmp_path / "p.h5", [0.05, 0.5, 0.95])
    assert list(q.columns) == ["q_0", "q_1", "q_2"]
    assert np.allclose(q.to_numpy(), pct[:, [4, 49, 94]])
    with pytest.raises(ValueError):
        perform.percentile_forecasts(tmp_path / "p.h5", [0.055])
    with pytest.raises(ValueError):
        perform.percentile_forecasts(tmp_path / "e.h5", [0.5])  # an ensemble file, not percentiles
    table, issue, fmeta = perform.read_forecasts(tmp_path / "p.h5")
    assert table.shape == (hours, 99) and fmeta["Facility.Name"] == "Site A" and issue.index.equals(table.index)
