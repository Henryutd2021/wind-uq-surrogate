"""Input directories, the command line, and the rolling-window targets."""

import json

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from wind_uq_surrogate.calibration import optimal_scale, rolling_optimal_scale
from wind_uq_surrogate.cli import main
from wind_uq_surrogate.io import read_plants, split_observed, write_plants
from wind_uq_surrogate.probabilistic import default_levels
from wind_uq_surrogate.synthetic import synthetic_fleet

LEVELS = default_levels(0.05)


def test_rolling_targets_match_window_fits():
    rng = np.random.default_rng(12)
    m = rng.uniform(0, 1, 40)
    y = m + rng.laplace(0, 0.1, 40)
    s, loss = rolling_optimal_scale(y, m, LEVELS, "laplace", window=7, bounds=(1e-4, 3.0))
    for t in (0, 3, 20, 39):
        sel = slice(max(0, t - 3), t + 4)
        ref, ref_loss = optimal_scale(y[sel], m[sel], LEVELS, "laplace", bounds=(1e-4, 3.0),
                                      groups=np.zeros(len(y[sel])))
        assert s[t] == pytest.approx(ref[0]) and loss[t] == pytest.approx(ref_loss[0])
    grid = np.linspace(1e-4, 3.0, 30001)
    q = m[17:24][None, :, None] + grid[:, None, None] * stats.laplace.ppf(LEVELS)[None, None, :]
    u = y[17:24][None, :, None] - q
    assert loss[20] <= np.maximum(LEVELS * u, (LEVELS - 1) * u).mean(axis=(1, 2)).min() + 1e-12
    one, _ = rolling_optimal_scale(y, m, LEVELS, "laplace", window=1, bounds=(1e-4, 3.0))
    assert np.allclose(one, optimal_scale(y, m, LEVELS, "laplace", bounds=(1e-4, 3.0))[0])
    with pytest.raises(ValueError):
        rolling_optimal_scale(y, m, LEVELS, window=8)


def test_round_trip(tmp_path):
    fleet = synthetic_fleet(n_plants=3, n_unobserved=1, n_steps=50, n_members=3, n_turbines=5, seed=5)
    write_plants(tmp_path, fleet.plants)
    back = read_plants(tmp_path)
    obs, unobs = split_observed(back)
    assert [p.plant_id for p in obs] == ["P01", "P02"] and [p.plant_id for p in unobs] == ["P03"]
    for a, b in zip(fleet.plants, back):
        pd.testing.assert_frame_equal(a.data, b.data, check_freq=False, check_names=False)
        assert a.capacity == b.capacity
        assert np.allclose(a.turbine_coords, b.turbine_coords)
        pd.testing.assert_frame_equal(a.turbine_field, b.turbine_field, check_freq=False, check_names=False)


def test_read_checks(tmp_path):
    (tmp_path / "series").mkdir()
    pd.DataFrame({"plant_id": ["A"], "capacity_mw": [50.0]}).to_csv(tmp_path / "plants.csv", index=False)
    with pytest.raises(FileNotFoundError):
        read_plants(tmp_path)
    pd.DataFrame({"t": [0, 1], "forecast": [1.0, 2.0]}).to_csv(tmp_path / "series" / "A.csv", index=False)
    with pytest.raises(ValueError):
        read_plants(tmp_path)
    pd.DataFrame({"time": [0, 1], "forecast": [1.0, 2.0], "actual": [np.nan, np.nan]}).to_csv(
        tmp_path / "series" / "A.csv", index=False)
    (p,) = read_plants(tmp_path)
    assert not p.observed and "actual" not in p.data.columns
    pd.DataFrame({"time": [1, 0], "forecast": [1.0, 2.0]}).to_csv(tmp_path / "series" / "A.csv", index=False)
    with pytest.raises(ValueError):
        read_plants(tmp_path)


def test_cli_synthetic_metrics_run(tmp_path, capsys):
    inp, out = tmp_path / "in", tmp_path / "out"
    assert main(["synthetic", str(inp), "--plants", "4", "--unobserved", "1", "--steps", "300", "--members", "4"]) == 0
    assert (inp / "plants.csv").exists() and (inp / "synthetic_truth.csv").exists()
    assert main(["metrics", str(inp), "--out-dir", str(out / "m"), "--window", "10", "--pair", "ws_a:ws_b",
                 "--members", "ens_", "--pi-history", "30"]) == 0
    feats = pd.read_csv(out / "m" / "features_P01.csv")
    assert "ws_a_vs_ws_b_nrmse" in feats.columns and "ens_pi_75_25" in feats.columns
    assert main(["run", str(inp), "--out-dir", str(out / "r"), "--window", "10", "--pair", "ws_a:ws_b",
                 "--members", "ens_", "--models", "ridge,constant", "--level-step", "0.1"]) == 0
    summary = json.loads((out / "r" / "summary.json").read_text())
    assert summary["unobserved_plants"] == ["P04"] and summary["chosen_model"] in ("ridge", "constant")
    pred = pd.read_csv(out / "r" / "unobserved_predictions.csv")
    assert "q50_mw" in pred.columns and "q10_mw" in pred.columns
    assert "chosen surrogate" in capsys.readouterr().out
    plants = pd.read_csv(inp / "plants.csv")
    plants["group"] = ["north", "north", "south", "south"]   # P04 (unobserved) is not used in validation
    plants.to_csv(inp / "plants.csv", index=False)
    assert main(["run", str(inp), "--out-dir", str(out / "g"), "--window", "10", "--models", "constant"]) == 0
    notes = json.loads((out / "g" / "summary.json").read_text())["notes"]
    assert any("at least two plants or groups" in n for n in notes)   # one group left in training: not validated
    with pytest.raises(SystemExit):
        main(["run", str(inp), "--out-dir", str(out)])   # --window is required
