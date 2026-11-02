"""Ensemble and spatial metrics against hand-computed values and brute-force definitions."""

import numpy as np
import pytest
from scipy import integrate

from wind_uq_surrogate.metrics_ensemble import (crps_ensemble, ensemble_metrics, observation_rank,
                                                predictability_index, rank_histogram, rank_histogram_summary,
                                                spread_index, total_spread_index)
from wind_uq_surrogate.metrics_spatial import (empirical_variogram, fit_variogram, pairwise_distances,
                                               variogram_model, variogram_over_time)


def test_spread_indices_hand_computed():
    members = np.array([[1.0, 2.0, 3.0, 4.0, 5.0],
                        [2.0, 2.0, 2.0, 2.0, 2.0]])
    # numpy linear percentiles of 1..5: q25 = 2, q75 = 4; mean 3
    assert spread_index(members, 75, 25)[0] == pytest.approx(100 * 2 / 3)
    assert total_spread_index(members)[0] == pytest.approx(100 * 4 / 3)
    assert spread_index(members)[1] == 0.0
    neg = np.array([[-1.0, 0.0, 1.0]])
    assert np.isnan(spread_index(neg)[0])  # zero mean
    with pytest.raises(ValueError):
        spread_index(members, 25, 75)
    with pytest.raises(ValueError):
        spread_index(np.ones((3, 1)))


def test_predictability_index():
    si = np.array([10.0, 20.0, 30.0, 40.0, 10.0])
    pi = predictability_index(si, 2)
    assert np.isnan(pi[:2]).all()
    assert pi[2] == pytest.approx(15.0 - 30.0)
    assert pi[3] == pytest.approx(25.0 - 40.0)
    assert pi[4] == pytest.approx(35.0 - 10.0)   # less spread than usual: positive
    si2 = si.copy()
    si2[1] = np.nan
    pi2 = predictability_index(si2, 2)
    assert np.isnan(pi2[2]) and np.isnan(pi2[3]) and np.isfinite(pi2[4])
    with pytest.raises(ValueError):
        predictability_index(si, 0)


def _crps_numeric(members, y):
    xs = np.sort(members)
    lo, hi = min(xs[0], y) - 1.0, max(xs[-1], y) + 1.0
    f = lambda x: (np.mean(xs <= x) - (y <= x)) ** 2
    pts = np.unique(np.concatenate([xs, [y]]))
    return integrate.quad(f, lo, hi, points=pts, limit=200)[0]


def test_crps_ensemble_matches_integral_and_fair_version():
    rng = np.random.default_rng(4)
    members = rng.normal(0, 1, size=(5, 7))
    obs = rng.normal(0, 1, size=5)
    got = crps_ensemble(members, obs)
    for t in range(5):
        assert got[t] == pytest.approx(_crps_numeric(members[t], obs[t]), rel=1e-6, abs=1e-8)
    fair = crps_ensemble(members, obs, fair=True)
    m = members.shape[1]
    pair = np.abs(members[:, :, None] - members[:, None, :]).sum(axis=(1, 2))
    assert np.allclose(fair, np.abs(members - obs[:, None]).mean(axis=1) - pair / (2 * m * (m - 1)))
    one = crps_ensemble(np.array([[2.0]]), np.array([5.0]))
    assert one[0] == pytest.approx(3.0)  # one member: absolute error


def test_rank_histogram_flat_for_exchangeable_members():
    rng = np.random.default_rng(5)
    members = rng.standard_normal((6000, 9))
    obs = rng.standard_normal(6000)
    counts = rank_histogram(members, obs, rng=1)
    assert counts.sum() == 6000 and len(counts) == 10
    summary = rank_histogram_summary(counts)
    assert summary["chi2"] < 30          # chi-square with 9 dof; 30 is far in the tail
    under = rank_histogram(members * 0.3, obs, rng=1)   # under-dispersed ensemble: U shape
    s2 = rank_histogram_summary(under)
    assert s2["outer_share"] > 2 * s2["outer_share_expected"]


def test_observation_rank_ties_and_missing():
    members = np.array([[1.0, 1.0, 1.0], [0.0, 1.0, np.nan]])
    obs = np.array([1.0, 0.5])
    ranks = observation_rank(members, obs, rng=0)
    assert 0 <= ranks[0] <= 3
    assert ranks[1] == -1


def test_ensemble_table():
    rng = np.random.default_rng(6)
    members = rng.uniform(4, 8, size=(30, 6))
    tab = ensemble_metrics(members, obs=members.mean(axis=1), si_pairs=((75, 25), (90, 10)), history=5,
                           prefix="e_")
    assert list(tab.columns) == ["e_mean", "e_si_75_25", "e_si_90_10", "e_si_total", "e_pi_75_25", "e_crps"]
    assert np.allclose(tab["e_si_90_10"], spread_index(members, 90, 10))
    tab2 = ensemble_metrics(members, history=5, si_pairs=((75, 25),), pi_pair=(90, 10))
    assert "pi_90_10" in tab2.columns


def test_distances():
    xy = np.array([[0.0, 0.0], [3.0, 4.0]])
    assert pairwise_distances(xy)[0, 1] == pytest.approx(5.0)
    latlon = np.array([[0.0, 0.0], [0.0, 1.0]])
    # one degree of longitude on the equator
    assert pairwise_distances(latlon, "haversine")[0, 1] == pytest.approx(2 * np.pi * 6371.0088 / 360, rel=1e-9)
    with pytest.raises(ValueError):
        pairwise_distances(xy, "manhattan")


def test_empirical_variogram_hand_computed():
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    z = np.array([1.0, 3.0, 4.0])
    ev = empirical_variogram(z, coords, [0.0, 1.5, 2.5])
    # bin 1: pairs (0,1) d=1, (1,2) d=1 -> (4 + 1) / (2 * 2); bin 2: pair (0,2) d=2 -> 9 / 2
    assert ev["gamma"].tolist() == pytest.approx([1.25, 4.5])
    assert ev["n_pairs"].tolist() == [2, 1]
    assert ev["lag"].tolist() == pytest.approx([1.0, 2.0])
    ev2 = empirical_variogram(z, coords, [0.0, 1.5, 2.5], min_pairs=2)
    assert len(ev2) == 1
    flat = empirical_variogram(z, coords, [0.0, 1.5, 2.5], detrend="linear")
    assert flat["gamma"].iloc[1] < 4.5   # the linear trend carried most of the long-distance difference


def test_variogram_models():
    for model in ("exponential", "gaussian", "spherical"):
        g = variogram_model(np.array([0.0, 1e-9, 5.0, 50.0]), 0.2, 1.0, 5.0, model)
        assert g[0] == 0.0
        assert g[1] == pytest.approx(0.2, abs=1e-6)
        assert g[3] == pytest.approx(1.2, abs=1e-6)
    assert variogram_model(5.0, 0.0, 1.0, 5.0, "exponential") == pytest.approx(1 - np.exp(-3))
    assert variogram_model(5.0, 0.0, 1.0, 5.0, "spherical") == pytest.approx(1.0)


def test_fit_variogram_recovers_parameters_from_exact_curve():
    h = np.linspace(0.3, 8.0, 15)
    g = variogram_model(h, 0.1, 0.9, 4.0, "exponential")
    for weighting in ("cressie", "pairs", "none"):
        fit = fit_variogram(h, g, np.full(15, 20), model="exponential", weighting=weighting)
        assert fit["nugget"] == pytest.approx(0.1, abs=1e-4)
        assert fit["sill"] == pytest.approx(1.0, abs=1e-4)
        assert fit["range"] == pytest.approx(4.0, rel=1e-3)
    with pytest.raises(ValueError):
        fit_variogram(h[:2], g[:2])


def test_variogram_over_time_tracks_field_amplitude():
    rng = np.random.default_rng(7)
    coords = rng.uniform(0, 6, size=(40, 2))
    d = pairwise_distances(coords)
    chol = np.linalg.cholesky(np.exp(-3 * d / 2.5) + 1e-9 * np.eye(40))
    amps = np.array([0.3, 0.3, 1.5, 1.5])
    field = np.array([a * chol @ rng.standard_normal(40) for a in amps])
    out = variogram_over_time(field, coords, np.linspace(0, 6, 9), prefix="vg_")
    assert list(out.columns) == ["vg_nugget", "vg_sill", "vg_range", "vg_vg_rmse"]
    assert out["vg_sill"].iloc[2:].mean() > 5 * out["vg_sill"].iloc[:2].mean()
