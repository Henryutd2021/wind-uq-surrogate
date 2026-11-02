"""Metrics of an ensemble: several members (or several data sources) for the same quantity at each time step.

``members`` is a 2-D array of shape ``(n_times, n_members)``. Definitions and references: ``docs/metrics.md``,
Section 4 (spread and predictability indices after Fernandez-Gonzalez et al., 2017; ensemble CRPS after Gneiting
and Raftery, 2007; rank histogram after Hamill, 2001).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def as_members(members, min_members: int = 2) -> np.ndarray:
    arr = np.asarray(members, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"members must be a 2-D array (n_times, n_members), got shape {arr.shape}")
    if arr.shape[1] < min_members:
        raise ValueError(f"at least {min_members} members are needed, got {arr.shape[1]}")
    return arr


def _check_percentile(p: float, name: str) -> float:
    p = float(p)
    if not 0.0 <= p <= 100.0:
        raise ValueError(f"{name} must be a percentile between 0 and 100, got {p}")
    return p


def _relative_to_mean(width: np.ndarray, mean: np.ndarray) -> np.ndarray:
    out = np.full(width.shape, np.nan)
    ok = mean > 0
    out[ok] = 100.0 * width[ok] / mean[ok]
    return out


def spread_index(members, upper: float = 75.0, lower: float = 25.0, method: str = "linear") -> np.ndarray:
    """Spread index SI(upper-lower): (q_upper - q_lower) / ensemble mean x 100, per time step.

    ``upper`` and ``lower`` are percentiles of the members at each time step (``numpy.percentile`` with the given
    ``method``). A non-positive ensemble mean gives ``NaN``. Rows with a missing member give ``NaN``.
    """
    arr = as_members(members)
    upper = _check_percentile(upper, "upper")
    lower = _check_percentile(lower, "lower")
    if upper <= lower:
        raise ValueError("upper must be larger than lower")
    q = np.percentile(arr, [lower, upper], axis=1, method=method)
    return _relative_to_mean(q[1] - q[0], arr.mean(axis=1))


def total_spread_index(members) -> np.ndarray:
    """Total spread index: (maximum member - minimum member) / ensemble mean x 100, per time step."""
    arr = as_members(members)
    return _relative_to_mean(arr.max(axis=1) - arr.min(axis=1), arr.mean(axis=1))


def predictability_index(si, history: int) -> np.ndarray:
    """Predictability index: mean of the spread index over the previous ``history`` steps minus its current value.

    ``PI_t = mean(SI_{t-history}, ..., SI_{t-1}) - SI_t``. Positive values: less spread than usual (a more
    predictable situation); negative values: more spread than usual. The first ``history`` steps, and steps whose
    history contains a missing value, give ``NaN``.
    """
    si = np.asarray(si, dtype=float)
    if si.ndim != 1:
        raise ValueError("si must be one-dimensional")
    if isinstance(history, (bool, np.bool_)) or not isinstance(history, (int, np.integer)) or history < 1:
        raise ValueError("history must be a positive integer number of steps")
    history = int(history)
    out = np.full(si.shape, np.nan)
    if len(si) <= history:
        return out
    csum = np.concatenate([[0.0], np.cumsum(np.nan_to_num(si))])
    nan_count = np.concatenate([[0], np.cumsum(np.isnan(si))])
    t = np.arange(history, len(si))
    past_mean = (csum[t] - csum[t - history]) / history
    past_nan = nan_count[t] - nan_count[t - history]
    out[t] = np.where(past_nan == 0, past_mean - si[t], np.nan)
    return out


def crps_ensemble(members, obs, fair: bool = False) -> np.ndarray:
    """Continuous ranked probability score of the ensemble's empirical distribution, per time step.

    ``CRPS = mean_i |x_i - y| - 1/(2 M^2) sum_i sum_j |x_i - x_j|`` (Gneiting and Raftery, 2007, Eq. 21 with the
    empirical distribution of the ``M`` members). With ``fair=True`` the second term uses ``1/(2 M (M - 1))``
    (Ferro, 2014), which removes the dependence on the ensemble size for exchangeable members.
    Same units as the data; lower is better.
    """
    arr = as_members(members, min_members=1 if not fair else 2)
    y = np.asarray(obs, dtype=float)
    if y.shape != (arr.shape[0],):
        raise ValueError("obs must have one value per time step")
    m = arr.shape[1]
    term1 = np.abs(arr - y[:, None]).mean(axis=1)
    xs = np.sort(arr, axis=1)
    weights = 2.0 * np.arange(1, m + 1) - m - 1  # sum_{i,j}|x_i - x_j| = 2 sum_i (2i - M - 1) x_(i)
    pair_sum = 2.0 * (xs * weights).sum(axis=1)
    denom = 2.0 * m * (m - 1) if fair else 2.0 * m * m
    return term1 - pair_sum / denom


def observation_rank(members, obs, rng=None) -> np.ndarray:
    """Rank of the observation among the members, 0 ... M, with ties between members and observation broken at
    random (Hamill, 2001). Rows with a missing value give -1."""
    arr = as_members(members, min_members=1)
    y = np.asarray(obs, dtype=float)
    if y.shape != (arr.shape[0],):
        raise ValueError("obs must have one value per time step")
    rng = np.random.default_rng(rng)
    below = (arr < y[:, None]).sum(axis=1)
    ties = (arr == y[:, None]).sum(axis=1)
    extra = np.array([rng.integers(0, t + 1) if t > 0 else 0 for t in ties])
    rank = below + extra
    bad = np.isnan(arr).any(axis=1) | np.isnan(y)
    rank[bad] = -1
    return rank.astype(int)


def rank_histogram(members, obs, rng=None) -> np.ndarray:
    """Counts of the observation rank (length ``M + 1``); flat for a statistically consistent ensemble."""
    arr = as_members(members, min_members=1)
    ranks = observation_rank(arr, obs, rng)
    ranks = ranks[ranks >= 0]
    return np.bincount(ranks, minlength=arr.shape[1] + 1)


def rank_histogram_summary(counts) -> dict:
    """Summary of a rank histogram: chi-square statistic against a flat histogram (degrees of freedom ``M``) and
    the share of observations in the two outer bins next to its expected value ``2 / (M + 1)``."""
    counts = np.asarray(counts, dtype=float)
    if counts.ndim != 1 or len(counts) < 2:
        raise ValueError("counts must be a 1-D array with at least two bins")
    n = counts.sum()
    if n <= 0:
        raise ValueError("the histogram is empty")
    expected = n / len(counts)
    chi2 = float(((counts - expected) ** 2 / expected).sum())
    return {"n": int(n), "bins": len(counts), "chi2": chi2, "dof": len(counts) - 1,
            "outer_share": float((counts[0] + counts[-1]) / n), "outer_share_expected": 2.0 / len(counts)}


def ensemble_metrics(members, obs=None, si_pairs=((75.0, 25.0),), total_spread: bool = True,
                     history: int | None = None, pi_pair=None, fair_crps: bool = False, prefix: str = "",
                     index=None) -> pd.DataFrame:
    """Ensemble metrics per time step as a table.

    Columns: ``<prefix>mean``, ``<prefix>si_<u>_<l>`` for each pair in ``si_pairs``, ``<prefix>si_total``; with
    ``history``, ``<prefix>pi_<u>_<l>`` for ``pi_pair`` (default: the first pair); with ``obs``, ``<prefix>crps``.
    Only ``mean``, ``si_*`` and ``pi_*`` can be computed where no observation exists.
    """
    arr = as_members(members)
    data = {f"{prefix}mean": arr.mean(axis=1)}
    names = {}
    for upper, lower in si_pairs:
        name = f"{prefix}si_{_fmt(upper)}_{_fmt(lower)}"
        data[name] = spread_index(arr, upper, lower)
        names[(float(upper), float(lower))] = name
    if total_spread:
        data[f"{prefix}si_total"] = total_spread_index(arr)
    if history is not None:
        if not si_pairs and pi_pair is None:
            raise ValueError("the predictability index needs a spread-index pair")
        pair = tuple(float(v) for v in (pi_pair if pi_pair is not None else si_pairs[0]))
        si = data[names[pair]] if pair in names else spread_index(arr, *pair)
        data[f"{prefix}pi_{_fmt(pair[0])}_{_fmt(pair[1])}"] = predictability_index(si, history)
    if obs is not None:
        data[f"{prefix}crps"] = crps_ensemble(arr, obs, fair=fair_crps)
    return pd.DataFrame(data, index=index)


def _fmt(p: float) -> str:
    p = float(p)
    return str(int(p)) if p.is_integer() else str(p).replace(".", "p")
