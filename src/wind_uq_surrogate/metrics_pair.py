"""Moving-window agreement between two series of the same quantity (data-source uncertainty).

Typical pairs: two meteorological data products at one location, a model series and a measurement, or two members
of an ensemble. The first argument is the reference. Definitions: ``docs/metrics.md``, Section 3.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .windows import check_window, rolling_pair


def _standardized_rows(a: np.ndarray):
    """Centre each window and divide by its largest absolute deviation; flag windows whose variation is at the
    level of floating-point rounding of their values (treated as constant)."""
    d = a - a.mean(axis=1, keepdims=True)
    dev = np.abs(d).max(axis=1, keepdims=True)
    floor = 64.0 * np.finfo(float).eps * np.abs(a).max(axis=1, keepdims=True)
    ok = (dev > floor)[:, 0]
    return np.divide(d, dev, out=np.zeros_like(d), where=dev > 0), ok


def _pearson_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    da, ok_a = _standardized_rows(a)
    db, ok_b = _standardized_rows(b)
    num = (da * db).sum(axis=1)
    den = np.sqrt((da**2).sum(axis=1) * (db**2).sum(axis=1))
    out = np.full(a.shape[0], np.nan)
    ok = ok_a & ok_b & (den > 0)
    out[ok] = num[ok] / den[ok]
    return np.clip(out, -1.0, 1.0)


def moving_correlation(reference, source, window: int) -> np.ndarray:
    """Pearson correlation in each trailing window (``NaN`` when either series is constant in the window)."""
    window = check_window(window, minimum=3)
    return rolling_pair(reference, source, window, _pearson_rows)


def moving_mape(reference, source, window: int, min_reference: float = 0.0) -> np.ndarray:
    """Mean absolute percentage error of ``source`` against ``reference`` in each window, in percent.

    Only time steps with ``|reference| > min_reference`` enter the mean (small reference values make the relative
    error explode); a window without such steps gives ``NaN``.
    """
    window = check_window(window, minimum=1)
    if min_reference < 0:
        raise ValueError("min_reference must be non-negative")

    def f(r, s):
        keep = np.abs(r) > min_reference
        rel = np.where(keep, np.abs(r - s) / np.where(keep, np.abs(r), 1.0), 0.0)
        count = keep.sum(axis=1)
        out = np.full(r.shape[0], np.nan)
        out[count > 0] = 100.0 * rel[count > 0].sum(axis=1) / count[count > 0]
        return out

    return rolling_pair(reference, source, window, f)


def _normalizer_rows(r: np.ndarray, normalizer) -> np.ndarray:
    if isinstance(normalizer, str):
        if normalizer == "range":
            den = r.max(axis=1) - r.min(axis=1)
        elif normalizer == "mean":
            den = r.mean(axis=1)
        else:
            raise ValueError("normalizer must be a positive number, 'range' or 'mean'")
        return np.where(den > 0, den, np.nan)
    value = float(normalizer)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("a numeric normalizer must be positive and finite")
    return np.full(r.shape[0], value)


def moving_nrmse(reference, source, window: int, normalizer) -> np.ndarray:
    """Root mean square difference in each window divided by ``normalizer`` (a fraction, not percent).

    ``normalizer`` is a positive number (for power: the installed capacity), ``"range"`` (max - min of the
    reference in the window) or ``"mean"`` (mean of the reference in the window). A zero range or a non-positive
    mean gives ``NaN``.
    """
    window = check_window(window, minimum=1)
    _normalizer_rows(np.ones((1, 1)), normalizer)  # validate early
    return rolling_pair(reference, source, window,
                        lambda r, s: np.sqrt(((r - s) ** 2).mean(axis=1)) / _normalizer_rows(r, normalizer))


def moving_nmae(reference, source, window: int, normalizer) -> np.ndarray:
    """Mean absolute difference in each window divided by ``normalizer`` (see :func:`moving_nrmse`)."""
    window = check_window(window, minimum=1)
    _normalizer_rows(np.ones((1, 1)), normalizer)
    return rolling_pair(reference, source, window,
                        lambda r, s: np.abs(r - s).mean(axis=1) / _normalizer_rows(r, normalizer))


PAIR_METRICS = ("corr", "mape", "nrmse", "nmae")


def agreement_metrics(reference, source, window: int, normalizer="range", min_reference: float = 0.0,
                      prefix: str = "", metrics=PAIR_METRICS, index=None) -> pd.DataFrame:
    """All moving-window agreement metrics of one pair as a table (columns ``<prefix><metric>``)."""
    unknown = set(metrics) - set(PAIR_METRICS)
    if unknown:
        raise ValueError(f"unknown agreement metrics: {sorted(unknown)}")
    funcs = {
        "corr": lambda: moving_correlation(reference, source, window),
        "mape": lambda: moving_mape(reference, source, window, min_reference=min_reference),
        "nrmse": lambda: moving_nrmse(reference, source, window, normalizer),
        "nmae": lambda: moving_nmae(reference, source, window, normalizer),
    }
    data = {f"{prefix}{m}": funcs[m]() for m in metrics}
    if index is None and isinstance(reference, pd.Series):
        index = reference.index
    return pd.DataFrame(data, index=index)
