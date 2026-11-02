"""Trailing moving windows.

Convention used everywhere in this package: the value reported at index ``t`` is computed from the window
``x[t - w + 1], ..., x[t]`` (the current value and the ``w - 1`` values before it). The first ``w - 1`` entries
have no complete window and are ``NaN``. A window that contains a missing value gives ``NaN``.

Trailing windows never look ahead, so a metric at time ``t`` only uses information available at ``t``.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def as_1d(x, name: str = "x") -> np.ndarray:
    """Return ``x`` as a 1-D float array (a copy is not guaranteed)."""
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {arr.shape}")
    return arr


def check_window(window: int, n: int | None = None, minimum: int = 2) -> int:
    """Validate a window length (an integer number of time steps)."""
    if isinstance(window, (bool, np.bool_)) or not isinstance(window, (int, np.integer)):
        raise TypeError(f"window must be an integer number of time steps, got {window!r}")
    window = int(window)
    if window < minimum:
        raise ValueError(f"window must be at least {minimum}, got {window}")
    if n is not None and window > n:
        raise ValueError(f"window ({window}) is longer than the series ({n})")
    return window


def trailing_windows(x, window: int) -> np.ndarray:
    """All complete trailing windows of ``x`` as a read-only array of shape ``(len(x) - window + 1, window)``.

    Row ``k`` is the window that ends at index ``k + window - 1``.
    """
    arr = as_1d(x)
    window = check_window(window, len(arr), minimum=1)
    return sliding_window_view(arr, window)


def pad_front(values: np.ndarray, window: int) -> np.ndarray:
    """Prepend ``window - 1`` NaNs so that a per-window result lines up with the time index of the series."""
    values = np.asarray(values, dtype=float)
    pad = np.full((window - 1,) + values.shape[1:], np.nan)
    return np.concatenate([pad, values], axis=0)


def rolling(x, window: int, func) -> np.ndarray:
    """Apply ``func`` to every trailing window and align the result with the series.

    ``func`` receives a 2-D array (one window per row) and must return one value per row. Rows that contain a
    missing value are set to ``NaN`` whatever ``func`` returns.
    """
    arr = as_1d(x)
    window = check_window(window, len(arr), minimum=1)
    win = sliding_window_view(arr, window)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.asarray(func(win), dtype=float)
    if out.shape != (win.shape[0],):
        raise ValueError("func must return one value per window")
    out = out.copy()
    out[np.isnan(win).any(axis=1)] = np.nan
    return pad_front(out, window)


def rolling_pair(a, b, window: int, func) -> np.ndarray:
    """Like :func:`rolling` for two aligned series; ``func(win_a, win_b)`` returns one value per window."""
    a = as_1d(a, "a")
    b = as_1d(b, "b")
    if a.shape != b.shape:
        raise ValueError(f"series must have the same length, got {len(a)} and {len(b)}")
    window = check_window(window, len(a), minimum=1)
    wa = sliding_window_view(a, window)
    wb = sliding_window_view(b, window)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.asarray(func(wa, wb), dtype=float)
    if out.shape != (wa.shape[0],):
        raise ValueError("func must return one value per window")
    out = out.copy()
    out[np.isnan(wa).any(axis=1) | np.isnan(wb).any(axis=1)] = np.nan
    return pad_front(out, window)
