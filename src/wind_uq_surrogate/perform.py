"""Reader for the ARPA-E PERFORM wind data (public; used by ``case_studies/perform_ercot``).

The PERFORM data sets (Sergi et al., 2022, https://doi.org/10.25984/1891136; licence CC BY 3.0 US) were produced
by the National Renewable Energy Laboratory for ARPA-E. For each wind plant of four U.S. system-operator regions they
hold, as HDF5 files on a public AWS S3 bucket:

- ``actuals``: simulated plant power in MW at 5-minute resolution (meteorology of the WIND Toolkit converted to
  power with plant configurations), with plant metadata;
- day-ahead power forecasts of the 51 members of the ECMWF ensemble (hourly, one issue per day);
- probabilistic day-ahead forecasts as the 1st to 99th percentiles (hourly), trained on the plant's own history.

This module reads these files into pandas objects and builds :class:`~wind_uq_surrogate.features.PlantSeries` in
the input format of ``docs/data_pipeline.md``. It needs ``h5py`` (``pip install "wind-uq-surrogate[perform]"``).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .features import PlantSeries

BUCKET_URL = "https://arpa-e-perform.s3.amazonaws.com"
DATASET_DOI = "https://doi.org/10.25984/1891136"
LABELS = ("beginning", "ending", "instant")

_SITE_FILE = re.compile(r"^Site_(?P<site>.+?)_wind_(?P<kind>actuals|day-ahead_fcst|2day-ahead_fcst)"
                        r"(?P<ecmwf>_ecmwf)?_(?P<year>\d{4})\.h5$")


def _h5py():
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError('reading PERFORM files needs h5py: pip install "wind-uq-surrogate[perform]"') from exc
    return h5py


def site_key(name: str) -> str:
    """Normalized site name used to match the files of one plant across folders (case, spaces and punctuation
    differ between the actuals, the ensemble forecasts and the probabilistic forecasts)."""
    text = re.sub(r"['\u2019]", "", str(name).lower())  # "Logan's Gap" and "Logans Gap" are the same site
    key = re.sub(r"[^0-9a-z]+", "_", text).strip("_")
    if not key:
        raise ValueError(f"empty site name: {name!r}")
    return key


def parse_site_file(filename: str) -> dict | None:
    """Parse a site-level PERFORM file name, e.g. ``Site_Aguayo_Wind_wind_actuals_2018.h5``.

    Returns ``{"key", "kind", "ecmwf", "year"}`` (``kind``: ``actuals``, ``day-ahead_fcst`` or
    ``2day-ahead_fcst``; ``ecmwf``: True for the 51-member files) or ``None`` for other files."""
    m = _SITE_FILE.match(Path(filename).name)
    if m is None:
        return None
    return {"key": site_key(m["site"]), "kind": m["kind"], "ecmwf": m["ecmwf"] is not None, "year": int(m["year"])}


def _text(values) -> np.ndarray:
    arr = np.asarray(values)
    if arr.dtype.kind in "SO":
        return np.array([v.decode() if isinstance(v, bytes) else str(v) for v in arr.ravel()]).reshape(arr.shape)
    return arr


def _times(values) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(pd.to_datetime(_text(values), utc=True))
    return idx.tz_convert("UTC").tz_localize(None)  # naive UTC, as in the input-directory format


def _meta(dataset) -> dict:
    rec = dataset[0]
    out = {}
    for name in rec.dtype.names:
        v = rec[name]
        out[name] = v.decode() if isinstance(v, bytes) else (v.item() if hasattr(v, "item") else v)
    return out


def read_actuals(path) -> tuple[pd.Series, dict]:
    """Simulated plant power (MW) with a naive-UTC time index, and the plant metadata of the file."""
    h5py = _h5py()
    with h5py.File(path, "r") as f:
        values = np.asarray(f["actuals"][...], dtype=float)
        index = _times(f["time_index"][...])
        meta = _meta(f["meta"])
    if values.ndim != 1:
        raise ValueError(f"{path}: expected one site per file, got actuals of shape {values.shape}")
    return pd.Series(values, index=index, name="actual"), meta


def read_forecasts(path) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Forecast table of one file (index = forecast time, naive UTC), its issue times and the metadata.

    Columns: ``m00`` ... ``m50`` for ensemble files (51 members), ``p01`` ... ``p99`` for percentile files."""
    h5py = _h5py()
    with h5py.File(path, "r") as f:
        values = np.asarray(f["forecasts"][...], dtype=float)
        index = _times(f["forecast_time"][...])
        issue = _times(f["issue_time"][...])
        meta = _meta(f["meta"])
    if values.ndim != 2:
        raise ValueError(f"{path}: forecasts must be 2-D (time steps, members or percentiles)")
    k = values.shape[1]
    cols = [f"p{j:02d}" for j in range(1, 100)] if k == 99 else [f"m{j:02d}" for j in range(k)]
    table = pd.DataFrame(values, index=pd.DatetimeIndex(index, name="time"), columns=cols)
    if not table.index.is_unique or not table.index.is_monotonic_increasing:
        raise ValueError(f"{path}: forecast times must be unique and increasing")
    return table, pd.Series(issue, index=table.index, name="issue_time"), meta


def hourly(actuals: pd.Series, label: str = "beginning") -> pd.Series:
    """Hourly values of a 5-minute series.

    ``"beginning"``: mean of the twelve values from ``h:00`` to ``h:55``, labelled ``h``;
    ``"ending"``: mean of the twelve values from ``(h-1):05`` to ``h:00``, labelled ``h``;
    ``"instant"``: the value at ``h:00``. Hours with fewer than twelve finite values are ``NaN``."""
    if label not in LABELS:
        raise ValueError(f"label must be one of {LABELS}")
    s = actuals.astype(float)
    if label == "instant":
        out = s[s.index.minute == 0]
        return out.rename("actual")
    closed = "left" if label == "beginning" else "right"
    grouped = s.resample("h", closed=closed, label=closed)
    mean, count = grouped.mean(), grouped.count()
    return mean.where(count == 12).rename("actual")


def alignment_errors(actuals: pd.Series, forecast: pd.Series) -> dict:
    """Mean absolute difference between an hourly forecast and the hourly actuals under each time label
    convention (on the common hours). The smallest value indicates the convention the forecast was made for."""
    out = {}
    for label in LABELS:
        a = hourly(actuals, label)
        both = pd.concat([a, forecast.rename("forecast")], axis=1, join="inner").dropna()
        out[label] = float((both["actual"] - both["forecast"]).abs().mean()) if len(both) else float("nan")
    return out


def plant_from_files(plant_id: str, actuals_path, ensemble_path, ensemble2_path=None, label: str = "beginning",
                     with_actual: bool = True) -> tuple[PlantSeries, dict]:
    """A :class:`PlantSeries` from the PERFORM files of one plant, and the plant metadata.

    Columns: ``forecast`` = mean of the 51 day-ahead ensemble members (MW); ``ens_00`` ... ``ens_50`` = the
    members (MW); ``fcst_2da`` = mean of the 2-day-ahead members for the same hour (MW), if ``ensemble2_path``;
    ``actual`` = hourly simulated power (MW, convention ``label``), if ``with_actual``. The time index is the
    hourly forecast time (naive UTC); hours without a complete set of values are dropped.
    """
    act5, meta = read_actuals(actuals_path)
    ens, _, _ = read_forecasts(ensemble_path)
    if not all(c.startswith("m") for c in ens.columns):
        raise ValueError(f"{ensemble_path}: not an ensemble-member file")
    data = pd.DataFrame(index=ens.index)
    data["forecast"] = ens.mean(axis=1)
    if ensemble2_path is not None:
        ens2, _, _ = read_forecasts(ensemble2_path)
        data["fcst_2da"] = ens2.mean(axis=1).reindex(data.index)
    for c in ens.columns:
        data[f"ens_{c[1:]}"] = ens[c]
    if with_actual:
        data["actual"] = hourly(act5, label).reindex(data.index)
    data = data.dropna()
    capacity = float(meta.get("Capacity", np.nan))
    plant = PlantSeries(str(plant_id), data, capacity)
    return plant, meta


def percentile_forecasts(path, levels) -> pd.DataFrame:
    """Percentile forecasts (MW) of one file at the quantile ``levels`` (each a multiple of 0.01 in (0, 1));
    columns ``q_0`` ... ``q_<k-1>`` in the order of ``levels``, index = forecast time (naive UTC)."""
    table, _, _ = read_forecasts(path)
    if not all(c.startswith("p") for c in table.columns):
        raise ValueError(f"{path}: not a percentile file")
    tau = np.asarray(levels, dtype=float)
    pct = np.rint(100 * tau).astype(int)
    if np.any(np.abs(100 * tau - pct) > 1e-9) or pct.min() < 1 or pct.max() > 99:
        raise ValueError("levels must be multiples of 0.01 between 0.01 and 0.99")
    out = table[[f"p{p:02d}" for p in pct]].copy()
    out.columns = [f"q_{j}" for j in range(len(tau))]
    return out
