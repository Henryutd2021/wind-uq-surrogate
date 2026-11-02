"""Input directories (layout and checks in ``docs/data_pipeline.md``, Section 3).

::

    plants.csv                        plant_id, capacity_mw[, x_km, y_km]
    series/<plant_id>.csv             time, forecast[, actual], other covariate columns
    turbines/<plant_id>.csv           turbine_id, x, y                       (optional, spatial metrics)
    turbine_series/<plant_id>.csv     time, one column per turbine_id        (optional, spatial metrics)

A plant is observed when its series has an ``actual`` column with at least one value.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .features import PlantSeries


def _parse_time(col: pd.Series) -> pd.Index:
    if pd.api.types.is_numeric_dtype(col):
        return pd.Index(col.to_numpy(), name="time")
    return pd.DatetimeIndex(pd.to_datetime(col, format="ISO8601"), name="time")


def read_plants(directory) -> list[PlantSeries]:
    """Read and check an input directory; returns one :class:`PlantSeries` per row of ``plants.csv``."""
    root = Path(directory)
    table = pd.read_csv(root / "plants.csv", dtype={"plant_id": str})
    for col in ("plant_id", "capacity_mw"):
        if col not in table.columns:
            raise ValueError(f"plants.csv needs a {col!r} column")
    if table["plant_id"].duplicated().any():
        raise ValueError("plants.csv: plant_id values must be unique")
    plants = []
    for row in table.itertuples(index=False):
        pid = str(row.plant_id)
        path = root / "series" / f"{pid}.csv"
        if not path.exists():
            raise FileNotFoundError(f"series file missing for plant {pid}: {path}")
        df = pd.read_csv(path)
        if "time" not in df.columns:
            raise ValueError(f"{path}: needs a 'time' column")
        df.index = _parse_time(df.pop("time"))
        if "actual" in df.columns and df["actual"].isna().all():
            df = df.drop(columns="actual")
        coords = field = None
        tpath, spath = root / "turbines" / f"{pid}.csv", root / "turbine_series" / f"{pid}.csv"
        if tpath.exists() != spath.exists():
            raise ValueError(f"plant {pid}: give both turbines/{pid}.csv and turbine_series/{pid}.csv, or neither")
        if tpath.exists():
            tur = pd.read_csv(tpath, dtype={"turbine_id": str})
            field = pd.read_csv(spath)
            field.index = _parse_time(field.pop("time"))
            field.columns = [str(c) for c in field.columns]
            missing = set(tur["turbine_id"]) - set(field.columns)
            if missing:
                raise ValueError(f"plant {pid}: turbine_series lacks turbines {sorted(missing)}")
            field = field[list(tur["turbine_id"])]
            coords = tur[["x", "y"]].to_numpy(dtype=float)
        get = row._asdict()
        plants.append(PlantSeries(pid, df, float(row.capacity_mw), x=float(get.get("x_km", np.nan)),
                                  y=float(get.get("y_km", np.nan)), turbine_coords=coords, turbine_field=field))
    return plants


def write_plants(directory, plants) -> Path:
    """Write plants in the input-directory layout (the inverse of :func:`read_plants`)."""
    root = Path(directory)
    (root / "series").mkdir(parents=True, exist_ok=True)
    rows = []
    for p in plants:
        rows.append({"plant_id": p.plant_id, "capacity_mw": p.capacity, "x_km": p.x, "y_km": p.y})
        df = p.data.copy()
        df.insert(0, "time", df.index)
        df.to_csv(root / "series" / f"{p.plant_id}.csv", index=False)
        if p.turbine_field is not None:
            (root / "turbines").mkdir(exist_ok=True)
            (root / "turbine_series").mkdir(exist_ok=True)
            pd.DataFrame({"turbine_id": p.turbine_field.columns, "x": p.turbine_coords[:, 0],
                          "y": p.turbine_coords[:, 1]}).to_csv(root / "turbines" / f"{p.plant_id}.csv", index=False)
            tf = p.turbine_field.copy()
            tf.insert(0, "time", tf.index)
            tf.to_csv(root / "turbine_series" / f"{p.plant_id}.csv", index=False)
    pd.DataFrame(rows).to_csv(root / "plants.csv", index=False)
    return root


def split_observed(plants) -> tuple[list, list]:
    """Observed plants (with an ``actual`` column) and unobserved plants."""
    plants = list(plants)
    return [p for p in plants if p.observed], [p for p in plants if not p.observed]
