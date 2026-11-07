"""Step B of the workflow: plant data containers and the surrogate's input features.

Features are UQ metrics of series that exist at every plant, observed or not (point forecasts or modelled power,
meteorological data products, ensemble members, turbine-level fields). The observed power never enters a feature:
it does not exist at an unobserved plant. All metrics use trailing windows, so a feature at time ``t`` uses data up
to ``t`` only. Details: ``docs/surrogate_workflow.md``, Section 3.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .calibration import block_labels, optimal_scale, rolling_optimal_scale
from .metrics_ensemble import ensemble_metrics
from .metrics_pair import PAIR_METRICS, agreement_metrics
from .metrics_single import SINGLE_METRICS, single_series_metrics
from .metrics_spatial import variogram_over_time

RESERVED = ("actual",)


@dataclass
class PlantSeries:
    """Time series of one plant.

    ``data``: index = time (unique, increasing); column ``forecast`` (point forecast or modelled power, MW) is
    required; ``actual`` (measured power, MW) is present only at observed plants; any other columns are covariates
    (for example wind speed from several data products, or ensemble members sharing a column prefix).
    ``capacity``: installed capacity in MW. ``x``, ``y``: plant location (informational).
    ``turbine_coords`` (n_turbines, 2) and ``turbine_field`` (index = time, one column per turbine): optional
    turbine-level values for the spatial metrics.
    """

    plant_id: str
    data: pd.DataFrame
    capacity: float
    x: float = float("nan")
    y: float = float("nan")
    turbine_coords: np.ndarray | None = None
    turbine_field: pd.DataFrame | None = None

    def __post_init__(self):
        self.plant_id = str(self.plant_id)
        if not isinstance(self.data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        if "forecast" not in self.data.columns:
            raise ValueError(f"plant {self.plant_id}: data needs a 'forecast' column")
        if not self.data.index.is_unique or not self.data.index.is_monotonic_increasing:
            raise ValueError(f"plant {self.plant_id}: the time index must be unique and increasing")
        if not np.isfinite(self.data["forecast"].to_numpy(dtype=float)).all():
            raise ValueError(f"plant {self.plant_id}: 'forecast' must be finite")
        if not (np.isfinite(self.capacity) and self.capacity > 0):
            raise ValueError(f"plant {self.plant_id}: capacity must be positive")
        if (self.turbine_coords is None) != (self.turbine_field is None):
            raise ValueError(f"plant {self.plant_id}: give both turbine_coords and turbine_field, or neither")
        if self.turbine_field is not None:
            coords = np.asarray(self.turbine_coords, dtype=float)
            if coords.shape != (self.turbine_field.shape[1], 2):
                raise ValueError(f"plant {self.plant_id}: turbine_coords must have one row per turbine column")
            if not self.turbine_field.index.equals(self.data.index):
                raise ValueError(f"plant {self.plant_id}: turbine_field must share the time index of data")
            self.turbine_coords = coords

    @property
    def observed(self) -> bool:
        return "actual" in self.data.columns and self.data["actual"].notna().any()


@dataclass
class SpatialSpec:
    """Settings of the turbine-level semivariogram features (see :func:`variogram_over_time`)."""

    bin_edges: tuple
    model: str = "exponential"
    kind: str = "euclidean"
    detrend: str = "none"
    min_pairs: int = 1
    max_range: float | None = None


@dataclass
class FeatureSpec:
    """Which UQ metrics become surrogate inputs.

    ``window``: trailing window length (time steps) of the moving-window metrics; required.
    ``single_series``: columns that get the single-series metrics (``forecast`` is used per unit of capacity).
    ``pairs``: ``(reference, source)`` column pairs that get the agreement metrics.
    ``member_prefix``: columns starting with this prefix form an ensemble (spread and predictability indices).
    ``pi_history``: history (steps) of the predictability index; ``None`` leaves it out.
    ``spatial``: :class:`SpatialSpec` for plants with a turbine field; ``None`` leaves the spatial metrics out.
    ``include_level``: add the per-unit forecast itself as a feature.
    ``per_unit``: covariate columns (power in MW, e.g. a second forecast) divided by the installed capacity before
    their metrics are computed, in single-series metrics and in pairs; ``"forecast"`` listed here is also per unit
    in pairs (in single-series metrics it always is).
    ``members_per_unit``: divide the ensemble members by the installed capacity (power ensembles in MW).
    ``si_relative``: spread indices relative to the ensemble mean (in %, the definition for wind-speed ensembles);
    ``False`` gives widths in the members' unit, for power ensembles whose mean is often zero.
    """

    window: int
    single_series: tuple = ("forecast",)
    single_metrics: tuple = SINGLE_METRICS
    pairs: tuple = ()
    pair_metrics: tuple = PAIR_METRICS
    pair_normalizer: object = "range"
    pair_min_reference: float = 0.0
    member_prefix: str | None = None
    si_pairs: tuple = ((75.0, 25.0),)
    pi_history: int | None = None
    spatial: SpatialSpec | None = None
    include_level: bool = True
    min_mean: float = 0.0
    per_unit: tuple = ()
    members_per_unit: bool = False
    si_relative: bool = True

    def __post_init__(self):
        used = set(self.single_series) | {c for pair in self.pairs for c in pair} | set(self.per_unit)
        if self.member_prefix is not None and "actual".startswith(self.member_prefix):
            raise ValueError("member_prefix must not select the 'actual' column")
        bad = used & set(RESERVED)
        if bad:
            raise ValueError("features may not use the observed power ('actual'): it does not exist at unobserved "
                             "plants")


def per_unit(plant: PlantSeries, column: str) -> pd.Series:
    return plant.data[column].astype(float) / plant.capacity


def member_columns(plant: PlantSeries, prefix: str) -> list[str]:
    cols = [c for c in plant.data.columns if str(c).startswith(prefix) and c not in RESERVED]
    if len(cols) < 2:
        raise ValueError(f"plant {plant.plant_id}: fewer than two columns start with {prefix!r}")
    return cols


def build_features(plant: PlantSeries, spec: FeatureSpec) -> pd.DataFrame:
    """Feature table of one plant (index = time). Rows inside the warm-up of a window are ``NaN``."""
    df = plant.data
    for col in spec.per_unit:
        if col not in df.columns:
            raise ValueError(f"plant {plant.plant_id}: column {col!r} not found")

    def values(col: str, in_pair: bool = False) -> np.ndarray:
        if (col == "forecast" and not in_pair) or col in spec.per_unit:
            return per_unit(plant, col).to_numpy()
        return df[col].to_numpy(dtype=float)

    parts = []
    for col in spec.single_series:
        if col not in df.columns:
            raise ValueError(f"plant {plant.plant_id}: column {col!r} not found")
        parts.append(single_series_metrics(values(col), spec.window, prefix=f"{col}_",
                                           metrics=spec.single_metrics, min_mean=spec.min_mean, index=df.index))
    for ref, src in spec.pairs:
        for c in (ref, src):
            if c not in df.columns:
                raise ValueError(f"plant {plant.plant_id}: column {c!r} not found")
        parts.append(agreement_metrics(values(ref, True), values(src, True), spec.window,
                                       normalizer=spec.pair_normalizer, min_reference=spec.pair_min_reference,
                                       prefix=f"{ref}_vs_{src}_", metrics=spec.pair_metrics, index=df.index))
    if spec.member_prefix is not None:
        cols = member_columns(plant, spec.member_prefix)
        members = df[cols].to_numpy(dtype=float)
        if spec.members_per_unit:
            members = members / plant.capacity
        ens = ensemble_metrics(members, si_pairs=spec.si_pairs, history=spec.pi_history, prefix="ens_",
                               index=df.index, relative=spec.si_relative)
        parts.append(ens.drop(columns=["ens_mean"]))
    if spec.spatial is not None:
        if plant.turbine_field is None:
            raise ValueError(f"plant {plant.plant_id}: spatial features need a turbine field")
        sp = spec.spatial
        vg = variogram_over_time(plant.turbine_field.to_numpy(dtype=float), plant.turbine_coords, sp.bin_edges,
                                 model=sp.model, kind=sp.kind, detrend=sp.detrend, min_pairs=sp.min_pairs,
                                 max_range=sp.max_range, prefix="vg_", index=df.index)
        parts.append(vg.drop(columns=["vg_vg_rmse"]))
    if spec.include_level:
        parts.append(pd.DataFrame({"level": per_unit(plant, "forecast").to_numpy()}, index=df.index))
    if not parts:
        raise ValueError("the feature specification selects no features")
    return pd.concat(parts, axis=1)


def calibration_targets(plant: PlantSeries, levels, dist: str = "laplace", block: int | None = None,
                        target_window: int | None = None, bounds=None) -> pd.DataFrame:
    """Per-unit pinball-loss-optimal scale at an observed plant (Step A), with the per-unit forecast and actual.

    Default: one target per time step from that step alone. ``block=k``: one scale per block of ``k``
    consecutive steps. ``target_window=k`` (odd): one scale per step, fitted on the centred window of ``k``
    steps. Time steps without an observation are dropped first.
    """
    if not plant.observed:
        raise ValueError(f"plant {plant.plant_id} has no observations")
    if block is not None and target_window is not None:
        raise ValueError("give block or target_window, not both")
    df = plant.data[["forecast", "actual"]].astype(float).dropna()
    f = df["forecast"].to_numpy() / plant.capacity
    a = df["actual"].to_numpy() / plant.capacity
    if target_window is not None:
        scale, loss = rolling_optimal_scale(a, f, levels, dist=dist, window=target_window, bounds=bounds)
    else:
        groups = None if block is None else block_labels(len(df), block)
        scale, loss = optimal_scale(a, f, levels, dist=dist, bounds=bounds, groups=groups)
    return pd.DataFrame({"forecast_pu": f, "actual_pu": a, "target_scale": scale, "target_loss": loss},
                        index=df.index)


def training_table(plants, spec: FeatureSpec, levels, dist: str = "laplace", block: int | None = None,
                   target_window: int | None = None, bounds=None) -> pd.DataFrame:
    """Long table for the observed plants: ``plant_id``, ``time``, features, per-unit forecast and actual, target
    (``target_scale``, ``target_loss``) and ``target_reach`` (how many rows ahead a target may use observations).

    Rows with a missing feature or target are dropped.
    """
    frames = []
    for plant in plants:
        feats = build_features(plant, spec)
        tgt = calibration_targets(plant, levels, dist=dist, block=block, target_window=target_window,
                                  bounds=bounds)
        tab = feats.join(tgt, how="inner")
        tab.insert(0, "time", tab.index)
        tab.insert(0, "plant_id", plant.plant_id)
        frames.append(tab.reset_index(drop=True))
    if not frames:
        raise ValueError("no observed plants")
    table = pd.concat(frames, ignore_index=True).dropna().reset_index(drop=True)
    # how many rows ahead a target may use observations (for purging time splits, validation.time_split);
    # a column, so that it survives saving and reading the table
    table["target_reach"] = 0 if target_window is None and block is None else \
        (target_window // 2 if target_window is not None else block - 1)
    return table


def feature_columns(table: pd.DataFrame) -> list[str]:
    """Feature columns of a table made by :func:`training_table` or :func:`prediction_table`."""
    skip = {"plant_id", "time", "forecast_pu", "actual_pu", "target_scale", "target_loss", "target_reach"}
    return [c for c in table.columns if c not in skip]


def prediction_table(plants, spec: FeatureSpec) -> pd.DataFrame:
    """Features of plants to predict (observed or not): ``plant_id``, ``time``, features, per-unit forecast.

    Rows inside a window warm-up are dropped.
    """
    frames = []
    for plant in plants:
        feats = build_features(plant, spec)
        feats["forecast_pu"] = per_unit(plant, "forecast").to_numpy()
        feats.insert(0, "time", feats.index)
        feats.insert(0, "plant_id", plant.plant_id)
        frames.append(feats.reset_index(drop=True))
    if not frames:
        raise ValueError("no plants to predict")
    return pd.concat(frames, ignore_index=True).dropna().reset_index(drop=True)
