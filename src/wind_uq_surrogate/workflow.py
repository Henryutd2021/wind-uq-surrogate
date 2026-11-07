"""The end-to-end surrogate workflow (Steps A-D, ``docs/surrogate_workflow.md``).

A. Calibrated uncertainty targets at observed plants (pinball-loss-optimal scale around the point forecast).
B. UQ-metric features from series that exist at every plant.
C. Surrogates from features to the target scale.
D. Validation on held-out plants; then the chosen surrogate, fitted on all observed plants, predicts the scale,
   and hence predictive quantiles, at unobserved plants.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .calibration import PITRecalibrator
from .features import FeatureSpec, feature_columns, prediction_table, training_table
from .probabilistic import available_coverages, check_levels, default_levels, quantiles
from .surrogate import SURROGATES, Surrogate
from .validation import (cross_validate, nested_feasible, oracle_scores, plant_splits, recalibrated_scores,
                         score_predictive, score_scale)


@dataclass
class WorkflowConfig:
    """Settings of :func:`run_workflow`.

    ``features``: the :class:`FeatureSpec`. ``levels``: quantile levels of the loss and of the outputs.
    ``dist``: predictive distribution family. ``block``: one target scale per block of consecutive steps;
    ``target_window``: one target per step from a centred window (odd length); neither: one target per step from
    that step alone. ``models``: surrogates to compare. ``gpr_max_train``: row cap for the GP surrogate.
    ``max_train``: optional row caps per surrogate (e.g. ``{"svr": 20000}``; a ``"gpr"`` entry replaces
    ``gpr_max_train``), for large tables. ``model_params``: optional keyword arguments per surrogate, passed to
    the scikit-learn estimator (e.g. ``{"random_forest": {"n_jobs": 8}}``); a ``random_state`` given there replaces
    ``random_state`` for that surrogate (row subset and estimator).
    ``n_splits``: plant folds (``None``: leave one plant out). ``split_groups``: optional mapping plant id ->
    group (e.g. region) so that whole groups are held out. ``recalibrate``: PIT recalibration of the chosen
    surrogate's predictive distributions, validated with nested plant folds (``inner_n_splits``).
    ``final_model``: surrogate used for the unobserved plants (``None``: the one with the lowest cross-validated
    pinball loss before recalibration; the choice uses the same folds as the reported scores, which makes the
    chosen surrogate's scores slightly optimistic).
    """

    features: FeatureSpec
    levels: tuple = tuple(default_levels(0.05))
    dist: str = "laplace"
    block: int | None = None
    target_window: int | None = None
    scale_bounds: tuple | None = None
    models: tuple = ("gpr", "random_forest", "svr", "ridge", "constant")
    gpr_max_train: int | None = 600
    n_splits: int | None = None
    split_groups: dict | None = None
    recalibrate: bool = True
    inner_n_splits: int | None = 3
    coverages: tuple = (0.5, 0.9)
    final_model: str | None = None
    random_state: int = 0
    max_train: dict | None = None
    model_params: dict | None = None

    def __post_init__(self):
        check_levels(self.levels)
        missing = set(map(float, self.coverages)) - set(available_coverages(self.levels, self.coverages))
        if missing:
            raise ValueError(f"levels lack the interval bounds of coverages {sorted(missing)}")
        unknown = set(self.models) - set(SURROGATES)
        if unknown:
            raise ValueError(f"unknown surrogates: {sorted(unknown)}")
        if self.final_model is not None and self.final_model not in self.models:
            raise ValueError("final_model must be one of the compared models")
        for label, mapping in (("max_train", self.max_train), ("model_params", self.model_params)):
            extra = set(mapping or {}) - set(self.models)
            if extra:
                raise ValueError(f"{label} names surrogates that are not compared: {sorted(extra)}")
        for name, cap in (self.max_train or {}).items():
            if cap is not None and (isinstance(cap, bool) or not isinstance(cap, (int, np.integer)) or cap < 2):
                raise ValueError(f"max_train[{name!r}] must be an integer of at least 2 or None")

    def factories(self) -> dict:
        out = {}
        caps = dict(self.max_train or {})
        params = dict(self.model_params or {})
        for name in self.models:
            cap = caps.get(name, self.gpr_max_train if name == "gpr" else None)
            kw = {"random_state": self.random_state, **params.get(name, {})}  # a model's own seed wins
            out[name] = (lambda n=name, c=cap, k=kw: Surrogate(n, max_train_samples=c, **k))
        return out


@dataclass
class WorkflowResult:
    table: pd.DataFrame
    cv_predictions: pd.DataFrame
    scale_scores: pd.DataFrame
    predictive_scores: pd.DataFrame
    reliability: pd.DataFrame
    oracle: pd.DataFrame
    chosen_model: str
    final_model: Surrogate
    predictions: pd.DataFrame
    recalibrated_scores: pd.DataFrame | None = None
    recalibrated_reliability: pd.DataFrame | None = None
    notes: list = field(default_factory=list)


def run_workflow(observed, unobserved, config: WorkflowConfig) -> WorkflowResult:
    """Run Steps A-D. ``observed``: plants with an ``actual`` column (at least two); ``unobserved``: plants to
    predict (may be empty). Returns tables of every step and the predictions at the unobserved plants."""
    observed = list(observed)
    unobserved = list(unobserved)
    if len(observed) < 2:
        raise ValueError("at least two observed plants are needed (plants are held out in validation)")
    for p in observed:
        if not p.observed:
            raise ValueError(f"plant {p.plant_id} is listed as observed but has no 'actual' values")
    levels = np.asarray(config.levels, dtype=float)
    table = training_table(observed, config.features, levels, dist=config.dist, block=config.block,
                           target_window=config.target_window, bounds=config.scale_bounds)
    if table["plant_id"].nunique() < 2:
        raise ValueError("after the window warm-up fewer than two plants have complete rows")
    splits = plant_splits(table["plant_id"].to_numpy(), config.n_splits, config.split_groups)
    factories = config.factories()
    cv = cross_validate(table, factories, splits)
    scale_scores = score_scale(cv)
    pred_scores, rel = score_predictive(cv, levels, config.dist, config.coverages)
    oracle = oracle_scores(table, levels, config.dist, config.coverages)
    chosen = config.final_model or str(pred_scores.sort_values("pinball").iloc[0]["model"])
    rec_scores = rec_rel = None
    notes = []
    if config.recalibrate and not nested_feasible(table, splits, config.split_groups):
        notes.append("recalibration not validated: nested plant folds need at least two plants or groups in every "
                      "outer training set")
    elif config.recalibrate:
        rec_scores, rec_rel = recalibrated_scores(table, {chosen: factories[chosen]}, splits, levels, config.dist,
                                                  config.coverages, config.inner_n_splits, cv_predictions=cv,
                                                  groups=config.split_groups)
    final = factories[chosen]().fit(table[feature_columns(table)], table["target_scale"])
    notes += [f"{chosen}: {w}" for w in final.fit_warnings_]
    predictions = pd.DataFrame()
    if unobserved:
        ptab = prediction_table(unobserved, config.features)
        feats = feature_columns(ptab)
        missing = set(feature_columns(table)) - set(feats)
        if missing:
            raise ValueError(f"unobserved plants lack features: {sorted(missing)}")
        feats = feature_columns(table)
        scale = final.predict(ptab[feats])
        if config.recalibrate:
            # out-of-fold predictions at every observed plant, each from a surrogate that did not see that plant
            sub = cv[cv["model"] == chosen]
            rec = PITRecalibrator(config.dist).fit(sub["actual_pu"], sub["forecast_pu"], sub["pred_scale"])
            q = rec.quantiles(ptab["forecast_pu"], scale, levels)
        else:
            q = quantiles(ptab["forecast_pu"], scale, levels, config.dist)
        capacity = {p.plant_id: p.capacity for p in unobserved}
        cap = ptab["plant_id"].map(capacity).to_numpy()
        predictions = ptab[["plant_id", "time", "forecast_pu"]].copy()
        predictions["pred_scale_pu"] = scale
        for k, tau in enumerate(levels):
            predictions[f"q{_pct(tau)}_mw"] = q[:, k] * cap
    return WorkflowResult(table, cv, scale_scores, pred_scores, rel, oracle, chosen, final, predictions,
                          rec_scores, rec_rel, notes)


def _pct(tau: float) -> str:
    s = f"{100 * tau:.4g}"
    return s.replace(".", "p")


def save_result(result: WorkflowResult, out_dir, config: WorkflowConfig | None = None) -> list[Path]:
    """Write the result tables as CSV files and a ``summary.json``; returns the written paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tables = {
        "training_table.csv": result.table, "cv_predictions.csv": result.cv_predictions,
        "scale_scores.csv": result.scale_scores, "predictive_scores.csv": result.predictive_scores,
        "reliability.csv": result.reliability, "oracle_scores.csv": result.oracle,
        "unobserved_predictions.csv": result.predictions,
    }
    if result.recalibrated_scores is not None:
        tables["recalibrated_scores.csv"] = result.recalibrated_scores
        tables["recalibrated_reliability.csv"] = result.recalibrated_reliability
    written = []
    for name, df in tables.items():
        path = out / name
        df.to_csv(path, index=False)
        written.append(path)
    summary = {"chosen_model": result.chosen_model, "notes": result.notes,
               "observed_plants": sorted(result.table["plant_id"].unique().tolist()),
               "unobserved_plants": sorted(result.predictions["plant_id"].unique().tolist())
               if len(result.predictions) else [],
               "predictive_scores": result.predictive_scores.to_dict(orient="records")}
    if config is not None:
        cfg = asdict(config)
        summary["config"] = json.loads(json.dumps(cfg, default=str))
    path = out / "summary.json"
    path.write_text(json.dumps(summary, indent=2, default=float))
    written.append(path)
    return written
