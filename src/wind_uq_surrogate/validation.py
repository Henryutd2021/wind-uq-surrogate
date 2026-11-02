"""Step D of the workflow: validation by held-out plants, and scores of the scale and of the predictive
distributions it implies. Details: ``docs/surrogate_workflow.md``, Sections 4 and 5.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

from .calibration import IsotonicRecalibrator
from .features import feature_columns
from .probabilistic import crps, interval_scores, pinball_loss, quantiles, reliability
from .surrogate import Surrogate


def plant_splits(plant_ids, n_splits: int | None = None):
    """Train/test row indices that hold out whole plants: leave one plant out (``n_splits=None``) or
    ``n_splits`` folds of plants. A held-out plant plays the role of an unobserved plant."""
    groups = np.asarray(plant_ids)
    n_groups = len(np.unique(groups))
    if n_groups < 2:
        raise ValueError("at least two plants are needed to hold out plants")
    dummy = np.zeros((len(groups), 1))
    if n_splits is None:
        return list(LeaveOneGroupOut().split(dummy, groups=groups))
    if not 2 <= n_splits <= n_groups:
        raise ValueError(f"n_splits must be between 2 and the number of plants ({n_groups})")
    return list(GroupKFold(n_splits=n_splits).split(dummy, groups=groups))


def time_split(table: pd.DataFrame, test_fraction: float = 0.25):
    """Within-plant split: for every plant the last ``test_fraction`` of its rows (in time) form the test set.
    Rows near the boundary share window history, which a gap between the sets would remove."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be strictly between 0 and 1")
    train, test = [], []
    for _, idx in table.groupby("plant_id", sort=False).indices.items():
        order = idx[np.argsort(table["time"].to_numpy()[idx], kind="stable")]
        cut = int(round(len(order) * (1 - test_fraction)))
        train.extend(order[:cut])
        test.extend(order[cut:])
    return [(np.sort(np.array(train, dtype=int)), np.sort(np.array(test, dtype=int)))]


def cross_validate(table: pd.DataFrame, models: dict, splits, features=None) -> pd.DataFrame:
    """Fit every surrogate on each training split and predict the held-out rows.

    ``models`` maps a label to a factory returning an unfitted :class:`Surrogate`
    (e.g. ``{"gpr": lambda: Surrogate("gpr", max_train_samples=1500)}``). Returns one row per held-out sample and
    model: ``model``, ``fold``, ``plant_id``, ``time``, ``forecast_pu``, ``actual_pu``, ``target_scale``,
    ``pred_scale`` and, for GP surrogates, ``pred_log_std``.
    """
    feats = feature_columns(table) if features is None else list(features)
    out = []
    for label, factory in models.items():
        for fold, (tr, te) in enumerate(splits):
            model: Surrogate = factory()
            model.fit(table.iloc[tr][feats], table.iloc[tr]["target_scale"])
            part = table.iloc[te][["plant_id", "time", "forecast_pu", "actual_pu", "target_scale"]].copy()
            part["pred_scale"] = model.predict(table.iloc[te][feats])
            if model.name == "gpr":
                part["pred_log_std"] = model.predict_log_std(table.iloc[te][feats])
            part.insert(0, "fold", fold)
            part.insert(0, "model", label)
            out.append(part)
    return pd.concat(out, ignore_index=True)


def score_scale(pred: pd.DataFrame) -> pd.DataFrame:
    """Accuracy of the predicted scale against the calibrated target, per model: MAE and RMSE of the scale,
    RMSE of its logarithm, and the coefficient of determination of the log scale."""
    rows = []
    for label, g in pred.groupby("model", sort=False):
        t, p = g["target_scale"].to_numpy(), g["pred_scale"].to_numpy()
        lt, lp = np.log(t), np.log(p)
        ss_res = ((lt - lp) ** 2).sum()
        ss_tot = ((lt - lt.mean()) ** 2).sum()
        rows.append({"model": label, "n": len(g), "scale_mae": float(np.abs(t - p).mean()),
                     "scale_rmse": float(np.sqrt(((t - p) ** 2).mean())),
                     "log_scale_rmse": float(np.sqrt(((lt - lp) ** 2).mean())),
                     "log_scale_r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")})
    return pd.DataFrame(rows)


def score_predictive(pred: pd.DataFrame, levels, dist: str = "laplace", coverages=(0.5, 0.9),
                     scale_column: str = "pred_scale") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Scores of the predictive distributions ``F((y - forecast) / predicted scale)`` against the observations,
    per model: mean pinball loss over ``levels``, mean CRPS (closed form), mean absolute deviation of the
    reliability curve, and coverage and mean width of central intervals (all per unit of capacity).

    Returns the summary table and the reliability table (``model``, ``level``, ``observed``, ``deviation``).
    """
    summary, rel_tables = [], []
    for label, g in pred.groupby("model", sort=False):
        y, m, s = g["actual_pu"].to_numpy(), g["forecast_pu"].to_numpy(), g[scale_column].to_numpy()
        q = quantiles(m, s, levels, dist)
        rel = reliability(y, q, levels)
        rel.insert(0, "model", label)
        rel_tables.append(rel)
        row = {"model": label, "n": len(g), "pinball": pinball_loss(y, q, levels),
               "crps": float(crps(y, m, s, dist).mean()), "reliability_mad": float(rel["deviation"].abs().mean())}
        for _, r in interval_scores(y, q, levels, coverages).iterrows():
            pct = int(round(100 * r["nominal"]))
            row[f"coverage_{pct}"] = r["coverage"]
            row[f"width_{pct}"] = r["mean_width"]
        summary.append(row)
    return pd.DataFrame(summary), pd.concat(rel_tables, ignore_index=True)


def recalibrated_scores(pred: pd.DataFrame, levels, dist: str = "laplace", coverages=(0.5, 0.9)):
    """Cross-fitted isotonic recalibration: for each fold, fit the recalibrator on the held-out predictions of
    the other folds and apply it to this fold; then score as :func:`score_predictive` (without CRPS, which has
    no closed form after recalibration). Needs at least two folds."""
    summary, rel_tables = [], []
    for label, g in pred.groupby("model", sort=False):
        folds = g["fold"].unique()
        if len(folds) < 2:
            raise ValueError("recalibration needs at least two folds")
        y_all, q_all = [], []
        for f in folds:
            fit_rows, apply_rows = g[g["fold"] != f], g[g["fold"] == f]
            rec = IsotonicRecalibrator(dist).fit(fit_rows["actual_pu"], fit_rows["forecast_pu"], fit_rows["pred_scale"])
            q_all.append(rec.quantiles(apply_rows["forecast_pu"], apply_rows["pred_scale"], levels))
            y_all.append(apply_rows["actual_pu"].to_numpy())
        y, q = np.concatenate(y_all), np.vstack(q_all)
        rel = reliability(y, q, levels)
        rel.insert(0, "model", label)
        rel_tables.append(rel)
        row = {"model": label, "n": len(y), "pinball": pinball_loss(y, q, levels),
               "reliability_mad": float(rel["deviation"].abs().mean())}
        for _, r in interval_scores(y, q, levels, coverages).iterrows():
            pct = int(round(100 * r["nominal"]))
            row[f"coverage_{pct}"] = r["coverage"]
            row[f"width_{pct}"] = r["mean_width"]
        summary.append(row)
    return pd.DataFrame(summary), pd.concat(rel_tables, ignore_index=True)


def oracle_scores(table: pd.DataFrame, levels, dist: str = "laplace", coverages=(0.5, 0.9)) -> pd.DataFrame:
    """Scores obtained with the calibrated targets themselves, in sample. With single-step targets each target
    minimizes its own step's pinball loss, so the mean pinball loss is a lower bound for any choice of scales;
    with block or window targets it is only a reference."""
    pred = table[["plant_id", "time", "forecast_pu", "actual_pu", "target_scale"]].copy()
    pred["model"] = "target (in-sample)"
    pred["pred_scale"] = pred["target_scale"]
    return score_predictive(pred, levels, dist, coverages)[0]
