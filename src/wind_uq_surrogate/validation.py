"""Step D of the workflow: validation by held-out plants, and scores of the scale and of the predictive
distributions it implies. Details: ``docs/surrogate_workflow.md``, Sections 4 and 5.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

from .calibration import PITRecalibrator
from .features import feature_columns
from .probabilistic import crps, interval_scores, pinball_loss, quantiles, reliability
from .surrogate import Surrogate


def plant_splits(plant_ids, n_splits: int | None = None, groups=None):
    """Train/test row indices that hold out whole plants. A held-out plant plays the role of an unobserved plant.

    ``plant_ids``: the plant of each row. ``groups``: optional mapping plant id -> group label (for example a
    region); then whole groups are held out, which avoids testing on plants that share weather with training
    plants. ``n_splits=None``: leave one plant (or group) out; otherwise ``n_splits`` folds of plants (or groups).
    """
    plants = np.asarray(plant_ids).astype(str)
    if groups is not None:
        mapping = {str(k): v for k, v in dict(groups).items()}
        missing = sorted(set(plants) - set(mapping))
        if missing:
            raise ValueError(f"groups has no label for plants {missing}")
        labels = np.array([str(mapping[p]) for p in plants])
    else:
        labels = plants
    n_groups = len(np.unique(labels))
    if n_groups < 2:
        raise ValueError("at least two plants (or groups) are needed to hold them out")
    dummy = np.zeros((len(labels), 1))
    if n_splits is None:
        return list(LeaveOneGroupOut().split(dummy, groups=labels))
    if not 2 <= n_splits <= n_groups:
        raise ValueError(f"n_splits must be between 2 and the number of plants or groups ({n_groups})")
    return list(GroupKFold(n_splits=n_splits).split(dummy, groups=labels))


def time_split(table: pd.DataFrame, test_fraction: float = 0.25, gap: int | None = None):
    """Within-plant split: for every plant the last ``test_fraction`` of its rows (in time) form the test set,
    and the ``gap`` training rows just before the test period are left out.

    The gap removes training rows whose target depends on test-period observations (centred-window or block
    targets reach ``target_reach`` rows ahead; :func:`training_table` stores it in the column ``target_reach``,
    which survives a CSV round trip). ``gap=None`` uses that column and raises an error if it is missing. Test-row features use trailing windows that may start in the
    training period; that is information available at prediction time, not leakage.
    """
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be strictly between 0 and 1")
    if gap is None:
        if "target_reach" in table.columns:
            gap = int(table["target_reach"].max())
        elif "target_reach" in table.attrs:
            gap = int(table.attrs["target_reach"])
        else:
            raise ValueError("the table does not record how far its targets reach (column 'target_reach'); "
                             "give gap explicitly")
    if gap < 0:
        raise ValueError("gap must be non-negative")
    train, test = [], []
    for _, idx in table.groupby("plant_id", sort=False).indices.items():
        order = idx[np.argsort(table["time"].to_numpy()[idx], kind="stable")]
        cut = int(round(len(order) * (1 - test_fraction)))
        train.extend(order[:max(cut - gap, 0)])
        test.extend(order[cut:])
    return [(np.sort(np.array(train, dtype=int)), np.sort(np.array(test, dtype=int)))]


def cross_validate(table: pd.DataFrame, models: dict, splits, features=None) -> pd.DataFrame:
    """Fit every surrogate on each training split and predict the held-out rows.

    ``models`` maps a label to a factory returning an unfitted :class:`Surrogate`
    (e.g. ``{"gpr": lambda: Surrogate("gpr", max_train_samples=600)}``). Returns one row per held-out sample and
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
    """Agreement of the predicted scale with the calibrated target, per model: MAE and RMSE of the scale, RMSE
    of its logarithm, and the coefficient of determination of the log scale. The targets are noisy proxies of the
    unknown error scale, so these numbers describe agreement with the proxy, not accuracy for the true scale."""
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


def _summary_row(label, y, q, levels, coverages, crps_values=None):
    rel = reliability(y, q, levels)
    rel.insert(0, "model", label)
    row = {"model": label, "n": len(y), "pinball": pinball_loss(y, q, levels)}
    if crps_values is not None:
        row["crps"] = float(np.mean(crps_values))
    row["reliability_mad"] = float(rel["deviation"].abs().mean())
    for _, r in interval_scores(y, q, levels, coverages).iterrows():
        pct = int(round(100 * r["nominal"]))
        row[f"coverage_{pct}"] = r["coverage"]
        row[f"width_{pct}"] = r["mean_width"]
    return row, rel


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
        row, rel = _summary_row(label, y, q, levels, coverages, crps(y, m, s, dist))
        summary.append(row)
        rel_tables.append(rel)
    return pd.DataFrame(summary), pd.concat(rel_tables, ignore_index=True)


def nested_recalibrated_quantiles(table: pd.DataFrame, factory, splits, levels, dist: str = "laplace",
                                  inner_n_splits: int | None = 3, features=None, outer_predictions=None,
                                  groups=None) -> pd.DataFrame:
    """Recalibrated predictive quantiles at the held-out plants, without information from those plants.

    The guarantee holds for a fixed surrogate ``factory``; a surrogate chosen by comparing scores over all folds
    has used every fold's labels for the choice.
    For each outer fold, the recalibrator (:class:`PITRecalibrator`) is fitted on inner out-of-fold predictions
    that are computed within the outer training plants only (``inner_n_splits`` plant folds, ``None`` = leave one
    plant out); it is then applied to the outer test predictions. ``outer_predictions`` (rows of
    :func:`cross_validate` for this surrogate) avoids refitting the outer models.
    Returns ``fold``, ``plant_id``, ``time``, ``actual_pu`` and one column ``q_<k>`` per level.
    """
    feats = feature_columns(table) if features is None else list(features)
    tau = np.asarray(levels, dtype=float)
    out = []
    for fold, (tr, te) in enumerate(splits):
        train = table.iloc[tr].reset_index(drop=True)
        if groups is None:
            n_inner_groups = train["plant_id"].nunique()
        else:
            mapping = {str(key): v for key, v in dict(groups).items()}
            n_inner_groups = len({mapping[p] for p in train["plant_id"].astype(str).unique()})
        k = None if inner_n_splits is None or inner_n_splits >= n_inner_groups else inner_n_splits
        inner = cross_validate(train, {"m": factory}, plant_splits(train["plant_id"], k, groups), feats)
        rec = PITRecalibrator(dist).fit(inner["actual_pu"], inner["forecast_pu"], inner["pred_scale"])
        test = table.iloc[te]
        if outer_predictions is not None:
            scale = _aligned_scale(outer_predictions, fold, test)
        else:
            scale = factory().fit(train[feats], train["target_scale"]).predict(test[feats])
        q = rec.quantiles(test["forecast_pu"].to_numpy(), scale, tau)
        part = test[["plant_id", "time", "actual_pu"]].copy().reset_index(drop=True)
        part.insert(0, "fold", fold)
        for j in range(len(tau)):
            part[f"q_{j}"] = q[:, j]
        out.append(part)
    return pd.concat(out, ignore_index=True)


def _aligned_scale(outer_predictions: pd.DataFrame, fold: int, test: pd.DataFrame) -> np.ndarray:
    """Predicted scales of one outer fold matched to the test rows by plant and time (not by position)."""
    part = outer_predictions.loc[outer_predictions["fold"] == fold, ["plant_id", "time", "pred_scale"]]
    keys = ["plant_id", "time"]
    if part.duplicated(keys).any():
        raise ValueError(f"outer_predictions has duplicated (plant_id, time) rows in fold {fold}")
    merged = test[keys].merge(part, on=keys, how="left", validate="one_to_one")
    if len(part) != len(test) or merged["pred_scale"].isna().any():
        raise ValueError(f"outer_predictions do not match the test rows of fold {fold}")
    return merged["pred_scale"].to_numpy()


def nested_feasible(table: pd.DataFrame, splits, groups=None) -> bool:
    """True when every outer training set holds at least two plants (or groups), as nested folds need."""
    for tr, _ in splits:
        ids = table["plant_id"].iloc[tr].astype(str).unique()
        if groups is not None:
            mapping = {str(key): v for key, v in dict(groups).items()}
            ids = {mapping[p] for p in ids}
        if len(ids) < 2:
            return False
    return True


def recalibrated_scores(table: pd.DataFrame, models: dict, splits, levels, dist: str = "laplace",
                        coverages=(0.5, 0.9), inner_n_splits: int | None = 3, cv_predictions=None, groups=None):
    """Scores after recalibration, validated by nested plant folds (:func:`nested_recalibrated_quantiles`), for
    each surrogate in ``models`` (label -> factory). No CRPS: it has no closed form after recalibration.
    Returns the summary table and the reliability table."""
    summary, rel_tables = [], []
    for label, factory in models.items():
        outer = None if cv_predictions is None else cv_predictions[cv_predictions["model"] == label]
        qtab = nested_recalibrated_quantiles(table, factory, splits, levels, dist, inner_n_splits,
                                             outer_predictions=outer, groups=groups)
        q = qtab[[f"q_{j}" for j in range(len(levels))]].to_numpy()
        row, rel = _summary_row(label, qtab["actual_pu"].to_numpy(), q, levels, coverages)
        summary.append(row)
        rel_tables.append(rel)
    return pd.DataFrame(summary), pd.concat(rel_tables, ignore_index=True)


def oracle_scores(table: pd.DataFrame, levels, dist: str = "laplace", coverages=(0.5, 0.9)) -> pd.DataFrame:
    """Scores of the calibrated targets themselves, in sample. With single-step targets each target minimizes
    its own step's pinball loss within the same location-scale family, location, levels and bounds, so their mean
    pinball loss is the smallest any choice of scales in that family can reach (it is reached by the targets).
    With block or window targets it is only a reference."""
    pred = table[["plant_id", "time", "forecast_pu", "actual_pu", "target_scale"]].copy()
    pred["model"] = "target (in-sample)"
    pred["pred_scale"] = pred["target_scale"]
    return score_predictive(pred, levels, dist, coverages)[0]
