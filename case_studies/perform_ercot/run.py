"""ERCOT case study on public PERFORM data: the surrogate workflow under three validation designs, with benchmarks.

Designs (``--design``):

- ``plants10``  ten folds of plants: each held-out plant plays an unobserved plant whose neighbours are observed;
- ``zones``     leave one ERCOT weather zone out: no plant of the held-out zone is observed;
- ``proposed``  the 125 existing plants are observed and the 139 proposed plants are not: the surrogates are
                compared and the recalibration is fitted on the existing plants only (ten plant folds), then the
                chosen surrogate predicts the proposed plants, scored against their simulated power.

Methods, scored on the same rows (per unit of capacity; every method's quantiles are clipped to [0, 1]):

- the surrogates (``gpr``, ``random_forest``, ``svr``, ``ridge``, ``constant``): Laplace predictive distributions
  around the ensemble-mean forecast with the predicted scale;
- ``<chosen> + recalibration`` and ``constant + recalibration``: the surrogate with the lowest pinball loss, and
  the constant scale, after PIT recalibration fitted without the held-out plants (nested plant folds; in
  ``proposed``, out-of-fold predictions at existing plants). Their difference is what the UQ-metric features add;
- ``raw ensemble``: empirical quantiles of the 51 members (no plant data needed, like the surrogates).

References that need the held-out plant's own power (not options at an unobserved plant):

- ``plant constant (in-sample)``: one pinball-loss-optimal scale per plant, fitted on that plant's whole year;
- ``daily target (in-sample)``: the daily target scales themselves (the best daily scales around this forecast).

Scores: mean pinball loss over the levels 0.05, ..., 0.95 (``pinball``) and over 0.01, ..., 0.99 (``crps_q`` =
twice that mean, a quantile approximation of the CRPS), mean absolute error of the median (``mae_median``), mean
absolute deviation of the reliability curve, coverage and mean width of the central 50% and 90% intervals; and
the pinball loss per plant.

Usage:  python run.py PREPARED_DIR OUT_DIR --design plants10|zones|proposed [--jobs 16] [--folds 10] [--cache DIR]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import wind_uq_surrogate as w
from wind_uq_surrogate.features import feature_columns, member_columns

LEVELS = w.default_levels(0.05)                       # targets and the main pinball score
LEVELS_99 = np.round(np.arange(1, 100) / 100, 2)      # outputs and the CRPS approximation
I19 = np.array([int(round(100 * t)) - 1 for t in LEVELS])
I50 = (24, 74)                                        # levels 0.25 and 0.75
I90 = (4, 94)                                         # levels 0.05 and 0.95
DIST = "laplace"
MODELS = ("gpr", "random_forest", "svr", "ridge", "constant")


def feature_spec() -> w.FeatureSpec:
    """UQ metrics that exist at every plant: variability of the forecast, revision between the 2-day-ahead and
    day-ahead forecasts, spread of the 51 members (absolute, per unit), predictability index, forecast level."""
    return w.FeatureSpec(window=24, single_series=("forecast",), single_metrics=("std",),
                         pairs=(("forecast", "fcst_2da"),), pair_metrics=("nmae",), pair_normalizer=1.0,
                         per_unit=("forecast", "fcst_2da"), member_prefix="ens_", members_per_unit=True,
                         si_pairs=((75.0, 25.0), (95.0, 5.0)), si_relative=False, pi_history=48,
                         include_level=True)


def workflow_config(jobs: int, n_splits=10, groups=None) -> w.WorkflowConfig:
    return w.WorkflowConfig(features=feature_spec(), levels=tuple(LEVELS), dist=DIST, block=24, models=MODELS,
                            n_splits=n_splits, split_groups=groups, inner_n_splits=3,
                            max_train={"gpr": 1000, "svr": 5000, "random_forest": 200000},
                            model_params={"random_forest": {"n_jobs": jobs}})


class Scores:
    """Accumulates scores per method and plant from 99 quantiles per row (memory stays per plant)."""

    def __init__(self):
        self.acc = {}

    def add(self, method: str, plant: str, y: np.ndarray, q: np.ndarray):
        q = np.sort(np.clip(q, 0.0, 1.0), axis=1)
        u = y[:, None] - q
        pin = np.maximum(LEVELS_99 * u, (LEVELS_99 - 1.0) * u).sum(axis=0)
        below = (y[:, None] <= q).sum(axis=0)
        inside50 = ((y >= q[:, I50[0]]) & (y <= q[:, I50[1]])).sum()
        inside90 = ((y >= q[:, I90[0]]) & (y <= q[:, I90[1]])).sum()
        width50 = (q[:, I50[1]] - q[:, I50[0]]).sum()
        width90 = (q[:, I90[1]] - q[:, I90[0]]).sum()
        mae50 = np.abs(y - q[:, 49]).sum()
        key = (method, plant)
        old = self.acc.get(key)
        new = [len(y), pin, below, inside50, inside90, width50, width90, mae50]
        self.acc[key] = new if old is None else [a + b for a, b in zip(old, new)]

    @staticmethod
    def _row(n, pin, below, i50, i90, w50, w90, mae50) -> dict:
        p = pin / n
        rel = below / n - LEVELS_99
        return {"n": int(n), "pinball": float(p[I19].mean()), "crps_q": float(2 * p.mean()),
                "mae_median": mae50 / n, "reliability_mad": float(np.abs(rel[I19]).mean()),
                "coverage_50": i50 / n, "width_50": w50 / n, "coverage_90": i90 / n, "width_90": w90 / n}

    def summary(self) -> pd.DataFrame:
        methods = {}
        for (m, _), v in self.acc.items():
            methods[m] = v if m not in methods else [a + b for a, b in zip(methods[m], v)]
        return pd.DataFrame([{"method": m, **self._row(*v)} for m, v in methods.items()])

    def reliability(self) -> pd.DataFrame:
        rows = []
        methods = {}
        for (m, _), v in self.acc.items():
            methods[m] = v if m not in methods else [a + b for a, b in zip(methods[m], v)]
        for m, v in methods.items():
            obs = v[2] / v[0]
            rows += [{"method": m, "level": float(LEVELS_99[i]), "observed": float(obs[i])} for i in I19]
        return pd.DataFrame(rows)

    def per_plant(self) -> pd.DataFrame:
        rows = [{"method": m, "plant_id": p, **self._row(*v)} for (m, p), v in self.acc.items()]
        return pd.DataFrame(rows)[["method", "plant_id", "n", "pinball", "crps_q", "coverage_90"]]


def ensemble_quantiles(plant: w.PlantSeries) -> pd.DataFrame:
    """Per-unit empirical quantiles of the 51 members at the 99 levels (linear interpolation), index = time."""
    cols = member_columns(plant, "ens_")
    members = plant.data[cols].to_numpy(dtype=float) / plant.capacity
    return pd.DataFrame(np.quantile(members, LEVELS_99, axis=1).T, index=plant.data.index)


def score_ensemble(scores: Scores, rows: pd.DataFrame, plants: dict):
    """Add the raw ensemble on the rows (plant_id, time, actual_pu) of a design."""
    for pid, g in rows.groupby("plant_id", sort=False):
        raw = ensemble_quantiles(plants[pid])
        scores.add("raw ensemble", pid, g["actual_pu"].to_numpy(), raw.loc[pd.DatetimeIndex(g["time"])].to_numpy())


def score_references(scores: Scores, rows: pd.DataFrame):
    """In-sample references on rows with ``plant_id``, ``forecast_pu``, ``actual_pu`` and the daily
    ``target_scale``: one optimal scale per plant, and the daily targets themselves."""
    for pid, g in rows.groupby("plant_id", sort=False):
        f, y = g["forecast_pu"].to_numpy(), g["actual_pu"].to_numpy()
        s_p, _ = w.optimal_scale(y, f, LEVELS, DIST, groups=np.zeros(len(g)))
        scores.add("plant constant (in-sample)", pid, y, w.quantiles(f, s_p, LEVELS_99, DIST))
        scores.add("daily target (in-sample)", pid, y,
                   w.quantiles(f, g["target_scale"].to_numpy(), LEVELS_99, DIST))


def score_parametric(scores: Scores, label: str, pred: pd.DataFrame):
    for pid, g in pred.groupby("plant_id", sort=False):
        q = w.quantiles(g["forecast_pu"].to_numpy(), g["pred_scale"].to_numpy(), LEVELS_99, DIST)
        scores.add(label, pid, g["actual_pu"].to_numpy(), q)


def score_quantile_table(scores: Scores, label: str, qtab: pd.DataFrame):
    qcols = [f"q_{j}" for j in range(len(LEVELS_99))]
    for pid, g in qtab.groupby("plant_id", sort=False):
        scores.add(label, pid, g["actual_pu"].to_numpy(), g[qcols].to_numpy())


def pinball19(pred: pd.DataFrame) -> pd.Series:
    """Cross-validated pinball loss (19 levels, unclipped, as in the package) per model: the model choice."""
    out = {}
    for m, g in pred.groupby("model", sort=False):
        q = w.quantiles(g["forecast_pu"].to_numpy(), g["pred_scale"].to_numpy(), LEVELS, DIST)
        out[m] = w.pinball_loss(g["actual_pu"].to_numpy(), q, LEVELS)
    return pd.Series(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prepared_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--design", required=True, choices=("plants10", "zones", "proposed"))
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--folds", type=int, default=10, help="plant folds of plants10 and proposed (default 10)")
    ap.add_argument("--cache", help="directory for the cross-validated predictions (reused when present)")
    args = ap.parse_args()
    t0 = time.time()
    root, out = Path(args.prepared_dir), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(root / "inputs" / "plants.csv", dtype={"plant_id": str})
    meta.rename(columns={"group": "zone"})[["plant_id", "perform_site", "capacity_mw", "zone", "status"]].to_csv(
        out / "plants.csv", index=False)
    for name in ("forecast_summary.csv", "alignment.csv", "prepare_summary.json"):  # outputs of prepare.py
        if (root / name).exists():
            shutil.copyfile(root / name, out / name)
    plants = {p.plant_id: p for p in w.read_plants(root / "inputs")}
    zone = dict(zip(meta["plant_id"], meta["group"]))
    status = dict(zip(meta["plant_id"], meta["status"]))
    print(f"{len(plants)} plants read ({time.time() - t0:.0f} s)", flush=True)

    if args.design == "proposed":
        train_ids = [p for p in plants if status[p] == "Existing"]
        test_ids = [p for p in plants if status[p] == "Proposed"]
        config = workflow_config(args.jobs, n_splits=args.folds)
    else:
        train_ids, test_ids = list(plants), []
        groups = zone if args.design == "zones" else None
        config = workflow_config(args.jobs, n_splits=None if groups else args.folds, groups=groups)
    spec = config.features
    table = w.training_table([plants[p] for p in train_ids], spec, LEVELS, dist=DIST, block=config.block)
    feats = feature_columns(table)
    print(f"training table: {len(table)} rows, {table['plant_id'].nunique()} plants, features {feats} "
          f"({time.time() - t0:.0f} s)", flush=True)
    splits = w.plant_splits(table["plant_id"].to_numpy(), config.n_splits, config.split_groups)
    factories = config.factories()
    cache = Path(args.cache) / f"cv_{args.design}_{args.folds}.pkl" if args.cache else None
    if cache is not None and cache.exists():
        cv = pd.read_pickle(cache)
        print(f"cross-validated predictions read from {cache}", flush=True)
    else:
        cv = w.cross_validate(table, factories, splits)
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cv.to_pickle(cache)
    print(f"cross-validation done ({time.time() - t0:.0f} s)", flush=True)
    choice = pinball19(cv)
    chosen = str(choice.idxmin())
    scale_scores = w.score_scale(cv)

    scores = Scores()
    if args.design != "proposed":
        for m in MODELS:
            score_parametric(scores, m, cv[cv["model"] == m])
        qtab = w.nested_recalibrated_quantiles(table, factories[chosen], splits, LEVELS_99, DIST,
                                               config.inner_n_splits, outer_predictions=cv[cv["model"] == chosen],
                                               groups=config.split_groups)
        score_quantile_table(scores, f"{chosen} + recalibration", qtab)
        if chosen != "constant":
            qtab = w.nested_recalibrated_quantiles(table, factories["constant"], splits, LEVELS_99, DIST,
                                                   config.inner_n_splits,
                                                   outer_predictions=cv[cv["model"] == "constant"],
                                                   groups=config.split_groups)
            score_quantile_table(scores, "constant + recalibration", qtab)
        score_references(scores, table)
        score_ensemble(scores, table[["plant_id", "time", "actual_pu"]], plants)
    else:
        test = [plants[p] for p in test_ids]
        unobs = [w.PlantSeries(p.plant_id, p.data.drop(columns="actual"), p.capacity) for p in test]
        ptab = w.prediction_table(unobs, spec)
        actual = pd.concat([pd.DataFrame({"plant_id": p.plant_id, "time": p.data.index,
                                          "actual_pu": p.data["actual"].to_numpy() / p.capacity}) for p in test])
        ptab = ptab.merge(actual, on=["plant_id", "time"], how="inner", validate="one_to_one")
        recal = {m: w.PITRecalibrator(DIST).fit(sub["actual_pu"], sub["forecast_pu"], sub["pred_scale"])
                 for m in {chosen, "constant"} for sub in [cv[cv["model"] == m]]}
        for m in MODELS:
            model = factories[m]().fit(table[feats], table["target_scale"])
            pred = ptab[["plant_id", "time", "forecast_pu", "actual_pu"]].copy()
            pred["pred_scale"] = model.predict(ptab[feats])
            score_parametric(scores, m, pred)
            if m in recal:
                for pid, g in pred.groupby("plant_id", sort=False):
                    q = recal[m].quantiles(g["forecast_pu"].to_numpy(), g["pred_scale"].to_numpy(), LEVELS_99)
                    scores.add(f"{m} + recalibration", pid, g["actual_pu"].to_numpy(), q)
        targets = pd.concat([w.calibration_targets(p, LEVELS, DIST, block=config.block)
                             .assign(plant_id=p.plant_id).rename_axis("time").reset_index() for p in test])
        ref = ptab[["plant_id", "time"]].merge(targets, on=["plant_id", "time"], how="inner", validate="one_to_one")
        score_references(scores, ref)
        score_ensemble(scores, ptab[["plant_id", "time", "actual_pu"]], plants)
    print(f"scoring done ({time.time() - t0:.0f} s)", flush=True)

    tag = args.design
    summary = scores.summary()
    base = float(summary.loc[summary["method"] == "constant", "pinball"].iloc[0])
    summary["skill_vs_constant"] = 1.0 - summary["pinball"] / base
    summary.to_csv(out / f"scores_{tag}.csv", index=False, float_format="%.6g")
    scores.reliability().to_csv(out / f"reliability_{tag}.csv", index=False, float_format="%.6g")
    per_plant = scores.per_plant()
    per_plant["zone"] = per_plant["plant_id"].map(zone)
    per_plant["status"] = per_plant["plant_id"].map(status)
    per_plant.to_csv(out / f"per_plant_{tag}.csv", index=False, float_format="%.6g")
    scale_scores.to_csv(out / f"scale_scores_{tag}.csv", index=False, float_format="%.6g")
    corr = table[feats].corrwith(np.log(table["target_scale"]), method="spearman")
    corr.rename("spearman_with_log_target").rename_axis("feature").reset_index().to_csv(
        out / f"feature_correlation_{tag}.csv", index=False, float_format="%.4g")
    info = {"design": tag, "train_plants": len(train_ids), "test_plants": len(test_ids) or len(train_ids),
            "training_rows": int(len(table)), "folds": len(splits), "features": feats,
            "cv_pinball_unclipped": choice.to_dict(), "chosen_model": chosen,
            "target_scale_median": float(table["target_scale"].median()),
            "config": {"window": spec.window, "pi_history": spec.pi_history, "block": config.block, "dist": DIST,
                       "levels_target": [float(x) for x in LEVELS], "max_train": config.max_train,
                       "inner_n_splits": config.inner_n_splits, "random_state": config.random_state},
            "runtime_s": round(time.time() - t0), "package_version": w.__version__}
    (out / f"design_{tag}.json").write_text(json.dumps(info, indent=2))
    print(summary.to_string(index=False))
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
