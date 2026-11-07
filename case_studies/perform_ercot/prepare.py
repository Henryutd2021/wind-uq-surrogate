"""Build the case-study input directory from the downloaded PERFORM files (see download.py).

Writes, under OUT_DIR:

- ``inputs/plants.csv``        plant_id, capacity_mw, group (ERCOT weather zone), status (Existing or Proposed),
                               latitude, longitude, perform_site (the PERFORM site name)
- ``inputs/series/<id>.csv``   time (UTC, hourly), forecast (mean of the 51 day-ahead members, MW),
                               fcst_2da (mean of the 2-day-ahead members, MW), ens_00 ... ens_50 (MW),
                               actual (hourly simulated power, MW)
- ``alignment.csv``            per plant: mean absolute difference between the hourly ensemble-mean forecast and
                               the hourly actuals under each time-label convention
- ``forecast_summary.csv``     per plant, per unit of capacity: mean actual and forecast, bias (forecast - actual),
                               mean absolute error and correlation of the ensemble-mean forecast, largest actual

Plant ids w001 ... w264 follow the alphabetical order of the normalized PERFORM site names. The time-label
convention of the hourly actuals is the one with the smallest median difference over all plants (printed and
stored in ``prepare_summary.json``) unless --label is given.

Usage:  python prepare.py RAW_DIR OUT_DIR [--label beginning|ending|instant] [--workers 16]
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from wind_uq_surrogate.perform import LABELS, alignment_errors, parse_site_file, plant_from_files, read_actuals, \
    read_forecasts

FOLDERS = {
    ("actuals", False): "ERCOT/2018/Wind/Actuals/Site_level",
    ("day-ahead_fcst", True): "ERCOT/2018/ECMWF/Wind_51MemberForecasts/Day_ahead/Site_level",
    ("2day-ahead_fcst", True): "ERCOT/2018/ECMWF/Wind_51MemberForecasts/2Day_ahead/Site_level",
}
NAMES = {("actuals", False): "act", ("day-ahead_fcst", True): "ens", ("2day-ahead_fcst", True): "ens2"}


def site_files(raw: Path) -> pd.DataFrame:
    """One row per site key with the path of each of the three files."""
    rows = {}
    for kind, folder in FOLDERS.items():
        for path in sorted((raw / folder).glob("*.h5")):
            info = parse_site_file(path.name)
            if info is None or (info["kind"], info["ecmwf"]) != kind or info["year"] != 2018:
                continue
            slot = rows.setdefault(info["key"], {})
            if NAMES[kind] in slot:
                raise ValueError(f"two files for site {info['key']} in {folder}")
            slot[NAMES[kind]] = path
    table = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    complete = table.dropna()
    if len(complete) < len(table):
        print(f"sites without all three files (skipped): {sorted(set(table.index) - set(complete.index))}")
    return complete


def check_alignment(row) -> dict:
    act, _ = read_actuals(row["act"])
    ens, _, _ = read_forecasts(row["ens"])
    return alignment_errors(act, ens.mean(axis=1))


def build(args) -> dict:
    pid, row, label, out = args
    plant, act_meta = plant_from_files(pid, row["act"], row["ens"], row["ens2"], label=label)
    _, _, fmeta = read_forecasts(row["ens"])
    series = plant.data.copy()
    series.insert(0, "time", series.index)
    series.to_csv(out / "inputs" / "series" / f"{pid}.csv", index=False, float_format="%.6g")
    a = plant.data["actual"] / plant.capacity
    f = plant.data["forecast"] / plant.capacity
    stats = {"mean_actual_pu": a.mean(), "mean_forecast_pu": f.mean(), "bias_pu": (f - a).mean(),
             "mae_pu": (f - a).abs().mean(), "corr": float(np.corrcoef(f, a)[0, 1]), "max_actual_pu": a.max()}
    return {"plant_id": pid, "capacity_mw": plant.capacity, "group": act_meta.get("Region", ""),
            "status": act_meta.get("Group", ""), "latitude": act_meta.get("latitude"),
            "longitude": act_meta.get("longitude"), "perform_site": fmeta.get("Facility.Name", ""),
            "hours": len(plant.data), **stats}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raw_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--label", choices=LABELS)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    raw, out = Path(args.raw_dir), Path(args.out_dir)
    files = site_files(raw)
    ids = {key: f"w{n:03d}" for n, key in enumerate(files.index, 1)}
    print(f"{len(files)} sites with all three files")
    with ProcessPoolExecutor(args.workers) as pool:
        errs = list(pool.map(check_alignment, [files.loc[k] for k in files.index]))
    align = pd.DataFrame(errs, index=[ids[k] for k in files.index]).rename_axis("plant_id").reset_index()
    out.mkdir(parents=True, exist_ok=True)
    align.to_csv(out / "alignment.csv", index=False)
    medians = align[list(LABELS)].median().to_dict()
    label = args.label or min(medians, key=medians.get)
    print(f"median absolute difference by time-label convention (MW): {medians}; using {label!r}")
    (out / "inputs" / "series").mkdir(parents=True, exist_ok=True)
    jobs = [(ids[k], files.loc[k], label, out) for k in files.index]
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(build, jobs))
    plants = pd.DataFrame(rows)
    stats = ["mean_actual_pu", "mean_forecast_pu", "bias_pu", "mae_pu", "corr", "max_actual_pu"]
    plants[["plant_id", "status", "group"] + stats].to_csv(out / "forecast_summary.csv", index=False,
                                                           float_format="%.5g")
    plants.drop(columns=["hours"] + stats).to_csv(out / "inputs" / "plants.csv", index=False)
    summary = {"label": label, "median_alignment_mw": medians, "plants": len(plants),
               "status": plants["status"].value_counts().to_dict(), "groups": plants["group"].value_counts().to_dict(),
               "hours_min": int(plants["hours"].min()), "hours_max": int(plants["hours"].max()),
               "median_over_plants": {k: float(plants[k].median()) for k in stats}}
    (out / "prepare_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
