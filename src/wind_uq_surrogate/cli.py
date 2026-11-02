"""Command line: ``wind-uq-surrogate synthetic | metrics | run`` (see README, "Command line")."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from .features import FeatureSpec, SpatialSpec, build_features
from .io import read_plants, split_observed, write_plants
from .probabilistic import available_coverages, default_levels
from .surrogate import SURROGATES
from .synthetic import synthetic_fleet
from .workflow import WorkflowConfig, run_workflow, save_result


def _pairs(values):
    out = []
    for v in values or []:
        if ":" not in v:
            raise argparse.ArgumentTypeError(f"pair {v!r} must look like reference:source")
        a, b = v.split(":", 1)
        out.append((a, b))
    return tuple(out)


def _spec(args) -> FeatureSpec:
    spatial = None
    if args.spatial_bins:
        edges = tuple(float(x) for x in args.spatial_bins.split(","))
        spatial = SpatialSpec(bin_edges=edges, model=args.spatial_model)
    return FeatureSpec(window=args.window, single_series=tuple(args.single or ["forecast"]), pairs=_pairs(args.pair),
                       member_prefix=args.members, pi_history=args.pi_history, spatial=spatial,
                       pair_normalizer=args.pair_normalizer)


def _add_feature_args(p):
    p.add_argument("--window", type=int, required=True, help="trailing window length in time steps")
    p.add_argument("--single", action="append", help="column for single-series metrics (repeatable; default forecast)")
    p.add_argument("--pair", action="append", help="reference:source columns for agreement metrics (repeatable)")
    p.add_argument("--pair-normalizer", default="range", help="'range', 'mean' or a positive number")
    p.add_argument("--members", default=None, help="column prefix of ensemble members")
    p.add_argument("--pi-history", type=int, default=None, help="history (steps) of the predictability index")
    p.add_argument("--spatial-bins", default=None, help="comma-separated lag-bin edges for the turbine variogram")
    p.add_argument("--spatial-model", default="exponential", choices=["exponential", "gaussian", "spherical"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="wind-uq-surrogate", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("synthetic", help="write an input directory with invented plants")
    s.add_argument("out_dir")
    s.add_argument("--plants", type=int, default=8)
    s.add_argument("--unobserved", type=int, default=2)
    s.add_argument("--steps", type=int, default=24 * 42)
    s.add_argument("--members", type=int, default=10)
    s.add_argument("--turbines", type=int, default=0)
    s.add_argument("--signal", type=float, default=1.0)
    s.add_argument("--seed", type=int, default=0)

    m = sub.add_parser("metrics", help="write the feature (UQ-metric) table of every plant")
    m.add_argument("in_dir")
    m.add_argument("--out-dir", required=True)
    _add_feature_args(m)

    r = sub.add_parser("run", help="calibrate, validate the surrogates, predict unobserved plants")
    r.add_argument("in_dir")
    r.add_argument("--out-dir", required=True)
    _add_feature_args(r)
    r.add_argument("--dist", default="laplace", choices=["laplace", "gaussian"])
    r.add_argument("--level-step", type=float, default=0.05, help="spacing of the quantile levels")
    r.add_argument("--block", type=int, default=None, help="one target scale per block of steps")
    r.add_argument("--target-window", type=int, default=None, help="one target per step from a centred window (odd)")
    r.add_argument("--models", default=",".join(SURROGATES))
    r.add_argument("--n-splits", type=int, default=None, help="plant folds (default: leave one plant out)")
    r.add_argument("--gpr-max-train", type=int, default=1500)
    r.add_argument("--no-recalibrate", action="store_true")

    args = ap.parse_args(argv)
    if getattr(args, "pair_normalizer", None) not in (None, "range", "mean"):
        args.pair_normalizer = float(args.pair_normalizer)

    if args.cmd == "synthetic":
        fleet = synthetic_fleet(n_plants=args.plants, n_unobserved=args.unobserved, n_steps=args.steps,
                                n_members=args.members, n_turbines=args.turbines, signal=args.signal, seed=args.seed)
        root = write_plants(args.out_dir, fleet.plants)
        fleet.truth.to_csv(Path(root) / "synthetic_truth.csv", index=False)
        print(f"wrote {len(fleet.plants)} invented plants ({len(fleet.unobserved)} unobserved) to {root}")
        return 0

    plants = read_plants(args.in_dir)
    spec = _spec(args)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.cmd == "metrics":
        for p in plants:
            feats = build_features(p, spec)
            feats.insert(0, "time", feats.index)
            feats.to_csv(out / f"features_{p.plant_id}.csv", index=False)
        print(f"wrote feature tables of {len(plants)} plants to {out}")
        return 0

    observed, unobserved = split_observed(plants)
    levels = default_levels(args.level_step)
    config = WorkflowConfig(features=spec, levels=tuple(levels), dist=args.dist,
                            coverages=available_coverages(levels, (0.5, 0.8, 0.9)),
                            block=args.block, target_window=args.target_window, models=tuple(m.strip() for m in args.models.split(",") if m.strip()),
                            gpr_max_train=args.gpr_max_train, n_splits=args.n_splits,
                            recalibrate=not args.no_recalibrate)
    result = run_workflow(observed, unobserved, config)
    save_result(result, out, config)
    with pd.option_context("display.width", 140, "display.max_columns", 20):
        print(result.predictive_scores.round(5).to_string(index=False))
    print(f"chosen surrogate: {result.chosen_model}; outputs in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
