"""Figures of the ERCOT case study from the result tables written by run.py (needs matplotlib).

- ``fig_skill.png``        pinball-loss skill of every method against the constant scale, per design
- ``fig_reliability.png``  reliability curves of the main methods, per design
- ``fig_plants.png``       per-plant skill of the recalibrated surrogate (design ``plants10``) against the plant's
                           forecast bias (``forecast_summary.csv`` from prepare.py, copied to RESULTS_DIR)

The chosen surrogate can differ between designs; its name is given in each panel title.

Usage:  python figures.py RESULTS_DIR
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

CHOSEN = "chosen surrogate + recalibration"

DESIGNS = {"plants10": "held-out plants\n(10 plant folds)", "zones": "held-out weather zones",
           "proposed": "existing -> proposed plants"}
COLORS = {"constant": "#9e9e9e", "ridge": "#bdbdbd", "svr": "#90a4ae", "random_forest": "#78909c",
          "gpr": "#546e7a", "recal": "#1565c0", "raw ensemble": "#ef6c00", "plant constant (in-sample)": "#8e24aa",
          "daily target (in-sample)": "#2e7d32"}


def color(method: str) -> str:
    if method == "constant + recalibration":
        return "#90caf9"
    return COLORS["recal"] if method.endswith("+ recalibration") else COLORS.get(method, "#333333")


def order(methods) -> list[str]:
    base = ["constant", "ridge", "svr", "random_forest", "gpr"]
    recal = sorted((m for m in methods if m.endswith("+ recalibration")), key=lambda m: m.startswith("constant"))
    refs = ("raw ensemble", "plant constant (in-sample)", "daily target (in-sample)")
    return [m for m in base if m in methods] + recal + [m for m in refs if m in methods]


def relabel(methods, chosen: str):
    """Name the chosen surrogate's recalibrated row the same way in every design."""
    return [CHOSEN if m == f"{chosen} + recalibration" and chosen != "constant" else m for m in methods]


def main() -> int:
    res = Path(sys.argv[1])
    designs = [d for d in DESIGNS if (res / f"scores_{d}.csv").exists()]
    chosen = {d: json.loads((res / f"design_{d}.json").read_text())["chosen_model"] for d in designs}
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

    fig, axes = plt.subplots(1, len(designs), figsize=(2.7 * len(designs) + 2.0, 3.6), sharey=True, squeeze=False)
    for ax, d in zip(axes[0], designs):
        s = pd.read_csv(res / f"scores_{d}.csv").set_index("method")
        ms = order(s.index)
        y = 100 * s.loc[ms, "skill_vs_constant"].to_numpy()
        ax.barh(range(len(ms)), y, color=[color(m) for m in ms])
        ax.set_yticks(range(len(ms)), relabel(ms, chosen[d]))
        ax.axvline(0, color="black", lw=0.6)
        ax.set_title(f"{DESIGNS[d]}\nchosen surrogate: {chosen[d]}", fontsize=8)
        ax.set_xlabel("pinball skill vs constant scale (%)")
        ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(res / "fig_skill.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(designs), figsize=(3.2 * len(designs), 3.2), sharey=True, squeeze=False)
    for ax, d in zip(axes[0], designs):
        r = pd.read_csv(res / f"reliability_{d}.csv")
        keep = [m for m in order(r["method"].unique())
                if m in ("constant", "raw ensemble") or m.endswith("+ recalibration")]
        for m in keep:
            g = r[r["method"] == m]
            ax.plot(g["level"], g["observed"], marker="o", ms=2.5, lw=1.2, color=color(m),
                    label=relabel([m], chosen[d])[0])
        ax.plot([0, 1], [0, 1], color="black", lw=0.6, ls=":")
        ax.set_title(f"{DESIGNS[d]}\nchosen surrogate: {chosen[d]}", fontsize=8)
        ax.set_xlabel("nominal level")
        ax.set_aspect("equal")
    axes[0][0].set_ylabel("observed share below the quantile")
    axes[0][-1].legend(fontsize=7, loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(res / "fig_reliability.png", dpi=150)
    plt.close(fig)

    if (res / "per_plant_plants10.csv").exists() and (res / "forecast_summary.csv").exists():
        p = pd.read_csv(res / "per_plant_plants10.csv")
        wide = p.pivot(index="plant_id", columns="method", values="pinball")
        bias = pd.read_csv(res / "forecast_summary.csv").set_index("plant_id")["bias_pu"]
        rec = f"{chosen['plants10']} + recalibration"
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        for base, col, lab in (("constant", COLORS["recal"], f"{CHOSEN} vs constant"),
                               ("constant + recalibration", "#90caf9", f"{CHOSEN} vs constant + recalibration")):
            skill = 100 * (1 - wide[rec] / wide[base])
            ax.scatter(100 * bias.loc[skill.index], skill, s=8, color=col, label=lab, zorder=3)
        ax.axhline(0, color="black", lw=0.6)
        ax.axvline(0, color="black", lw=0.6, ls=":")
        ax.set_xlabel("plant's forecast bias: mean(forecast - actual), % of capacity")
        ax.set_ylabel("per-plant pinball skill (%)")
        ax.set_title(f"{DESIGNS['plants10'].replace(chr(10), ' ')}; chosen surrogate: {chosen['plants10']}",
                     fontsize=8)
        ax.legend(fontsize=7, frameon=False, loc="upper right")
        fig.tight_layout()
        fig.savefig(res / "fig_plants.png", dpi=150)
        plt.close(fig)
    print("figures written to", res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
