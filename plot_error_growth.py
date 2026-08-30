"""Plot forecast error growth with lead time.

Collects every forecast state in outputs/ valid at one time (from
initialisations at different lead times), scores each against the open
data analysis, and draws RMSE against lead time as small multiples —
one panel per field, since the units differ.
"""

import argparse
import datetime
import re
from pathlib import Path

import earthkit.data as ekd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from verify_forecast import fetch_truth

# (field, panel title, unit scale factor)
PANELS = [
    ("2t", "2 m temperature (K)", 1.0),
    ("t_850", "850 hPa temperature (K)", 1.0),
    ("msl", "Mean sea-level pressure (hPa)", 0.01),
    ("z_500", "500 hPa geopotential (m²/s²)", 1.0),
]

SERIES = "#2a78d6"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e5e4e0"
SURFACE = "#fcfcfb"


def collect_forecasts(out_dir, valid):
    """Map lead time in hours -> forecast file, for one valid time."""
    stamp = f"{valid:%Y%m%dT%H}"
    forecasts = {}
    for path in out_dir.glob(f"forecast_{stamp}_+*h.npz"):
        match = re.search(r"\+(\d+)h", path.name)
        forecasts[int(match.group(1))] = path
    return dict(sorted(forecasts.items()))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--valid", required=True, help="valid time, e.g. 2026-08-30T06")
    parser.add_argument("--source", default="ecmwf", choices=["ecmwf", "azure", "aws", "google"])
    parser.add_argument("--out-dir", default="outputs", type=Path)
    parser.add_argument("--plot", default=Path("docs/error-growth.png"), type=Path)
    args = parser.parse_args()

    ekd.config.set({"cache-policy": "user"})

    valid = datetime.datetime.fromisoformat(args.valid)
    forecasts = collect_forecasts(args.out_dir, valid)
    if not forecasts:
        raise SystemExit(f"no forecasts valid at {valid} in {args.out_dir}")
    print(f"Scoring {len(forecasts)} forecasts valid {valid} "
          f"(leads: {', '.join(f'{h}h' for h in forecasts)})\n")

    scores = {}
    for param, _, scale in PANELS:
        truth = fetch_truth(param, valid, args.source)
        rmse = {}
        for lead, path in forecasts.items():
            with np.load(path) as npz:
                diff = npz[param].astype(np.float64) - truth
            rmse[lead] = float(np.sqrt(np.nanmean(diff**2))) * scale
        scores[param] = rmse

    print(f"| lead | {' | '.join(p for p, _, _ in PANELS)} |")
    for lead in forecasts:
        row = " | ".join(f"{scores[p][lead]:.2f}" for p, _, _ in PANELS)
        print(f"| +{lead}h | {row} |")

    fig, axes = plt.subplots(2, 2, figsize=(8, 5.6), dpi=150, facecolor=SURFACE)
    for ax, (param, title, _) in zip(axes.flat, PANELS):
        leads = list(scores[param])
        values = [scores[param][lead] for lead in leads]
        ax.set_facecolor(SURFACE)
        ax.plot(leads, values, color=SERIES, linewidth=2, marker="o", markersize=6)
        for lead in (leads[0], leads[-1]):
            ax.annotate(f"{scores[param][lead]:.2f}", (lead, scores[param][lead]),
                        textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=8, color=TEXT)
        ax.set_title(title, fontsize=10, color=TEXT, loc="left")
        ax.set_ylim(bottom=0)
        ax.set_ylim(top=ax.get_ylim()[1] * 1.2)
        ax.set_xticks(leads)
        ax.tick_params(colors=TEXT_2, labelsize=8)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
    for ax in axes[1]:
        ax.set_xlabel("lead time (h)", fontsize=9, color=TEXT_2)
    fig.suptitle(f"AIFS Single 2.0 — global RMSE vs lead time, valid {valid:%Y-%m-%d %H} UTC",
                 fontsize=11, color=TEXT, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    args.plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.plot, facecolor=SURFACE)
    print(f"\nsaved {args.plot}")


if __name__ == "__main__":
    main()
