"""Compare local AIFS forecasts with ECMWF's operational IFS and AIFS.

For one valid time, scores three forecast sources against the same open
data analysis: the local AIFS runs (outputs/), ECMWF's operational IFS
(the physics model), and ECMWF's operational AIFS. Note the home
advantage: the analysis is produced by the IFS system, which flatters
IFS scores slightly.
"""

import argparse
import datetime
import tempfile
from pathlib import Path

import earthkit.data as ekd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ecmwf.opendata import Client

from initial_conditions import STANDARD_GRAVITY, regrid_to_n320
from plot_error_growth import collect_forecasts
from verify_forecast import fetch_truth

PANELS = [
    ("2t", "2 m temperature (K)", 1.0),
    ("t_850", "850 hPa temperature (K)", 1.0),
    ("msl", "Mean sea-level pressure (hPa)", 0.01),
    ("z_500", "500 hPa geopotential (m²/s²)", 1.0),
]

# Fixed entity -> colour assignment (never re-ordered).
SOURCES = [
    ("local AIFS", "#2a78d6"),
    ("ECMWF AIFS", "#1baf7a"),
    ("ECMWF IFS", "#eb6834"),
]

TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e5e4e0"
SURFACE = "#fcfcfb"


def fetch_operational(model, param, init, lead, source):
    """Fetch one field of an operational forecast and regrid to N320."""
    name, _, level = param.partition("_")
    request = dict(
        date=f"{init:%Y%m%d}", time=init.hour, type="fc", step=lead,
        param="gh" if (level and name == "z") else name,
    )
    if level:
        request["levelist"] = int(level)

    client = Client(source=source, model=model)
    with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as tmp:
        client.retrieve(target=tmp.name, **request)
        fields = list(ekd.from_source("file", tmp.name))
        assert len(fields) == 1, f"expected one field, got {len(fields)}"
        values = regrid_to_n320(fields[0].to_numpy())
    Path(tmp.name).unlink()
    if level and name == "z":
        values = values * STANDARD_GRAVITY
    return values


def rmse(forecast, truth):
    diff = forecast.astype(np.float64) - truth
    return float(np.sqrt(np.nanmean(diff**2)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--valid", required=True, help="valid time, e.g. 2026-08-30T06")
    parser.add_argument("--source", default="ecmwf", choices=["ecmwf", "azure", "aws", "google"])
    parser.add_argument("--out-dir", default="outputs", type=Path)
    parser.add_argument("--plot", default=Path("docs/model-comparison.png"), type=Path)
    args = parser.parse_args()

    ekd.config.set({"cache-policy": "user"})
    valid = datetime.datetime.fromisoformat(args.valid)
    local = collect_forecasts(args.out_dir, valid)
    if not local:
        raise SystemExit(f"no local forecasts valid at {valid}")
    leads = list(local)
    print(f"Valid {valid}, leads: {', '.join(f'{h}h' for h in leads)}\n")

    # scores[param][source_name][lead] = rmse
    scores = {p: {name: {} for name, _ in SOURCES} for p, _, _ in PANELS}
    for param, _, scale in PANELS:
        truth = fetch_truth(param, valid, args.source)
        for lead in leads:
            init = valid - datetime.timedelta(hours=lead)
            with np.load(local[lead]) as npz:
                scores[param]["local AIFS"][lead] = rmse(npz[param], truth) * scale
            for model, name in (("aifs-single", "ECMWF AIFS"), ("ifs", "ECMWF IFS")):
                values = fetch_operational(model, param, init, lead, args.source)
                scores[param][name][lead] = rmse(values, truth) * scale

    for param, title, _ in PANELS:
        print(title)
        for name, _ in SOURCES:
            row = "  ".join(f"+{lead}h {scores[param][name][lead]:8.2f}" for lead in leads)
            print(f"  {name:11s} {row}")

    fig, axes = plt.subplots(2, 2, figsize=(8, 5.8), dpi=150, facecolor=SURFACE)
    for ax, (param, title, _) in zip(axes.flat, PANELS):
        ax.set_facecolor(SURFACE)
        for name, color in SOURCES:
            values = [scores[param][name][lead] for lead in leads]
            ax.plot(leads, values, color=color, linewidth=2, marker="o",
                    markersize=5, label=name)
        ax.set_title(title, fontsize=10, color=TEXT, loc="left")
        ax.set_ylim(bottom=0)
        ax.set_ylim(top=ax.get_ylim()[1] * 1.15)
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
    axes[0, 0].legend(fontsize=8, frameon=False, labelcolor=TEXT_2)
    fig.suptitle(f"RMSE vs the analysis, valid {valid:%Y-%m-%d %H} UTC "
                 "(IFS analysis — home advantage for IFS)",
                 fontsize=10, color=TEXT, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    args.plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.plot, facecolor=SURFACE)
    print(f"\nsaved {args.plot}")


if __name__ == "__main__":
    main()
