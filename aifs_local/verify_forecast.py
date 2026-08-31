"""Verify a forecast state against the analysis for its valid time.

Downloads the same fields from ECMWF open data (step 0 of the cycle at
the forecast's valid time — only possible once that cycle is published)
and reports bias, MAE and RMSE. N320 cells are near-equal-area, so
unweighted means over the grid are a reasonable approximation of
area-weighted scores.
"""

import argparse
import datetime
from pathlib import Path

import earthkit.data as ekd
import numpy as np

from aifs_local.initial_conditions import STANDARD_GRAVITY, regrid_to_n320
from aifs_local.states import newest_forecast

# The fields scored throughout the repository, with the factor from the
# model's own unit to the unit used in every table and figure. One source
# of truth, so a score never means hPa in one place and Pa in another.
PANELS = [
    ("2t", "2 m temperature", 1.0, "K"),
    ("t_850", "850 hPa temperature", 1.0, "K"),
    ("msl", "Mean sea-level pressure", 0.01, "hPa"),
    ("z_500", "500 hPa geopotential", 1.0, "m²/s²"),
]
SCALE = {param: (scale, unit) for param, _, scale, unit in PANELS}


def fetch_truth(param, date, source):
    """Fetch one field at analysis time (step 0) and regrid to N320."""
    name, _, level = param.partition("_")
    request = dict(date=date, source=source)
    if level:
        # z on pressure levels is served as geopotential height (gh).
        request.update(param="gh" if name == "z" else name, levelist=[int(level)])
    else:
        request.update(param=name)

    data = ekd.from_source("ecmwf-open-data", **request)
    fields = [f for f in data]
    if len(fields) != 1:
        raise SystemExit(f"expected one field for {param} at {date}, got {len(fields)}")
    values = regrid_to_n320(fields[0].to_numpy())
    if level and name == "z":
        values = values * STANDARD_GRAVITY
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--forecast", type=Path, help="forecast .npz (default: newest in outputs/)")
    parser.add_argument("--params", default="2t,msl,z_500,t_850",
                        help="comma-separated fields to verify")
    parser.add_argument("--source", default="ecmwf", choices=["ecmwf", "azure", "aws", "google"])
    parser.add_argument("--out-dir", default="outputs", type=Path)
    args = parser.parse_args()

    ekd.config.set({"cache-policy": "user"})

    path = args.forecast or newest_forecast(args.out_dir)
    with np.load(path) as npz:
        valid = datetime.datetime.fromisoformat(str(npz["date"]))
        forecast = {p: npz[p] for p in args.params.split(",")}

    print(f"Verifying {path.name} against open data analysis at {valid}")
    print("One case only — a single valid time says nothing about average skill.\n")
    print(f"{'param':8s} {'unit':>6s} {'bias':>10s} {'mae':>10s} {'rmse':>10s}")
    for param, fc in forecast.items():
        truth = fetch_truth(param, valid, args.source)
        scale, unit = SCALE.get(param, (1.0, ""))
        diff = (fc.astype(np.float64) - truth) * scale
        bias = np.nanmean(diff)
        mae = np.nanmean(np.abs(diff))
        rmse = float(np.sqrt(np.nanmean(diff**2)))
        print(f"{param:8s} {unit:>6s} {bias:10.3f} {mae:10.3f} {rmse:10.3f}")


if __name__ == "__main__":
    main()
