"""Meteogram: one location's forecast as time-series panels.

Extracts the nearest N320 grid point from every state of one forecast
run and draws 2 m temperature, wind speed, 6-hourly precipitation and
mean sea-level pressure against valid time.
"""

import argparse
import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from aifs_local.plotstyle import SERIES, SURFACE, TEXT, style_axis
from aifs_local.states import collect_run


def nearest_point(latitudes, longitudes, lat, lon):
    """Index of the grid point closest to (lat, lon), great-circle."""
    lat_r, lon_r = np.deg2rad(lat), np.deg2rad(lon % 360)
    lats_r, lons_r = np.deg2rad(latitudes), np.deg2rad(longitudes % 360)
    # Haversine central angle; fine at this resolution.
    a = (np.sin((lats_r - lat_r) / 2) ** 2
         + np.cos(lat_r) * np.cos(lats_r) * np.sin((lons_r - lon_r) / 2) ** 2)
    return int(np.argmin(a))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--init", required=True, help="initialisation time, e.g. 2026-08-30T06")
    parser.add_argument("--lat", type=float, default=50.73)
    parser.add_argument("--lon", type=float, default=7.10)
    parser.add_argument("--place", default="Bonn")
    parser.add_argument("--out-dir", default="outputs", type=Path)
    parser.add_argument("--plot", default=None, type=Path)
    args = parser.parse_args()

    init = datetime.datetime.fromisoformat(args.init)
    files = collect_run(args.out_dir, init)
    if not files:
        raise SystemExit(f"no forecast states initialised {init} in {args.out_dir}")

    with np.load(files[min(files)]) as npz:
        point = nearest_point(npz["latitudes"], npz["longitudes"], args.lat, args.lon)
        grid_lat = float(npz["latitudes"][point])
        grid_lon = float(npz["longitudes"][point])
    grid_lon = grid_lon - 360 if grid_lon > 180 else grid_lon
    print(f"{args.place} ({args.lat}, {args.lon}) -> grid point {point} "
          f"({grid_lat:.2f}, {grid_lon:.2f})")

    times, t2m, wind, tp, msl = [], [], [], [], []
    for lead, path in files.items():
        with np.load(path) as npz:
            times.append(init + datetime.timedelta(hours=lead))
            t2m.append(npz["2t"][point] - 273.15)
            wind.append(float(np.hypot(npz["10u"][point], npz["10v"][point])))
            tp.append(npz["tp"][point] * 1000)
            msl.append(npz["msl"][point] / 100)

    panels = [
        ("2 m temperature (°C)", t2m, "line"),
        ("10 m wind speed (m/s)", wind, "line"),
        ("precipitation (mm / 6 h)", tp, "bar"),
        ("mean sea-level pressure (hPa)", msl, "line"),
    ]
    fig, axes = plt.subplots(4, 1, figsize=(8, 7.5), dpi=150,
                             sharex=True, facecolor=SURFACE)
    for ax, (title, values, kind) in zip(axes, panels):
        if kind == "bar":
            ax.bar(times, values, width=0.2, color=SERIES)
            ax.set_ylim(bottom=0)
        else:
            ax.plot(times, values, color=SERIES, linewidth=2)
        style_axis(ax, title)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    fig.suptitle(f"{args.place} — AIFS Single 2.0, init {init:%Y-%m-%d %H} UTC",
                 fontsize=11, color=TEXT, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    plot = args.plot or args.out_dir / f"meteogram_{args.place.lower()}_{init:%Y%m%dT%H}.png"
    plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot, facecolor=SURFACE)
    print(f"saved {plot}")


if __name__ == "__main__":
    main()
