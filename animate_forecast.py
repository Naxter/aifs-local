"""Render a forecast run as an animated GIF.

Collects the forecast states of one run (same initialisation time) from
outputs/ and draws one global map per step. The colour scale is fixed
across frames; for temperature it diverges around 0 °C.
"""

import argparse
import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from states import collect_run


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--init", required=True, help="initialisation time, e.g. 2026-08-30T06")
    parser.add_argument("--param", default="2t")
    parser.add_argument("--every", type=int, default=12, help="hours between frames")
    parser.add_argument("--out-dir", default="outputs", type=Path)
    parser.add_argument("--gif", default=None, type=Path)
    args = parser.parse_args()

    init = datetime.datetime.fromisoformat(args.init)
    frames = {lead: path for lead, path in collect_run(args.out_dir, init).items()
              if lead % args.every == 0}
    if not frames:
        raise SystemExit(f"no forecast states initialised {init} in {args.out_dir}")
    print(f"{len(frames)} frames (+{min(frames)}h .. +{max(frames)}h)")

    first = np.load(frames[min(frames)])
    latitudes = first["latitudes"]
    longitudes = np.where(first["longitudes"] > 180,
                          first["longitudes"] - 360, first["longitudes"])
    triangulation = tri.Triangulation(longitudes, latitudes)

    celsius = args.param == "2t"
    fig, ax = plt.subplots(figsize=(8, 4.4), dpi=90,
                           subplot_kw={"projection": ccrs.PlateCarree()})
    levels = np.linspace(-40, 40, 17) if celsius else 16
    label = f"{args.param} (°C)" if celsius else args.param
    contour = None
    colorbar_drawn = False

    def draw(lead):
        nonlocal contour, colorbar_drawn
        for coll in ax.collections:
            coll.remove()
        with np.load(frames[lead]) as npz:
            values = npz[args.param] - 273.15 if celsius else npz[args.param]
            valid = datetime.datetime.fromisoformat(str(npz["date"]))
        contour = ax.tricontourf(triangulation, values, levels=levels,
                                 cmap="RdBu_r", extend="both",
                                 transform=ccrs.PlateCarree())
        ax.coastlines(linewidth=0.5)
        ax.set_title(f"AIFS Single 2.0: {args.param}  init {init:%Y-%m-%d %H} UTC  "
                     f"+{lead:03d}h  valid {valid:%Y-%m-%d %H} UTC", fontsize=9)
        if not colorbar_drawn:
            fig.colorbar(contour, ax=ax, orientation="vertical", shrink=0.8, label=label)
            colorbar_drawn = True
        print(f"  frame +{lead:03d}h")

    animation = FuncAnimation(fig, draw, frames=list(frames), interval=400)
    gif = args.gif or Path("outputs") / f"{args.param}_{init:%Y%m%dT%H}.gif"
    gif.parent.mkdir(parents=True, exist_ok=True)
    animation.save(gif, writer=PillowWriter(fps=2.5))
    print(f"saved {gif} ({gif.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
