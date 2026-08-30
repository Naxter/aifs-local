"""Plot a field from a forecast state on a map.

Reads the .npz states written by run_forecast.py. The N320 grid is
unstructured from a plotting perspective, so earthkit-plots interpolates
the point cloud onto a regular grid internally.
"""

import argparse
import datetime
import re
from pathlib import Path

import numpy as np
from earthkit.plots import Figure

# Fields stored in Kelvin that read better in Celsius on a map.
KELVIN_FIELDS = {"2t", "2d", "skt", "stl1", "stl2"} | {f"t_{lev}" for lev in (
    1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50, 10)}


def newest_forecast(out_dir):
    files = sorted(out_dir.glob("forecast_*.npz"))
    if not files:
        raise SystemExit(f"no forecast_*.npz in {out_dir}; run run_forecast.py first")
    return files[-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--file", type=Path, help="forecast .npz (default: newest in outputs/)")
    parser.add_argument("--param", default="2t")
    parser.add_argument("--domain", default="Europe")
    parser.add_argument("--sum-run", action="store_true",
                        help="sum the field over all steps of the run up to this one "
                        "(for per-step fields like tp: accumulated precipitation)")
    parser.add_argument("--out-dir", default="outputs", type=Path)
    args = parser.parse_args()

    path = args.file or newest_forecast(args.out_dir)
    with np.load(path) as npz:
        date = datetime.datetime.fromisoformat(str(npz["date"]))
        latitudes = npz["latitudes"]
        longitudes = npz["longitudes"]
        if args.param not in npz.files:
            raise SystemExit(f"{args.param} not in {path.name}; available: {sorted(npz.files)}")
        values = npz[args.param]

    unit = ""
    accumulated = ""
    if args.sum_run:
        from animate_forecast import collect_run

        lead = int(re.search(r"\+(\d+)h", path.name).group(1))
        init = date - datetime.timedelta(hours=lead)
        values = np.zeros_like(values)
        for step, file in collect_run(args.out_dir, init).items():
            if step <= lead:
                with np.load(file) as npz:
                    values = values + npz[args.param]
        accumulated = f"0-{lead}h "
        if args.param in {"tp", "cp", "sf"}:
            values = values * 1000
            unit = " (mm)"
    elif args.param in KELVIN_FIELDS:
        values = values - 273.15
        unit = " (°C)"

    # With an unsized Figure, add_map only queues the subplot and returns
    # None; an explicit 1x1 layout makes it return the subplot directly.
    figure = Figure(rows=1, columns=1)
    subplot = figure.add_map(domain=args.domain)
    subplot.pcolormesh(x=longitudes, y=latitudes, z=values)
    subplot.coastlines()
    subplot.borders()
    subplot.legend(label=f"{accumulated}{args.param}{unit}")
    subplot.title(f"AIFS Single 2.0: {accumulated}{args.param} valid {date:%Y-%m-%d %H} UTC")

    suffix = "_sum" if args.sum_run else ""
    out = args.out_dir / f"{args.param}{suffix}_{args.domain.lower()}_{date:%Y%m%dT%H}.png"
    figure.save(str(out))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
