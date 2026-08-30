"""Download and prepare AIFS initial conditions from ECMWF open data.

AIFS Single 2.0 is initialised with two consecutive atmospheric states
(t-6h and t0). ECMWF open data serves fields on a regular 0.25 deg
lat/lon grid; the model expects its native N320 reduced Gaussian grid,
so every field is interpolated after download.

The parameter lists and transformations mirror the example notebook in
the ecmwf/aifs-single-2.0 model repository.
"""

import argparse
import datetime
import urllib.request
from collections import defaultdict
from pathlib import Path

import earthkit.data as ekd
import earthkit.regrid as ekr
import numpy as np
from ecmwf.opendata import Client

PARAM_SFC = ["10u", "10v", "2d", "2t", "msl", "skt", "sp", "tcw", "lsm", "z", "slor", "sdor", "sd"]
PARAM_SOIL = ["vsw", "sot"]
PARAM_WAVE = ["wmb", "h1012", "h1214", "h1417", "h1721", "h2125", "h2530", "mwd", "cdww", "mwp", "swh"]
PARAM_PL = ["gh", "t", "u", "v", "q"]
LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50, 10]
SOIL_LEVELS = [1, 2]

# Land-sea mask shipped with the checkpoint; used to blank land-only
# fields over the ocean.
LSM_URL = "https://huggingface.co/ecmwf/aifs-single-2.0/resolve/main/lsm.grib"

STANDARD_GRAVITY = 9.80665


def get_open_data(date, source, param, levelist=(), **kwargs):
    """Fetch one parameter set for t-6h and t0 and regrid to N320."""
    fields = defaultdict(list)
    for step_date in [date - datetime.timedelta(hours=6), date]:
        data = ekd.from_source(
            "ecmwf-open-data", date=step_date, param=param, levelist=list(levelist),
            source=source, **kwargs,
        )
        for f in data:
            # Open data longitudes run -180..180; the regrid matrices
            # expect 0..360, hence the half-width roll.
            values = f.to_numpy()
            assert values.shape == (721, 1440), f"unexpected grid {values.shape}"
            values = np.roll(values, -values.shape[1] // 2, axis=1)
            values = ekr.interpolate(values, {"grid": (0.25, 0.25)}, {"grid": "N320"})
            name = f"{f.metadata('param')}_{f.metadata('levelist')}" if levelist else f.metadata("param")
            fields[name].append(values)

    return {name: np.stack(values) for name, values in fields.items()}


def fetch_lsm(path):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(LSM_URL, path)
    return path


def fetch_fields(date, source, data_dir):
    """Download all input fields and apply the checkpoint's transformations."""
    fields = {}

    print("Downloading surface fields ...")
    fields.update(get_open_data(date, source, param=PARAM_SFC, levtype="sfc"))
    missing = set(PARAM_SFC) - set(fields)
    assert not missing, f"missing surface parameters: {missing}"

    print("Downloading wave fields ...")
    fields.update(get_open_data(date, source, param=PARAM_WAVE, stream="wave"))
    missing = set(PARAM_WAVE) - set(fields)
    assert not missing, f"missing wave parameters: {missing}"

    print("Downloading soil fields ...")
    soil = get_open_data(date, source, param=PARAM_SOIL, levelist=SOIL_LEVELS)
    soil_names = [f"{p}_{lev}" for p in PARAM_SOIL for lev in SOIL_LEVELS]
    missing = set(soil_names) - set(soil)
    assert not missing, f"missing soil parameters: {missing}"

    print("Downloading pressure-level fields ...")
    fields.update(get_open_data(date, source, param=PARAM_PL, levelist=LEVELS))
    pl_names = [f"{p}_{lev}" for p in PARAM_PL for lev in LEVELS]
    missing = set(pl_names) - set(fields)
    assert not missing, f"missing pressure-level parameters: {missing}"

    # Mean wave direction is an angle; the model was trained on its
    # sine/cosine components.
    mwd_rad = np.deg2rad(fields.pop("mwd"))
    fields["cos_mwd"] = np.cos(mwd_rad)
    fields["sin_mwd"] = np.sin(mwd_rad)

    # Open data serves soil fields under different names than the
    # training data used.
    mapping = {"sot_1": "stl1", "sot_2": "stl2", "vsw_1": "swvl1", "vsw_2": "swvl2"}
    for src, dst in mapping.items():
        fields[dst] = soil[src]

    # Specific humidity at 10 and 50 hPa is not a prognostic variable of
    # the model.
    fields.pop("q_10", None)
    fields.pop("q_50", None)

    # Snow depth and soil moisture are only defined over land.
    lsm_path = fetch_lsm(data_dir / "lsm.grib")
    mask = np.equal(ekd.from_source("file", str(lsm_path))[0].to_numpy(flatten=True), 0)
    for name in ["sd", "swvl1", "swvl2"]:
        fields[name][:, mask] = np.nan

    # Open data provides geopotential height (gh, in m); the model was
    # trained on geopotential (z, in m^2 s^-2).
    for level in LEVELS:
        fields[f"z_{level}"] = fields.pop(f"gh_{level}") * STANDARD_GRAVITY

    return fields


def summarise(fields):
    print(f"\n{len(fields)} fields, shape {next(iter(fields.values())).shape} (2 dates x grid points)")
    for name in sorted(fields):
        arr = fields[name]
        print(f"  {name:12s} min {np.nanmin(arr):14.4f}  max {np.nanmax(arr):14.4f}  NaNs {np.isnan(arr).sum():7d}")


def save_state(fields, date, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, date=date.isoformat(), **fields)
    print(f"\nSaved input state to {path} ({path.stat().st_size / 1e6:.0f} MB)")


def resolve_date(spec, source):
    if spec == "latest":
        return Client(source).latest()
    return datetime.datetime.fromisoformat(spec)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--date", default="latest",
                        help="initial time, ISO format e.g. 2026-08-29T12 (default: latest cycle)")
    parser.add_argument("--source", default="ecmwf", choices=["ecmwf", "azure", "aws", "google"])
    parser.add_argument("--data-dir", default="data", type=Path)
    args = parser.parse_args()

    ekd.config.set({"cache-policy": "user"})

    date = resolve_date(args.date, args.source)
    print(f"Initial conditions for {date} from source '{args.source}'")

    fields = fetch_fields(date, args.source, args.data_dir)
    summarise(fields)
    save_state(fields, date, args.data_dir / f"state_{date:%Y%m%dT%H}.npz")


if __name__ == "__main__":
    main()
