"""Initial conditions from ERA5 via the CDS API — any date back to 1940.

Same output as initial_conditions.py (a state_*.npz that run_forecast.py
reads), different source, for dates beyond open data's ~4-day window.
Requires a CDS account, an accepted ERA5 licence and `pip install cdsapi`.

ERA5 differences handled here:

- atmosphere arrives on 0.25 deg, ocean waves on 0.5 deg (own regrid);
- longitudes start at 0° here but at -180° in open data — both headers
  are truthful, and to_zero_first() normalises by the header; the
  geography guard below catches any orientation mistake regardless;
- z on pressure levels is served directly (no gh conversion);
- the six wave-period-band heights (h1012..h2530) do not exist anywhere
  in ERA5. They are set to zero over sea. Measured cost of that
  substitution at +24 h: 0.13 K rms on 2t (locally up to 1.6 K), wave
  outputs untrustworthy for the first day or two. See NOTES.md.
"""

import argparse
import datetime
from pathlib import Path

import earthkit.data as ekd
import numpy as np

from aifs_local.initial_conditions import (
    LEVELS,
    fetch_lsm,
    regrid_to_n320,
    save_state,
    summarise,
    to_zero_first,
)

# CDS request names by dataset; fields are read back via GRIB shortNames,
# which match what initial_conditions.py produces from open data.
SFC_VARIABLES = [
    "10m_u_component_of_wind", "10m_v_component_of_wind",
    "2m_dewpoint_temperature", "2m_temperature", "mean_sea_level_pressure",
    "skin_temperature", "surface_pressure", "total_column_water",
    "land_sea_mask", "geopotential", "slope_of_sub_gridscale_orography",
    "standard_deviation_of_orography", "snow_depth",
    "soil_temperature_level_1", "soil_temperature_level_2",
    "volumetric_soil_water_layer_1", "volumetric_soil_water_layer_2",
]
WAVE_VARIABLES = [
    "significant_height_of_combined_wind_waves_and_swell",
    "mean_wave_direction", "mean_wave_period",
    "coefficient_of_drag_with_waves", "model_bathymetry",
]
PL_VARIABLES = ["geopotential", "temperature", "u_component_of_wind",
                "v_component_of_wind", "specific_humidity"]

MISSING_BANDS = ["h1012", "h1214", "h1417", "h1721", "h2125", "h2530"]

def upsample_wave(values):
    """Nearest-neighbour 0.5 deg -> 0.25 deg for the ERA5 wave grid.

    earthkit-regrid ships no 0.5->N320 matrix, so the wave fields go
    through the proven 0.25->N320 path instead. Nearest neighbour keeps
    the NaN coastline intact and invents no values; the resulting 0.25
    deg of positional blur is far below the uncertainty already carried
    by the zeroed period bands.
    """
    if values.shape != (361, 720):
        raise ValueError(f"expected the 0.5 deg wave grid (361, 720), got {values.shape}")
    rows = np.clip(np.round(np.arange(721) / 2).astype(int), 0, 360)
    cols = np.round(np.arange(1440) / 2).astype(int) % 720
    return values[np.ix_(rows, cols)]


def fetch_era5(dataset, variables, dates, target, levels=None):
    """One CDS request covering all needed datetimes; cached on disk."""
    if target.exists():
        print(f"  cached: {target.name}")
        return
    import cdsapi

    request = {
        "product_type": ["reanalysis"],
        "variable": variables,
        "year": sorted({f"{d:%Y}" for d in dates}),
        "month": sorted({f"{d:%m}" for d in dates}),
        "day": sorted({f"{d:%d}" for d in dates}),
        "time": sorted({f"{d:%H}:00" for d in dates}),
        "data_format": "grib",
    }
    if levels:
        request["pressure_level"] = [str(level) for level in levels]
    print(f"  requesting {dataset} ({len(variables)} variables) ...")
    target.parent.mkdir(parents=True, exist_ok=True)
    cdsapi.Client().retrieve(dataset, request, str(target))




def read_fields(path, dates, wave=False):
    """{name: (2, N) array} for the two input dates, regridded to N320."""
    per_date = {i: {} for i in range(len(dates))}
    for f in ekd.from_source("file", str(path)):
        valid = datetime.datetime.strptime(
            f"{f.metadata('validityDate')}{f.metadata('validityTime'):04d}", "%Y%m%d%H%M")
        if valid not in dates:
            continue
        values = to_zero_first(f)
        if wave:
            values = upsample_wave(values)
        values = regrid_to_n320(values)
        level = f.metadata("levelist", default=None)
        name = f"{f.metadata('shortName')}_{level}" if level else f.metadata("shortName")
        per_date[dates.index(valid)][name] = values

    names = per_date[0].keys()
    return {name: np.stack([per_date[i][name] for i in range(len(dates))]) for name in names}


def fetch_fields(date, data_dir, spectra_grib=None):
    dates = [date - datetime.timedelta(hours=6), date]
    era5_dir = data_dir / "era5"
    stamp = f"{date:%Y%m%dT%H}"

    sfc = era5_dir / f"sfc_{stamp}.grib"
    wave = era5_dir / f"wave_{stamp}.grib"
    pl = era5_dir / f"pl_{stamp}.grib"
    fetch_era5("reanalysis-era5-single-levels", SFC_VARIABLES, dates, sfc)
    fetch_era5("reanalysis-era5-single-levels", WAVE_VARIABLES, dates, wave)
    fetch_era5("reanalysis-era5-pressure-levels", PL_VARIABLES, dates, pl,
               levels=LEVELS)

    fields = {}
    fields.update(read_fields(sfc, dates))
    fields.update(read_fields(wave, dates, wave=True))
    fields.update(read_fields(pl, dates))

    # Same transformations as the open data path.
    mwd_rad = np.deg2rad(fields.pop("mwd"))
    fields["cos_mwd"] = np.cos(mwd_rad)
    fields["sin_mwd"] = np.sin(mwd_rad)
    fields.pop("q_10", None)
    fields.pop("q_50", None)

    # ERA5 has no wave-period-band heights. Default: zero over sea (where
    # the wave model is defined), NaN over land like every other wave
    # field. With --spectra-grib they are integrated properly from the
    # archived 2D spectra instead.
    sea = fields["swh"]
    if spectra_grib:
        from aifs_local.spectra import load_bands

        wave_grid_bands = load_bands(spectra_grib, wave, dates)
        for name in MISSING_BANDS:
            n320 = np.stack([regrid_to_n320(upsample_wave(
                np.nan_to_num(wave_grid_bands[name][i])))
                for i in range(len(dates))])
            fields[name] = np.where(np.isnan(sea), np.nan, n320)
    else:
        for name in MISSING_BANDS:
            fields[name] = np.where(np.isnan(sea), np.nan, 0.0)

    lsm_path = fetch_lsm(data_dir / "lsm.grib")
    reference = ekd.from_source("file", str(lsm_path))[0].to_numpy(flatten=True)
    verify_geography(fields, reference)
    mask = np.equal(reference, 0)
    for name in ["sd", "swvl1", "swvl2"]:
        fields[name][:, mask] = np.nan

    return fields


def verify_geography(fields, reference_lsm):
    """Refuse to produce a geographically scrambled state.

    A rotated or flipped grid still yields plausible-looking forecasts —
    the model just simulates a different planet. Comparing the state's
    own land-sea mask against the checkpoint's catches every such
    orientation mistake loudly.
    """
    agreement = np.mean((fields["lsm"][1] > 0.5) == (reference_lsm > 0.5))
    if agreement < 0.99:
        raise SystemExit(
            f"geography check failed: state lsm agrees with the checkpoint lsm at only "
            f"{100 * agreement:.1f}% of points (expected >= 99%) — grid orientation is wrong"
        )
    print(f"geography check: land/sea agreement {100 * agreement:.2f}%")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--date", required=True, help="initial time, ISO format e.g. 2021-07-13T00")
    parser.add_argument("--data-dir", default="data", type=Path)
    parser.add_argument("--spectra-grib", type=Path,
                        help="2D wave spectra GRIB (param 140251) covering both input "
                        "times; reconstructs the wave-band heights instead of zeroing them")
    args = parser.parse_args()

    ekd.config.set({"cache-policy": "user"})
    date = datetime.datetime.fromisoformat(args.date)
    print(f"ERA5 initial conditions for {date}")

    fields = fetch_fields(date, args.data_dir, spectra_grib=args.spectra_grib)
    summarise(fields)
    suffix = "_spectra" if args.spectra_grib else ""
    save_state(fields, date, args.data_dir / f"state_{date:%Y%m%dT%H}{suffix}.npz")


if __name__ == "__main__":
    main()
