"""Reconstruct the wave-period-band heights from ERA5 2D wave spectra.

ERA5 has no h1012..h2530 fields, but it archives the full 2D wave
spectrum (param 140251, MARS/tape only): 24 directions x 30 frequencies
per time, values stored as log10 of the spectral density in
m^2 s rad^-1, missing where the density was below threshold (read: zero)
or over land. The frequency grid is f(n) = 0.03453 * 1.1^(n-1) Hz.

A band height is 4 sqrt(E) with E the density integrated over all
directions and the band's frequency window [1/T2, 1/T1]; frequency bins
take edges at the geometric means of neighbours, and bins straddling a
window edge contribute their overlapping fraction.

The same integration over the whole grid must reproduce ERA5's own swh
field — reconstruct() computes that as a built-in correctness check.
"""

import datetime
from pathlib import Path

import earthkit.data as ekd
import numpy as np

from aifs_local.initial_conditions import to_zero_first

FREQUENCIES = 0.03453 * 1.1 ** np.arange(30)
_EDGE = np.sqrt(1.1)
BIN_LOW = FREQUENCIES / _EDGE
BIN_HIGH = FREQUENCIES * _EDGE
DF = BIN_HIGH - BIN_LOW
DTHETA = 2.0 * np.pi / 24.0

PERIOD_BANDS = {
    "h1012": (10.0, 12.0), "h1214": (12.0, 14.0), "h1417": (14.0, 17.0),
    "h1721": (17.0, 21.0), "h2125": (21.0, 25.0), "h2530": (25.0, 30.0),
}


def band_fractions():
    """Per band: the fraction of each frequency bin inside its window."""
    fractions = {}
    for name, (t_short, t_long) in PERIOD_BANDS.items():
        f_low, f_high = 1.0 / t_long, 1.0 / t_short
        overlap = np.clip(np.minimum(BIN_HIGH, f_high) - np.maximum(BIN_LOW, f_low),
                          0.0, None)
        fractions[name] = overlap / DF
    return fractions


def reconstruct(path, dates):
    """Integrate the spectra file into band heights for the given datetimes.

    Returns ({band: (len(dates), 361, 720) heights in m}, swh of the same
    shape). Land (no spectral bin present at all) is NaN.
    """
    fractions = band_fractions()
    shape = (len(dates), 361, 720)
    band_energy = {name: np.zeros(shape) for name in PERIOD_BANDS}
    total_energy = np.zeros(shape)
    bins_seen = np.zeros(shape, dtype=np.int32)

    used = 0
    for field in ekd.from_source("file", str(path)):
        valid = datetime.datetime.strptime(
            f"{field.metadata('validityDate')}{field.metadata('validityTime'):04d}",
            "%Y%m%d%H%M")
        if valid not in dates:
            continue
        i = dates.index(valid)
        n = int(field.metadata("frequencyNumber")) - 1
        log_density = to_zero_first(field)
        finite = np.isfinite(log_density)
        # missing bins are discarded-as-too-small, i.e. zero energy
        contribution = np.where(finite, 10.0 ** log_density, 0.0) * DF[n] * DTHETA
        total_energy[i] += contribution
        bins_seen[i] += finite
        for name, fraction in fractions.items():
            if fraction[n] > 0.0:
                band_energy[name][i] += contribution * fraction[n]
        used += 1

    expected = len(dates) * 24 * 30
    if used != expected:
        raise SystemExit(f"{path} holds {used} matching spectral fields, expected {expected}")

    land = bins_seen == 0
    heights = {name: np.where(land, np.nan, 4.0 * np.sqrt(energy))
               for name, energy in band_energy.items()}
    swh = np.where(land, np.nan, 4.0 * np.sqrt(total_energy))
    return heights, swh


def check_against_swh(swh_reconstructed, wave_grib, dates):
    """Relative error of the reconstructed swh against ERA5's own field."""
    reference = np.full(swh_reconstructed.shape, np.nan)
    for field in ekd.from_source("file", str(wave_grib)):
        if field.metadata("shortName") != "swh":
            continue
        valid = datetime.datetime.strptime(
            f"{field.metadata('validityDate')}{field.metadata('validityTime'):04d}",
            "%Y%m%d%H%M")
        if valid in dates:
            reference[dates.index(valid)] = to_zero_first(field)

    both = np.isfinite(swh_reconstructed) & np.isfinite(reference) & (reference > 0.1)
    relative = np.abs(swh_reconstructed[both] - reference[both]) / reference[both]
    return float(np.mean(relative)), float(np.median(relative))


def load_bands(spectra_grib, wave_grib, dates):
    """Band heights for era5_conditions, with the swh self-check applied."""
    heights, swh = reconstruct(Path(spectra_grib), dates)
    mean_err, median_err = check_against_swh(swh, wave_grib, dates)
    print(f"spectra check: reconstructed swh vs ERA5 swh — mean relative error "
          f"{100 * mean_err:.1f}%, median {100 * median_err:.1f}%")
    if mean_err > 0.10:
        raise SystemExit("spectral integration disagrees with ERA5 swh by more "
                         "than 10% — refusing to use the reconstructed bands")
    return heights
