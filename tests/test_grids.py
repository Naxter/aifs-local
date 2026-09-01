import numpy as np
import pytest

pytest.importorskip("earthkit.regrid")
pytest.importorskip("ecmwf.opendata")

from aifs_local.era5_conditions import upsample_wave, verify_geography
from aifs_local.initial_conditions import regrid_to_n320
from aifs_local.meteogram import nearest_point


def test_upsample_wave_shape_and_nearest_mapping():
    source = np.arange(361, dtype=float)[:, None] * np.ones((361, 720))
    out = upsample_wave(source)
    assert out.shape == (721, 1440)
    # every output row holds the value of its nearest source row
    assert out[2, 0] == source[1, 0]
    assert out[720, 0] == source[360, 0]


def test_upsample_wave_preserves_nan_coastline():
    source = np.zeros((361, 720))
    source[0, 0] = np.nan
    out = upsample_wave(source)
    assert np.isnan(out[0, 0])
    assert np.isnan(out).sum() >= 1
    # NaN stays local: the far side of the grid is untouched
    assert not np.isnan(out[400:, :]).any()


def test_upsample_wave_rejects_wrong_grid():
    with pytest.raises(ValueError):
        upsample_wave(np.zeros((721, 1440)))


def test_regrid_rejects_wrong_grid():
    with pytest.raises(ValueError):
        regrid_to_n320(np.zeros((361, 720)))


def test_verify_geography_accepts_matching_mask(capsys):
    rng = np.random.default_rng(0)
    reference = (rng.random(1000) > 0.7).astype(float)
    fields = {"lsm": np.stack([reference, reference])}
    verify_geography(fields, reference)
    assert "geography check" in capsys.readouterr().out


def test_verify_geography_rejects_rotated_planet():
    rng = np.random.default_rng(0)
    reference = (rng.random(1000) > 0.7).astype(float)
    rotated = np.roll(reference, 500)
    fields = {"lsm": np.stack([rotated, rotated])}
    with pytest.raises(SystemExit):
        verify_geography(fields, reference)


def test_nearest_point_picks_closest():
    lats = np.array([0.0, 10.0, 20.0])
    lons = np.array([0.0, 10.0, 20.0])
    assert nearest_point(lats, lons, 11.0, 9.0) == 1


def test_nearest_point_handles_date_line_wrap():
    lats = np.array([0.0, 0.0])
    lons = np.array([359.75, 180.0])
    assert nearest_point(lats, lons, 0.0, -0.1) == 0


class _FakeField:
    def __init__(self, values, first_lon):
        self._values = values
        self._first = first_lon

    def to_numpy(self):
        return self._values

    def metadata(self, key):
        assert key == "longitudeOfFirstGridPointInDegrees"
        return self._first


def test_to_zero_first_keeps_zero_origin():
    from aifs_local.initial_conditions import to_zero_first

    values = np.arange(8, dtype=float).reshape(2, 4)
    out = to_zero_first(_FakeField(values, 0.0))
    assert np.array_equal(out, values)


def test_to_zero_first_rolls_180_origin():
    from aifs_local.initial_conditions import to_zero_first

    values = np.arange(8, dtype=float).reshape(2, 4)
    out = to_zero_first(_FakeField(values, 180.0))
    assert np.array_equal(out[0], [2.0, 3.0, 0.0, 1.0])


def test_to_zero_first_rejects_odd_origins():
    from aifs_local.initial_conditions import to_zero_first

    with pytest.raises(ValueError):
        to_zero_first(_FakeField(np.zeros((2, 4)), 90.0))
