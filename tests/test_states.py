import datetime

import pytest

from aifs_local.states import collect_forecasts, collect_run, newest_forecast, parse


def test_parse():
    valid, lead = parse("forecast_20260830T06_+024h.npz")
    assert valid == datetime.datetime(2026, 8, 30, 6)
    assert lead == 24


def test_parse_rejects_other_names():
    with pytest.raises(ValueError):
        parse("state_20260830T06.npz")


def touch(directory, name):
    (directory / name).write_bytes(b"")


def test_collect_run_groups_by_initialisation(tmp_path):
    touch(tmp_path, "forecast_20260830T12_+006h.npz")  # init 06 UTC
    touch(tmp_path, "forecast_20260830T18_+012h.npz")  # init 06 UTC
    touch(tmp_path, "forecast_20260830T12_+012h.npz")  # init 00 UTC
    run = collect_run(tmp_path, datetime.datetime(2026, 8, 30, 6))
    assert list(run) == [6, 12]
    assert run[6].name == "forecast_20260830T12_+006h.npz"


def test_collect_forecasts_groups_by_valid_time(tmp_path):
    touch(tmp_path, "forecast_20260830T12_+006h.npz")
    touch(tmp_path, "forecast_20260830T12_+012h.npz")
    touch(tmp_path, "forecast_20260830T18_+012h.npz")
    forecasts = collect_forecasts(tmp_path, datetime.datetime(2026, 8, 30, 12))
    assert list(forecasts) == [6, 12]


def test_newest_forecast_fails_loudly_on_empty_dir(tmp_path):
    with pytest.raises(SystemExit):
        newest_forecast(tmp_path)
