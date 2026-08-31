"""Finding the forecast states written by run_forecast.py.

States are named forecast_<valid time>_+<lead>h.npz, which makes both
questions answerable from the filename alone: which states belong to one
run (same initialisation), and which forecasts are valid at one time.
"""

import datetime
import re
from pathlib import Path

NAME = re.compile(r"forecast_(\d{8}T\d{2})_\+(\d+)h")


def parse(path):
    """(valid time, lead hours) from a state's filename."""
    match = NAME.match(Path(path).name)
    if not match:
        raise ValueError(f"not a forecast state filename: {path}")
    return datetime.datetime.strptime(match.group(1), "%Y%m%dT%H"), int(match.group(2))


def collect_run(out_dir, init):
    """Lead hours -> file for every state of the run started at `init`."""
    frames = {}
    for path in Path(out_dir).glob("forecast_*_+*h.npz"):
        valid, lead = parse(path)
        if valid - datetime.timedelta(hours=lead) == init:
            frames[lead] = path
    return dict(sorted(frames.items()))


def collect_forecasts(out_dir, valid):
    """Lead hours -> file for every forecast valid at `valid`."""
    forecasts = {}
    for path in Path(out_dir).glob(f"forecast_{valid:%Y%m%dT%H}_+*h.npz"):
        forecasts[parse(path)[1]] = path
    return dict(sorted(forecasts.items()))


def newest_forecast(out_dir):
    files = sorted(Path(out_dir).glob("forecast_*.npz"))
    if not files:
        raise SystemExit(f"no forecast_*.npz in {out_dir}; run run_forecast.py first")
    return files[-1]
