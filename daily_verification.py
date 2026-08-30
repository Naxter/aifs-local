"""Daily driver: run the latest forecast and score matured ones.

Meant for a scheduler (cron, Windows Task Scheduler). Each run:

1. runs a 24 h forecast from the latest open data cycle, unless its
   output already exists;
2. scores every +24 h forecast whose analysis is available, and appends
   the results to scores.csv (one row per field; already-scored rows are
   skipped, so re-running is safe).

scores.csv is tracked in git — commit it whenever you like; a few weeks
of rows turn single-case verification into averaged scores.
"""

import csv
import datetime
import re
import subprocess
import sys
from pathlib import Path

import earthkit.data as ekd
import numpy as np
from ecmwf.opendata import Client

from verify_forecast import fetch_truth

PARAMS = ["2t", "t_850", "msl", "z_500"]
SCORES = Path("scores.csv")
FIELDS = ["valid", "init", "lead_h", "param", "bias", "mae", "rmse"]


def existing_rows():
    if not SCORES.exists():
        return set()
    with SCORES.open() as f:
        return {(row["valid"], row["lead_h"], row["param"]) for row in csv.DictReader(f)}


def append_rows(rows):
    new_file = not SCORES.exists()
    with SCORES.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerows(rows)


def main():
    ekd.config.set({"cache-policy": "user"})
    out_dir = Path("outputs")

    latest = Client("ecmwf").latest()
    target = out_dir / f"forecast_{latest + datetime.timedelta(hours=24):%Y%m%dT%H}_+024h.npz"
    if target.exists():
        print(f"forecast for {latest} already exists, skipping run")
    else:
        print(f"running 24 h forecast from {latest}")
        subprocess.run([sys.executable, "run_forecast.py",
                        "--date", latest.isoformat(), "--lead-time", "24"], check=True)

    done = existing_rows()
    now = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
    for path in sorted(out_dir.glob("forecast_*_+024h.npz")):
        valid = datetime.datetime.strptime(
            re.match(r"forecast_(\d{8}T\d{2})", path.name).group(1), "%Y%m%dT%H")
        if valid > now:
            continue
        missing = [p for p in PARAMS if (valid.isoformat(), "24", p) not in done]
        if not missing:
            continue
        rows = []
        try:
            for param in missing:
                truth = fetch_truth(param, valid, "ecmwf")
                with np.load(path) as npz:
                    diff = npz[param].astype(np.float64) - truth
                rows.append(dict(
                    valid=valid.isoformat(), init=(valid - datetime.timedelta(hours=24)).isoformat(),
                    lead_h="24", param=param,
                    bias=f"{np.nanmean(diff):.4f}", mae=f"{np.nanmean(np.abs(diff)):.4f}",
                    rmse=f"{np.sqrt(np.nanmean(diff ** 2)):.4f}",
                ))
        except Exception as error:
            print(f"skipping {path.name}: {error}")
            continue
        append_rows(rows)
        print(f"scored {path.name} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
