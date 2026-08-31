"""Poor man's ensemble: combine forecasts from successive initialisations.

All forecasts valid at one time (from different init times) form a
lagged ensemble. Reports each member's RMSE against the analysis, the
ensemble mean's RMSE, and the mean spread — the cheap version of why
ensembles exist: averaging out uncorrelated errors.
"""

import argparse
import datetime
from pathlib import Path

import earthkit.data as ekd
import numpy as np

from aifs_local.states import collect_forecasts
from aifs_local.verify_forecast import fetch_truth


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--valid", required=True, help="valid time, e.g. 2026-08-30T06")
    parser.add_argument("--params", default="2t,msl,z_500,t_850")
    parser.add_argument("--source", default="ecmwf", choices=["ecmwf", "azure", "aws", "google"])
    parser.add_argument("--out-dir", default="outputs", type=Path)
    args = parser.parse_args()

    ekd.config.set({"cache-policy": "user"})
    valid = datetime.datetime.fromisoformat(args.valid)
    members = collect_forecasts(args.out_dir, valid)
    if len(members) < 2:
        raise SystemExit(f"need at least two forecasts valid {valid}, found {len(members)}")
    print(f"Lagged ensemble valid {valid}: {len(members)} members "
          f"(leads {', '.join(f'{h}h' for h in members)})\n")

    def rmse(diff):
        return float(np.sqrt(np.nanmean(diff**2)))

    for param in args.params.split(","):
        truth = fetch_truth(param, valid, args.source)
        fields = {}
        for lead, path in members.items():
            with np.load(path) as npz:
                fields[lead] = npz[param].astype(np.float64)

        stack = np.stack(list(fields.values()))
        mean_rmse = rmse(stack.mean(axis=0) - truth)
        spread = float(np.nanmean(stack.std(axis=0)))
        best = min(rmse(f - truth) for f in fields.values())

        print(param)
        for lead, f in fields.items():
            print(f"  member +{lead:3d}h   rmse {rmse(f - truth):8.3f}")
        beats = "beats every member" if mean_rmse < best else "does not beat the best member"
        print(f"  ensemble mean  rmse {mean_rmse:8.3f}   ({beats})")
        print(f"  mean spread         {spread:8.3f}\n")


if __name__ == "__main__":
    main()
