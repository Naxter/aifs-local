"""The aifs-local command: one entry point, one subcommand per tool.

Each subcommand dispatches to a module's main() and passes the remaining
arguments through untouched, so `aifs-local run --lead-time 72` behaves
exactly like the module invoked directly. Imports happen per command:
`aifs-local --help` must not pay for torch or matplotlib.
"""

import importlib
import sys

COMMANDS = {
    "fetch": ("initial_conditions", "initial conditions from ECMWF open data"),
    "era5": ("era5_conditions", "initial conditions from ERA5 via CDS (any date back to 1940)"),
    "run": ("run_forecast", "run the model"),
    "plot": ("plot_forecast", "map a field (--sum-run for accumulated precipitation)"),
    "verify": ("verify_forecast", "score one forecast against the analysis"),
    "growth": ("plot_error_growth", "RMSE vs lead time across initialisations"),
    "compare": ("compare_models", "local AIFS vs operational IFS and AIFS"),
    "meteogram": ("meteogram", "one location's forecast as time-series panels"),
    "animate": ("animate_forecast", "a whole run as a GIF"),
    "ensemble": ("lagged_ensemble", "ensemble statistics from successive runs"),
    "daily": ("daily_verification", "scheduler-friendly daily run + scoring"),
}


def usage():
    lines = ["usage: aifs-local <command> [options]", "", "commands:"]
    for name, (_, help_text) in COMMANDS.items():
        lines.append(f"  {name:10s} {help_text}")
    lines.append("")
    lines.append("aifs-local <command> --help shows that command's options.")
    return "\n".join(lines)


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(usage())
        return
    command = sys.argv[1]
    if command not in COMMANDS:
        print(f"unknown command: {command}\n\n{usage()}", file=sys.stderr)
        raise SystemExit(2)

    module = importlib.import_module(f"aifs_local.{COMMANDS[command][0]}")
    sys.argv = [f"aifs-local {command}"] + sys.argv[2:]
    module.main()


if __name__ == "__main__":
    main()
