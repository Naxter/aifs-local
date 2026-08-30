# aifs-local

![ci](https://github.com/Naxter/aifs-local/actions/workflows/ci.yml/badge.svg)

Run ECMWF's AIFS Single 2.0 machine-learning weather model on a consumer
GPU, end to end: initial conditions from ECMWF open data, forecast with
anemoi-inference, plots with earthkit.

![2m temperature over Europe](docs/example-2t-europe.png)

## Usage

One command from date to forecast (first run downloads the ~1 GB
checkpoint from Hugging Face):

```sh
python run_forecast.py --date latest --lead-time 24
```

Or step by step:

```sh
python initial_conditions.py --date 2026-08-30T06
python run_forecast.py --lead-time 12
python plot_forecast.py --param 2t --domain Europe
```

Forecast states are written to `outputs/` as compressed .npz (values,
latitudes, longitudes per 6 h step). Useful flags on `run_forecast.py`:
`--device cpu`, `--attention sdpa` (runs without flash-attn),
`--precision 16`, `--num-chunks N` for tighter memory.

On an RTX 3060 (12 GB) a 6 h step takes ~13 s at ~6 GB peak GPU memory.

See NOTES.md for how the pieces fit together and FRICTION.md for the
sharp edges hit along the way.

## Requirements

- Linux x86_64 (developed under WSL2 / Ubuntu 24.04), Python 3.12
- NVIDIA GPU, Ampere or newer, or CPU fallback
- ~5 GB disk for the environment, ~2 GB for checkpoint and data

## Setup

```sh
python3.12 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
pip install --no-deps "flash-attn @ https://github.com/cathalobrien/get-flash-attn/releases/download/v0.1-alpha/flash_attn-2.8.3+cu12torch2.7cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
```

The flash-attn wheel is the prebuilt one referenced by the
[aifs-single-2.0](https://huggingface.co/ecmwf/aifs-single-2.0) model repo;
it avoids compiling from source. Without a CUDA GPU, skip it — the runner
can fall back to PyTorch's built-in attention (see NOTES.md).

## Data attribution

Initial conditions come from [ECMWF open data](https://www.ecmwf.int/en/forecasts/datasets/open-data)
(CC BY 4.0). The model checkpoint is © ECMWF, also CC BY 4.0.

## License

MIT — see [LICENSE](LICENSE). This is an independent project, not
affiliated with or endorsed by ECMWF.
