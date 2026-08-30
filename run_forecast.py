"""Run an AIFS Single 2.0 forecast from a prepared initial state.

Loads the input state produced by initial_conditions.py, downloads the
checkpoint from Hugging Face on first use (~1 GB, cached), and rolls the
model forward in 6-hour steps. Each output state is written to
outputs/ as a compressed .npz for inspection and plotting.
"""

import argparse
import datetime
import os
import time
from pathlib import Path

import numpy as np

CHECKPOINT = {"huggingface": "ecmwf/aifs-single-2.0"}


def load_state(path):
    with np.load(path) as npz:
        date = datetime.datetime.fromisoformat(str(npz["date"]))
        fields = {name: npz[name] for name in npz.files if name != "date"}
    return dict(date=date, fields=fields)


def newest_state(data_dir):
    states = sorted(data_dir.glob("state_*.npz"))
    if not states:
        raise SystemExit(f"no state_*.npz in {data_dir}; run initial_conditions.py first")
    return states[-1]


def save_state(state, out_dir, step_hours):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"forecast_{state['date']:%Y%m%dT%H}_+{step_hours:03d}h.npz"
    fields = {name: values.astype(np.float32) for name, values in state["fields"].items()}
    np.savez_compressed(
        path,
        date=state["date"].isoformat(),
        latitudes=state["latitudes"].astype(np.float32),
        longitudes=state["longitudes"].astype(np.float32),
        **fields,
    )
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--date", help="fetch initial conditions for this time (ISO format or "
                        "'latest') instead of reading a saved state")
    parser.add_argument("--source", default="ecmwf", choices=["ecmwf", "azure", "aws", "google"])
    parser.add_argument("--state", type=Path, help="input state .npz (default: newest in data/)")
    parser.add_argument("--lead-time", type=int, default=12, help="forecast length in hours")
    parser.add_argument("--device", default=None, help="cuda, cpu or auto (default)")
    parser.add_argument("--precision", default=None, help="e.g. 16 or bf16 (default: checkpoint setting)")
    parser.add_argument("--num-chunks", type=int, default=16,
                        help="split encoder/decoder into chunks to reduce peak memory")
    parser.add_argument("--attention", default="flash", choices=["flash", "sdpa"],
                        help="sdpa switches to PyTorch attention (no flash-attn needed, works on CPU)")
    parser.add_argument("--data-dir", default="data", type=Path)
    parser.add_argument("--out-dir", default="outputs", type=Path)
    parser.add_argument("--raw-dir", type=Path,
                        help="also write states in the raw-output format anemoi-inference "
                        "can read back (see plugin/)")
    args = parser.parse_args()

    # Both variables must be set before the CUDA context is created.
    os.environ.setdefault("ANEMOI_INFERENCE_NUM_CHUNKS", str(args.num_chunks))
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import torch
    from anemoi.inference.outputs.printer import print_state
    from anemoi.inference.runners.simple import SimpleRunner

    if args.date:
        import earthkit.data as ekd

        import initial_conditions as ic

        ekd.config.set({"cache-policy": "user"})
        date = ic.resolve_date(args.date, args.source)
        print(f"Fetching initial conditions for {date} from '{args.source}'")
        fields = ic.fetch_fields(date, args.source, args.data_dir)
        ic.save_state(fields, date, args.data_dir / f"state_{date:%Y%m%dT%H}.npz")
        input_state = dict(date=date, fields=fields)
    else:
        input_state = load_state(args.state or newest_state(args.data_dir))
    print(f"Input state: {input_state['date']}, {len(input_state['fields'])} fields")

    # The checkpoint's stored config selects flash_attention; patching the
    # metadata swaps the wrapper without touching the file.
    patch = {}
    if args.attention == "sdpa":
        patch = {"config": {"model": {"processor": {
            "attention_implementation": "scaled_dot_product_attention"}}}}

    # allow_nans: snow depth and soil moisture are NaN over the ocean by
    # construction (see initial_conditions.py).
    runner = SimpleRunner(
        CHECKPOINT,
        device=args.device,
        precision=args.precision,
        allow_nans=True,
        patch_metadata=patch,
    )

    def write_raw(date, fields, latitudes, longitudes):
        args.raw_dir.mkdir(parents=True, exist_ok=True)
        restate = {f"field_{name}": values for name, values in fields.items()}
        restate["date"] = np.array(str(date), dtype=str)
        np.savez_compressed(args.raw_dir / f"{date:%Y%m%d%H%M%S}.npz",
                            latitudes=latitudes, longitudes=longitudes, **restate)

    last = time.perf_counter()
    for state in runner.run(input_state=input_state, lead_time=args.lead_time):
        step_hours = int((state["date"] - input_state["date"]).total_seconds() // 3600)
        print_state(state)
        path = save_state(state, args.out_dir, step_hours)
        if args.raw_dir:
            if step_hours == 6:
                # The grid only becomes known with the first output state;
                # write the two input dates then, for replay via the raw input.
                for i, lag in enumerate((-6, 0)):
                    write_raw(input_state["date"] + datetime.timedelta(hours=lag),
                              {k: v[i] for k, v in input_state["fields"].items()},
                              state["latitudes"], state["longitudes"])
            # Forecast states carry no static fields (lsm, orography, ...),
            # but a replay needs them at its start date — merge them in.
            constants = {k: v[1] for k, v in input_state["fields"].items()
                         if k not in state["fields"]}
            write_raw(state["date"], {**constants, **state["fields"]},
                      state["latitudes"], state["longitudes"])
        now = time.perf_counter()
        print(f"+{step_hours}h took {now - last:.1f} s -> {path}")
        last = now

    if torch.cuda.is_available():
        print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
