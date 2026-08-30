# Notes

Working notes on how AIFS and the anemoi stack actually fit together,
written while getting a forecast to run locally. Sections get filled in as
each part is verified against real data and the real checkpoint — nothing
here is copied from docs without checking.

## The input state

Verified by downloading the 2026-08-30 06 UTC cycle: the input is a dict
of 97 named fields, each an array of shape (2, 542080) — two times
(t-6h, t0) by 542,080 N320 grid points. No channel tensor, no image.

Composition:

- 13 surface fields: winds (10u/10v), 2 m temperature and dewpoint
  (2t/2d), mean sea-level and surface pressure (msl/sp), skin
  temperature (skt), total column water (tcw), plus static fields:
  land-sea mask (lsm), orography as geopotential (z), and slope/std of
  sub-grid orography (slor/sdor), snow depth (sd).
- 12 wave fields (swh, mwp, spectral bands h1012..h2530, wmb, cdww, and
  mean wave direction as cos_mwd/sin_mwd — an angle is discontinuous at
  360°, so it is fed as its sine and cosine).
- 4 soil fields, two levels each: temperature (stl1/stl2) and moisture
  (swvl1/swvl2).
- 68 pressure-level fields: z, t, u, v on 14 levels
  (1000..10 hPa), q on 12 (specific humidity at 10 and 50 hPa is not a
  prognostic of the model).

Everything is in raw physical units: Kelvin, Pa, kg/kg, m/s, and
geopotential in m^2/s^2 (height times g = 9.80665). Normalisation is the
model's job, not the data pipeline's (see below).

NaNs are meaningful, not missing data: wave fields are NaN over land
(~30% of points), snow depth and soil moisture are NaN over the ocean
(~69%, applied deliberately via the checkpoint's lsm.grib mask). The
runner is told allow_nans=True.

One trap: pressure-level values exist even below the ground. t_1000 has
a minimum of 236 K (-37°C) — over high terrain like Antarctica the
1000 hPa surface is underground and values are extrapolated by the
producing system.

## The grid: why not a rectangular image

A regular lat/lon grid oversamples the poles: meridians converge, so a
0.25° cell at 80°N is far smaller than one at the equator. A reduced
Gaussian grid (ECMWF's N320 here) keeps cell areas roughly equal by using
fewer points per latitude ring near the poles. The numbers make the point:
open data's regular 0.25° grid is 721 x 1440 = 1,038,240 points, N320
covers the same sphere with 542,080 — half the memory for the same
effective resolution (~31 km).

Consequence for anyone with image intuitions: the state is not a (H, W)
raster but a flat 1D array of grid points, and every field is a vector
indexed by grid point, not a 2D tensor. Neighbourhood structure lives in
a precomputed graph (anemoi-graphs), not in convolution kernels. Plotting
needs the latitude/longitude arrays and triangulation (or a library that
understands the grid) — there is no reshape back to an image.

## Normalisation

Normalisation lives inside the model, not in the data pipeline. The
checkpoint's metadata carries the full training config plus per-variable
statistics of the training dataset; anemoi-models instantiates an
InputNormalizer from them as a layer of the model interface. Inputs are
normalised on the way in, outputs de-normalised on the way out — every
state dict on the outside is in plain physical units (Kelvin, Pa).

It is not one scheme for all 134 variables
(config/data/processors/normalizer in the metadata):

- default: mean-std (z-score) for most fields;
- max-scaling for the static orography fields (z, sdor, slor);
- none for fields that are already bounded or encoded: sin/cos forcings,
  cloud fractions, soil moisture, snow, the land-sea mask;
- std-only (no mean shift) for precipitation-like fields (tp, cp, sf,
  ro, ...) — zero must stay exactly zero for sparse fields;
- remap: convective precipitation (cp) and snowfall (sf) borrow total
  precipitation's statistics instead of their own.

## The variable ledger

The checkpoint knows 134 variables; we supply 97; forecasts contain 120.
The arithmetic (all from the checkpoint metadata):

- 92 prognostic variables: predicted each step and fed back in rollout;
- 5 constant fields supplied with the input (lsm, z, sdor, slor, wmb) —
  the "coupled forcings not supported" warning from SimpleRunner refers
  to these, and is harmless because they never change;
- 14 computed forcings the runner generates itself per step (solar
  geometry: sin/cos of latitude, longitude, julian day, local time,
  insolation) — this is why a forecast needs no further downloads;
- 28 diagnostic variables: produced as output only (precipitation,
  cloud cover, 100 m winds, ...). 92 + 28 = the 120 fields per output
  state; diagnostics have no t0 input, so the model can output variables
  it never sees.

## 6-hour steps and rollout

The checkpoint maps (t-6h, t0) -> t+6h in one forward pass
(config/data/timestep = 6h, multistep_input = 2 in the metadata). Longer
forecasts are rollout: feed the prediction back as the new t0. Errors
compound; this is the ML analogue of a numerical model's timestep, chosen
at training time and baked into the checkpoint.

Measured on an RTX 3060 (12 GB), flash-attention,
ANEMOI_INFERENCE_NUM_CHUNKS=16: first step 117 s (checkpoint load and
setup included), subsequent steps ~13 s, peak GPU memory 6.2 GB. A
10-day forecast (40 steps) is roughly a 10-minute job on this card.

## anemoi framework vs. ECMWF plumbing

*To fill in as the pieces are used.* First data point: the model repo's
`inference.yaml` uses `input: opendata`, which is provided not by
anemoi-inference itself but by the separate `anemoi-plugins-ecmwf-inference`
package — the framework/plumbing split is visible right in the packaging.

## Attention backends

anemoi-models 0.9.3 has two attention implementations
(`models/layers/attention.py`): `flash_attention` (the default the
checkpoint was trained with, needs the flash-attn CUDA package) and
`scaled_dot_product_attention` (plain PyTorch, CPU-capable). The choice is
read from the model config stored inside the checkpoint
(config/model/processor/attention_implementation), and anemoi-inference's
`patch_metadata` runner argument can override it without touching the
file — that is what `run_forecast.py --attention sdpa` does.

The processor itself (from the same metadata): a 16-layer transformer,
16 heads, GELU, sliding-window attention with window 1120 — attention is
local along the grid-point sequence, not global.

Measured on the RTX 3060 for one 6 h step of aifs-single-2.0:
flash-attention 13.4 s at 6.18 GB peak; SDPA 21.5 s at 5.92 GB peak.
So for this checkpoint flash-attn is a ~60% speed advantage, not a
memory requirement — reports of SDPA needing far more memory predate
PyTorch's fused SDPA kernels and were measured on older checkpoints.
Field values agree to within GPU non-determinism.

## Where image/tensor intuitions break

Running list:

1. No (H, W) spatial axes — the grid is an unstructured 1D point cloud with
   graph connectivity (see grid section).
2. NaN is a value, not a bug — it encodes "this variable does not exist
   here" (waves on land, soil moisture at sea). Nothing imputes them.
3. No normalisation in the data pipeline — fields arrive in Kelvin and
   Pascals spanning five orders of magnitude across variables.
4. "Channels" have per-variable identities and physics (an angle arrives
   as sin/cos, a height became a geopotential) — there is no uniform RGB-
   like treatment across fields.
5. Some values are fictional by construction: pressure surfaces below
   terrain are extrapolated, not observed (see t_1000 in Antarctica).
