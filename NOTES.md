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

The split is visible in the packaging: anemoi-inference provides the
runner, the registries and generic inputs/outputs (grib, netcdf, raw,
printer); everything ECMWF-specific — the `opendata` input, multio
outputs, polytope — lives in the separate
`anemoi-plugins-ecmwf-inference` package and hooks in via plain
setuptools entry points (groups like `anemoi.inference.inputs`).

Writing a third-party plugin is genuinely one entry point plus one
class: plugin/ in this repo implements a `raw` input (reads the .npz
states the built-in raw output writes) in ~100 lines by subclassing
`EkdInput`, which does the heavy lifting of assembling the two-date
input state. Replaying a stored forecast through the standard CLI
reproduced the original run to within GPU non-determinism.

The replay also makes the variable ledger tangible: forecast states
carry only prognostics + diagnostics, so a replay input additionally
needs the static fields (lsm, orography, ...) merged into its start
state — the runner requests constants from the input separately.

## Attention backends

anemoi-models 0.9.3 has two attention implementations
(`models/layers/attention.py`): `flash_attention` (the default the
checkpoint was trained with, needs the flash-attn CUDA package) and
`scaled_dot_product_attention` (plain PyTorch, CPU-capable). The choice is
read from the model config stored inside the checkpoint
(config/model/processor/attention_implementation), and anemoi-inference's
`patch_metadata` runner argument can override it without touching the
file — that is what `aifs-local run --attention sdpa` does.

The processor itself (from the same metadata): a 16-layer transformer,
16 heads, GELU, sliding-window attention with window 1120 — attention is
local along the grid-point sequence, not global.

Measured on the RTX 3060 for one 6 h step of aifs-single-2.0:
flash-attention 13.4 s at 6.18 GB peak; SDPA 21.5 s at 5.92 GB peak.
So for this checkpoint flash-attn is a ~60% speed advantage, not a
memory requirement — reports of SDPA needing far more memory predate
PyTorch's fused SDPA kernels and were measured on older checkpoints.
Field values agree to within GPU non-determinism.

## Precipitation accounting

The model's tp/cp/sf outputs are per-6-hour-step, not accumulated from
forecast start — verified empirically: the global mean of tp sits near
0.75 mm per step at every lead (~3 mm/day, the textbook global
precipitation rate), instead of growing with lead time. GRIB convention
however is accumulation from forecast start, which is exactly what the
`accumulate_from_start_of_forecast` post-processor in the model card's
inference.yaml exists to produce. Working with raw states means summing
steps yourself (`aifs-local plot --sum-run`).

## The lagged ensemble lesson

Averaging forecasts from successive initialisations (a lagged ensemble)
looked like free skill, but with members of very unequal age the
equal-weight mean beat the +48 h and +72 h members while losing to the
freshest one (z500 RMSE: mean 64.9 vs members 41.1/69.8/122.7). The
classic result — the ensemble mean beating every member — needs members
of comparable skill or lead-dependent weights. Spread grew with the
error (0.22 K mean 2t spread vs 0.74-1.19 K member RMSEs), which is the
signal real ensemble systems calibrate against.

## Hindcasting from ERA5

Initialising the model from ERA5 instead of open data works — with
three lessons attached.

First, the missing inputs: ERA5 contains no wave-period-band heights
(h1012..h2530), anywhere — checked in the CDS forms and the ERA5
documentation's 50-parameter wave table. Zeroing them costs a measured
0.13 K rms on +24 h 2 m temperature (locally up to 1.6 K) and makes the
wave outputs untrustworthy for a day or two; the atmosphere otherwise
keeps full skill (+24 h z500 RMSE vs the analysis: 37 m²/s², the same
band as operational-window runs). Others hit the ERA5-initialisation
wall before (anemoi-inference#278, closed unresolved; a related
soil-representation discussion on the aifs-single-1.0 Hugging Face
page) — the band gap and its measured cost were the missing pieces.

The bands can be done properly: ERA5 archives the full 2D wave spectrum
(24 directions x 30 frequencies, log10-encoded, MARS/tape only), and
integrating it over each band's frequency window reproduces the missing
fields — spectra.py implements this with a built-in check that the
whole-spectrum integral matches ERA5's own swh (1.5% mean error). The
punchline is a lesson in itself: proper bands change the wave forecasts
substantially (swh rms 0.18 m at +24 h) and the atmosphere almost not
at all (2t rms 0.06 K); the Ahr precipitation is identical to the
decimal. For atmospheric case studies the cheap substitution was fine —
but now that is measured, not hoped.

Second, the trap that ate an afternoon: open data GRIBs start their
longitudes at -180°, ERA5-via-CDS GRIBs at 0° — both headers truthful —
and a half-width roll for the former was hiding inside this repo's
regrid helper, which silently rotated the ERA5 planet by 180° (full
post-mortem in FRICTION.md). The rotated states produced
plausible-looking forecasts — sane ranges, a realistic storm track,
normal global-mean precipitation — because the model was simply
simulating a self-consistent wrong Earth. Geographic correctness cannot
be eyeballed and cannot be checked against data that went through the
same code path; the ERA5 fetcher now validates every state's land-sea
mask against the checkpoint's own lsm.grib (an independently oriented
reference) and refuses below 99% agreement.

Third, the payoff: initialised the day before the July 2021 Ahr valley
flood, the model puts 90 mm of event rain where ERA5's own analysis has
100 mm, peak timing correct (see README). Grid-scale, not point-scale —
but from a 12 GB GPU and a reanalysis that did not exist as an input
option for this checkpoint.

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
