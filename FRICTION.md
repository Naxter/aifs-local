# Friction log

Every point where docs were wrong or missing, an error was unhelpful, or a
constraint was undocumented. Kept as a candidate list for upstream issues
and PRs. Dates are when the problem was hit.

## 2026-08-30 — Version pins are not in the model card

The [aifs-single-2.0](https://huggingface.co/ecmwf/aifs-single-2.0) README
names no package versions at all. The actual pins live in three places in
the same repo: `pyproject.toml`, `uv.lock`, and the install cells of
`run_AIFS_v2.0.ipynb` — and they disagree in places (see flash-attn entry).
What I did: took `pyproject.toml` as authoritative, cross-checked against
the notebook. Possible PR: a version table in the model card README.

## 2026-08-30 — Linux-only support is expressed only in uv metadata

The model is unsupported outside Linux, but the only place that says so is
`environments = ["sys_platform == 'linux' ..."]` under `[tool.uv]` in the
model repo's `pyproject.toml`. Nothing in the README or model card prose.
A Windows or macOS user finds out via resolver/install errors. What I did:
chose WSL2/Ubuntu 24.04. Possible PR: one sentence in the model card.

## 2026-08-30 — Notebook promises an SDPA fallback snippet that isn't there

`run_AIFS_v2.0.ipynb` (markdown cell before section 9) says: "The code
snippet below shows how to overwrite a model from a checkpoint to use
SDPA." No such code cell exists in the notebook — the next cell is the
next section heading. The cell only links to
[anemoi-inference#119](https://github.com/ecmwf/anemoi-inference/issues/119),
where the snippet buried in a comment targets the old aifs-single-0.2.1
checkpoint (hard-coded TransformerProcessor arguments that don't match
v2.0). With anemoi-models 0.9.3 the supported route appears to be the
runner's `patch_metadata` on `attention_implementation` instead — to be
verified. Possible PR: fix the notebook, document the modern override.

## 2026-08-30 — Open data retention is undocumented (and bites at t-6h first)

Fetching initial conditions for 2026-08-26 06 UTC (4 days back) failed
with `requests.exceptions.HTTPError: 404 Client Error: Not Found for
url: https://data.ecmwf.int/forecasts/20260826/00z/ifs/0p25/oper/20260826000000-0h-oper-fc.index`.
Note the cycle: the 06z data may still have existed, but AIFS needs the
t-6h state too, so the 00z file rolling out of retention ends the run.
The ecmwf-opendata README documents no retention period at all; in
practice it is a ~4-day rolling window, effectively 6 hours shorter for
AIFS initialisation. Possible PR: one sentence on retention in the
ecmwf-opendata README. Error-growth experiments beyond ~3.75 days need
archived initial conditions (MARS) instead.

## 2026-08-30 — The model card's own CLI path is broken with its own pins

`anemoi-inference run` with `input: opendata` (the setup in the model
repo's `inference.yaml`) crashes on construction with `KeyError: 'z'`
in `anemoi/transform/filters/orog_to_z.py`. Root cause:
`InferenceOrography` (anemoi-plugins-ecmwf-inference 0.2.1,
`opendata/geopotential_height.py`) overrides
`optional_inputs = {"orography": "gh", "geopotential": "z"}`, but the
parent `Orography` filter in the pinned anemoi-transform 0.1.16.post2
has `optional_inputs = {"orog": "orog", "z": "z"}` and its
`forward_select`/`backward_select` read `self.orog`/`self.z` — attribute
names from a different anemoi-transform version. The subclass also
rejects `orog`/`z` as explicit kwargs ("Unknown input(s)"), so there is
no config-level workaround. The two packages pinned together by the
model card's `pyproject.toml` cannot run the `input: opendata` path; the
example notebook avoids it by converting gh to z manually in Python.
Worked around by feeding the model from stored raw states instead.
Issue candidate with exact versions and traceback.

## 2026-08-30 — Built-in raw output: shorthand config always crashes

With anemoi-inference 0.8.3, the natural config

```yaml
output:
  raw: data/raw
```

fails with `TypeError: RawOutput.__init__() missing 1 required positional
argument: 'dir'`. Cause: `RawOutput` is decorated with
`@main_argument("path")` but its required constructor argument is named
`dir`, so the shorthand scalar never reaches it. Workaround: spell it as
`raw: {dir: data/raw}`. One-line fix upstream
(`@main_argument("dir")` in `outputs/raw.py`) — PR candidate.

## 2026-08-30 — The official example input plugin's test cannot pass

`plugins/inference/inputs/example/tests/test_plugin.py` in
ecmwf/anemoi-plugins calls `create_input(TestingContext(), "example")`.
Two independent problems: `TestingContext` is an empty class while
`Input.__init__` reads `context.checkpoint` (AttributeError), and
`ExamplePlugin` implements none of `Input`'s abstract methods, so it
cannot be instantiated at all. Hit while modelling my own plugin test on
it; had to stub a checkpoint with a `default_namer()` and implement the
abstract methods to get a passing test. PR candidate.

## 2026-08-30 — earthkit-plots papercuts (v0.6.1)

- The deprecation warning in `quickmap` says "will be removed in
  earthkit-plots 0.4" — in a package released as 0.6.1. Meanwhile the
  readthedocs "latest" documents the 1.x API, but the model card pins
  `earthkit-plots<1`, so the docs describe a different library than the
  one installed. Reading the installed source was faster than the docs.
- `Figure().add_map(domain=...)` returns `None` (the subplot is only
  queued), so the natural next line fails with
  `AttributeError: 'NoneType' object has no attribute 'pcolormesh'`.
  Workaround: `Figure(rows=1, columns=1)` makes `add_map` return the
  subplot. The library's own quickmap module works around the same
  behaviour by calling the private `figure._release_queue()`.
- Plotting the N320 point cloud via `pcolormesh(x=..., y=..., z=...)`
  works, but emits "pcolormesh failed with raw data, attempting
  interpolation to structured grid" plus a NaN re-interpolation warning
  — for what appears to be the intended code path for unstructured data.

## 2026-08-30 — flash-attn: pinned version and installed version disagree

`pyproject.toml` declares `flash-attn==2.7.4`, but its `[tool.uv]` sources
section silently substitutes a prebuilt `2.8.3` wheel from a personal
GitHub release (`cathalobrien/get-flash-attn`). The notebook's Colab cell
installs yet another version (`2.7.4.post1`, compiled from source). So
three different flash-attn versions are referenced for the same checkpoint,
and the wheel's provenance is not explained anywhere. What I did: used the
prebuilt 2.8.3 wheel (it is what ECMWF's own uv setup resolves to) and
noted the discrepancy here.
