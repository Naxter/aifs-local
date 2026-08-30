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
