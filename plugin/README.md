# anemoi-inference-input-raw-plugin

An input plugin for [anemoi-inference](https://anemoi.readthedocs.io/projects/inference/):
reads the compressed `.npz` state files written by the built-in `raw`
output, so a forecast can be replayed or continued from stored states
without re-fetching the original source data.

```yaml
# write states with the built-in output ...
output:
  raw: data/raw

# ... and read them back as input in another run
input:
  raw:
    path: data/raw
```

The file layout matches the `raw` output: one `{date}.npz` per state,
fields stored as `field_{name}` plus `date`, `latitudes` and
`longitudes`. To start a forecast at date t0, files for every input lag
of the checkpoint must exist (t0 and t0-6h for AIFS Single).

Install with `pip install -e plugin` from the repository root.
