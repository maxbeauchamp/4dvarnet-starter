# WGVarFM — WeatherGenerator × Variational Flow Matching

Coupling of the WeatherGenerator (WG) latent representation with CROSCIM-style
flow-matching reconstruction, inside this repo's Lightning/Hydra environment.

Target chain (long term):

```
Y_t → WG encoder → z_t → WG forecast → z^b → WG TargetPredictionEngine → (h^b, X^b)
    → FM-4DVarNet coarse (v_ψ(X_s, s, h^b, ∇J)) → CROSCIM patchwise FM (medium → fine)
```

Two configurations:

1. **WG pre-training** (`config/xp/WGVarFM/wg_pretrain_croscim_sic.yaml`):
   self-supervised masked reconstruction on the raw sensor files, one stream per
   sensor at its own resolution. *(implemented)*
2. **CROSCIM conditioned on the frozen WG latent** / features `h^WG`. *(next)*

## Layout

```
contrib/WGVarFM/
├── weathergen_ext/        # vendored WG subset (see below)
├── tokenization.py        # points (lat, lon, values) -> WG source tokens / target points, cell masking
├── batch.py               # minimal ModelBatch stand-in + collate
├── data.py                # generic WGDataset / WGDataModule (reader -> masked tokens)
├── readers/croscim.py     # raw CROSCIM daily NetCDF reader (one sensor = one stream)
├── models/lit_wgfm.py     # LitWGFM: generic LightningModule around WG `Model`
└── tests/                 # standalone smoke tests
```

### Data pipeline

- A **stream** = one sensor (`asip_sic`, `cimr_SIC`, …) or covariate group. Streams
  are defined once in the xp config (`streams:`) and shared by reader, tokenizer
  and model.
- The reader returns, per stream and per day, the **valid points at the sensor's
  own grid**: no interpolation onto a common grid. `coarsen: f` applies an
  optional NaN-aware f×f block mean (e.g. ASIP 500 m → 2.5 km); positions are
  averaged on the unit sphere.
- Points are grouped by HEALPix cell and split into tokens of `token_size` points
  (per stream: denser sensors use larger tokens).
- **Masking** (WG pre-training): for each sample and each stream with
  `source: true` and `target: true`, a fraction `masking_rate` of its non-empty
  HEALPix cells is removed from the input and becomes the target. Streams with
  `target: false` (covariates) are input only. Masks are random for train and
  seeded by sample index for val/test.
- Loss: MSE between prediction (ensemble mean) and target points, averaged over
  target streams, in normalized units.

## `weathergen_ext/` — vendored WeatherGenerator subset

- Source: https://github.com/ecmwf/WeatherGenerator, branch `develop`,
  commit `1719b8e` (2026-09-18). Apache-2.0, see `weathergen_ext/LICENSE` and
  `weathergen_ext/NOTICE`.
- Copied verbatim: `src/weathergen/model/{attention,blocks,embeddings,encoder,engines,layers,model,norms,parametrised_prob_dist,positional_encoding,utils}.py`
  and `src/weathergen/datasets/{utils,tokenizer,tokenizer_utils}.py`.
- Only change: `weathergen.*` imports rewritten to
  `contrib.WGVarFM.weathergen_ext.*`. Symbols from WG packages that are not
  vendored (`Config`, `IOReaderData`, `ModelBatch`, `get_dtype`, `is_root`, ...)
  come from `weathergen_ext/_compat.py`.
- WG readers, anemoi, `multi_stream_data_sampler` and masking strategies are
  **not** vendored.

To update: re-copy the same files from a newer WG commit, re-apply the import
rewrite, update the commit hash above.

## Extra dependencies

- `astropy_healpix`
- `flash-attn` (CUDA only). WG's varlen attention asserts flash, so the model
  does not run on CPU as is.
- torch ≥ 2.5 (`flex_attention`).

## Tests

Run from the repo root (GPU + flash-attn):

```bash
python contrib/WGVarFM/tests/smoke_wg_encoder.py    # encoder alone, random tokens
python contrib/WGVarFM/tests/smoke_wg_pretrain.py   # full xp config on synthetic CROSCIM-like files
```

## Roadmap

1. Vendored WG encoder builds and runs forward/backward. *(done)*
2. Generic data pipeline + `LitWGFM`; WG self-supervised pre-training on raw
   CROSCIM sensors (SIC). *(done, synthetic smoke test only)*
3. Second config: CROSCIM FM conditioned on frozen WG features `h^WG`;
   ablation `p(X_10|X_50)` vs `p(X_10|X_50, h^WG)`.
4. Later: latent forecast, variational guidance `∇J` (FM-4DVarNet), patchwise.
