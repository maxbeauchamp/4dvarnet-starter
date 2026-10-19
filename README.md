# CROSCIM — multi-scale flow-matching cascade for sea ice

Probabilistic estimation and simulation of the Arctic sea-ice state from
heterogeneous satellite observations (ASIP SAR, CIMR, CRISTAL) and ERA5
forcing, with a cascade of three conditional flow-matching generators on a
nested polar-stereographic grid (EPSG:3411):

| level | grid | state | targets (v1) |
|---|---|---|---|
| C | 32 km | SIC, SIT, snow + 9 sub-grid statistics S_C | ASIP aggregated (SIC), CRISTAL (SIT, snow); CIMR is an input (prior) only |
| M | 4 km | δSIC, δSIT | CRISTAL SIT (as L2 points), ASIP aggregated |
| H | 1 km | δSIC | ASIP |

The state is decomposed exactly, `X = (X_C, δX_M, δX_H)` with `R δX = 0`
(no learned decoder). Two networks are trained in turn: N0 (analysis, the
cascade G_C → G_M → G_H) and N1 (one-day transition of `Z_C = (X_C, S_C, ψ_C)`,
rolled out in mode A). Notation and choices: [docs/formulation.tex](docs/formulation.tex).

Everything is computed on the fly, per batch: input / target partition of the
observations, gridding, coverage masks, detail targets and sub-grid
statistics. No interpolation: aligned fields are block-averaged, other dense
fields remapped by exact overlap matrices, points binned.

## Install

```bash
conda create -n croscim
conda activate croscim
mamba env update -f environment.yaml
```

## Repository

```
main.py                 Hydra entry point: python main.py xp=N0 | xp=N1
config/xp/N0.yaml       N0 (phase 3 = model.levels=[C] model.pyramid=false)
config/xp/N1.yaml       N1 (needs model.n0_ckpt)
croscim/
  grid.py               nested 32 / 4 / 1 km grid, R, P, Pi0, decompose / recompose
  obs.py                gridding without interpolation, partition, detail targets, S_C
  ocean.py              ocean fraction from Natural Earth (assets/ne_10m_land.zip)
  data.py               Zarr dataset + GPU batch building (horizon K for N1 sequences)
  encoders.py           point tokens per cell, masked CNNs, ASIP encoder, Pi^up, Pi^down
  fm.py, backbone.py    flow-matching blocks and Swin U-Net backbone
  generators.py         one FM generator (Pi0 projection, censoring, Heun, guidance)
  cascade.py            N0 (LitCascadeN0)
  transition.py         N1 (LitCascadeN1)
  rollout.py            modes A / B, interventions, diagnostics
  train.py              base_training (fit + test)
scripts/convert_sources.py   NetCDF -> Zarr stores (crop + format change only)
scripts/inspect_data.py      dataloader on real data: maps, coverage, timings, normalisation
scripts/overfit.py           overfit N0 on one fixed batch (loss, memory, time per step)
scripts/convert_all.sh       one conversion job per source over a period
scripts/run_pipeline.py      whole pipeline from config/pipeline/*.yaml: phases 3-6, then evaluation (resumable)
scripts/plot_panels.py       inference panels (C / M / H, ensembles, S_C, psi) and metrics of an N0
scripts/plot_forecast.py     forecast panels (mode A rollouts, fine levels on demand) and lead-time scores of an N1
scripts/scenarios.py         scenario experiments with an N1 (perturbed S, forcing, do(S), nudging, downscaling, I^up / I^down)
scripts/submit_odyssey.sbatch  sbatch wrapper: sbatch scripts/submit_odyssey.sbatch <script.py> [args]
tests/                  pytest suite (synthetic data, CPU)
```

## Workflow

1. Convert the sources (one job per source; stores written next to the NetCDF
   directories, e.g. `ASIP_L3_zarr/`):
   ```bash
   scripts/convert_all.sh 2022-01-01 2022-12-31
   ```
2. Run the whole pipeline described by a config of `config/pipeline/`
   (N0: G_C alone, G_M + G_H, full cascade with pyramid; N1; then the
   evaluation scripts on the test period). Resumable: submit the same command
   again after the job time limit.
   ```bash
   sbatch scripts/submit_odyssey.sbatch scripts/run_pipeline.py --config config/pipeline/2022_v4.yaml
   ```
   `config/pipeline/smoke.yaml` runs the same chain on one month in a few
   minutes per phase (end-to-end check).
3. Single phases can also be run with `main.py`, e.g.
   `python main.py xp=N0 model.levels=[C] model.pyramid=false datamodule.train_dates=[...] ...`

Tests: `pytest tests` (about 1 min on CPU).
