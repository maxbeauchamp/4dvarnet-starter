# CROSCIM — multi-scale flow-matching cascade for sea ice

Probabilistic estimation and simulation of the Arctic sea-ice state from
heterogeneous satellite observations (ASIP SAR, CIMR, CRISTAL) and ERA5
forcing, with a cascade of three conditional flow-matching generators on a
nested polar-stereographic grid (EPSG:3411):

| level | grid | state | observations (v1) |
|---|---|---|---|
| C | 32 km | SIC, SIT, snow + 9 sub-grid statistics S_C | CIMR (SIC, thin SIT), CRISTAL snow |
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
scripts/submit_odyssey.sbatch  sbatch wrapper: sbatch scripts/submit_odyssey.sbatch <script.py> [args]
tests/                  pytest suite (synthetic data, CPU)
```

## Workflow

1. Convert the sources once (stores written next to the NetCDF directories,
   e.g. `ASIP_L3_zarr/`):
   ```bash
   python scripts/convert_sources.py --src-root /Odyssey/public/CROSCIM_dataset \
       --start 2021-01-01 --end 2023-12-31 --workers 16
   ```
2. Train N0, first G_C alone, then the full cascade:
   ```bash
   python main.py xp=N0 model.levels=[C] model.pyramid=false \
       datamodule.train_dates=[...] datamodule.val_dates=[...] datamodule.test_dates=[...]
   python main.py xp=N0 ...
   ```
3. Train N1 from the N0 checkpoint: `python main.py xp=N1 model.n0_ckpt=... ...`
4. Scenarios and diagnostics: `croscim/rollout.py`.

Tests: `pytest tests` (about 1 min on CPU).
