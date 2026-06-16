# CLAUDE.md — Agent Work Log

This file tracks all modifications made by the AI coding agent across sessions.

---

## Session Summary

### 1. Inference Notebook Fixes

**`main_inference_ice_sheet.ipynb`**
- Cells 7 & 12: updated variable references and fixed import/path issues

**`main_inference_sea_ice.ipynb`**
- Made ICDR variable name robust to dataset naming variations

---

### 2. Config YAML Fixes (5 files)

- `val_total_loss` → `val_loss` across all config YAML files in `config/`

---

### 3. UNet v5 Notebook

**`4dvarnet-starter-devs/Notebooks/`** (UNet v5)
- Added `_compute_obs_weights` method
- Updated input channels to `3C+1`

---

### 4. DynCM Consistency Model — Full v6 Redesign

#### Root Cause of Identity Collapse
- `total_training_steps=10_000` was hardcoded → N (number of discretisation steps) saturated at max (~50) after only ~50–100 epochs out of 2500
- Once N=50, dt≈0.02 → interpolated target ≈ interpolated input → `f(x)=x` trivially minimises loss
- Loss → 0 but inference from noise produces pure noise (identity propagates noise unchanged)

#### Correct Design (v6)

| Variable | Range | Meaning |
|---|---|---|
| Solver progress `s` | 0 → 1 | 0 = pure noise, 1 = clean |
| Diffusion time `t` | 1 → 0 | 1 = pure noise, 0 = clean |
| **Spin-up** | s ∈ [0, 0.3], t ∈ [0.7, 1] | pure noise → IC (frame 0) |
| **Physical** | s ∈ [0.3, 1], t ∈ [0, 0.7] | IC → frame 1 → … → frame C-1 |

- `spinup_boundary = 0.7` (diffusion-time threshold)
- `SPINUP_FRAC = 1 - 0.7 = 0.3`
- `NSTEPS = 4*C+1` → IC appears at step ~C+1 (s ≈ 0.3) ✓

#### File: `consistency/consistency_models/consistency_models_DynCM.py`

Complete rewrite with the following changes:

- **`skip_spinup(t, physical_lag=0.7, steepness=100)`**
  - Gate = **1 at large t** (spin-up regime), **0 at small t** (physical regime)
  - Formula: `1 / (1 + exp(-steepness*(t - physical_lag)))` — exponent sign CORRECTED vs old code

- **`physical_steps = linspace(0, spinup_boundary, C)` (INCREASING)**
  - `physical_steps[k]` ↔ frame `C-1-k`: k=0 (t=0) → last frame; k=C-1 (t=0.7) → IC

- **`_make_regime_input`**
  - Spin-up mask: `times > physical_lag` (inverted from old code)
  - Frame indexing: `x_prev = x[:,C-idx,:]` (next fwd-time), `x_curr = x[:,C-1-idx,:]` (IC-side)
  - `anchor_mode=True`: returns `x_prev` (next clean frame) — **prevents identity collapse**

- **`ConsistencyTrainingDynamicalSystems.__call__`**
  - `physical_steps = linspace(0, physical_lag, C)`
  - `valid_spinup = range(0, k_IC-1)` (large t, early solver steps)
  - `valid_physical = range(k_IC+1, N-2)` (small t, late solver steps)
  - `is_physical = (current_times <= physical_lag)`
  - Physical target: `gt_next` (anchor frame) — never interpolated

- **`spinup_boundary=0.7` passed explicitly** through all functions (replaces old hardcoded `1/C`)

#### File: `consistency/Notebooks/Notebooks_GP/Notebook_consistency_model_DynCM_spde.ipynb`

- **`LitConsistencyModel._obs_loss`** (cell `#VSC-8c04f12e`, lines 690–718):
  - Uses `sb = self.consistency_training.spinup_boundary`
  - `t_k = sb * (C-1-k) / max(C-1, 1)` — DECREASING from `sb` → 0
  - `t_{k+1} = sb * (C-2-k) / max(C-1, 1)`
  - Calls wrapper with explicit `spinup_boundary=sb`

- **`configure_optimizers`**: dynamic `total_training_steps` via `trainer.estimated_stepping_batches` (N now grows across the full 2500-epoch run)

- **Training cell** (`#VSC-b5dcd726`):
  ```python
  SPINUP_BOUNDARY = 0.7
  ct = ConsistencyTrainingDynamicalSystems(
      sigma_min=0.002, sigma_max=10.0, rho=7.0, sigma_data=1.0,
      initial_timesteps=5, final_timesteps=50,
      spinup_boundary=SPINUP_BOUNDARY,
  )
  LOG_VERSION = "v6"
  RESET_TRAINING = True
  ```

- **Sampler cell** (`#VSC-00362a01`):
  ```python
  NSTEPS = 4 * datamodule.window_size + 1
  SPINUP_BOUNDARY = ct.spinup_boundary   # 0.7
  consistency_sampling = ConsistencySamplingAndEditingDynamicalSystems(
      sigma_min=0.002, sigma_max=10.0, sigma_data=1.0,
      spinup_boundary=SPINUP_BOUNDARY,
  )
  ```

- **Plot cell** (`#VSC-e10b9cf0`):
  ```python
  SPINUP_FRAC = 1.0 - consistency_sampling.spinup_boundary   # 0.3
  # steelblue = s ≤ SPINUP_FRAC (spin-up), tomato = s > SPINUP_FRAC (physical)
  ```

- **Markdown cell** (`#VSC-3abe2ab5`): updated with correct regime layout table

---

### 5. Verification (Passed ✅)

```python
# skip_spinup gate direction
skip_spinup(tensor(0.9), 0.7)  # → 1.0      (spin-up ON at large t) ✅
skip_spinup(tensor(0.1), 0.7)  # → ~0.0     (spin-up OFF at small t) ✅

# _make_regime_input frame indexing
# At t=sb → frame 0 (IC)         ✅
# At t~0  → frame C-1 (last)     ✅

# Class attributes
ConsistencyTrainingDynamicalSystems(spinup_boundary=0.7).spinup_boundary  # → 0.7  ✅
ConsistencySamplingAndEditingDynamicalSystems(spinup_boundary=0.7).spinup_boundary  # → 0.7  ✅
```

---

## Key Parameters (v6)

| Parameter | Value |
|---|---|
| `spinup_boundary` | 0.7 |
| `SPINUP_FRAC` | 0.3 |
| `NSTEPS` | `4*C+1` |
| IC step (s) | ≈ 0.30 |
| `initial_timesteps` | 5 |
| `final_timesteps` | 50 |
| `max_epochs` | 2500 |
| `total_training_steps` | dynamic (`trainer.estimated_stepping_batches`) |

---

## Next Steps

- [ ] Launch v6 training (run training cell with `LOG_VERSION="v6"`, `RESET_TRAINING=True`)
- [ ] Monitor loss curves: loss should NOT collapse to 0 with v6
- [ ] Evaluate inference quality after ~500 epochs
