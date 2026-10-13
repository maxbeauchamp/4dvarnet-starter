# AGENTS.md

Operating guidelines for AI agents working on Croscim. Codex reads this file
natively; Claude Code reads `CLAUDE.md`. Both files share the behavioral
guidelines in section 1; this file adds project context and review instructions
that Codex needs as a standalone reference.

**Tradeoff:** these guidelines bias toward caution over speed. For trivial
tasks, use judgment.

---

## 1. Behavioral Guidelines

### 1.1 Think Before Coding

Do not assume. Do not hide confusion. Surface tradeoffs.

Before implementing:

- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them. Do not pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what is confusing. Ask.

### 1.2 Simplicity First

Minimum code that solves the problem. Nothing speculative.

- No features beyond what was asked.
- No abstractions for single-use code.
- No flexibility or configurability that was not requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Test: would a senior engineer say this is overcomplicated? If yes, simplify.

### 1.3 Surgical Changes

Touch only what you must. Clean up only your own mess.

When editing existing code:

- Do not improve adjacent code, comments, or formatting.
- Do not refactor things that are not broken.
- Match existing style, even if you would do it differently.
- If you notice unrelated dead code, mention it. Do not delete it.

When your changes create orphans:

- Remove imports, variables, or functions that your changes made unused.
- Do not remove pre-existing dead code unless asked.

The test: every changed line should trace directly to the user's request.

### 1.4 Goal-Driven Execution

Define success criteria. Loop until verified.

Transform tasks into verifiable goals:

- "Add validation" -> "Write tests for invalid inputs, then make them pass"
- "Fix the bug" -> "Write a test that reproduces it, then make it pass"
- "Refactor X" -> "Ensure tests pass before and after"

For multi-step tasks, state a brief plan with verification per step.

### 1.5 Project Conventions

**Language and style**

- Chat with the user in French unless they ask otherwise.
- Code and comments should follow the existing file language and style.
- Keep prose direct. If a sentence does not add information, delete it.
- Avoid broad rewrites when a targeted edit is enough.

**Data safety**

- Never commit data files, model checkpoints, generated figures, Hydra outputs,
  TensorBoard logs, or large run logs.
- Treat `/Odyssey/public/CROSCIM_dataset` and `/Odyssey/private/m19beauc`
  as machine-specific operational paths. Do not run destructive operations
  there unless the user explicitly asks.

**What not to do**

- Do not silently change scientific assumptions, channel order, normalization,
  patch geometry, or loss definitions.
- Do not make scripts portable by inventing new path logic unless requested.
- Do not touch application code during documentation-only tasks.
- Do not use raw notes as source of truth without checking current code.

---

## 2. Project Context

### 2.1 What This Project Is

CROSCIM estimates and simulates the Arctic sea-ice state with a cascade of
conditional flow-matching generators on a nested EPSG:3411 grid
(C = 32 km, M = 4 km, H = 1 km), from ASIP, CIMR and CRISTAL observations and
ERA5 forcing. Two networks: N0 (analysis, G_C -> G_M -> G_H) and N1 (one-day
transition of the coarse augmented state). Reference for notation and v1
choices: `docs/formulation.tex` (section 8 for implementation choices).

### 2.2 Stack

- Python, PyTorch, PyTorch Lightning, Hydra/OmegaConf
- Zarr (training data), xarray / netCDF4 (conversion only), pyproj, geopandas
- TensorBoard for monitoring

### 2.3 Repository Map

- `main.py`: Hydra entry point (`python main.py xp=N0`).
- `config/xp/N0.yaml`, `config/xp/N1.yaml`: experiment configs.
- `croscim/`: the package (grid, obs, data, encoders, generators, N0, N1, rollouts).
- `scripts/convert_sources.py`: NetCDF -> Zarr conversion (format change only).
- `tests/`: pytest suite on synthetic data.
- `assets/ne_10m_land.zip`: Natural Earth land polygons (ocean fraction).
- `docs/formulation.tex`: formulation and notation.

### 2.4 Important Runtime Paths

- Source data and Zarr stores: `/Odyssey/public/CROSCIM_dataset`
  (`ASIP_L3`, `data_noise` (CIMR), `out_CRISTAL`, `atm_data` (ERA5), each with
  its `*_zarr` store next to it).
- Results: `/Odyssey/private/m19beauc/results` (`paths.src` in the configs).

### 2.5 Commands

```bash
pytest tests
python scripts/convert_sources.py --start 2021-01-01 --end 2023-12-31 --workers 16
python main.py xp=N0 model.levels=[C] model.pyramid=false datamodule.train_dates=[...] ...
python main.py xp=N1 model.n0_ckpt=/path/to/n0.ckpt ...
```

### 2.6 Technical Caveats

- Nothing is precomputed offline: partition, gridding, masks, detail targets
  and S_C are built per batch on the GPU (`CascadeDataModule.on_after_batch_transfer`).
  The Zarr conversion only crops and rechunks.
- Grid edges are aligned: one H cell is exactly a 2 x 2 ASIP block; fine
  generators run on crops aligned on C cells so that Pi0 stays exact.
- Workers use torch RNG (reseeded per worker), not numpy.
- In DDP, drop `shuffle` from `dl_kw` (Lightning sets the sampler) and set
  `find_unused_parameters: true` when `model.levels` is a subset.
- `OBS_SIGMA`, the S_C normalisation and the detail scales are provisional.

### 2.7 Documentation Sources

Authoritative: `README.md`, `AGENTS.md`, `CLAUDE.md`, `docs/`.

---

## 3. Review Instructions

These instructions are used when Codex is asked to review changes. They
override generic review heuristics.

### 3.1 Diff Review Priorities

Surface these first:

1. **Behavioral regressions** introduced by the diff: a path, env var, config
   key, function signature, tensor shape, channel ordering, DAW crop, or
   dataloader behavior that changed and could break callers elsewhere.
2. **Silent scientific failures**: code paths that swallow missing data,
   mismatched stats, invalid masks, broken interpolation, or broken patch
   geometry and continue with invalid results.
3. **Channel layout consistency** across `croscim/data.py`,
   `croscim/cascade.py`, `croscim/transition.py`, configs, scripts, and docs
   (state channels Z_NAMES, context channels, S_C order).
4. **Numerical correctness**: normalization, masking, target fusion, padding,
   time encoding, device placement, dtype mismatches, accidental precision loss.
5. **HPC-specific correctness**: SLURM scripts, DDP settings, worker counts,
   machine-specific paths, module/venv order, account settings.
6. **New imports or dependencies** that may not exist in `environment.yaml` or
   on Odyssey.

### 3.2 Holistic Review Constraints

A holistic review is wanted, but it must be short and selective.

- Maximum 3 holistic findings per review.
- Prioritize: dependency graph health, modules that cause concrete maintenance
  pain, unvalidated env vars or paths where failure would be silent/confusing.
- Do not raise generic process suggestions unless the user asked for them:
  adding CI, adding pre-commit, adding type checkers, adding packaging metadata,
  or converting standalone tests to pytest.
- Do not raise repository hygiene complaints unless they affect the requested
  task or create a concrete failure.
- Do not raise stylistic refactors when behavior is unchanged.
- Do not use "consider" or "you might want to" findings. Raise concrete,
  demonstrable issues only.

### 3.3 Output Format

For review outputs, use this structure:

1. **Diff findings** ordered by severity: `critical -> high -> medium -> low`.
2. **Open questions / assumptions** when needed.
3. **Summary** with a brief count and overall assessment.

If the diff has no in-scope issues, say so clearly and mention remaining test
gaps or residual risk.

### 3.4 Review Style

- Terse. One or two sentences per finding when possible.
- State what is wrong and what to change.
- Severity definitions:
  - **critical**: data corruption, silent wrong scientific results, security.
  - **high**: regression that breaks working code or training pipelines.
  - **medium**: subtle bug, fragile assumption, hidden coupling.
  - **low**: minor but concrete improvement.

---

## 4. Notes For Working On The Codebase

- The repository contains `CLAUDE.md` with the same behavioral guidelines as
  section 1. If you edit one, consider whether the other should follow.
- The repo is developed on macOS but runs on Linux machines. Avoid macOS-only
  flags in scripts meant for Odyssey.
- When updating docs, keep public information in `README.md`, operational agent
  details in `AGENTS.md`, and the formulation in `docs/formulation.tex`.
