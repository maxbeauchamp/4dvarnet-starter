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
- Do not delete raw notes in `notes/`. Move or summarize important information
  into maintained docs instead.
- Treat `/nwp/sst_malegu`, `/dmidata/projects/4dvarnet`, and
  `/dcai/projects/cu_0026` as machine-specific operational paths. Do not run
  destructive operations there unless the user explicitly asks.

**What not to do**

- Do not silently change scientific assumptions, channel order, normalization,
  patch geometry, or loss definitions.
- Do not make scripts portable by inventing new path logic unless requested.
- Do not touch application code during documentation-only tasks.
- Do not use raw notes as source of truth without checking current code.

---

## 2. Project Context

### 2.1 What This Project Is

Croscim trains a multi-resolution 4D-VarNet model for global Sea Surface
Temperature reconstruction from sparse satellite observations.

The active pipeline uses daily Zarr files at three resolutions:

- `x10`: 50 km, 15-day DAW, coarse global context.
- `x3`: 15 km, central 9-day DAW, residual refinement.
- `x1`: 5 km, central 5-day DAW, final high-resolution reconstruction.

The model runs coarse to fine. For x3 and x1, the code interpolates the coarser
prediction onto the finer grid, converts the finer batch to anomalies, predicts
a residual, then adds residual plus interpolated coarse prediction.

### 2.2 Stack

- Python, PyTorch, PyTorch Lightning
- Hydra/OmegaConf for experiment configuration
- Zarr, xarray, Dask, NumPy, SciPy, pandas
- TensorBoard for monitoring
- Shell and SLURM scripts for local and Gefion runs

### 2.3 Repository Map

- `main.py`: Hydra entry point. Calls `hydra.utils.call(cfg.entrypoints)`.
- `environment.yaml`: conda environment specification.
- `config/main.yaml`: root Hydra config. Requires `xp=...`.
- `config/xp/SST/*.yaml`: active experiment configs.
- `src/`: base 4D-VarNet framework, entrypoints, utilities, metrics.
- `contrib/SST/`: active SST datasets, model, solver, normalization.
- `scripts/local/`: DMI/Ohm launch scripts.
- `scripts/gefion/`: Gefion launch scripts.
- `data/`: preprocessing scripts for Zarr generation.
- `tests/`: standalone validation scripts, not a pytest suite.
- `tools/`: plotting, timing, and data inspection utilities.
- `docs/`: maintained documentation.
- `notes/`: raw context and session notes, not authoritative.
- `archive/`: legacy code and historical docs.

### 2.4 Important Runtime Paths

Known paths in scripts and configs:

- DMI data: `/nwp/sst_malegu`
- DMI outputs/checkpoints/logs: `/dmidata/projects/4dvarnet`
- Local script code path: `/home/malegu/4D-MLG/Croscim`
- Some configs use: `/dmidata/users/malegu/4D-MLG/Croscim`
- Gefion code: `/dcai/users/guimae/4dvarnet-mlg/Croscim`
- Gefion data: `/dcai/projects/cu_0026/data_sst`
- Gefion outputs/checkpoints/tmp: `/dcai/projects/cu_0026/guimae/croscim`
- Gefion env setup: `source scripts/gefion/env.sh` from the Croscim root.

These paths are not fully harmonized. Check scripts and YAML before launching
long runs.

### 2.5 Commands

Environment:

```bash
conda activate croscim
export PYTHONPATH=$PWD:$PYTHONPATH
```

Local smoke run:

```bash
./scripts/local/run_train_lite.sh 0
```

Local full run:

```bash
./scripts/local/run.sh 0
```

Resume local checkpoint:

```bash
./scripts/local/run.sh 0 /path/to/checkpoint.ckpt
```

Test checkpoint:

```bash
./scripts/local/run_test_checkpoint.sh /path/to/checkpoint.ckpt
```

Gefion single-GPU experiment:

```bash
sbatch scripts/gefion/submit_gefion_single.sh exp_name
```

Gefion DDP:

```bash
sbatch scripts/gefion/train_gefion.sh
```

Gefion ResUNet-prior experiment:

```bash
sbatch scripts/gefion/train_gefion_resunet.sh
```

### 2.6 Technical Caveats

- Dask plus xarray plus PyTorch DataLoader workers can deadlock. The active code
  uses pure Zarr in hot paths and sets Dask scheduler to `synchronous` in
  datamodule setup and worker init.
- In DDP, do not add explicit `shuffle=True` in train dataloaders; Lightning
  manages distributed sampling.
- Gefion DDP uses `persistent_workers: false` to avoid worker deadlocks.
- Tests are standalone scripts, not a pytest suite.
- Many tests and tools require production data under `/nwp/sst_malegu`.
- `format_batch_for_solver()` currently builds `8*T + 4` input channels:
  x10 = 124, x3 = 76, x1 = 44. Active configs are aligned on this layout.
- Older notes mention dimensions 139/85/49. Treat those as legacy.
- `lat` and `lon` are normalized model channels. `lat_geo` and `lon_geo` are
  degree coordinates and must be used for interpolation.
- `tgt_sst` is the masked fusion used as solver input during SSL.
  `tgt_sst_full` is the complete fused target used for loss/evaluation.
- x3/x1 training uses residual targets: temperature inputs and targets are
  converted to anomalies relative to the interpolated coarse prediction, then
  reconstructed as `coarse + residual`.
- Active configs use the shared `sst_common` z-score scale generated by
  `compute_statistics.py` for mean-temperature fields (`aasti.av`, `avhrr.av`,
  `pmw.av`, `slstr.av`, `tgt_sst`). Satellite `_std` fields keep their own
  generated stats. Regenerate these values before changing the training data
  range substantially.
- Complete Gefion data currently used by the ResUNet experiment spans
  2017–2024. Years 2014–2015 have no SLSTR, and 2016 has incomplete SLSTR
  coverage; do not include them without revisiting target availability.
- `multires_gefion_resunet` currently uses batch size 2 with gradient
  accumulation 6, but this still exhausted an 80 GB H100 during the validation
  sanity check. The open issue is graph retention across unrolled solver steps,
  not simply the input batch size.
- Gefion checkpoint evaluation should be submitted with
  `scripts/gefion/test_checkpoint_gefion.slurm`; `run_test_checkpoint_gefion.sh`
  is the worker used inside that allocation.
- `scripts/gefion/run_gefion_single.sh` is inconsistent with the Gefion single
  config and paths.
- `notes/Point.txt` is old SSH/NATL60 learning context, not the current SST
  project.

### 2.7 Documentation Sources

Authoritative:

- `README.md`
- `AGENTS.md`
- `CLAUDE.md`
- `docs/`
- folder READMEs as local indexes

Context only:

- `notes/*.txt`
- `notes/*.md`
- `archive/docs/*`

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
3. **Channel layout consistency** across `contrib/SST/data.py`,
   `contrib/SST/models.py`, `contrib/SST/solver.py`, configs, scripts, and
   docs.
4. **Numerical correctness**: normalization, masking, target fusion, padding,
   time encoding, device placement, dtype mismatches, accidental precision loss.
5. **HPC-specific correctness**: SLURM scripts, DDP settings, Dask scheduler,
   worker counts, machine-specific paths, module/venv order, account settings.
6. **New imports or dependencies** that may not exist in `environment.yaml` or
   on Gefion.

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
  flags in scripts meant for DMI or Gefion.
- Notebooks are exploratory and may have stale paths. Do not treat them as
  maintained workflows.
- When updating docs, keep public information in `README.md`, operational agent
  details in `AGENTS.md` and `docs/current-state.md`, and raw history in
  `notes/` or `archive/`.
