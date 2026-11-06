# Two-layer QG testbed: 4DVarNet vs 4DVar, and the Lyapunov view of the gain

Goal: explain *mechanistically* why end-to-end learned DA (4DVarNet) beats a classical
4DVar in a chaotic system with model error. Both reconstruction errors are projected
on the backward / covariant Lyapunov vectors (BLV / CLV) of the DA model: does
4DVarNet win on the **unstable** component (better control of the growing
directions) or on the **stable** one (where the 4DVar model error accumulates and
the learned prior acts)?

## Pipeline

```bash
python main.py xp=qg2l_generate          # 1. truth + nadir/SWOT obs + figures/animation
python main.py xp=qg2l_4dvarnet          # 2. train/test 4DVarNet -> data/qg2l/reconstructions/4dvarnet.nc
python main.py xp=qg2l_4dvar             # 3. cycled 4DVar (strong)  -> .../4dvar_strong_test.nc
python main.py xp=qg2l_4dvar qg2l_4dvar.mode=weak   #              (weak)    -> .../4dvar_weak_test.nc
python main.py xp=qg2l_diag              # 4. BLV/CLV projections -> data/qg2l/diag/
```

Install: see `environment.yaml` (conda/mamba), then run from the repository root.

Quick end-to-end check on CPU (tiny grid, a few minutes per step):

```bash
for xp in qg2l_generate qg2l_4dvarnet qg2l_4dvar qg2l_diag; do
  python main.py xp=$xp +params=qg2l_smoke; done
```

Notebook (same pipeline; small 32x32 / 500 km configuration by default, sized to run in minutes per step on a GPU, hours for `SIZE="full"`):
`contrib/qg2l/qg2l_pipeline.ipynb` (`*.ipynb` is git-ignored in this repo: use `git add -f` to commit changes to it).

Tests (numpy core always; torch / Lightning tests when available):

```bash
cd contrib/qg2l/tests && python -m pytest -q
```

## Files

| file | content |
|---|---|
| `contrib/qg2l/qgmodel.py` | 2-layer QG on a doubly periodic beta-plane, pseudo-spectral, RK4, pyqg-like exponential filter. numpy **or** torch backend (same code): torch gives the 4DVar adjoint and the exact TLM (`torch.func.jvp`) |
| `contrib/qg2l/simulate.py` | truth run (HR `nx_truth=128` spectrally truncated to the DA grid `nx=64` -> structured model error of the LR DA model), presets, OSSE netCDF, lambda_1 / revisit summary |
| `contrib/qg2l/obs.py` | synthetic altimetry: Jason-3, Sentinel-3A/B (nadir) + SWOT (2 x 50 km KaRIn swaths, 20 km nadir gap + nadir), repeat cycles, interleaved sub-cycles, white noise (+ optional roll error) |
| `contrib/qg2l/viz.py` | snapshots, spectra, coverage / revisit maps, energy, animations (mp4, gif fallback) |
| `contrib/qg2l/data.py` | `QG2LDataModule`: full-domain sliding windows; state = (psi_1, psi_2) in SSH-equivalent units, obs on layer 1 only (NaN on layer 2), mask-resampling augmentation |
| `contrib/qg2l/models.py` | `LitQG2L` (subclass of `src.models.Lit4dVarNet`): time blending of windows, scores, export in the 4DVar format; `periodic()` -> circular conv padding |
| `contrib/qg2l/fourdvar.py` | cycled 4DVar, strong or weak constraint (daily model-error forcing), spectral isotropic 2-layer B (control-variable transform = preconditioning); solvers: nonlinear L-BFGS or incremental Gauss-Newton + preconditioned CG (TL / adjoint by autograd) |
| `contrib/qg2l/lyapunov.py` | Benettin QR (BLV, exponents) + Ginelli backward pass (CLV) in the energy norm; on-the-fly projection of errors; BLV/CLV decompositions |
| `contrib/qg2l/diag.py` | the diagnostic: `summary.json`, `lyapunov_projection.nc`, figures, comparison animation |
| `contrib/qg2l/pipeline.py` | cached scenario pipeline (generation, 4DVarNet with checkpoint resume, 4DVar / hybrid, diagnostic), keyed by configuration hash |
| `contrib/qg2l/stats.py` | paired moving-block bootstrap: confidence intervals of the error energies per Lyapunov subspace and of the ratios between methods |
| `contrib/qg2l/metrics.py` | RMSE, mu = 1 - RMSE/RMS, effective resolution lambda_x from the isotropic PSD score |

## Default set-up and measured regime

`energetic` preset (pyqg "eddy" parameters with doubled shear U1 = 5 cm/s), L = 1000 km,
DA grid 64^2 (dx = 15.6 km), R_d = 15 km, DA dt = 2 h. Checked here (numpy):

* stable, saturated turbulence, SSH std ~ 7 cm, eddies 100-200 km, CFL ~ 0.3;
* leading exponent of the 64^2 DA model lambda_1 ~ 0.08-0.10 /day (Lyapunov time ~ 10-13 days);
* median revisit time of the constellation 7.6 days -> **lambda_1 tau_obs ~ 0.6-0.8**: the
  intermediate regime where neither the dynamics nor the statistics alone suffice;
* the Lyapunov spectrum is **very flat** (extensive chaos: every eddy contributes its own
  unstable directions). With k = 256 vectors (50-day average) **all 256 exponents are
  positive**, decreasing from 0.104 to 0.017 /day (lambda_50 = 0.074, lambda_100 = 0.054,
  lambda_200 = 0.029); extrapolating the tail gives an unstable dimension n_0 ~ 300-400 out of
  8192. Hence **k must exceed it**: default `qg2l_diag.k=512` (if `n_unstable == k` the run
  warns: increase k). This is a real difference with Lorenz-96 (13 / 40) worth stating in the
  paper: the unstable subspace scales with the number of eddies, i.e. with the domain area,
  so "AUS-like" low-rank arguments only hold relative to n_0, not in absolute dimension.

`preset=eddy` (weaker shear) gives a less chaotic regime (lambda_1 ~ 0.035 /day) to sweep
lambda_1 tau_obs; the constellation (revisit) can be changed in `obs.DEFAULT_CONSTELLATION`
or through `entrypoints[0].constellation`.

## Realism scenarios (`contrib/qg2l/qg2l_realism_sweep.ipynb`)

The default OSSE is a favourable case for 4DVar (same equations / parameters / forcing for truth and DA model,
observations without representativeness error, known diagonal R, B estimated from the truth, perfect H, periodic
domain, unlimited optimisation budget). The second notebook switches these assumptions off, one at a time and all
together, retrains 4DVarNet on each dataset and compares 4DVarNet, strong / weak / low-budget 4DVar and the hybrid
4DVarNet -> 4DVar, with scores and error energy per Lyapunov subspace. Options (also usable in `simulate.generate`
and `fourdvar.run_4dvar`):

Crash-safe: re-running a notebook or an xp reuses finished steps and resumes unfinished ones (4DVarNet from `last.ckpt`, cycled 4DVar from its last window, Lyapunov sweep from its last 10-day checkpoint).

Progress of a sweep (from another terminal while the notebook runs): `python -m contrib.qg2l.pipeline data/qg2l_scenarios_small` (datasets ready, 4DVarNet epoch / best val_mse, 4DVar and diagnostic done per scenario).

| option | effect |
|---|---|
| `truth_overrides=dict(rek=..., beta=..., U1=...)` | truth physics differs from the DA model (parameter / structural model error) |
| `forcing=dict(ssh_std, wavelength_km)` | unknown band-limited stochastic forcing in the truth only |
| `obs_source="hr_block"` | obs sample the block-averaged high-resolution SSH (representativeness error) |
| `unresolved_signal=dict(ssh_std, wavelength_km, decorrelation_days)` | internal-tide-like signal in the obs only |
| `constellation[...]["roll_std"]` | correlated SWOT cross-track error (4DVar keeps a diagonal R) |
| `run_4dvar(b_source="free_run")` | B from a free run of the DA model instead of the truth |
| `run_4dvar(solver="incremental", n_outer, n_inner)` | operational-style incremental 4DVar: Gauss-Newton outer loops + conjugate gradient on the B^{1/2}-preconditioned quadratic cost (TL / adjoint by `torch.func.jvp` / `vjp`); the default `lbfgs` minimises the nonlinear preconditioned cost to near convergence |
| `run_4dvar(r_mode="desroziers" \| "oracle")` | R (per satellite) from the Desroziers diagnostic E[d_a d_b] = R on the validation split (no truth), or from the actual obs - truth std (upper bound); default `nominal` = instrumental noise only |
| `run_4dvar(background_path=4dvarnet.nc, spinup_days=0)` | hybrid: 4DVarNet reconstruction as background of each window |

## What the diagnostic reports

For each method and test day, with e = x_rec - x_truth (both layers, energy norm):

* BLV (orthogonal, additive): unstable / neutral / stable-resolved / residual (outside the
  span of the k leading BLVs, i.e. the most stable directions) — absolute energies and fractions;
* CLV subspaces: orthogonal projections of the resolved error on the unstable / neutral / stable CLV subspaces (bounded, not additive) and the smallest unstable-stable angle. The oblique split is not reported: near-tangencies (angles of a few degrees, Lorenz-96 and QG) make it ill-conditioned;
* error energy along the Lyapunov index (`lyapunov_spectrum_error_distribution.png`);
* `gain_vs_<first method>`: relative gain per subspace (1 - E_ref / E_other).

Sanity check performed on synthetic errors: an error *grown* by the dynamics over 20 days
puts ~90x more energy fraction in the leading BLVs than white noise of the same energy.

## Design choices / caveats

* Truth-model vs DA-model: model error = unresolved scales (HR 128 -> LR 64). Set
  `qg2l.nx_truth=64` for a perfect-model twin, or use `truth_overrides` in
  `simulate.generate` for parameter error.
* Lyapunov vectors are those of the **LR DA model** along the **true** trajectory.
* 4DVarNet reconstructs both layers from SSH only (`qg2l_net.layers=[1,2]`); with `[1]` it is the
  standard SSH-only set-up and the Lyapunov diagnostic is skipped for it.
* 4DVar: B = b_scale^2 x climatological spectral covariance (homogeneous, isotropic, with
  inter-layer correlation). Tune `b_scale`, `q_scale`, `window_days` on the validation split
  before comparing (`qg2l_4dvar.split=val`). Strong-constraint windows longer than
  ~1 Lyapunov time hit the non-convexity issue — that is part of the story, so sweep it.
* Both methods are smoothers within their window (4DVarNet +-10 days with `n_time=21`, 4DVar its window but with the memory of all past obs through the cycled background). Keep the 4DVarNet window >= 2-3 revisit times, otherwise it is starved of observations compared to the cycled 4DVar; the window length is itself a parameter worth sweeping.
* `tlm=fd` (central differences, float64) is the default for the Lyapunov vectors; `jvp` is
  exact and faster on GPU (validated against fd in the tests).
* 4DVarNet test windows extend `n_time - 1` days on each side of the test period (observations only),
  so every test day is reconstructed from full windows; without it the first / last days of the test
  period carry large edge errors that dominate the mean scores.
