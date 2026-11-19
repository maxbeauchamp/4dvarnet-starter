# Pairwise consistency models (ICLR 2027 submission)

Code of *Pairwise Consistency Models for Bayesian Transport and Variational Data Assimilation*: CM, VarCM, DynCM,
DynVarCM, their flow-matching teachers (FM, DynFM) and the UNet / 4DVarNet baselines, on three benchmarks (GP, SSH,
SIC). The state at the rebuttal is tagged `iclr2027-rebuttal`.

| Folder | Content |
|---|---|
| `consistency_models/`, `flowmatching_models/` | models, losses, samplers; `sic_costs.py` (SIC observation costs), `teacher_targets.py` |
| `calibration.py`, `spectral_utils.py` | CRPS, spread-skill ratio, rank histograms; spectral score and resolved scale |
| `Notebooks/Notebooks_{GP,SSH_GF,SIC}/` | one notebook per method and benchmark (training, sampling, metrics); first cell = parameters |
| `sbatch_submission/` | `run_training.py`, `run_metrics.py`, `run_nfe_sweep.py` execute a notebook with the parameters of `methods.yaml` |
| `paper_results/` | CSVs of the final tables and NFE studies (copied from `sbatch_submission/results/`, which is not tracked) |
| `scripts/` | figures of the paper, exact GP posterior (`gp_exact_posterior.py`), supplementary archive (`make_supplementary.py`) |

Data (`../data/`, not tracked): `SPDE_diffusion_dataset_1.nc` (GP), `natl_gf_w_5nadirs_swot.nc` (SSH),
`asip_database_daw15_sparse.nc` (SIC). Checkpoints: `Notebooks/Notebooks_<XP>/<log_dir>/` (not tracked), with the
`log_dir` of `methods.yaml`.

```bash
cd consistency/sbatch_submission
python run_training.py --xp SSH_GF --method FM_aug --reset     # train
python run_metrics.py  --xp SSH_GF --method FM_aug             # 20 members, full test set -> results/SSH_GF/
python run_metrics.py  --xp SSH_GF --method CMT_aug --set NSTEPS_EVAL=5 --out-tag _nfe5   # parameter override
```

On Odyssey the runs go through the pilot (`../odyssey/`, run from the Mac): one Slurm job per lane, the Odyssey
working copy is the git worktree `4dvarnet-starter-devs` (fast-forwarded at each submission).
