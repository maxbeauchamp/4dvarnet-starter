# ICLR 2027 rebuttal: running the experiments on Odyssey from the Mac

Same pilot as in `physical-generator-matching`, adapted to this repository (branch `rebuttal-iclr`).

```
Mac (SSH key for `mee`, GitHub)                                Odyssey (/Odyssey/private/m19beauc)
  pilot.py tick (launchd, every 30 min)                          login node
    ├─ rsync code (no .git, data, logs*, *.ckpt) ─────────────▶   iclr-rebuttal/4dvarnet-starter-iclr
    ├─ sbatch odyssey/job.sbatch (ONE job, no dependency) ─────▶   GPU node: run_step.py --cmd "..."
    ├─ squeue / sacct ◀─────────────────────────────────────────     runs/<run_id>/{run.json, metrics.json, log.txt}
    ├─ rsync runs/<run_id> + sbatch_submission/results (csv/tex/png/pdf) ◀──
    ├─ gate (plan.json) → submit the next step, or pause + macOS notification
    └─ git commit + push to origin/rebuttal-iclr ─▶ Claude reads the results, updates the paper repo
```

* **One `icl-*` job at a time, never a dependency chain** (shared cluster).
* The Mac is the only machine that talks to GitHub. Odyssey gets the code by rsync into a **separate folder**
  (`4dvarnet-starter-iclr`); the existing clone with the submitted paper's checkpoints (`ODY_OLD`) is only read.
* Switching the Mac off is safe: the job continues, and the next tick fetches it.

## First use (on the Mac)

```bash
cd ~/Desktop/4dvarnet-starter-croscim            # your main clone
git fetch origin rebuttal-iclr
git worktree add ../4dvarnet-iclr rebuttal-iclr   # separate working copy for the rebuttal
cd ../4dvarnet-iclr
ssh -o BatchMode=yes mee hostname                 # must answer without asking anything (key already set up for PGM)
python3 odyssey/pilot.py next                     # submits p0_env
python3 odyssey/pilot.py tick                     # later: fetch, gate, push, submit the next step
./odyssey/install_launchd.sh                      # then tick automatically every 30 min
python3 odyssey/pilot.py watch                    # live dashboard
```

Commands: `status`, `next`, `submit <id>`, `tick`, `approve <id>`, `retry`, `pause`, `resume`, `skip <id>`, `push`, `watch`.

## Files

| File | Where | Role |
|---|---|---|
| `config.env` (+ `config.local.env`, not committed) | Mac | SSH alias, remote paths, Slurm settings |
| `plan.json` | Mac | ordered steps: shell command, time limit, gate, `confirm` |
| `pilot.py` | Mac | sync, submit, poll, fetch, gate, push; state in `.pilot/state.json`, log in `RUNLOG.md` |
| `job.sbatch` | Odyssey | one step on one H100 (conda env put on `PATH`, `BASH_ENV=` empty) |
| `run_step.py` | Odyssey | runs the command, writes `run.json` and `metrics.json` (rows of the CSVs written during the step) |
| `inventory.sh` | Odyssey | step `p0_inventory`: lists the existing checkpoints and metrics CSVs (read only) |

Gates: `status == 'ok'`, plus `col('<xp>/<file>.csv', '<column>')`, the mean of a column of a CSV written during
the step under `consistency/sbatch_submission/results/`.

## Plan (filled in as the code lands)

1. `p0_env`, `p0_inventory`: environment check; where are the FM/DynFM checkpoints (teachers) → set `ODY_OLD`.
2. GP: metrics of the submitted models with the new calibration metrics (spread–skill ratio, exact-posterior
   RMSE/CRPS/spread), then the corrected CM / VarCM / DynCM / DynVarCM (ODE-teacher targets + tangent condition).
3. SSH_GF, then SIC if the GPU budget allows.
4. Tables and figures → paper repository (Overleaf).
