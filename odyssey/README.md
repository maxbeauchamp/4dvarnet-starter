# Running the experiments on Odyssey from the Mac

```
Mac (SSH key for `mee`, GitHub)                                Odyssey (/Odyssey/private/m19beauc)
  pilot.py submit <id> / tick                                    login node
    ├─ ssh: git fetch + merge --ff-only (SYNC_MODE=git) ─────▶   4dvarnet-starter-devs (git worktree of the branch;
    │                                                              data/ and the checkpoints logs_* are untracked)
    ├─ sbatch odyssey/job.sbatch (ONE job per lane) ─────────▶   GPU node: run_step.py --cmd "..."
    ├─ squeue / sacct ◀─────────────────────────────────────     runs/<run_id>/{run.json, metrics.json, log.txt}
    └─ rsync runs/<run_id> + sbatch_submission/results ◀─────     (kept locally on the Mac, not committed)
```

* **One `icl-*` job per lane, never a dependency chain** (shared cluster).
* Push the code to GitHub before `submit`: the Odyssey worktree is fast-forwarded to `origin/consistency_models`;
  a local change of a tracked file there stops the submission.
* Results stay local (`runs/`, `results/`, `RUNLOG.md` are git-ignored); `PUSH_PATHS` in `config.env` lists what
  `pilot.py push` may commit (nothing by default). The CSVs of the paper are copied by hand to
  `consistency/paper_results/`.

## First use (on the Mac)

```bash
cd ~/Projects/4dvarnet-iclr                      # working copy of the branch on the Mac
git pull
ssh -o BatchMode=yes mee hostname                 # must answer without asking anything (key already set up for PGM)
python3 odyssey/pilot.py submit env_check         # checks env, data and one checkpoint
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
| `job.sbatch` | Odyssey | one step on one GPU (A100 by default) (conda env put on `PATH`, `BASH_ENV=` empty) |
| `run_step.py` | Odyssey | runs the command, writes `run.json` and `metrics.json` (rows of the CSVs written during the step) |

Gates: `status == 'ok'`, plus `col('<xp>/<file>.csv', '<column>')`, the mean of a column of a CSV written during
the step under `consistency/sbatch_submission/results/`.

