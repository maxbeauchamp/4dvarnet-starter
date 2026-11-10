"""Run one plan step on Odyssey and record what it produced.

    python odyssey/run_step.py --run-dir runs/<id> --cmd "<shell command>"

The command runs from the repository root with $ODY_* in the environment. Afterwards:
* <run-dir>/run.json   : plan id, command, commit, host, Slurm job, duration, status;
* <run-dir>/metrics.json: the rows of every CSV written under consistency/sbatch_submission/results/
  during the step (key = path relative to that folder), so gates in plan.json can read them,
  e.g. ``status == 'ok' and col('GP/CM_metrics.csv', 'crps') < 0.05``.
Standard library only.
"""
import argparse, csv, json, os, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "consistency" / "sbatch_submission" / "results"


def num(v):
    try:
        f = float(v)
        return int(f) if f.is_integer() and "." not in str(v) else f
    except (TypeError, ValueError):
        return v


def harvest(since):
    out = {}
    if RESULTS.exists():
        for p in sorted(RESULTS.rglob("*.csv")):
            if p.stat().st_mtime >= since:
                with p.open() as f:
                    rows = [{k: num(v) for k, v in r.items()} for r in csv.DictReader(f)]
                if len(rows) <= 500:
                    out[str(p.relative_to(RESULTS))] = rows
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--cmd", required=True)
    a = ap.parse_args()
    run_dir = Path(a.run_dir).resolve(); run_dir.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["cat", str(REPO / "odyssey" / "COMMIT")], capture_output=True, text=True).stdout.strip() or None
    meta = dict(plan_id=os.environ.get("ICL_PLAN_ID"), cmd=a.cmd, commit=commit, host=os.uname().nodename,
                slurm_job=os.environ.get("SLURM_JOB_ID"), started=time.strftime("%Y-%m-%dT%H:%M:%S"))
    (run_dir / "run.json").write_text(json.dumps(meta, indent=2))
    t0 = time.time()
    env = dict(os.environ, RUN_DIR=str(run_dir))
    rc = subprocess.run(a.cmd, shell=True, executable="/bin/bash", cwd=REPO, env=env).returncode   # bash, not dash
    (run_dir / "metrics.json").write_text(json.dumps(harvest(t0 - 1), indent=1))
    meta.update(status="ok" if rc == 0 else f"exit {rc}", seconds=round(time.time() - t0, 1),
                finished=time.strftime("%Y-%m-%dT%H:%M:%S"))
    (run_dir / "run.json").write_text(json.dumps(meta, indent=2))
    print(f"[run_step] {meta['status']} in {meta['seconds']} s -> {run_dir}")
    sys.exit(rc)


if __name__ == "__main__":
    main()
