#!/usr/bin/env python3
"""Drive the ICLR rebuttal experiments on Odyssey from the Mac, one Slurm job at a time.

    python3 odyssey/pilot.py connect          # open the shared SSH connection (asks for your password / OTP once)
    python3 odyssey/pilot.py status           # plan position, current job, last results
    python3 odyssey/pilot.py next             # sync the code and submit the next step of the plan
    python3 odyssey/pilot.py submit <id>      # sync and submit a given step
    python3 odyssey/pilot.py tick             # what launchd runs every 30 min: poll, fetch, report, gate, submit next
    python3 odyssey/pilot.py approve <id>     # allow a step marked "confirm" to start
    python3 odyssey/pilot.py retry            # resubmit the current step after a failure
    python3 odyssey/pilot.py pause | resume   # stop / restart automatic submission
    python3 odyssey/pilot.py push             # commit and push results/ and paper/results/ to GitHub
    python3 odyssey/pilot.py watch [sec]      # live dashboard: plan progress, current job, log tail (default 60 s)

Rules: never more than one icl job per lane: the main lane, plus side lanes 2..N (steps marked "lane": n run on an
extra GPU, in parallel with the main lane; 2 lanes by the user's decision of 2026-11-14, 3 then 4 on 2026-11-16); no dependency chains; a failed job is retried once,
then the pilot pauses; a failed gate pauses; steps marked "confirm" wait for `approve`. Every action is
appended to RUNLOG.md, and anything needing a human writes results/ATTENTION.md and posts a macOS
notification. Standard library only.
"""
import json, os, re, shlex, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HERE = REPO / "odyssey"
STATE = REPO / ".pilot" / "state.json"
RUNLOG = REPO / "RUNLOG.md"


# ----------------------------------------------------------------------------- configuration
def load_env():
    env = {}
    for name in ("config.env", "config.local.env"):
        p = HERE / name
        if p.exists():
            for line in p.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    env[k.strip()] = v.strip()
    return env


CFG = load_env()
PLAN = json.loads((HERE / "plan.json").read_text())["steps"]
IDS = [s["id"] for s in PLAN]
SSH_OPTS = ["-o", "ForwardX11=no", "-o", "ControlMaster=auto", "-o", "ControlPath=~/.ssh/cm-icl-%r@%h:%p", "-o", "ControlPersist=8h"]


def step(sid):
    for s in PLAN:
        if s["id"] == sid:
            return s
    sys.exit(f"unknown step {sid}; known: {', '.join(IDS)}")


# ----------------------------------------------------------------------------- state and log
def load_state():
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"next": IDS[0], "job": None, "run_dir": None, "step": None, "retries": 0, "paused": False,
            "approved": [], "history": []}


def save_state(st):
    STATE.parent.mkdir(exist_ok=True)
    STATE.write_text(json.dumps(st, indent=1))


def log(msg):
    stamp = time.strftime("%Y-%m-%d %H:%M")
    if not RUNLOG.exists():
        RUNLOG.write_text("# Run log\n\nWritten by `odyssey/pilot.py`; newest last.\n\n")
    with RUNLOG.open("a") as f:
        f.write(f"- {stamp} {msg}\n")
    print(msg)


def attention(msg):
    p = REPO / "results" / "ATTENTION.md"
    p.parent.mkdir(exist_ok=True)
    p.write_text(f"# Needs attention ({time.strftime('%Y-%m-%d %H:%M')})\n\n{msg}\n")
    log(f"**ATTENTION** {msg}")
    if sys.platform == "darwin":
        subprocess.run(["osascript", "-e", f'display notification {json.dumps(msg[:200])} with title "ICLR pilot"'])


# ----------------------------------------------------------------------------- remote helpers
def ssh(cmd, check=True, interactive=False):
    args = ["ssh", *SSH_OPTS] + ([] if interactive else ["-o", "BatchMode=yes"]) + [CFG["ODY_HOST"], cmd]
    r = subprocess.run(args, capture_output=not interactive, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"ssh failed ({r.returncode}): {cmd}\n{(r.stderr or '').strip()}")
    return (r.stdout or "").strip()


def rsync(src, dst, extra=()):
    e = "ssh " + " ".join(shlex.quote(o) for o in SSH_OPTS + ["-o", "BatchMode=yes"])
    subprocess.run(["rsync", "-az", "-e", e, *extra, src, dst], check=True)


def remote_repo():
    r = CFG["ODY_REPO"].rstrip("/")
    if not r.endswith("4dvarnet-starter-iclr"):
        sys.exit("ODY_REPO must end with '4dvarnet-starter-iclr' (safety check before rsync --delete)")
    return r


def git_pull():
    """Fast-forward the working copy to origin/<branch> (new code and plan from Claude); never merges or overwrites."""
    br = CFG.get("GIT_BRANCH", "consistency_models")
    r = subprocess.run(["git", "-C", str(REPO), "pull", "-q", "--rebase", "--autostash", "origin", br],
                       capture_output=True, text=True)
    if r.returncode != 0:
        log(f"git pull failed: {(r.stderr or '').strip()[:200]}")


def sync_code():
    git_pull()
    r = remote_repo()
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout.strip()
    (HERE / "COMMIT").write_text(head + ("-dirty" if dirty else "") + "\n")
    ssh(f"mkdir -p {shlex.quote(r)}/runs")
    rsync(f"{REPO}/", f"{CFG['ODY_HOST']}:{r}/",
          ["--delete", "--exclude", ".git/", "--exclude", "/data", "--exclude", "runs/", "--exclude", ".pilot/",
           "--exclude", "results/", "--exclude", "logs*", "--exclude", "lightning_logs", "--exclude", "*.ckpt",
           "--exclude", "consistency/sbatch_submission/figures/", "--exclude", "consistency/cartopy/",
           "--exclude", "/consistency/Notebooks/*/figures/",          # figures made on Odyssey: never overwrite
           "--exclude", "/consistency/Notebooks/*/*.ipynb_out",
           "--exclude", "__pycache__/", "--exclude", "*.pyc", "--exclude", "*.ipynb_checkpoints"])


def our_jobs():
    # check=True: if SSH fails we must NOT conclude that the job has finished
    out = ssh(f"squeue -h -u {CFG['ODY_USER']} -o '%i %j %T'", check=True)
    return [l.split() for l in out.splitlines() if len(l.split()) == 3 and l.split()[1].startswith("icl-")]


def job_state(job):
    out = ssh(f"sacct -n -X -j {job} -o State,Elapsed,ExitCode -P", check=False)
    line = out.splitlines()[0] if out else ""
    return line.split("|") if line else ["UNKNOWN", "", ""]


# ----------------------------------------------------------------------------- actions
SIDE_LANES = (2, 3, 4)


def lane_of(sid):
    n = step(sid).get("lane", 1)
    return n if n in SIDE_LANES else 1


def side_jobs(st):
    return {(st.get(f"lane{n}") or {}).get("job") for n in SIDE_LANES} - {None}


def choose_resources(base):
    """GPU type and node exclusion for a new job (user, 16 Nov: 4 GPUs, preferably not on the nodes already used,
    H100 if available, else A100). For each candidate (SLURM_GRES_CHOICES, in order of preference), with and then
    without SLURM_EXCLUDE, ask Slurm when the job would start (`sbatch --test-only`). The earliest start with the
    exclusion is taken unless it is more than 30 min later than the earliest start without it; ties go to the
    earlier candidate (H100 first). Falls back to SLURM_GRES."""
    from datetime import datetime, timedelta
    gres = [g.strip() for g in CFG.get("SLURM_GRES_CHOICES", CFG["SLURM_GRES"]).split(",") if g.strip()]
    excl = CFG.get("SLURM_EXCLUDE", "").strip()
    found = []
    for ex in ([excl, ""] if excl else [""]):
        for g in gres:
            args = base + [f"--gres={g}"] + ([f"--exclude={ex}"] if ex else []) + ["--test-only", "odyssey/job.sbatch", "x"]
            out = ssh(f"cd {shlex.quote(remote_repo())} && " + " ".join(shlex.quote(a) for a in args) + " 2>&1", check=False)
            m = re.search(r"to start at (\S+)", out)
            if m:
                try:
                    found.append((datetime.fromisoformat(m.group(1)), g, ex))
                except ValueError:
                    pass
    if not found:
        return CFG["SLURM_GRES"], ""
    best_all = min(found, key=lambda f: f[0])          # min keeps the first of equal starts (preference order)
    with_ex = [f for f in found if f[2]]
    best = best_all
    if with_ex:
        b = min(with_ex, key=lambda f: f[0])
        if b[0] <= best_all[0] + timedelta(minutes=30):
            best = b
    print(f"resources: {best[1]}{' excluding ' + best[2] if best[2] else ''} (expected start {best[0]})")
    return best[1], best[2]


def submit(sid, st):
    s = step(sid)
    lane = lane_of(sid)
    l2 = st.get(f"lane{lane}") or {} if lane != 1 else {}
    others = [j for j in our_jobs() if j[0] not in side_jobs(st)] if lane == 1 else []
    if lane == 1 and others:
        sys.exit("an icl job is already in the main lane: one job at a time per lane")
    if lane != 1 and l2.get("job"):
        sys.exit(f"lane {lane} is busy with job {l2['job']} ({l2.get('step')})")
    sync_code()
    run_id = time.strftime("%Y%m%d-%H%M%S") + "_" + sid
    run_dir = f"runs/{run_id}"
    exports = ["ALL", "BASH_ENV=", f"ODY_REPO={remote_repo()}", f"ODY_CONDA_ENV={CFG['ODY_CONDA_ENV']}",
               f"ICL_PLAN_ID={sid}"]
    for k in ("ODY_OLD", "ODY_DATA"):
        if CFG.get(k):
            exports.append(f"{k}={CFG[k]}")
    base = [f"--time={s.get('time', '04:00:00')}", f"--partition={CFG['SLURM_PARTITION']}",
            f"--account={CFG['SLURM_ACCOUNT']}", f"--cpus-per-task={CFG.get('SLURM_CPUS', '8')}"]
    gres, excl = choose_resources(["sbatch"] + base)
    sb = ["sbatch", "--parsable", f"--job-name=icl-{sid}", *base, f"--gres={gres}",
          *([f"--exclude={excl}"] if excl else []),
          "--export=" + ",".join(exports), "odyssey/job.sbatch", run_dir]
    sb += ["--cmd", s["cmd"]]
    job = ssh(f"cd {shlex.quote(remote_repo())} && " + " ".join(shlex.quote(a) for a in sb)).split(";")[0]
    if lane != 1:
        st[f"lane{lane}"] = dict(job=job, run_dir=run_dir, step=sid, retries=l2.get("retries", 0) if l2.get("step") == sid else 0)
    else:
        st.update(job=job, run_dir=run_dir, step=sid)
    save_state(st)
    log(f"submitted `{sid}` as job {job} -> `{run_dir}`" + (f" (lane {lane})" if lane != 1 else "")
        + f" [{gres}{', excl. ' + excl if excl else ''}]")


def fetch(run_dir, job=None):
    local = REPO / run_dir
    local.mkdir(parents=True, exist_ok=True)
    ssh(f"mkdir -p {shlex.quote(remote_repo() + '/' + run_dir)}")   # a job cancelled before it started has no run dir
    rsync(f"{CFG['ODY_HOST']}:{remote_repo()}/{run_dir}/", f"{local}/",
          ["--exclude", "*.pt", "--exclude", "*.npz", "--exclude", "*.nc", "--exclude", "_source.py"])
    job_out = ssh(f"ls {shlex.quote(remote_repo())}/runs/slurm-icl-*-{job or load_state().get('job')}.out 2>/dev/null", check=False)
    if job_out:
        rsync(f"{CFG['ODY_HOST']}:{job_out.splitlines()[0]}", f"{local}/slurm.out")
    # figures written by the notebooks on Odyssey since the rebuttal started -> results/figures/<xp>/<tag>/
    figs = ssh(f"cd {shlex.quote(remote_repo())}/consistency/Notebooks && "
               "find . -path '*/figures/*' -name '*.png' -newermt 2026-11-09 2>/dev/null", check=False)
    if figs.strip():
        lst = HERE / ".figlist"
        lst.write_text(figs)
        dst = REPO / "consistency/sbatch_submission/results/figures"
        dst.mkdir(parents=True, exist_ok=True)
        rsync(f"{CFG['ODY_HOST']}:{remote_repo()}/consistency/Notebooks/", f"{dst}/", ["--files-from", str(lst)])
        lst.unlink(missing_ok=True)
    # small result files written by the sbatch_submission scripts (CSV, LaTeX, figures; no notebooks)
    res = "consistency/sbatch_submission/results/"
    (REPO / res).mkdir(parents=True, exist_ok=True)
    ssh(f"mkdir -p {shlex.quote(remote_repo() + '/' + res)}")          # may not exist before the first training step
    rsync(f"{CFG['ODY_HOST']}:{remote_repo()}/{res}", f"{REPO / res}",
          ["--include", "*/", "--include", "*.csv", "--include", "*.tex", "--include", "*.json",
           "--include", "*.png", "--include", "*.pdf", "--include", "*.npz", "--exclude", "*"])


def mean(rows, key):
    vals = [r[key] for r in rows if isinstance(r, dict) and isinstance(r.get(key), (int, float))]
    return sum(vals) / len(vals) if vals else float("nan")


def evaluate(expr, run_dir):
    d = REPO / run_dir
    metrics = json.loads((d / "metrics.json").read_text()) if (d / "metrics.json").exists() else {}
    meta = json.loads((d / "run.json").read_text()) if (d / "run.json").exists() else {}
    col = lambda path, key: mean(metrics.get(path, []), key)          # mean of a CSV column, e.g. col('GP/CM_metrics.csv', 'crps')
    env = dict(status=meta.get("status", "missing"), metrics=metrics, col=col, mean=mean, abs=abs, min=min, max=max,
               len=len, all=all, any=any)
    try:
        return bool(eval(expr, {"__builtins__": {}}, env)), None
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def push():
    paths = [p for p in ("results", "runs", "consistency/sbatch_submission/results", "RUNLOG.md") if (REPO / p).exists()]
    if not paths:
        return
    subprocess.run(["git", "-C", str(REPO), "add", *paths], check=True)
    if subprocess.run(["git", "-C", str(REPO), "diff", "--cached", "--quiet"]).returncode == 0:
        return
    msg = f"results: update ({time.strftime('%Y-%m-%d %H:%M')})"
    subprocess.run(["git", "-C", str(REPO), "commit", "-q", "-m", msg], check=True)
    br = CFG.get("GIT_BRANCH", "consistency_models")
    r = subprocess.run(["git", "-C", str(REPO), "pull", "-q", "--rebase", "--autostash", "origin", br],
                       capture_output=True, text=True)
    if r.returncode != 0:
        log(f"git pull --rebase failed: {(r.stderr or '').strip()[:300]}")
    r = subprocess.run(["git", "-C", str(REPO), "push", "-q", "origin", f"HEAD:{br}"], capture_output=True, text=True)
    if r.returncode != 0:
        attention(f"git push failed, results not on GitHub: {(r.stderr or '').strip()[:300]}")


def reload_plan():
    """Re-read plan.json (it may have changed with the last git pull)."""
    global PLAN, IDS
    PLAN = json.loads((HERE / "plan.json").read_text())["steps"]
    IDS = [s["id"] for s in PLAN]


def first_pending(st, lane=1):
    """First step of plan.json, in its current order, of the given lane that has not completed yet."""
    done = {h["step"] for h in st.get("history", []) if str(h.get("state", "")).startswith("COMPLETED")}
    busy = ({(st.get(f"lane{n}") or {}).get("step") for n in SIDE_LANES if (st.get(f"lane{n}") or {}).get("job")}
            | {st.get("step") if st.get("job") else None})   # a failed step (no job) is not busy: it is resubmitted
    return next((i for i in IDS if i not in done and lane_of(i) == lane and (lane == 1 or i not in busy)), None)


def tick_lane(st, n):
    """Side GPU n: poll / fetch / gate its job, then submit its next step. Never pauses the main lane."""
    l2 = st.get(f"lane{n}") or {}
    if l2.get("job"):
        if any(j[0] == l2["job"] for j in our_jobs()):
            print(f"lane {n}: job {l2['job']} ({l2['step']}) still running")
            return
        state, elapsed, code = job_state(l2["job"])
        sid, run_dir = l2["step"], l2["run_dir"]
        fetch(run_dir, l2["job"])
        st["history"].append(dict(step=sid, job=l2["job"], state=state, elapsed=elapsed, run_dir=run_dir, lane=n))
        st[f"lane{n}"] = dict(step=sid, retries=l2.get("retries", 0)); save_state(st)
        ok, err = (False, None)
        if state.startswith("COMPLETED"):
            ok, err = evaluate(step(sid).get("gate", "status == 'ok'"), run_dir)
        if not ok:
            if not state.startswith("COMPLETED"):
                st["history"][-1]["state"] = state
            st[f"lane{n}_paused"] = True; save_state(st)
            attention(f"lane {n}: `{sid}` ended {state} after {elapsed}"
                      f"{' (gate failed: ' + str(err) + ')' if err else ''}; see `{run_dir}`. Lane paused "
                      f"(`pilot.py resume` re-enables it; the failed step is resubmitted).")
            if state.startswith("COMPLETED"):         # gate failed: do not count the step as done
                st["history"][-1]["state"] = "GATE_FAILED"; save_state(st)
            push(); return
        log(f"lane {n}: `{sid}` done in {elapsed}, gate passed")
        st[f"lane{n}"] = None; save_state(st)
        push()
    if st.get("paused") or st.get(f"lane{n}_paused"):
        return
    nxt = first_pending(st, lane=n)
    if not nxt:
        return
    if step(nxt).get("confirm") and nxt not in st["approved"]:
        if st.get("waiting") != nxt:
            st["waiting"] = nxt; save_state(st)
            attention(f"next lane-{n} step `{nxt}` needs approval (`python3 odyssey/pilot.py approve {nxt}`).")
        return
    submit(nxt, st)


def advance(st, sid):
    st["next"] = first_pending(st)
    st.update(job=None, run_dir=None, step=None, retries=0)
    save_state(st)


def tick(st):
    """Exclusive: a second tick (e.g. `tick` by hand while `watch` ticks) returns at once instead of submitting the
    same step twice; the state is re-read under the lock so that it is never stale."""
    import fcntl
    STATE.parent.mkdir(exist_ok=True)
    with open(STATE.parent / "tick.lock", "w") as lk:
        try:
            fcntl.flock(lk, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            print("another tick is running (watch?): nothing done")
            return
        fresh = load_state()
        st.clear(); st.update(fresh)
        return _tick(st)


def _tick(st):
    git_pull(); reload_plan()          # Claude may have pushed code or a reordered plan
    r = subprocess.run(["ssh", *SSH_OPTS, "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", CFG["ODY_HOST"], "true"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"cannot reach {CFG['ODY_HOST']} without a password: {(r.stderr or '').strip()[:200]}\n"
              "nothing changed; fix SSH (key loaded? network?) and tick again")
        return
    for n in SIDE_LANES:
        tick_lane(st, n)
    if st.get("job"):
        if any(j[0] == st["job"] for j in our_jobs()):
            print(f"job {st['job']} ({st['step']}) still running")
            return
        state, elapsed, code = job_state(st["job"])
        sid, run_dir = st["step"], st["run_dir"]
        fetch(run_dir, st["job"])
        st["history"].append(dict(step=sid, job=st["job"], state=state, elapsed=elapsed, run_dir=run_dir))
        st["job"] = None; save_state(st)
        if not state.startswith("COMPLETED"):
            if st["retries"] < 1 and not state.startswith("CANCELLED"):
                log(f"`{sid}` ended {state} after {elapsed}; retrying once")
                st["retries"] += 1; save_state(st)
                submit(sid, st)
            else:
                st["paused"] = True; save_state(st)
                attention(f"`{sid}` ended {state} after {elapsed}; see `{run_dir}`. Pilot paused "
                          f"(`pilot.py retry` to resubmit, `resume` to continue).")
            push(); return
        ok, err = evaluate(step(sid).get("gate", "status == 'ok'"), run_dir)
        if not ok:
            st["paused"] = True; save_state(st)
            attention(f"`{sid}` finished but its gate failed{f' ({err})' if err else ''}: "
                      f"`{step(sid).get('gate')}`. Results in `{run_dir}`. Pilot paused.")
            push(); return
        log(f"`{sid}` done in {elapsed}, gate passed")
        advance(st, sid)
        push()
    if st.get("paused"):
        print("pilot paused"); return
    nxt = first_pending(st)           # always follow the current order of plan.json
    if nxt != st.get("next"):
        st["next"] = nxt; save_state(st)
        if nxt:
            log(f"plan.json changed; next is `{nxt}`")
    if not nxt:
        print("plan complete"); return
    if step(nxt).get("confirm") and nxt not in st["approved"]:
        if st.get("waiting") != nxt:
            st["waiting"] = nxt; save_state(st)
            attention(f"next step `{nxt}` needs approval (`python3 odyssey/pilot.py approve {nxt}`).")
        return
    submit(nxt, st)


def watch(interval=60, once=False):
    """Terminal dashboard refreshed every `interval` seconds (Ctrl-C to quit)."""
    plist = Path.home() / "Library" / "LaunchAgents" / "org.icl.pilot.plist"
    while True:
        st = load_state()
        done = {h["step"] for h in st.get("history", []) if str(h.get("state", "")).startswith("COMPLETED")}
        lines = [f"ICLR pilot  {time.strftime('%Y-%m-%d %H:%M:%S')}   "
                 f"automatic ticks: {'on' if plist.exists() else 'OFF (./odyssey/install_launchd.sh)'}   "
                 f"{'PAUSED' if st.get('paused') else ''}", ""]
        n_done = sum(1 for i in IDS if i in done)
        bar = int(30 * n_done / len(IDS))
        lines.append(f"plan  [{'#' * bar}{'.' * (30 - bar)}]  {n_done}/{len(IDS)} steps done")
        # every lane: main (st["job"]) and side lanes (st["laneN"]["job"])
        running = {}                                   # step -> (lane, job, run_dir)
        if st.get("job"):
            running[st.get("step")] = (1, st["job"], st.get("run_dir"))
        failed = set()
        for n in SIDE_LANES:
            l2 = st.get(f"lane{n}") or {}
            if l2.get("job"):
                running[l2.get("step")] = (n, l2["job"], l2.get("run_dir"))
            elif l2.get("step") and st.get(f"lane{n}_paused"):
                failed.add(l2["step"])
        nexts = {first_pending(st, lane=n) for n in SIDE_LANES if not st.get(f"lane{n}_paused")}
        nexts.add(st.get("next"))
        for sid in IDS:
            if sid in done:
                mark = "done"
            elif sid in running:
                mark = "RUN "
            elif sid in failed:
                mark = "FAIL"
            elif sid == st.get("waiting"):
                mark = "WAIT"
            elif sid in nexts:
                mark = "next"
            else:
                mark = "    "
            lane = lane_of(sid)
            lines.append(f"  {mark}  {sid}" + (f"   [lane {lane}]" if lane != 1 else ""))
        lines.append("")
        if running:
            try:
                ids = ",".join(j for _, j, _ in running.values())
                q = {}
                for l in (ssh(f"squeue -h -j {ids} -o '%i %T %M %l %R'", check=False) or "").splitlines():
                    parts = l.split(None, 1)
                    if len(parts) == 2:
                        q[parts[0]] = parts[1]
                for sid, (lane, job, run_dir) in sorted(running.items(), key=lambda kv: kv[1][0]):
                    lines.append(f"lane {lane}  job {job} ({sid}): {q.get(job, 'finished (fetched at next tick)')}")
                    tail = ssh(f"tail -n 3 {shlex.quote(remote_repo())}/{run_dir}/log.txt 2>/dev/null", check=False)
                    lines += ["      " + l[:140] for l in tail.splitlines()]
            except Exception as e:
                lines.append(f"Odyssey unreachable: {e}")
        else:
            lines.append("no job running")
        att = REPO / "results" / "ATTENTION.md"
        if att.exists():
            lines += ["", "ATTENTION:"] + ["  " + l for l in att.read_text().splitlines()[2:8]]
        if RUNLOG.exists():
            lines += ["", "recent log:"] + ["  " + l for l in RUNLOG.read_text().splitlines()[-5:]]
        print("\033[2J\033[H" + "\n".join(lines), flush=True)
        if once:
            return
        time.sleep(interval)


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    cmd, args = sys.argv[1], sys.argv[2:]
    st = load_state()
    if cmd == "connect":
        ssh("hostname", interactive=True); print("SSH connection open (kept 8 h)")
    elif cmd == "status":
        print(json.dumps({k: st[k] for k in ("next", "job", "step", "run_dir", "retries", "paused")}, indent=1))
        for n in SIDE_LANES:
            if st.get(f"lane{n}") or st.get(f"lane{n}_paused"):
                print(f"lane {n}:", json.dumps(st.get(f"lane{n}")), "PAUSED" if st.get(f"lane{n}_paused") else "")
        if st.get("job") or side_jobs(st):
            print("queue:", our_jobs())
    elif cmd == "next":
        if not st["next"]:
            sys.exit("plan complete")
        submit(st["next"], st)
    elif cmd == "submit":
        submit(args[0], st)
    elif cmd == "tick":
        tick(st)
    elif cmd == "approve":
        st["approved"].append(args[0]); st.pop("waiting", None); save_state(st); log(f"approved `{args[0]}`")
    elif cmd == "retry":                     # resubmit the failed step and resume the plan after it
        st["paused"] = False; st["retries"] = 0; save_state(st)
        (REPO / "results" / "ATTENTION.md").unlink(missing_ok=True)
        submit(st.get("step") or st["next"], st)
    elif cmd == "watch":
        try:
            watch(int(args[0]) if args else 60)
        except KeyboardInterrupt:
            pass
    elif cmd == "pause":
        st["paused"] = True; save_state(st); log("pilot paused")
    elif cmd == "resume":
        st["paused"] = False
        for n in SIDE_LANES:
            st[f"lane{n}_paused"] = False
        save_state(st); log("pilot resumed")
        (REPO / "results" / "ATTENTION.md").unlink(missing_ok=True)
    elif cmd == "skip":
        # mark the step as done without running it (cancel its job first if it is running: `scancel <job>`)
        st["history"].append(dict(step=args[0], job=None, state="COMPLETED (skipped)", elapsed="0", run_dir=None))
        if st.get("step") == args[0]:
            st.update(job=None, run_dir=None, step=None)
        for n in SIDE_LANES:
            if (st.get(f"lane{n}") or {}).get("step") == args[0]:
                st[f"lane{n}"] = None
        st["paused"] = False
        log(f"skipped `{args[0]}`"); advance(st, args[0])
    elif cmd == "push":
        push()
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
