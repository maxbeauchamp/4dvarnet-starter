"""
Run the 4 CROSCIM SIT forecast experiments over the 25 date sequences and collect
the produced NetCDF files (x50 / x10) under NetCDF_tests/ with clear names.

For each (experiment, sequence) we call main.py once, overriding ONLY:
  - the test period       -> datamodule.domains.test.time
  - the output directory  -> entrypoints.0.save_dir   (unique per run)

The model writes  save_dir/lightning_logs/version_<rnd>/test_data_<start>_<end>_patch_x{res}.nc
which we then copy/rename to:
  NetCDF_tests/<experiment>_seq<ii>_<start>_<end>_x{res}.nc

Usage (from anywhere):
  python run_benchmark_sequences.py                 # all 4 experiments x 25 sequences
  python run_benchmark_sequences.py --dry-run       # print the commands, run nothing
  python run_benchmark_sequences.py --experiments UNet_UOAI --seqs 1
                                                    # single combination (test one run)
  python run_benchmark_sequences.py --gpus 2 3      # run in parallel on GPUs 2 and 3
                                                    # (default: all GPUs seen by nvidia-smi)
python contrib/CROSCIM/scripts/run_benchmark_sequences.py \
    --experiments UNet_UOAI \
    --gpus 2 3
    --refresh-stale
"""
import argparse
import datetime
import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
from glob import glob
from pathlib import Path

# Reuse the exact same sequence definition as the extraction script.
sys.path.append(str(Path(__file__).resolve().parent))
from extract_files_inference import generate_sequences, validate_and_adjust_sequences

ROOT = Path(__file__).resolve().parents[3]          # 4dvarnet-starter
NETCDF_TESTS = ROOT / "NetCDF_tests"
RUNS_DIR = NETCDF_TESTS / "_runs"                    # per-run working dirs + logs

YEAR = 2022
N_SEQUENCES = 25
SEQUENCE_LENGTH_DAYS = 15                            # = patch_dims.time

# Watchdog: the multiprocessing DataLoader intermittently deadlocks (workers idle,
# main waiting) and the run then never progresses. A healthy run writes to its log
# continuously (setup prints + tqdm); a hung one goes silent. So we detect a HANG as
# "log file stopped growing for STALL_TIMEOUT_S", kill the process group, and retry
# on a fresh process. num_workers is left untouched (kept fast).
STALL_TIMEOUT_S = 300                               # no log growth for 5 min ⇒ hung
POLL_S = 5
MAX_ATTEMPTS = 3

# experiment name -> (hydra xp path, resolutions produced, checkpoint path rel. to ROOT)
EXPERIMENTS = {
    "UNet_UOAI":           ("CROSCIM/UNet_solvers/base_arctic_croscim_test_sit_UOAI_supervised_forecast",            [50, 10], "ckpt/CROSCIM/base_croscim_UNet_sit_UOAI_supervised_forecast.ckpt"),
    "UNet_UOAI_res10":     ("CROSCIM/UNet_solvers/base_arctic_croscim_test_sit_UOAI_supervised_forecast_res10",      [10],     "ckpt/CROSCIM/base_croscim_UNet_sit_UOAI_supervised_forecast_res10.ckpt"),
    "UNet_unrolling":      ("CROSCIM/UNet_unrolling_solvers/base_arctic_croscim_test_sit_supervised_forecast",       [50, 10], "ckpt/CROSCIM/base_croscim_UNet_unrolling_sit_supervised_forecast.ckpt"),
    "UNet_unrolling_res10":("CROSCIM/UNet_unrolling_solvers/base_arctic_croscim_test_sit_supervised_forecast_res10", [10],     "ckpt/CROSCIM/base_croscim_UNet_unrolling_sit_supervised_forecast_res10.ckpt"),
}


def build_sequences():
    """25 windows, validated/shifted to complete data, as (idx, start, end_inclusive)."""
    seqs = generate_sequences(YEAR, N_SEQUENCES, SEQUENCE_LENGTH_DAYS)
    seqs = validate_and_adjust_sequences(seqs, SEQUENCE_LENGTH_DAYS)
    out = []
    for idx, (start, _end_excl) in enumerate(seqs, 1):
        # The config test slice is inclusive on both ends and 15 days long,
        # so the inclusive end is start + (seq_len - 1) days.
        end_incl = (datetime.datetime.strptime(start, "%Y-%m-%d")
                    + datetime.timedelta(days=SEQUENCE_LENGTH_DAYS - 1)).strftime("%Y-%m-%d")
        out.append((idx, start, end_incl))
    return out


def final_paths(exp, idx, start, end_incl, resolutions):
    return {res: NETCDF_TESTS / f"{exp}_seq{idx:02d}_{start}_{end_incl}_x{res}.nc"
            for res in resolutions}


def detect_gpus():
    """Return GPU indices visible via nvidia-smi, or [] if none / no driver."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    return [int(x) for x in out.split()]


def _kill_tree(proc):
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()


def _run_with_watchdog(cmd, log_file, env=None):
    """
    Run cmd, watching its log for liveness. If the log stops growing for
    STALL_TIMEOUT_S the run is considered hung: its process group is killed and the
    attempt retried (up to MAX_ATTEMPTS). A genuine non-zero exit is returned as-is
    (no retry). Returns the exit code, or "hung" if every attempt stalled.
    """
    for attempt in range(1, MAX_ATTEMPTS + 1):
        with open(log_file, "w") as lf:
            lf.write(f"# attempt {attempt}/{MAX_ATTEMPTS}\n")
            lf.flush()
            proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=lf,
                                    stderr=subprocess.STDOUT, start_new_session=True,
                                    env=env)

        last_size, last_change = -1, time.time()
        stalled = False
        while proc.poll() is None:
            try:
                size = log_file.stat().st_size
            except OSError:
                size = last_size
            now = time.time()
            if size != last_size:
                last_size, last_change = size, now
            elif now - last_change > STALL_TIMEOUT_S:
                print(f"    ⏱ attempt {attempt}/{MAX_ATTEMPTS}: no progress for "
                      f"{STALL_TIMEOUT_S}s (DataLoader hang) — killing & retrying")
                _kill_tree(proc)
                stalled = True
                break
            time.sleep(POLL_S)

        if not stalled:
            return proc.returncode          # finished on its own (0 or genuine error)
    return "hung"


def run_one(exp, xp, resolutions, ckpt, idx, start, end_incl, dry_run=False,
            gpu_id=None, refresh_stale=False):
    targets = final_paths(exp, idx, start, end_incl, resolutions)
    tag = f"gpu{gpu_id} " if gpu_id is not None else ""

    if all(p.exists() for p in targets.values()):
        # Optionally rebuild if the checkpoint is newer than the produced outputs.
        ckpt_path = ROOT / ckpt
        stale = (refresh_stale and ckpt_path.exists()
                 and ckpt_path.stat().st_mtime
                     > min(p.stat().st_mtime for p in targets.values()))
        if not stale:
            print(f"  ✓ [{tag}{exp} seq{idx:02d}] already done — skipping")
            return "skipped"
        print(f"  ↻ [{tag}{exp} seq{idx:02d}] checkpoint newer than outputs — rebuilding")
        for p in targets.values():
            p.unlink(missing_ok=True)

    run_dir = RUNS_DIR / f"{exp}_seq{idx:02d}_{start}_{end_incl}"
    cmd = [
        sys.executable, str(ROOT / "main.py"),
        f"xp={xp}",
        f"++datamodule.domains.test.time._args_=['{start}', '{end_incl}']",
        # persistent_workers=True reproducibly deadlocks the test DataLoader at the
        # 2nd batch (esp. single-resolution res10). Disabling it keeps num_workers
        # (speed) while removing the hang.
        "++datamodule.dl_kw.persistent_workers=False",
        f"++entrypoints.0.save_dir={run_dir}",
        f"hydra.run.dir={run_dir}/hydra",
    ]

    print(f"\n  ▶ [{tag}{exp} seq{idx:02d}] {start} → {end_incl}")
    print("    " + " ".join(cmd))
    if dry_run:
        return "dry"

    env = None
    if gpu_id is not None:
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}

    # Start each run from a clean working dir, otherwise stale version_* dirs from
    # previous runs accumulate and the glob below can copy an old NetCDF.
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = run_dir / "run.log"
    rc = _run_with_watchdog(cmd, log_file, env=env)
    if rc == "hung":
        print(f"    ✗ all {MAX_ATTEMPTS} attempts hung — see {log_file}")
        return "failed"
    if rc != 0:
        print(f"    ✗ run FAILED (exit {rc}) — see {log_file}")
        return "failed"

    # Collect produced NetCDFs and copy with clear names.
    ok = True
    for res, dst in targets.items():
        matches = sorted(glob(str(run_dir / "lightning_logs" / "version_*" /
                                  f"test_data_*_patch_x{res}.nc")))
        if not matches:
            print(f"    ✗ no output for x{res} — see {log_file}")
            ok = False
            continue
        src = matches[-1]                 # newest if several
        shutil.copy2(src, dst)
        print(f"    ✓ x{res} → {dst.name}")
    return "done" if ok else "partial"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments", nargs="+", choices=list(EXPERIMENTS),
                    default=list(EXPERIMENTS), help="subset of experiments to run")
    ap.add_argument("--seqs", nargs="+", type=int, default=None,
                    help="subset of sequence indices (1-based); default = all")
    ap.add_argument("--dry-run", action="store_true",
                    help="print commands without running")
    ap.add_argument("--gpus", nargs="+", type=int, default=None,
                    help="GPU ids to run on in parallel (one run per GPU); "
                         "default = all GPUs seen by nvidia-smi")
    ap.add_argument("--refresh-stale", action="store_true",
                    help="rebuild an existing output when its checkpoint is newer")
    args = ap.parse_args()

    NETCDF_TESTS.mkdir(parents=True, exist_ok=True)
    sequences = build_sequences()
    if args.seqs:
        sequences = [s for s in sequences if s[0] in args.seqs]

    # One worker per GPU; each run is pinned to its GPU via CUDA_VISIBLE_DEVICES.
    gpus = args.gpus if args.gpus is not None else detect_gpus()
    if not gpus:
        gpus = [None]                       # no GPU pinning / single worker

    # (exp, xp, resolutions, ckpt, idx, start, end_incl)
    jobs = [(exp, *EXPERIMENTS[exp], idx, start, end_incl)
            for exp in args.experiments
            for idx, start, end_incl in sequences]

    print(f"\n{'='*70}")
    print(f"BENCHMARK RUNS: {len(args.experiments)} experiments × {len(sequences)} sequences "
          f"= {len(jobs)} runs")
    print(f"Parallelism: {len(gpus)} worker(s) on GPU(s) "
          f"{[g for g in gpus if g is not None] or 'n/a'}")
    print(f"Output: {NETCDF_TESTS}")
    print("="*70)

    job_q = queue.Queue()
    for job in jobs:
        job_q.put(job)

    stats = {}
    stats_lock = threading.Lock()

    def worker(gpu_id):
        while True:
            try:
                exp, xp, resolutions, ckpt, idx, start, end_incl = job_q.get_nowait()
            except queue.Empty:
                return
            try:
                status = run_one(exp, xp, resolutions, ckpt, idx, start, end_incl,
                                 args.dry_run, gpu_id, args.refresh_stale)
            except Exception as e:                       # keep the pool alive
                print(f"    ✗ [{exp} seq{idx:02d}] crashed: {e}")
                status = "failed"
            with stats_lock:
                stats[status] = stats.get(status, 0) + 1

    threads = [threading.Thread(target=worker, args=(g,)) for g in gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print(f"\n{'='*70}")
    print("SUMMARY: " + ", ".join(f"{k}={v}" for k, v in sorted(stats.items())))
    print(f"Files in: {NETCDF_TESTS}")
    print("="*70)


if __name__ == "__main__":
    main()
