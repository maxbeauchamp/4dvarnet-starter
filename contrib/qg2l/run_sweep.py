"""
Run (part of) the realism sweep outside the notebook, e.g. as separate GPU jobs in parallel.

    python -m contrib.qg2l.run_sweep --size small --scenario realistic             # everything for one scenario
    python -m contrib.qg2l.run_sweep --size small --scenario baseline --only seeds  # extra 4DVarNet seeds only
    python -m contrib.qg2l.run_sweep --size small --scenario obs_signal --only tuning

All steps are cached and crash-safe (see pipeline.py): the notebook then only reads the results.
Do not run two jobs on scenarios sharing the same dataset (e.g. baseline and B_free_run) *at the same time*,
except with different --only parts that do not overlap (seeds vs tuning).
"""
import argparse

from . import pipeline
from .sweep_config import config, tuned_methods


def run_one(C, name, device="cuda", only="all"):
    sc = C["SCENARIOS"][name]
    gen = {**C["BASE_GEN"], **sc.get("gen", {})}
    var_over = sc.get("var", {})
    root = C["ROOT"]
    seeds = C["SEEDS"] if name in C["SEED_SCENARIOS"] else C["SEEDS"][:1]
    data = pipeline.generate_cached(root, gen, device=device)
    if only == "seeds":
        for sd in seeds:
            pipeline.train_4dvarnet(root, data, C["NET"], seed=sd)
        return None
    tuning = None
    if C["TUNE"]:
        tuning = pipeline.tune_4dvar(root, name, data, base=dict(spinup_days=C["SPIN_VAR"], **var_over),
                                     device=device, **C["TUNE"])
        print(f"[tuning] {name}: best strong {tuning['best_strong']}, best weak {tuning['best_weak']}", flush=True)
    if only == "tuning":
        return tuning
    methods = [{**m, **var_over} for m in C["METHODS"]] + tuned_methods(tuning, C["SPIN_VAR"], var_over)
    res = pipeline.run_scenario(root, name, gen, methods, net=C["NET"], lyap=C["LYAP"], device=device, seeds=seeds)
    res["tuning"] = tuning
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="small", choices=["small", "full"])
    ap.add_argument("--scenario", required=True, nargs="+")
    ap.add_argument("--only", default="all", choices=["all", "seeds", "tuning"])
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    C = config(a.size)
    for name in a.scenario:
        print(f"\n========== {name} ({a.only}) ==========", flush=True)
        run_one(C, name, device=a.device, only=a.only)
    pipeline.progress(C["ROOT"])


if __name__ == "__main__":
    main()
