"""
Link between the code (contrib/qg2l) and the paper: tables, figures and in-text numbers are generated
from the cached results of the realism sweep, never typed by hand.

  # on the cluster, after the sweep (needs the repo environment: xarray, ...):
  python paper/scripts/make_paper_assets.py collect --root data/qg2l_scenarios_small --out paper/results/results_small.json
  python paper/scripts/make_paper_assets.py collect --root data/qg2l_scenarios_full  --out paper/results/results_full.json
  # anywhere (numpy + matplotlib only):
  python paper/scripts/make_paper_assets.py render --results paper/results/results_small.json [--full paper/results/results_full.json]

`render` writes paper/tables/{macros,scores,cost,robustness}.tex and paper/figures/ratios.pdf.
"""
import argparse
import json
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1]
REPO = PAPER.parent

SCEN = {"baseline": "Base", "B_free_run": "Bfree", "obs_signal": "ObsSig", "param_error": "ParErr",
        "realistic": "Real", "obs_hr": "ObsHR", "swot_roll": "Roll", "forcing": "Forc", "nadir_only": "Nadir"}
SCEN_LABEL = {"baseline": "baseline", "B_free_run": "free-run $\\mathbf{B}$", "obs_signal": "unresolved signal",
              "param_error": "parameter error", "realistic": "realistic", "obs_hr": "HR obs",
              "swot_roll": "SWOT roll", "forcing": "forcing", "nadir_only": "nadir only"}
METH = {"4dvarnet": "Net", "4dvar_strong": "Var", "4dvar_tuned": "Tuned", "4dvar_tuned_weak": "TunedW",
        "4dvar_R_desroziers": "RD", "4dvar_R_oracle": "RO", "hybrid": "HybSeven", "hybrid_w3": "HybThree",
        "hybrid_w3_Rdes": "Hyb", "4dvar_weak": "Weak", "4dvar_incr": "Incr", "4dvar_incr_budget": "Budget"}
SUB = {"unstable": "Unst", "neutral": "Neu", "stable_resolved": "Stab", "residual": "Res", "total": "Tot"}
PAPER_METHODS = [  # rows of the scores table (kept if present)
    ("4dvarnet", "4DVarNet"),
    ("4dvar_tuned", "4DVar, tuned ($\\mathbf{R}_{\\mathrm{D}}$)"),
    ("4dvar_strong", "4DVar"),
    ("4dvar_R_desroziers", "4DVar, $\\mathbf{R}_{\\mathrm{D}}$"),
    ("4dvar_incr", "4DVar incr.\\ $3{\\times}20$"),
    ("4dvar_incr_budget", "4DVar incr.\\ $1{\\times}5$"),
    ("hybrid_w3_Rdes", "Hybrid (3\\,d, $\\mathbf{R}_{\\mathrm{D}}$)"),
]
PAPER_SCENARIOS = ["baseline", "obs_signal", "param_error", "realistic"]


# ---------------------------------------------------------------------------
# collect (cluster side)
# ---------------------------------------------------------------------------
def collect(root, out):
    import xarray as xr

    sys.path.insert(0, str(REPO))
    from contrib.qg2l.stats import subspace_ci_from_file

    root = Path(root)
    res = dict(size=root.name.replace("qg2l_scenarios_", ""), source=str(root), scenarios={})
    for sc in sorted((root / "scenarios").glob("*")):
        summ = sc / "diag" / "summary.json"
        if not summ.exists():
            continue
        s = json.loads(summ.read_text())
        data = json.loads((sc / "diag" / "summary.json.json").read_text())["data"]
        ds = xr.open_dataset(data, decode_times=False)
        obs = json.loads(ds.attrs["summary"])
        res["k"] = s["k"]
        res["state_dim"] = int(2 * ds.sizes["x"] * ds.sizes["y"])
        recs = {p.stem: p for p in sc.glob("*.nc")}
        gen_hash = Path(data).parent.name
        for r in sorted((root / "4dvarnet").glob(f"{gen_hash}_*")):
            if (r / "4dvarnet.nc").exists() and (r / "run.json").exists():
                seed = json.loads((r / "run.json").read_text()).get("seed", 333)
                recs["4dvarnet" if seed == 333 else f"4dvarnet_s{seed}"] = r / "4dvarnet.nc"
        recs.setdefault("4dvarnet", next((root / "4dvarnet").glob(f"{gen_hash}_*/4dvarnet.nc"), None))
        scores, cost, r_est = {}, {}, {}
        for m, p in recs.items():
            if p is None:
                continue
            a = xr.open_dataset(p, decode_times=False).attrs
            scores[m] = json.loads(a["scores"])
            if "cost" in a:
                cost[m] = json.loads(a["cost"])
            info = json.loads(a.get("config", "{}")).get("r_info", {})
            if info.get("desroziers"):
                r_est.setdefault("desroziers", list(info["desroziers"][-1].values()))
            if info.get("oracle"):
                r_est.setdefault("oracle", list(info["oracle"].values()))
        proj = sc / "diag" / "lyapunov_projection.nc"
        avail = json.loads(xr.open_dataset(proj, decode_times=False).attrs["methods"])
        ci = {}
        for ref in ["4dvar_tuned", "4dvar_strong", "4dvarnet"]:
            if ref in avail:
                ci[ref] = subspace_ci_from_file(proj, ref=ref, block=10, n_boot=2000, level=0.9)
        tuning = json.loads((sc / "tuning" / "tuning.json").read_text()) if (sc / "tuning" / "tuning.json").exists() \
            else None
        res["scenarios"][sc.name] = dict(
            lambda1=obs["lambda1_per_day_LRmodel"], revisit=obs["median_revisit_days"], ssh_std=obs["ssh_std_m"],
            obs_minus_truth_std=obs.get("obs_minus_truth_std_m"), n_unstable=s["n_unstable"],
            energy={m: v for m, v in s["methods"].items()}, scores=scores, ci=ci, cost=cost, r_est=r_est,
            tuning=tuning)
    res["ref"] = "4dvar_tuned" if any("4dvar_tuned" in v["ci"] for v in res["scenarios"].values()) else "4dvar_strong"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res, indent=1))
    print("wrote", out)


# ---------------------------------------------------------------------------
# render (paper side)
# ---------------------------------------------------------------------------
def _seed_stats(scores, key):
    import numpy as np

    vals = [v[key] for m, v in scores.items() if m == "4dvarnet" or m.startswith("4dvarnet_s")]
    return float(np.mean(vals)), (float(np.std(vals, ddof=1)) if len(vals) > 1 else None), len(vals)


def render(results, full=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    R = json.loads(Path(results).read_text())
    F = json.loads(Path(full).read_text()) if full else None
    tab, fig = PAPER / "tables", PAPER / "figures"
    tab.mkdir(exist_ok=True)
    fig.mkdir(exist_ok=True)
    ref = R["ref"]
    scen = [s for s in PAPER_SCENARIOS if s in R["scenarios"]]

    # ---- macros ------------------------------------------------------------------
    L = ["% generated by paper/scripts/make_paper_assets.py -- do not edit",
         f"% source: {R.get('source', '')}",
         f"\\newcommand{{\\refmethod}}{{{dict(PAPER_METHODS).get(ref, ref)}}}",
         f"\\newcommand{{\\kLyap}}{{{R.get('k', '?')}}}",
         f"\\newcommand{{\\stateDim}}{{{R.get('state_dim', '?')}}}",
         f"\\newcommand{{\\resultsSize}}{{{R.get('size', '')}}}"]

    def d(name, val):
        L.append(f"\\expandafter\\def\\csname {name}\\endcsname{{{val}}}")

    for s, v in R["scenarios"].items():
        S = SCEN.get(s, s)
        d(f"nu@{S}", v["n_unstable"])
        d(f"lam@{S}", f"{v['lambda1']:.3f}")
        d(f"lamtau@{S}", f"{v['lambda1'] * v['revisit']:.2f}")
        d(f"rev@{S}", f"{v['revisit']:.1f}")
        if v.get("obs_minus_truth_std") is not None:
            d(f"omt@{S}", f"{100 * v['obs_minus_truth_std']:.2f}")
        for refm, rows in v.get("ci", {}).items():
            tag = "" if refm == ref else "@vs" + METH.get(refm, refm)
            for r in rows:
                if r["method"] not in METH:
                    continue
                key = f"{METH[r['method']]}@{SUB[r['subspace']]}@{S}{tag}"
                d(f"r@{key}", f"{r['ratio']:.2f}\\,[{r['ratio_lo']:.2f},\\,{r['ratio_hi']:.2f}]")
                d(f"rv@{key}", f"{r['ratio']:.2f}")
        for m, sc in v["scores"].items():
            if m in METH:
                d(f"ssh@{METH[m]}@{S}", f"{100 * sc['rmse_ssh']:.2f}")
                d(f"ltwo@{METH[m]}@{S}", f"{100 * sc['rmse_layer2']:.2f}")
                d(f"lx@{METH[m]}@{S}", f"{sc['lx_ssh_km']:.0f}")
        if v.get("tuning") and v["tuning"].get("best_strong"):
            b = v["tuning"]["best_strong"]
            d(f"tunb@{S}", b["b_scale"])
            d(f"tunw@{S}", b["window_days"])
    (tab / "macros.tex").write_text("\n".join(L) + "\n")

    # ---- scores table ------------------------------------------------------------------
    rows = [(m, lab) for m, lab in PAPER_METHODS if any(m in R["scenarios"][s]["scores"] for s in scen)]
    T = ["% generated by paper/scripts/make_paper_assets.py -- do not edit",
         "\\begin{tabular}{l" + "cc" * len(scen) + "}", "\\toprule",
         " & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{SCEN_LABEL[s]}}}" for s in scen) + " \\\\",
         " ".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(scen))),
         " & " + " & ".join(["SSH & layer 2"] * len(scen)) + " \\\\", "\\midrule"]
    for m, lab in rows:
        cells = []
        for s in scen:
            sc = R["scenarios"][s]["scores"]
            for k in ["rmse_ssh", "rmse_layer2"]:
                if m == "4dvarnet" and any(x.startswith("4dvarnet_s") for x in sc):
                    mu, sd, n = _seed_stats(sc, k)
                    cells.append(f"{100 * mu:.2f}$\\pm${100 * sd:.2f}")
                elif m in sc:
                    best = min(v[k] for mm, v in sc.items() if mm in dict(rows))
                    val = f"{100 * sc[m][k]:.2f}"
                    cells.append(f"\\textbf{{{val}}}" if sc[m][k] == best else val)
                else:
                    cells.append("--")
        T.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    T += ["\\bottomrule", "\\end{tabular}"]
    (tab / "scores.tex").write_text("\n".join(T) + "\n")

    # ---- cost table --------------------------------------------------------------------
    cost = next(iter(R["scenarios"].values()))["cost"]
    C = ["% generated -- do not edit", "\\begin{tabular}{lrrrr}", "\\toprule",
         "method & nonlinear & tangent linear & adjoint & total \\\\", "\\midrule"]
    for m, lab in PAPER_METHODS:
        if m in cost:
            c = cost[m]
            C.append(f"{lab} & {c['n_nonlinear']} & {c['n_tl']} & {c['n_adjoint']} & "
                     f"{c['n_nonlinear'] + c['n_tl'] + c['n_adjoint']} \\\\")
    C += ["\\bottomrule", "\\end{tabular}"]
    (tab / "cost.tex").write_text("\n".join(C) + "\n")

    # ---- robustness (seeds, tuning, full size) -------------------------------------------
    Rb = ["% generated -- do not edit"]
    seeds = {s: v for s, v in R["scenarios"].items() if any(m.startswith("4dvarnet_s") for m in v["scores"])}
    if seeds:
        Rb.append("\\begin{tabular}{lccc}\\toprule scenario & seeds & SSH RMSE (cm) & unstable ratio \\\\ \\midrule")
        for s, v in seeds.items():
            mu, sd, n = _seed_stats(v["scores"], "rmse_ssh")
            ratios = [r["ratio"] for r in v["ci"].get(ref, []) if r["subspace"] == "unstable"
                      and (r["method"] == "4dvarnet" or r["method"].startswith("4dvarnet_s"))]
            rr = f"{np.mean(ratios):.2f}$\\pm${np.std(ratios, ddof=1):.2f}" if len(ratios) > 1 else "--"
            Rb.append(f"{SCEN_LABEL.get(s, s)} & {n} & {100 * mu:.2f}$\\pm${100 * sd:.2f} & {rr} \\\\")
        Rb.append("\\bottomrule\\end{tabular}")
    else:
        Rb.append("\\pend{seed-to-seed variability: pending (sweep in progress)}")
    if F:
        Rb.append("")
        for s, v in F["scenarios"].items():
            rows_f = {r["method"]: r for r in v["ci"].get(F["ref"], []) if r["subspace"] in ("unstable", "total")
                      and r["method"] == "4dvarnet"}
            Rb.append(f"% full size {s}: " + json.dumps({k: v2['ratio'] for k, v2 in rows_f.items()}))
    (tab / "robustness.tex").write_text("\n".join(Rb) + "\n")

    # ---- figure: ratios to the reference per subspace -------------------------------------
    show = [("4dvarnet", "4DVarNet", "tab:red", "o"), ("4dvar_R_desroziers", "4DVar, $R_D$", "tab:blue", "s"),
            ("hybrid_w3_Rdes", "hybrid (3 d, $R_D$)", "tab:purple", "D")]
    if ref == "4dvar_tuned":
        show[1] = ("4dvar_tuned_weak", "4DVar tuned, weak", "tab:blue", "s")
    scen_f = [s for s in ["baseline", "B_free_run", "obs_signal", "param_error", "realistic"] if s in R["scenarios"]]
    subs = [("unstable", "unstable"), ("stable_resolved", "stable (resolved)"), ("residual", "residual (most stable)"),
            ("total", "total")]
    plt.rcParams.update({"font.size": 8})
    f, axes = plt.subplots(1, 4, figsize=(7.2, 2.55), sharey=True, layout="constrained")
    for ax, (sub, title) in zip(axes, subs):
        for j, (m, lab, col, mk) in enumerate(show):
            x, y, lo, hi = [], [], [], []
            for i, s in enumerate(scen_f):
                for r in R["scenarios"][s]["ci"].get(ref, []):
                    if r["method"] == m and r["subspace"] == sub:
                        x.append(i + (j - 1) * 0.18)
                        y.append(r["ratio"])
                        lo.append(r["ratio"] - r["ratio_lo"])
                        hi.append(r["ratio_hi"] - r["ratio"])
            ax.errorbar(x, y, yerr=[lo, hi], fmt=mk, color=col, ms=3.5, capsize=1.5, lw=0.9, label=lab)
        ax.axhline(1, color="0.5", lw=0.7)
        ax.set_yscale("log")
        ax.set_yticks([0.8, 1, 1.5, 2, 3, 4, 6])
        ax.set_yticklabels(["0.8", "1", "1.5", "2", "3", "4", "6"])
        ax.set_xticks(range(len(scen_f)))
        ax.set_xticklabels([SCEN_LABEL[s].replace("$\\mathbf{B}$", "B") for s in scen_f], rotation=45, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", lw=0.3, alpha=0.5)
    axes[0].set_ylabel(f"$E_{{\\mathrm{{ref}}}}\\,/\\,E_{{\\mathrm{{method}}}}$")
    h, lab = axes[0].get_legend_handles_labels()
    f.legend(h, lab, loc="outside upper center", ncol=len(show), fontsize=7, frameon=False)
    f.savefig(fig / "ratios.pdf")
    print("rendered tables and figures (reference:", ref, ")")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    c = sp.add_parser("collect")
    c.add_argument("--root", required=True)
    c.add_argument("--out", required=True)
    r = sp.add_parser("render")
    r.add_argument("--results", default=str(PAPER / "results" / "results_small.json"))
    r.add_argument("--full", default=None)
    a = ap.parse_args()
    collect(a.root, a.out) if a.cmd == "collect" else render(a.results, a.full)


if __name__ == "__main__":
    main()
