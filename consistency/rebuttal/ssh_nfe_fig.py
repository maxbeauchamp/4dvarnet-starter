"""SSH NFE figure (appendix): RMSE, CRPS, SSR versus NFE, full test set, 20 members, same checkpoints as Table 3.

Reads results/SSH_GF/<method>{,_nfe<tag>}_metrics_calibration.csv (run_metrics.py --set NSTEPS_EVAL=n --out-tag _nfe<n>)
and converts the step counts to network evaluations with the sampler of each model (checked on the GP sweeps, where
the NFE was counted by wrapping the network):
  CM, DynCM, DynVarCM (uniform grid): nsteps - 1;  VarCM: nsteps;  FM: 2 nsteps (Heun);
  DynFM: 2 x (Heun steps of the boundary-aware grid);  DynVarCM frame grid: K x C (C = 9 frames).

    python ssh_nfe_fig.py RESULTS_DIR OUT.pdf [OUT.csv]
"""
import glob, os, re, sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

R = sys.argv[1] if len(sys.argv) > 1 else "results/SSH_GF"
OUT = sys.argv[2] if len(sys.argv) > 2 else "ssh_nfe.pdf"
CSV = sys.argv[3] if len(sys.argv) > 3 else os.path.splitext(OUT)[0] + ".csv"
C = 9


def dynfm_nfe(n, sf=0.3, c=C):
    t = {i / n for i in range(n + 1)} | {sf} | {sf + (1 - sf) * k / (c - 1) for k in range(c)}
    return 2 * (len(t) - 1)


# name, method key, default nsteps of Table 3, nfe(n), style
MODELS = [
    ("FM", "FM_aug", 20, lambda n: 2 * n),
    ("DynFM", "DynFM_v2", 4 * C + 1, dynfm_nfe),
    ("CM", "CMT_aug", 10, lambda n: n - 1),
    ("DynCM", "DynCMT", 4 * C + 1, lambda n: n - 1),
    ("VarCM", "VarCM_aug_student", 15, lambda n: n),
    ("DynVarCM (uniform grid)", "VarDynCM_scratch", 5 * C, lambda n: n - 1),
]
col = {"FM": "#555555", "CM": "#2a6fb0", "VarCM": "#2e8b57"}


def read(path):
    d = pd.read_csv(path)
    return dict(rmse=float(d.rmse[0]), crps=float(d.crps_all[0]), ssr=float(d.ssr[0]), spread=float(d.spread[0]))


rows = []
for name, key, n0, nfe in MODELS:
    pts = []
    base = f"{R}/{key}_metrics_calibration.csv"
    if os.path.exists(base):
        pts.append((n0, read(base)))
    for f in glob.glob(f"{R}/{key}_nfe[0-9]*_metrics_calibration.csv"):
        n = int(re.search(r"_nfe(\d+)_metrics", f).group(1))
        pts.append((n, read(f)))
    for n, v in pts:
        rows.append(dict(model=name, nsteps=n, nfe=nfe(n), **v))
for k in (1, 2):
    f = f"{R}/VarDynCM_scratch_nfe_fr{k}_metrics_calibration.csv"
    if os.path.exists(f):
        rows.append(dict(model="DynVarCM (frame grid)", nsteps=k, nfe=k * C, **read(f)))
df = pd.DataFrame(rows).sort_values(["model", "nfe"])
df.to_csv(CSV, index=False)
print(df.to_string(index=False))

plt.rcParams.update({"font.size": 9, "font.family": "serif", "mathtext.fontset": "cm"})
fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.4), sharex=True)
handles = []
ORDER = ["FM", "CM", "VarCM", "DynFM", "DynCM", "DynVarCM (uniform grid)", "DynVarCM (frame grid)"]
for name in [m for m in ORDER if m in set(df.model)]:
    d = df[df.model == name]
    base = name.split(" ")[0].replace("Dyn", "") or "FM"
    dyn = name.startswith("Dyn")
    fr = "frame" in name
    kw = dict(color=col[base], ls=":" if fr else ("--" if dyn else "-"), marker="D" if fr else ("s" if dyn else "o"),
              ms=3.5, lw=1.4, label=name)
    h, = axs[0].plot(d.nfe, d.rmse, **kw); handles.append(h)
    axs[1].plot(d.nfe, d.crps, **kw)
    axs[2].plot(d.nfe, d.ssr, **kw)
axs[0].set_ylabel("RMSE (m)"); axs[1].set_ylabel("CRPS (m)"); axs[2].set_ylabel("spread–skill ratio")
axs[2].axhline(1.0, color="k", lw=0.8, ls=":")
for a in axs:
    a.set_xscale("log"); a.set_xlabel("network evaluations (NFE)")
    a.grid(alpha=0.3, lw=0.5)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.0), fontsize=8,
           columnspacing=1.0, handlelength=2.2)
fig.tight_layout(rect=(0, 0.14, 1, 1))
fig.savefig(OUT, bbox_inches="tight")
print("saved", OUT)
