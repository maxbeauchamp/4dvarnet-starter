"""
Mechanistic diagnostic: projection of the reconstruction errors of several DA
methods (4DVarNet, 4DVar strong / weak, ...) onto the Lyapunov vectors of the DA
model along the true trajectory.

For each method m and each test day n, with e = x_m - x_truth (both layers):
    BLV (orthogonal) split, energy norm:
        |e|^2 = |e_unstable|^2 + |e_neutral|^2 + |e_stable,resolved|^2 + |e_residual|^2
        (residual = outside the span of the k leading BLVs = most stable directions)
    projections on the unstable / neutral / stable CLV subspaces of the resolved part,
    and the smallest angle between the unstable and stable CLV subspaces.
The key outputs are the *absolute* error energies per subspace: if 4DVarNet beats
4DVar mostly on the unstable component, its gain comes from better controlling
the growing directions (observability / optimisation); if it wins on the stable
part, the gain comes from the prior / from model error accumulating in directions
the dynamics would otherwise damp.

Run:  python main.py xp=qg2l_diag
"""
import json
from pathlib import Path

import numpy as np

from .qgmodel import QG2L
from . import lyapunov as LY

METHOD_COLORS = {"4dvarnet": "tab:red", "4dvar_strong": "tab:blue", "4dvar_weak": "tab:cyan"}


def _color(name, i):
    return METHOD_COLORS.get(name, f"C{i + 3}")


def load_reconstructions(recs, t0, t1, truth_ds):
    import xarray as xr

    out = {}
    for name, path in recs.items():
        if path is None or not Path(path).exists():
            print(f"[diag] skip {name}: {path} not found")
            continue
        ds = xr.open_dataset(path, decode_times=False).load()
        if 2 not in json.loads(ds.attrs.get("layers_estimated", "[1, 2]")):
            print(f"[diag] skip {name}: the deep layer is not estimated (layers=[1])")
            continue
        rec = ds.psi.sel(time=truth_ds.time.values[t0:t1], method="nearest")
        if rec.sizes["time"] != t1 - t0 or not np.allclose(rec.time.values, truth_ds.time.values[t0:t1]):
            raise ValueError(
                f"{name} ({path}) covers days [{rec.time.values.min():.0f}, {rec.time.values.max():.0f}] "
                f"(file: [{ds.time.values.min():.0f}, {ds.time.values.max():.0f}]) but the diagnostic needs "
                f"[{t0}, {t1 - 1}]: it was probably computed with other splits; delete it and recompute")
        out[name] = rec.values.astype(np.float64)
    return out


def kaplan_yorke(exps):
    e = np.sort(np.asarray(exps))[::-1]
    c = np.cumsum(e)
    if c[-1] >= 0:
        return np.nan  # spectrum truncated before the sum turns negative
    j = np.where(c >= 0)[0].max()
    return j + 1 + c[j] / abs(e[j + 1])


def run_diag(data_path, recs, out_dir, k=256, n_transient_fwd=30, n_transient_bwd=60, t_range=None,
             backend="torch", device="cuda", tlm="fd", neutral_tol=0.01, snapshot_offset=60,
             animation_days=60, make_figures=True, figure_methods=None):
    """
    data_path: OSSE netCDF;  recs: dict name -> reconstruction netCDF (psi, time, layer, y, x)
    t_range:   (t0, t1) days to diagnose (default: test split). Truth must be available on
               [t0 - n_transient_fwd, t1 + n_transient_bwd].
    figure_methods: subset of methods shown in the maps / animation (all by default; every method is
               always in the numbers, the spectrum and the subspace figures)
    k:         number of Lyapunov vectors; must exceed the number of unstable exponents
               (check ``n_unstable`` in summary.json: if n_unstable == k, increase k).
    """
    import xarray as xr

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = xr.open_dataset(data_path, decode_times=False).load()
    kw = json.loads(ds.attrs["model_kw"])
    splits = json.loads(ds.attrs["splits"])
    t0, t1 = t_range or splits["test"]
    a, b = t0 - n_transient_fwd, t1 + n_transient_bwd - 1
    if a < 0 or b >= ds.sizes["time"]:
        raise ValueError(f"truth needed on [{a}, {b}] but the dataset has {ds.sizes['time']} days")
    truth = ds.psi.values.astype(np.float64)
    rec = load_reconstructions(recs, t0, t1, ds)
    if not rec:
        raise ValueError("no usable reconstruction")

    if backend == "torch":
        import torch

        if str(device).startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
    m = QG2L(backend=backend, device=device, dtype="float64", **kw)
    traj_np = truth[a:b + 1]
    traj = m.xp.asarray(traj_np)
    errors = {}
    for name, r in rec.items():
        e = np.full(traj_np.shape, np.nan)
        e[n_transient_fwd:n_transient_fwd + (t1 - t0)] = r - truth[t0:t1]
        errors[name] = e
    tlm_fn = LY.make_tlm_torch(m) if tlm == "jvp" else LY.make_tlm_fd(m)
    snap = n_transient_fwd + min(snapshot_offset, t1 - t0 - 1)
    ck_dir = out_dir / "lyap_checkpoint"
    ck_key = json.dumps(dict(data=str(data_path), recs={nm: [str(p), Path(p).stat().st_mtime] for nm, p in recs.items()
                                                        if nm in rec},
                             k=k, fwd=n_transient_fwd, bwd=n_transient_bwd, t=[int(t0), int(t1)], tlm=tlm,
                             snap=int(snap)), sort_keys=True)
    lyap = LY.lyapunov_vectors(m, traj, k=k, tlm=tlm_fn, n_transient_fwd=n_transient_fwd,
                               n_transient_bwd=n_transient_bwd, errors=errors, snapshot_index=[snap],
                               checkpoint_dir=ck_dir, checkpoint_key=ck_key)
    dec = {name: LY.decompose(lyap, name, neutral_tol) for name in rec}
    uns, neu, sta = LY.classify_exponents(lyap["exponents"], neutral_tol)

    # ---- save -------------------------------------------------------------------
    days = ds.time.values[a + lyap["index"]]
    keys = ["total", "blv_unstable", "blv_neutral", "blv_stable_resolved", "residual",
            "blv_unstable_abs", "blv_neutral_abs", "blv_stable_resolved_abs", "residual_abs",
            "clv_unstable", "clv_neutral", "clv_stable_resolved", "angle_us_deg"]
    res = xr.Dataset(
        {f"{name}__{key}": (("time",), dec[name][key]) for name in dec for key in keys if key in dec[name]},
        coords=dict(time=days, k=np.arange(1, k + 1)),
    )
    res["exponents"] = ("k", lyap["exponents"])
    res["local_exp"] = (("time", "k"), lyap["local_exp"][-len(days):])
    for name in dec:
        c = lyap["proj"][name]["c"]
        res[f"{name}__coef2_blv"] = (("time", "k"), c**2)
    res.attrs.update(model_kw=json.dumps(kw), neutral_tol=neutral_tol, methods=json.dumps(list(dec)))
    res.to_netcdf(out_dir / "lyapunov_projection.nc")

    summary = dict(
        k=k, n_unstable=int(uns.sum()), n_neutral=int(neu.sum()), lambda1_per_day=float(lyap["exponents"][0]),
        kaplan_yorke_dim=float(kaplan_yorke(lyap["exponents"])),
        warning=("n_unstable == k: the unstable subspace is not fully resolved, increase k"
                 if uns.sum() == k else None),
        methods={},
    )
    for name in dec:
        d = dec[name]
        summary["methods"][name] = {
            "mean_error_energy": float(np.nanmean(d["total"])),
            **{f"mean_{key}": float(np.nanmean(d[key])) for key in keys[1:] if key in d},
        }
    names = list(dec)
    if len(names) > 1:
        ref = names[0]
        summary["gain_vs_" + ref] = {
            other: {sub: float(1 - np.nanmean(dec[ref][sub + "_abs"]) / np.nanmean(dec[other][sub + "_abs"]))
                    for sub in ["blv_unstable", "blv_neutral", "blv_stable_resolved", "residual"]}
            for other in names[1:]
        }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))

    import shutil

    shutil.rmtree(ck_dir, ignore_errors=True)  # results are saved: the forward-sweep checkpoint is no longer needed
    if make_figures:
        figures(out_dir, lyap, dec, rec, truth, ds, kw, t0, t1, snap, n_transient_fwd, a, neutral_tol,
                animation_days, figure_methods)
    return summary


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def figures(out_dir, lyap, dec, rec, truth, ds, kw, t0, t1, snap, n_fwd, a, neutral_tol, animation_days,
            figure_methods=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from . import viz

    names = list(dec)
    exps = lyap["exponents"]
    k = len(exps)
    uns, neu, sta = LY.classify_exponents(exps, neutral_tol)
    days = ds.time.values[a + lyap["index"]]
    f0g = kw["f0"] / kw["g"]

    # 1. Lyapunov spectrum + error distribution along it --------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.3), constrained_layout=True)
    idx = np.arange(1, k + 1)
    ax[0].plot(idx, exps, "k.-", ms=3)
    ax[0].axhline(0, c="gray", lw=0.8)
    ax[0].fill_between(idx, exps.min(), exps.max(), where=uns, color="tab:red", alpha=0.08, label="unstable")
    ax[0].set_xlabel("index i")
    ax[0].set_ylabel(r"$\lambda_i$ [day$^{-1}$]")
    ax[0].set_title(f"Lyapunov spectrum of the DA model (n_unstable={uns.sum()}/{k})")
    ax[0].legend()
    for i, name in enumerate(names):
        c2 = lyap["proj"][name]["c"] ** 2
        ax[1].semilogy(idx, np.nanmean(c2, 0), c=_color(name, i), label=name)
    ax[1].axvspan(0.5, uns.sum() + 0.5, color="tab:red", alpha=0.08)
    ax[1].set_xlabel("BLV index i")
    ax[1].set_ylabel(r"mean $\langle e, \mathrm{BLV}_i\rangle_E^2$")
    ax[1].set_title("error energy along the Lyapunov spectrum")
    ax[1].legend()
    fig.savefig(out_dir / "lyapunov_spectrum_error_distribution.png", dpi=120)
    plt.close(fig)

    # 2. absolute error energy per subspace (time series) ---------------------------------
    subs = [("blv_unstable_abs", "unstable"), ("blv_neutral_abs", "neutral"),
            ("blv_stable_resolved_abs", "stable (resolved)"), ("residual_abs", "residual (beyond k)")]
    fig, ax = plt.subplots(len(subs) + 1, 1, figsize=(11, 11), sharex=True, constrained_layout=True)
    for i, name in enumerate(names):
        ax[0].semilogy(days, dec[name]["total"], c=_color(name, i), label=name)
        for j, (key, lab) in enumerate(subs):
            ax[j + 1].semilogy(days, np.maximum(dec[name][key], 1e-20), c=_color(name, i))
    ax[0].set_title(r"total error energy $|e|_E^2$")
    ax[0].legend(ncol=len(names))
    for j, (_, lab) in enumerate(subs):
        ax[j + 1].set_title(lab)
    ax[-1].set_xlabel("day")
    fig.savefig(out_dir / "error_energy_by_subspace.png", dpi=120)
    plt.close(fig)

    # 3. summary bars: mean absolute energies and fractions -------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3), constrained_layout=True)
    w = 0.8 / len(names)
    for i, name in enumerate(names):
        vals = [np.nanmean(dec[name][key]) for key, _ in subs]
        ax[0].bar(np.arange(len(subs)) + i * w, vals, w, color=_color(name, i), label=name)
        fr = [np.nanmean(dec[name][key.replace("_abs", "")]) for key, _ in subs]
        ax[1].bar(np.arange(len(subs)) + i * w, fr, w, color=_color(name, i))
        if "clv_unstable" in dec[name]:
            cl = [np.nanmean(dec[name][f"clv_{s}"]) for s in ["unstable", "neutral", "stable_resolved"]]
            ax[2].bar(np.arange(3) + i * w, cl, w, color=_color(name, i))
    for a_, labs in [(ax[0], [s[1] for s in subs]), (ax[1], [s[1] for s in subs]),
                     (ax[2], ["unstable", "neutral", "stable (resolved)"])]:
        a_.set_xticks(np.arange(len(labs)) + 0.4 - w / 2)
        a_.set_xticklabels(labs, rotation=15)
    ax[0].set_yscale("log")
    ax[0].set_title("mean error energy per subspace (BLV)")
    ax[0].legend()
    ax[1].set_title("fraction of error energy (BLV, sums to 1)")
    ax[2].set_title(r"projection on CLV subspaces, $|P_X e|^2 / |Pe|^2$")
    fig.savefig(out_dir / "subspace_summary.png", dpi=120)
    plt.close(fig)

    map_names = [n for n in names if not figure_methods or n in figure_methods]

    # 4. maps: leading CLVs/BLVs and errors at the snapshot day --------------------------------
    if snap in lyap["snapshots"]:
        V = lyap["snapshots"][snap]  # (k, 2, n, n) BLVs
        j = snap - lyap["index"][0]
        clv = np.einsum("k...,kj->j...", V.astype(np.float64), lyap["C"][j]) if "C" in lyap else V
        t_abs = a + snap
        n_m = len(map_names)
        fig, ax = plt.subplots(2, 4 + n_m, figsize=(3.3 * (4 + n_m), 6.4), constrained_layout=True)
        ext = [0, kw["L"] / 1e3, 0, kw["L"] / 1e3]
        panels = [("SSH truth", truth[t_abs, 0], truth[t_abs, 1], "psi"),
                  ("CLV 1", clv[0, 0], clv[0, 1], "v"), (f"CLV {max(uns.sum() // 2, 1)}",
                                                         clv[max(uns.sum() // 2, 1) - 1, 0],
                                                         clv[max(uns.sum() // 2, 1) - 1, 1], "v"),
                  (f"CLV {min(uns.sum() + 5, k)}", clv[min(uns.sum() + 5, k) - 1, 0],
                   clv[min(uns.sum() + 5, k) - 1, 1], "v")]
        for name in map_names:
            e = rec[name][t_abs - t0] - truth[t_abs]
            panels.append((f"error {name}", e[0], e[1], "e"))
        emax = max(np.abs(rec[nm][t_abs - t0] - truth[t_abs]).max() for nm in map_names)
        for col, (title, l1, l2, kind) in enumerate(panels):
            for row, fld in enumerate([l1, l2]):
                if kind == "psi":
                    v = np.abs(truth[t_abs]).max()
                elif kind == "e":
                    v = emax
                else:
                    v = np.abs(np.stack([l1, l2])).max()
                ax[row, col].imshow(fld, origin="lower", extent=ext, cmap="RdBu_r", vmin=-v, vmax=v)
                ax[row, col].set_xticks([])
                ax[row, col].set_yticks([])
                ax[row, col].set_title(f"{title} - layer {row + 1}", fontsize=9)
        fig.suptitle(f"day {ds.time.values[t_abs]:.0f}: covariant Lyapunov vectors and reconstruction errors (psi)")
        fig.savefig(out_dir / "clv_and_error_maps.png", dpi=110)
        plt.close(fig)

    # 5. animation: truth / obs / reconstructions / errors (SSH) --------------------------------
    if animation_days:
        fields = {}
        T = ds.sizes["time"]
        for name in map_names:
            full = np.full((T,) + rec[name].shape[-2:], np.nan)
            full[t0:t1] = f0g * rec[name][:, 0]
            fields[f"SSH {name}"] = full
        for name in map_names:
            err = np.full((T,) + rec[name].shape[-2:], np.nan)
            err[t0:t1] = f0g * (rec[name][:, 0] - truth[t0:t1, 0])
            fields[f"SSH err {name}"] = err
        viz.animate(ds, out_dir / "reconstructions.mp4", days=range(t0, min(t0 + animation_days, t1)),
                    fields=fields)
