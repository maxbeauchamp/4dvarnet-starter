#!/usr/bin/env python3
"""Scenario experiments with a trained N1 (formulation section 7): how S_C,
the forcing and the coarse state act on each other.

For each start date t (C tile with the most ASIP observations at t) and each
member s, a reference mode-A rollout of --lead days (free simulation, w = 0)
and modified rollouts with the same noise (paired: the differences isolate
the effect of the modification), all from the same N0 analysis at t:

    ic         S_C^0 + delta on shape channels (default edge_density,
               sic_std; delta = --delta x the channel's normalisation std),
               then free: response of X_C, memory delta_S(tau) / delta_S(0)
    placebo    spatial permutation of S_C^0: response to compare with ic
    forcing    T2m + --d-t2m, then winds rotated by --rotate
    prescribe  do(S = S_ref + delta) after every step
    nudge      S <- S + (S_ref + delta - S) / tau after every step
    downscale  fine levels of the 256 km zoom from Phi and Phi + delta (same
               noise): controllability slope dS_C(X_H) / dS* (~1 expected)
    gaps       I^up, I^down_f, I^down_X (loss gaps, groups set to null)
    modes      mode A vs mode B on the SAR coverage (full cascade at every
               step: heavy, not in "all")

Outputs in --out: scenarios.json (ocean means of the responses per lead,
member spread, memory, slopes, loss gaps, fraction of cells clipped by Pi_S),
responses.png, response_maps.png (last lead), downscale.png.

Usage:
    sbatch scripts/submit_odyssey.sbatch scripts/scenarios.py --ckpt <N1 last.ckpt> \\
        --start 2023-01-15 --end 2023-03-31 --n 4 --lead 7 --members 8 \\
        --out /Odyssey/private/$USER/results/scenarios
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.append(REPO)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from croscim import data as D  # noqa: E402
from croscim import grid as g  # noqa: E402
from croscim import obs  # noqa: E402
from croscim import rollout as R  # noqa: E402
from croscim.cascade import CROP_C, N_X, X_NAMES, Z_NAMES, crop  # noqa: E402
from croscim.ocean import ocean_fraction_h  # noqa: E402
from croscim.transition import load_n1  # noqa: E402
from plot_forecast import downscale_zoom  # noqa: E402
from plot_panels import best_crops, best_tile, extent, masked, panel  # noqa: E402

EXPS = ("ic", "placebo", "forcing", "prescribe", "nudge", "downscale", "gaps")
FM, FH = g.factor("M", "C"), g.factor("H", "M")


def ocean_mean(x, ocean):
    """Mean over the ocean C cells of x [..., 64, 64]."""
    return (x * ocean).sum((-2, -1)) / ocean.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="N1 checkpoint (Lightning .ckpt)")
    ap.add_argument("--n0-ckpt", default=None, help="N0 checkpoint, if it moved since N1 was trained")
    ap.add_argument("--root", default="/Odyssey/public/CROSCIM_dataset")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--n", type=int, default=3, help="number of start dates")
    ap.add_argument("--lead", type=int, default=7)
    ap.add_argument("--members", type=int, default=8)
    ap.add_argument("--exp", nargs="+", default=list(EXPS), choices=[*EXPS, "modes"])
    ap.add_argument("--channels", nargs="+", default=["edge_density", "sic_std"], help="S_C channels perturbed")
    ap.add_argument("--delta", type=float, default=1.0, help="perturbation, in normalisation stds of each channel")
    ap.add_argument("--d-t2m", type=float, default=2.0, help="forcing scenario: T2m increment (K)")
    ap.add_argument("--rotate", type=float, default=30.0, help="forcing scenario: wind rotation (degrees)")
    ap.add_argument("--tau", type=float, default=3.0, help="nudging time scale (days)")
    ap.add_argument("--n-steps", type=int, default=None, help="Heun steps (default: the model's)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if dev.type == "cuda" else torch.autocast("cpu", enabled=False)

    n1 = load_n1(args.ckpt, args.n0_ckpt).to(dev).eval()
    n0, K = n1.n0, args.lead
    n_steps = args.n_steps or n1.hparams.n_steps
    idx = R.channels(args.channels)
    delta = torch.tensor([args.delta * n0.nrm.norm[c][1] for c in args.channels], device=dev).view(1, -1, 1, 1)
    omega_h = ocean_fraction_h(os.path.join(REPO, "assets", "ne_10m_land.zip"))
    omega_t = torch.from_numpy(omega_h)
    omega_c = g.restrict(omega_t, torch.ones_like(omega_t), g.factor("H", "C"))[0].numpy()
    dm = D.CascadeDataModule(args.root, None, *[[args.start, args.end]] * 3, horizon=K)
    starts = dm._days("val")
    if not starts:
        raise SystemExit(f"no start date t in [{args.start}, {args.end}] with t..t+{K} in the stores")
    starts = [starts[int(i)] for i in np.linspace(0, len(starts) - 1, min(args.n, len(starts)))]

    resp = {e: [] for e in ("ic", "placebo", "forcing", "prescribe", "nudge")}  # [dates x members] of [K+1, 12]
    maps, results, fine_fig = {}, {"settings": vars(args), "dates": []}, None
    for t0 in starts:
        tic = time.perf_counter()
        date = str(D.EPOCH + dt.timedelta(t0))
        iy0, ix0 = best_tile(args.root, t0, omega_c)
        ds = D.CascadeDataset(args.root, [t0], omega_h, np.array([[iy0, ix0]]), train=False, horizon=K)
        days = D.build_batch({k: v.to(dev) for k, v in torch.utils.data.default_collate([ds[0]]).items()})
        ocean = (days[0]["omega_c"][0] > 0).float()
        rec = {"date": date, "tile": [iy0, ix0]}
        with torch.no_grad(), amp:
            for s in range(args.members):
                base, psi = R.rollout(n1, days, K, w=0.0, seed=s, sampler_steps=n_steps)
                z0, p0 = base[:, 0], psi[:, 0]

                def run(intervention=None, d=days):
                    z, _ = R.rollout(n1, d, K, w=0.0, seed=s, intervention=intervention, z0=z0, psi0=p0,
                                     sampler_steps=n_steps)
                    return (z - base)[0].float()                       # [K+1, 12, 64, 64]

                ref_s = lambda k: base[:, k][:, idx] + delta  # noqa: E731
                todo = {"ic": lambda: run(R.perturb_ic(delta, args.channels)),
                        "placebo": lambda: run(R.permute_s(seed=s)),
                        "forcing": lambda: run(d=R.modify_forcing(days, args.d_t2m, args.rotate)),
                        "prescribe": lambda: run(R.prescribe(ref_s, args.channels)),
                        "nudge": lambda: run(R.nudge(ref_s, args.channels, 1.0 / args.tau))}
                for e, fn in todo.items():
                    if e in args.exp:
                        diff = fn()
                        resp[e].append(ocean_mean(diff, ocean).cpu())
                        if s == 0 and t0 == starts[0]:
                            maps[e] = (diff[K].cpu(), ocean.cpu(), iy0, ix0)
            if "ic" in args.exp:
                # garde-fou: fraction of ocean cells where Pi_S clipped the perturbation
                pert = z0.clone()
                pert[:, idx] = pert[:, idx] + delta
                clipped = ((R.project_s(pert) - pert)[:, idx].abs() > 1e-6).float()
                rec["ic_clipped_fraction"] = float(ocean_mean(clipped[0], ocean).mean())
            if "downscale" in args.exp:
                oy, ox, hy, hx = best_crops(days[0])
                o = lambda v: torch.tensor([v], device=dev)  # noqa: E731
                w_h = crop(days[0]["omega_h"], o(hy), o(hx), CROP_C["H"], FM * FH)[0]
                ocean_z = crop(days[0]["omega_c"], o(hy), o(hx), CROP_C["H"], 1)[0] > 0.5
                slopes, fine_ref, fine_mod = {}, None, None
                for c, name in zip(idx, args.channels):
                    if c - N_X >= 8:  # sit_std is not a statistic of the 1 km SIC
                        continue
                    z1 = z0.clone()
                    z1[:, c] += delta[0, args.channels.index(name), 0, 0]
                    f0 = downscale_zoom(n0, days[0], z0, p0, oy, ox, hy, hx, n_steps, 0).to(dev)
                    f1 = downscale_zoom(n0, days[0], R.project_s(z1), p0, oy, ox, hy, hx, n_steps, 0).to(dev)
                    s0, s1 = obs.sic_stats(f0.double(), w_h.double())[c - N_X], obs.sic_stats(f1.double(), w_h.double())[c - N_X]
                    slopes[name] = float(((s1 - s0)[ocean_z] / delta[0, args.channels.index(name), 0, 0]).mean())
                    fine_ref, fine_mod = (fine_ref, fine_mod) if fine_ref is not None else (f0.cpu(), f1.cpu())
                rec["controllability_slope"] = slopes
                if fine_fig is None and fine_ref is not None:
                    fine_fig = (fine_ref, fine_mod, extent(iy0, ix0, CROP_C["H"], hy, hx),
                                (w_h > 0).cpu().numpy(), args.channels[0])
            if "gaps" in args.exp:
                rec["loss_gaps"] = {k: float(v) for k, v in R.loss_gaps(n1, days).items()}
            if "modes" in args.exp:
                rec["mode_A_vs_B"] = [{k: (float(v) if torch.is_tensor(v) else v) for k, v in d.items()}
                                      for d in R.compare_modes(n1, days, K)]
        rec["time_s"] = round(time.perf_counter() - tic, 1)
        results["dates"].append(rec)
        print(f"{date}: scenarios in {rec['time_s']} s", flush=True)

    # ---- summary: ocean-mean responses per lead (mean / std over dates x members) ----
    results["responses"] = {}
    for e, r in resp.items():
        if not r:
            continue
        r = torch.stack(r)                                              # [n, K+1, 12]
        results["responses"][e] = {name: {"mean": r[:, :, c].mean(0).tolist(), "std": r[:, :, c].std(0).tolist()}
                                   for c, name in enumerate(Z_NAMES)}
        if e == "ic":
            d0 = r[:, 0, idx].mean(0)
            results["responses"]["ic"]["memory"] = {n: (r[:, :, c].mean(0) / d0[i]).tolist()
                                                    for i, (n, c) in enumerate(zip(args.channels, idx))}
    with open(os.path.join(args.out, "scenarios.json"), "w") as f:
        json.dump(results, f, indent=2)
    figures(results, maps, fine_fig, args, os.path.join(args.out, ""))


def figures(results, maps, fine_fig, args, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rs = results["responses"]
    if rs:
        names = list(X_NAMES) + [args.channels[0]]
        fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 3.2))
        for ax, name in zip(axes, names):
            for e, r in rs.items():
                m, sd = np.array(r[name]["mean"]), np.array(r[name]["std"])
                lead = np.arange(len(m))
                ax.plot(lead, m, "-o", ms=3, label=e)
                ax.fill_between(lead, m - sd, m + sd, alpha=0.15)
            ax.axhline(0, color="k", lw=0.5)
            ax.set_title(f"response of {name} (ocean mean)", fontsize=9)
            ax.set_xlabel("lead (days)", fontsize=8)
            ax.tick_params(labelsize=7)
        axes[0].legend(fontsize=7)
        fig.suptitle("paired-noise responses (perturbed - reference), mean +- std over dates x members", fontsize=10)
        fig.tight_layout()
        fig.savefig(out + "responses.png", dpi=110)
        plt.close(fig)
    if maps:
        rows = []
        for e, (diff, ocean, iy0, ix0) in maps.items():
            e_c = extent(iy0, ix0, D.N_C)
            land = ocean.numpy() > 0
            c = Z_NAMES.index(args.channels[0])
            rows.append((e, [(f"dSIC t+{args.lead}", masked(diff[0].numpy(), land), e_c, dict(cmap="RdBu_r", vmin=-0.2, vmax=0.2), None, ~land),
                             (f"dSIT t+{args.lead} (m)", masked(diff[1].numpy(), land), e_c, dict(cmap="RdBu_r", vmin=-0.3, vmax=0.3), None, ~land),
                             (f"d{args.channels[0]} t+{args.lead}", masked(diff[c].numpy(), land), e_c, dict(cmap="PuOr"), None, ~land)]))
        panel(out + "response_maps.png", f"responses at t+{args.lead} (first start date, member 1)", rows)
    if fine_fig is not None:
        f0, f1, ext, ocean, name = fine_fig
        sic = dict(cmap="Blues_r", vmin=0, vmax=1)
        panel(out + "downscale.png", f"fine levels on demand (256 km zoom): Phi vs Phi + delta on {name}, same noise",
              [("H 1 km", [("SIC, reference Phi", masked(f0.numpy(), ocean), ext, sic, None, ~ocean),
                           (f"SIC, Phi + delta {name}", masked(f1.numpy(), ocean), ext, sic, None, ~ocean),
                           ("difference", masked((f1 - f0).numpy(), ocean), ext, dict(cmap="RdBu_r", vmin=-0.3, vmax=0.3), None, ~ocean)])])


if __name__ == "__main__":
    main()
