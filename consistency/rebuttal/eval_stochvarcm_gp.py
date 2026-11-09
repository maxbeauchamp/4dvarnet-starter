#!/usr/bin/env python3
"""Full-test-set evaluation of the existing stochastic VarCM checkpoints on GP (no retraining).

Answers XETn W4 ("App. D's stochastic variants are never evaluated"). The model code is taken verbatim from
Notebooks_GP/Notebook_consistency_model_StochVarCM_spde.ipynb: its definition cells are executed, then the saved
EMA model and costs (<LOG_DIR>/best_model) are loaded and sampled in ODE and SDE mode on every test window.

    python consistency/rebuttal/eval_stochvarcm_gp.py --cond grad obs --members 20 --nsteps 40

Writes, under consistency/sbatch_submission/results/GP/:
    StochVarCM_<cond>_<ODE|SDE>_metrics_calibration.csv   (spread, RMSE, SSR, CRPS all px, rank deviation, ranks)
    StochVarCM_metrics_summary.csv                         (one row per variant, + RMSE to the OI field)
"""
import argparse, csv, json, os, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONS = HERE.parent
NB_DIR = CONS / "Notebooks" / "Notebooks_GP"
NB = NB_DIR / "Notebook_consistency_model_StochVarCM_spde.ipynb"
OUT = CONS / "sbatch_submission" / "results" / "GP"
DEF_CELLS = [3, 5, 8, 10, 12, 14, 15]          # imports, datamodule, UNet blocks, UNet, costs, stoch_forward, Lit


def load_namespace():
    nb = json.load(open(NB))
    ns = {"__name__": "stochvarcm_eval", "display": print}
    os.chdir(NB_DIR)                               # the notebook's relative paths (data, sys.path) assume this
    import matplotlib
    matplotlib.use("Agg")
    for i in DEF_CELLS:
        src = "".join(nb["cells"][i]["source"])
        src = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith(("!", "%")))
        exec(compile(src, f"<cell {i}>", "exec"), ns)
    sampler = "".join(nb["cells"][25]["source"]).split("# ── Generate ensembles")[0]
    exec(compile(sampler, "<cell 25: sampler>", "exec"), ns)
    return ns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", nargs="+", default=["grad", "obs"])
    ap.add_argument("--members", type=int, default=20)
    ap.add_argument("--nsteps", type=int, default=40)
    ap.add_argument("--max-windows", type=int, default=0, help="0 = whole test set")
    a = ap.parse_args()

    ns = load_namespace()
    torch, np, nn = ns["torch"], ns["np"], ns["nn"]
    sys.path.insert(0, str(CONS))
    from calibration import calib_stats, calib_summary, write_calibration_csv

    dm = ns["datamodule"]
    m_norm, s_norm = dm.norm_stats()
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    C = dm.window_size
    summary_rows = []

    for cond in a.cond:
        log_dir = NB_DIR / f"logs_CT_VarCM_stoch_spde_{cond}"
        mp = log_dir / "best_model"
        obs_cost = ns["BaseObsCost"](sigma_obs=1.0)
        prior_cost = ns["BilinAEPriorCost"](dim_in=C)
        if (mp / "model.pt").exists() and (mp / "costs.pt").exists():
            unet = ns["UNet"].from_pretrained(str(mp))
            costs = torch.load(mp / "costs.pt", map_location="cpu")
            obs_cost.load_state_dict(costs["obs_cost"]); prior_cost.load_state_dict(costs["prior_cost"])
            src = str(mp)
        else:
            # best_model is only written at the end of trainer.fit; the June runs stopped before that,
            # so take the EMA student and the costs from the last Lightning checkpoint instead.
            ck = log_dir / "checkpoints" / "last.ckpt"
            if not ck.exists():
                print(f"[skip] no best_model nor {ck}"); continue
            sd = torch.load(ck, map_location="cpu", weights_only=False)["state_dict"]
            part = lambda pre: {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
            unet = ns["UNet"](ns["UNetConfig"](channels=C, stoch=True))
            unet.load_state_dict(part("ema_student_model."))
            obs_cost.load_state_dict(part("obs_cost.")); prior_cost.load_state_dict(part("prior_cost."))
            costs = dict(lambda_reg=sd["lambda_reg"], sigma_noise=10.0, schedule_power=1.0, conditioning_mode=cond)
            src = f"{ck} (epoch {torch.load(ck, map_location='cpu', weights_only=False).get('epoch')})"
        unet = unet.eval().to(device=device, dtype=dtype)
        obs_cost = obs_cost.to(device=device, dtype=dtype)
        prior_cost = prior_cost.to(device=device, dtype=dtype)
        obs_cost.eval(); prior_cost.eval()
        lam = nn.Parameter(costs["lambda_reg"].to(device=device, dtype=torch.float32))
        kw = dict(nsteps=a.nsteps, sigma_noise=costs.get("sigma_noise", 10.0),
                  schedule_power=costs.get("schedule_power", 1.0),
                  conditioning_mode=costs.get("conditioning_mode", cond))
        print(f"[model] {src}: {kw}", flush=True)

        for mode in ("ODE", "SDE"):
            ns["seed_everything"](42)
            stats, rmse_oi, n_win, t0 = [], [], 0, time.time()
            for k, tb in enumerate(dm.test_dataloader()):
                if a.max_windows and n_win >= a.max_windows:
                    break
                y = tb.input.to(device=device, dtype=dtype)
                gt = tb.tgt[0].float().numpy() * s_norm + m_norm                         # (C, H, W)
                ens = torch.stack([ns["sample_stoch_varcm"](unet, obs_cost, prior_cost, lam, y,
                                                            stoch=(mode == "SDE"), **kw)[0]
                                   for _ in range(a.members)], 0).squeeze(1).float().numpy() * s_norm + m_norm
                start = dm.test_ds.indices[k]
                oi = dm.oi[start:start + C].astype(np.float32)
                for t in range(C):
                    valid = np.isfinite(gt[t])
                    stats.append(calib_stats(ens[:, t], gt[t], valid))
                    rmse_oi.append(float(np.sqrt(np.mean((ens[:, t].mean(0)[valid] - oi[t][valid]) ** 2))))
                n_win += 1
            s = calib_summary(stats)
            name = f"StochVarCM_{cond}_{mode}"
            write_calibration_csv(OUT / f"{name}_metrics_calibration.csv", name, s, a.members)
            row = dict(method=name, windows=n_win, members=a.members, nsteps=a.nsteps,
                       rmse=s["rmse"], spread=s["spread"], ssr=s["ssr"], crps_all=s["crps_all"],
                       rank_dev=s["rank_dev"], rmse_to_oi=float(np.mean(rmse_oi)),
                       seconds=round(time.time() - t0, 1))
            summary_rows.append(row)
            print(f"[{name}] " + "  ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                                           for k, v in row.items()), flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "StochVarCM_metrics_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
        w.writeheader(); w.writerows(summary_rows)


if __name__ == "__main__":
    main()
