"""(rebuttal) Why is FM SIC under-dispersed? Diagnostics on the trained FM_v2 SIC model, no retraining.

  1. bounds    : SSR / rank histogram on all valid pixels vs excluding pixels where the truth is at a bound (0 or 1);
  2. distance  : SSR / rank deviation by distance (pixels) to the nearest SAR observation of the same frame;
  3. ODE steps : 20 vs 50 Heun steps (best model);
  4. epochs    : SSR of every checkpoint kept on disk (logs_FM_sic*/checkpoints, EMA weights), 20 steps.
Everything on the test set, ensembles of M members, same data module as the notebook.
Run from consistency/Notebooks/Notebooks_SIC:  python ../../rebuttal/diag_fm_sic.py [--members 20] [--batches 0]
Writes ../../sbatch_submission/results/SIC/FM_v2_diag.json and prints a summary.
"""
import argparse
import functools as ft
import glob
import json
import os
import re
import sys
import zipfile

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

sys.path.append("../..")
sys.path.insert(0, "../../..")
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
from flowmatching_models.flowmatching_models_FM import UNetFM, UNetFMConfig, heun_sample  # noqa: E402
from src.dataloader_SIC import BaseDataModule, TrainingItem_4da  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--members", type=int, default=20)
ap.add_argument("--batches", type=int, default=0, help="0 = whole test set")
ap.add_argument("--log", default="logs_FM_sic_v2")
args = ap.parse_args()
M = args.members
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CroppedDataModule(BaseDataModule):
    def __init__(self, *a, t_crop=9, **k):
        super().__init__(*a, **k)
        self.t_crop = t_crop

    def build_batch(self):
        return ft.partial(ft.reduce, lambda i, f: f(i), [TrainingItem_4da._make])

    def setup(self, stage="test"):
        super().setup(stage)
        for ds in [self.train_ds, self.val_ds, self.test_ds]:
            ds.db = ds.db.isel(time=slice(-self.t_crop, None))


dm = CroppedDataModule(
    asip_paths="../../../data/asip_database_daw15_sparse.nc",
    split_train=slice(0, 300), split_val=slice(300, 350), split_test=slice(350, 393),
    da=True, norm_stats=[0, 1],
    norm_stats_covs=[{"t2m": 270.08, "istl1": 267.68, "siconc": 0, "sst": 276.97, "skt": 270.50},
                     {"t2m": 14.67, "istl1": 7.80, "siconc": 1, "sst": 6.82, "skt": 15.21}],
    t_crop=9)
dm.setup()
batches = list(dm.test_dataloader())
if args.batches:
    batches = batches[:args.batches]
print(f"test batches: {len(batches)}, members: {M}", flush=True)


def ensemble(net, n_steps, seed=42):
    """List of (ens (M,B,C,H,W) np, gt (B,C,H,W) np with NaN, sar (B,C,H,W) np with NaN)."""
    out = []
    g = torch.Generator(device=dev).manual_seed(seed)
    for b in batches:
        gt = b.asip.to(dev).float()
        y = torch.cat((b.input, b.osisaf), 1).to(dev).float()
        mem = []
        with torch.no_grad():
            for _ in range(M):
                z = torch.randn(gt.shape, generator=g, device=dev)
                mem.append(heun_sample(net, z, y, n_steps=n_steps)[0].float().cpu().numpy())
        out.append((np.stack(mem), gt.cpu().numpy(), b.input.numpy()))
    return out


def stats(ens, gt, sel):
    """SSR, spread, rmse, bias, normalised rank histogram on the selected pixels."""
    e = ens[:, sel]                          # (M, n)
    x = gt[sel]
    if x.size < 100:
        return None
    mean = e.mean(0)
    var = e.var(0, ddof=1).mean()
    mse = ((mean - x) ** 2).mean()
    ranks = (e < x[None]).sum(0) + np.random.default_rng(0).integers(0, (e == x[None]).sum(0) + 1)
    h = np.bincount(ranks, minlength=M + 1).astype(float)
    h /= h.mean()
    return dict(n=int(x.size), ssr=float(np.sqrt(var / mse) * np.sqrt((M + 1) / M)), spread=float(np.sqrt(var)),
                rmse=float(np.sqrt(mse)), bias=float((mean - x).mean()), rank_dev=float(np.abs(h - 1).mean()),
                rank_lo=float(h[0]), rank_hi=float(h[-1]))


def analyse(res, detailed=True):
    ens = np.concatenate([r[0] for r in res], 1)      # (M, N, C, H, W)
    gt = np.concatenate([r[1] for r in res], 0)
    sar = np.concatenate([r[2] for r in res], 0)
    valid = np.isfinite(gt)
    out = {"all": stats(ens, gt, valid)}
    if not detailed:
        return out
    eps = 1e-3
    at_bound = (gt <= eps) | (gt >= 1 - eps)
    out["truth_at_bound_frac"] = float(at_bound[valid].mean())
    out["interior"] = stats(ens, gt, valid & ~at_bound)
    out["at_bound"] = stats(ens, gt, valid & at_bound)
    out["miz"] = stats(ens, gt, valid & (gt > 0.15) & (gt < 0.85))
    obs = np.isfinite(sar)
    dist = np.zeros(gt.shape, dtype=np.float32)
    for i in range(gt.shape[0]):
        for k in range(gt.shape[1]):
            dist[i, k] = distance_transform_edt(~obs[i, k]) if obs[i, k].any() else 1e9
    out["by_distance"] = {}
    for lo, hi in [(0, 0.5), (0.5, 5), (5, 20), (20, 60), (60, 1e10)]:
        out["by_distance"][f"{lo}-{hi}"] = stats(ens, gt, valid & ~at_bound & (dist >= lo) & (dist < hi))
    return out


def ema_from_ckpt(path):
    sd = torch.load(path, map_location="cpu")["state_dict"]
    w = {k[len("ema_model.module."):]: v for k, v in sd.items() if k.startswith("ema_model.module.")}
    cfg = json.load(open(os.path.join(args.log, "best_model", "config.json")))
    net = UNetFM(UNetFMConfig(**cfg))
    net.load_state_dict(w)
    return net.eval().to(dev)


def valid_zip(p):
    try:
        with zipfile.ZipFile(p) as z:
            z.testzip()
        return True
    except Exception:
        return False


report = {}
best = UNetFM.from_pretrained(os.path.join(args.log, "best_model")).eval().to(dev)
for n in (20, 50):
    report[f"best_heun{n}"] = analyse(ensemble(best, n), detailed=(n == 20))
    print(f"best, {n} Heun steps:", json.dumps(report[f'best_heun{n}']['all']), flush=True)
report["checkpoints"] = {}
for p in sorted(glob.glob("logs_FM_sic*/checkpoints/*.ckpt")):
    if not valid_zip(p):
        continue
    try:
        ep = torch.load(p, map_location="cpu").get("epoch")
        r = analyse(ensemble(ema_from_ckpt(p), 20), detailed=False)["all"]
    except Exception as e:                      # e.g. a checkpoint of another architecture
        print(f"skip {p}: {e!r}"[:200], flush=True)
        continue
    report["checkpoints"][p] = dict(epoch=ep, **r)
    print(f"{p} (epoch {ep}):", json.dumps(r), flush=True)

os.makedirs("../../sbatch_submission/results/SIC", exist_ok=True)
json.dump(report, open("../../sbatch_submission/results/SIC/FM_v2_diag.json", "w"), indent=1)
b = report["best_heun20"]
print("\n== SUMMARY ==")
print(f"all          SSR {b['all']['ssr']:.3f}  rank lo/hi {b['all']['rank_lo']:.2f}/{b['all']['rank_hi']:.2f}")
print(f"truth at a bound: {100 * b['truth_at_bound_frac']:.1f}% of valid pixels")
for k in ("interior", "at_bound", "miz"):
    if b[k]:
        print(f"{k:12s} SSR {b[k]['ssr']:.3f}  rmse {b[k]['rmse']:.4f}  rank lo/hi {b[k]['rank_lo']:.2f}/{b[k]['rank_hi']:.2f}")
for k, v in b["by_distance"].items():
    if v:
        print(f"dist {k:9s} SSR {v['ssr']:.3f}  rmse {v['rmse']:.4f}  n {v['n']}")
print(f"50 Heun steps: SSR {report['best_heun50']['all']['ssr']:.3f}")
for p, v in sorted(report["checkpoints"].items(), key=lambda kv: kv[1]["epoch"] or 0):
    print(f"epoch {v['epoch']}: SSR {v['ssr']:.3f}  rmse {v['rmse']:.4f}  ({p})")
