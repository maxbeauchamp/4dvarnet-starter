"""
Cached end-to-end pipeline for scenario studies (used by qg2l_realism_sweep.ipynb).

Everything is keyed by a hash of its configuration, so scenarios that only differ in
DA settings share the same data and the same trained 4DVarNet, and re-running a
notebook only recomputes what changed:

    <root>/data/<gen_hash>/                      OSSE netCDF + figures
    <root>/4dvarnet/<gen_hash>_<net_hash>/       checkpoints, logs, 4dvarnet.nc
    <root>/scenarios/<name>/<method>.nc          4DVar / hybrid analyses
    <root>/scenarios/<name>/diag/                Lyapunov diagnostic
"""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np


def _hash(cfg):
    return hashlib.sha1(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:10]


def _sidecar_ok(path, cfg):
    side = Path(str(path) + ".json")
    return Path(path).exists() and side.exists() and json.loads(side.read_text()) == json.loads(
        json.dumps(cfg, sort_keys=True, default=str))


def _write_sidecar(path, cfg):
    Path(str(path) + ".json").write_text(json.dumps(cfg, sort_keys=True, default=str, indent=1))


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def generate_cached(root, gen, device="cuda", force=False):
    """gen: kwargs of simulate.generate (preset, nx, nx_truth, L, splits, constellation, truth_overrides,
    forcing, obs_source, unresolved_signal, ...). Returns the netCDF path."""
    from . import simulate

    h = _hash(gen)
    out_dir = Path(root) / "data" / h
    gen = dict(gen)
    path = out_dir / f"qg2l_{gen.get('preset', 'energetic')}_nx{gen.get('nx', 64)}_truth{gen.get('nx_truth', 128)}.nc"
    if not force and _sidecar_ok(path, gen):
        return path
    out_dir.mkdir(parents=True, exist_ok=True)
    p = simulate.generate(out_dir=str(out_dir), backend="torch", device=device, **gen)
    _write_sidecar(p, gen)
    return Path(p)


# ---------------------------------------------------------------------------
# 4DVarNet
# ---------------------------------------------------------------------------
DEFAULT_NET = dict(n_time=21, epochs=120, n_step=15, batch_size=16, hidden_prior=64, hidden_grad=96,
                   aug_factor=5, downsamp=None, layers=[1, 2], lr=1e-3, num_workers=0)


def train_4dvarnet(root, data_path, net=None, retrain=False, accelerator="auto"):
    """Train (or resume / reuse) 4DVarNet on ``data_path``; returns the path of the test reconstruction."""
    import torch
    import pytorch_lightning as pl

    import src.models
    from .data import QG2LDataModule, qg_rec_weight, n_channels
    from .models import LitQG2L, periodic

    net = {**DEFAULT_NET, **(net or {})}
    gen_hash = Path(data_path).parent.name
    run_dir = Path(root) / "4dvarnet" / f"{gen_hash}_{_hash(net)}"
    rec_path = run_dir / "4dvarnet.nc"
    ckpt_dir = run_dir / "checkpoints"
    nx = json.loads(__import__("xarray").open_dataset(data_path, decode_times=False).attrs["model_kw"])["nx"]

    pl.seed_everything(333)
    C = n_channels(net["n_time"], net["layers"])
    dm = QG2LDataModule(str(data_path), n_time=net["n_time"], layers=net["layers"], aug_factor=net["aug_factor"],
                        dl_kw=dict(batch_size=net["batch_size"], num_workers=net["num_workers"]))
    solver = src.models.GradSolver(
        prior_cost=periodic(src.models.BilinAEPriorCost(C, net["hidden_prior"], downsamp=net["downsamp"],
                                                        bilin_quad=False)),
        obs_cost=src.models.BaseObsCost(),
        grad_mod=periodic(src.models.ConvLstmGradModel(C, net["hidden_grad"], downsamp=net["downsamp"])),
        n_step=net["n_step"], lr_grad=1e3,
    )

    def opt_fn(lit):
        lr = net["lr"]
        opt = torch.optim.Adam([
            {"params": lit.solver.grad_mod.parameters(), "lr": lr},
            {"params": lit.solver.obs_cost.parameters(), "lr": lr},
            {"params": lit.solver.prior_cost.parameters(), "lr": lr / 2},
        ])
        return {"optimizer": opt, "lr_scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=net["epochs"])}

    lit = LitQG2L(solver=solver, rec_weight=qg_rec_weight(net["n_time"], net["layers"], nx), opt_fn=opt_fn,
                  persist_rw=False, export_path=str(rec_path))
    last = ckpt_dir / "last.ckpt"
    resume = None if retrain or not last.exists() else last
    epoch = torch.load(resume, map_location="cpu", weights_only=False)["epoch"] if resume else -1
    done = epoch + 1 >= net["epochs"]
    trainer = pl.Trainer(max_epochs=net["epochs"], accelerator=accelerator, devices=1, inference_mode=False,
                         gradient_clip_val=0.5, logger=pl.loggers.CSVLogger(str(run_dir / "logs"), name=""),
                         callbacks=[pl.callbacks.ModelCheckpoint(
                             dirpath=str(ckpt_dir), monitor="val_mse", save_top_k=1, save_last=True,
                             filename="best-{epoch:03d}-{val_mse:.5f}")],
                         enable_progress_bar=False)
    if done:
        best = sorted(ckpt_dir.glob("best-*.ckpt")) or [last]
        if not rec_path.exists():
            trainer.test(lit, datamodule=dm, ckpt_path=str(best[-1]))
        print(f"[4dvarnet] reuse {run_dir}")
    else:
        print(f"[4dvarnet] {'resume from epoch ' + str(epoch + 1) if resume else 'train'} -> {run_dir}")
        trainer.fit(lit, datamodule=dm, ckpt_path=str(resume) if resume else None)
        trainer.test(lit, datamodule=dm, ckpt_path="best")
    _write_sidecar(rec_path, dict(data=str(data_path), net=net))
    return rec_path


def training_curves(rec_path):
    import pandas as pd

    runs = sorted((Path(rec_path).parent / "logs").glob("**/metrics.csv"), key=lambda q: q.stat().st_mtime)
    return pd.concat([pd.read_csv(r) for r in runs]).groupby("epoch").last() if runs else pd.DataFrame()


# ---------------------------------------------------------------------------
# 4DVar / hybrid
# ---------------------------------------------------------------------------
DEFAULT_VAR = dict(mode="strong", window_days=7, b_scale=0.5, q_scale=0.05, r_inflation=1.0, n_iter=40,
                   spinup_days=30, b_source="truth")


def run_4dvar_cached(data_path, out_path, var, background_path=None, device="cuda", force=False):
    from .fourdvar import run_4dvar

    var = {**DEFAULT_VAR, **var}
    var.pop("name", None)
    bg = var.pop("background", None)
    cfg = dict(data=str(data_path), background=str(background_path) if bg else None, **var)
    if not force and _sidecar_ok(out_path, cfg):
        print(f"[4dvar] reuse {out_path}")
        return Path(out_path)
    run_4dvar(str(data_path), str(out_path), device=device, background_path=background_path if bg else None, **var)
    _write_sidecar(out_path, cfg)
    return Path(out_path)


# ---------------------------------------------------------------------------
# one scenario
# ---------------------------------------------------------------------------
def run_scenario(root, name, gen, methods, net=None, lyap=None, device="cuda", force_diag=False):
    """gen: generation kwargs; methods: list of dicts with 'name' + run_4dvar kwargs
    (optional 'background': '4dvarnet' for the hybrid). Returns a dict with paths, scores and diag summary."""
    import xarray as xr

    from .diag import run_diag

    data = generate_cached(root, gen, device=device)
    rec_net = train_4dvarnet(root, data, net)
    sc_dir = Path(root) / "scenarios" / name
    sc_dir.mkdir(parents=True, exist_ok=True)
    recs = {"4dvarnet": rec_net}
    for m in methods:
        m = copy.deepcopy(m)
        mname = m["name"]
        recs[mname] = run_4dvar_cached(data, sc_dir / f"{mname}.nc", m,
                                       background_path=rec_net if m.get("background") == "4dvarnet" else None,
                                       device=device)
    scores = {k: json.loads(xr.open_dataset(v, decode_times=False).attrs["scores"]) for k, v in recs.items()}
    summary = None
    if lyap is not None:
        lyap = dict(lyap)
        diag_dir = sc_dir / "diag"
        cfg = dict(recs={k: [str(v), Path(v).stat().st_mtime] for k, v in recs.items()}, data=str(data), **lyap)
        summ = diag_dir / "summary.json"
        if force_diag or not _sidecar_ok(summ, cfg):
            run_diag(str(data), {k: str(v) for k, v in recs.items()}, str(diag_dir), device=device,
                     k=lyap.get("k", 192), n_transient_fwd=lyap.get("n_fwd", 30),
                     n_transient_bwd=lyap.get("n_bwd", 60), animation_days=lyap.get("animation_days", 30))
            _write_sidecar(summ, cfg)
        summary = json.loads(summ.read_text())
    ds = xr.open_dataset(data, decode_times=False)
    return dict(name=name, data=data, recs=recs, scores=scores, lyap=summary,
                obs_summary=json.loads(ds.attrs["summary"]), diag_dir=sc_dir / "diag")
