#!/usr/bin/env python3
"""Run a whole pipeline described by a config of config/pipeline/: the
training phases (N0 G_C alone, G_M + G_H, full cascade with pyramid, N1),
each initialised from the previous one, then the evaluation scripts
(plot_panels, plot_forecast, scenarios) on the test period.

Resumable: <out>/report.json records the finished phases. Run the same
command again (e.g. after the job time limit) and finished phases are
skipped, an interrupted phase resumes from its last.ckpt.

Usage:
    sbatch scripts/submit_odyssey.sbatch scripts/run_pipeline.py --config config/pipeline/2022.yaml
    sbatch scripts/submit_odyssey.sbatch scripts/run_pipeline.py --config config/pipeline/2022.yaml --redo-eval
    python scripts/run_pipeline.py --config config/pipeline/smoke.yaml --root <stores> --tiny \\
        --set out=/tmp/smoke dates.train=[2022-01-05,2022-01-06]      # local test
"""
from __future__ import annotations

import argparse
import gc
import inspect
import json
import os
import subprocess
import sys
import time

import hydra
import pytorch_lightning as pl
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.append(REPO)

TINY = ["base_dim=8", "encoder_depths=[1,1,1,1]", "decoder_depths=[1,1,1]", "num_heads=[1,1,1,1]",
        "time_dim=16", "modulation_rank=4"]


def load_cfg(xp: str, overrides: list[str]):
    with initialize_config_dir(config_dir=os.path.join(REPO, "config"), version_base="1.3"):
        return compose("main", overrides=[f"xp={xp}", *overrides])


class Progress(pl.Callback):
    """One line every `every` steps and one per epoch in the job log (the
    Lightning progress bar is off: it floods the log files)."""

    def __init__(self, name: str, every: int = 100):
        self.name, self.every = name, every

    def on_train_epoch_start(self, trainer, pl_module):
        self.t_epoch = self.t_step = time.perf_counter()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if (batch_idx + 1) % self.every == 0:
            dt = (time.perf_counter() - self.t_step) / self.every
            self.t_step = time.perf_counter()
            loss = trainer.callback_metrics.get("train_loss")
            print(f"{self.name} epoch {trainer.current_epoch} step {batch_idx + 1}: "
                  f"train_loss {float(loss) if loss is not None else float('nan'):.4f}, {dt:.2f} s/step", flush=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        m = {k: float(v) for k, v in trainer.callback_metrics.items()
             if k in ("train_loss", "val_loss") or k.startswith(("val_C_rmse_SIC", "val_rmse_SIC_k"))}
        print(f"{self.name} epoch {trainer.current_epoch} end ({(time.perf_counter() - self.t_epoch) / 60:.1f} min): "
              + ", ".join(f"{k} {v:.4f}" for k, v in sorted(m.items())), flush=True)


def floats(metrics: dict) -> dict:
    return {k: round(float(v), 5) for k, v in metrics.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="config/pipeline/*.yaml")
    ap.add_argument("--root", default="/Odyssey/public/CROSCIM_dataset")
    ap.add_argument("--set", nargs="*", default=[], help="overrides of the pipeline config (key=value)")
    ap.add_argument("--tiny", action="store_true", help="tiny networks on CPU (local test of this script)")
    ap.add_argument("--redo-eval", action="store_true", help="run the evaluation scripts again (phases done are skipped)")
    args = ap.parse_args()
    pcfg = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(args.set))
    out = pcfg.out
    os.makedirs(out, exist_ok=True)
    cuda = torch.cuda.is_available() and not args.tiny
    path = os.path.join(out, "report.json")
    report = json.load(open(path)) if os.path.exists(path) else {}
    if args.redo_eval:
        report = {k: v for k, v in report.items() if not k.startswith("eval_")}

    def save():
        with open(path, "w") as f:
            json.dump(report, f, indent=2)

    d = pcfg.dates
    common = [f"datamodule.root={args.root}", f"datamodule.natural_earth_zip={REPO}/assets/ne_10m_land.zip",
              f"datamodule.train_dates=[{d.train[0]},{d.train[1]}]", f"datamodule.val_dates=[{d.val[0]},{d.val[1]}]",
              f"datamodule.test_dates=[{d.test[0]},{d.test[1]}]", *pcfg.get("common", [])]
    tiny_n0 = [f"swin_config.{t}" for t in TINY] + [f"swin_config_fine.{t}" for t in TINY] if args.tiny else []
    if args.tiny:
        common += ["datamodule.dl_kw.num_workers=0", "datamodule.dl_kw.pin_memory=false"]

    def ckpt_of(name):
        r = report[name]
        return r["best"] or r["last"]

    for ph in pcfg.phases:
        name = ph.name
        if report.get(name, {}).get("done"):
            print(f"{name}: done, skipped", flush=True)
            continue
        overrides = common + list(ph.get("overrides", []))
        if ph.xp == "N0":
            overrides += tiny_n0
        else:
            overrides.append(f"model.n0_ckpt='{ckpt_of(ph.n0)}'")  # quoted: checkpoint names contain "="
        cfg = load_cfg(ph.xp, overrides)
        dm = hydra.utils.instantiate(cfg.datamodule)
        model = hydra.utils.instantiate(cfg.model)
        ckpt_dir = os.path.join(out, name, "checkpoints")
        resume = os.path.join(ckpt_dir, "last.ckpt")
        if not os.path.exists(resume):
            resume = None
            init = ckpt_of(ph.init) if ph.get("init") else None
            if ph.get("init_from"):  # a phase of another pipeline run, e.g. a G_C that stays valid
                old = json.load(open(ph.init_from.report))[ph.init_from.phase]
                init = old["best"] or old["last"]
            if init:
                sd = torch.load(init, map_location="cpu", weights_only=False)["state_dict"]
                missing, _ = model.load_state_dict(sd, strict=False)
                print(f"{name}: initialised from {init} ({len(missing)} tensors left at random init)", flush=True)
            if ph.get("init_overlay"):  # then some sub-networks from a phase of another run, e.g. G_M / G_H
                ov = ph.init_overlay
                old = json.load(open(ov.report))[ov.phase]
                path = old["best"] or old["last"]
                sd = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
                sd = {k: v for k, v in sd.items() if any(f".{m}." in f".{k}" for m in ov.modules)}
                model.load_state_dict(sd, strict=False)
                print(f"{name}: {len(sd)} tensors of {list(ov.modules)} from {path}", flush=True)
        kw = dict(OmegaConf.to_container(ph.trainer), enable_progress_bar=False, num_sanity_val_steps=0)
        try:
            from pytorch_lightning.loggers import TensorBoardLogger
            import tensorboard  # noqa: F401
            kw["logger"] = TensorBoardLogger(out, name=name)
        except ImportError:
            kw["logger"] = pl.loggers.CSVLogger(out, name=name)
        if not cuda:
            kw.update(accelerator="cpu", devices=1, precision="32")
        trainer = hydra.utils.instantiate(cfg.trainer, **kw)
        trainer.callbacks.append(Progress(name))
        trainer.checkpoint_callback.dirpath = ckpt_dir
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        print(f"{name}: {'resumed from ' + resume if resume else 'started'}", flush=True)
        t0 = time.perf_counter()
        fit_kw = {"ckpt_path": resume}
        if resume and "weights_only" in inspect.signature(trainer.fit).parameters:
            fit_kw["weights_only"] = False  # our own checkpoints hold Hydra configs (torch >= 2.6)
        trainer.fit(model, datamodule=dm, **fit_kw)
        t_fit = time.perf_counter() - t0
        val = floats(trainer.callback_metrics)
        trainer.test(model, datamodule=dm)
        cb = trainer.checkpoint_callback
        report[name] = {"done": True, "fit_s": round(t_fit, 1), "val": val, "test": floats(trainer.callback_metrics),
                        "peak_gpu_GiB": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if cuda else None,
                        "best": cb.best_model_path, "last": cb.last_model_path, "log_dir": trainer.logger.log_dir}
        save()
        print(f"{name}: done in {t_fit / 3600:.2f} h, best {cb.best_model_path}", flush=True)
        # free this phase (model on GPU, loaders) before the next one: phase 6
        # loads two N0 copies and larger items
        del trainer, model, dm, cb
        gc.collect()
        if cuda:
            torch.cuda.empty_cache()

    # ---- evaluation on the test period ----
    ev = pcfg.get("evaluation", {})
    for script, cfg_ev in ev.items():
        key = f"eval_{script}"
        if report.get(key, {}).get("done"):
            continue
        cfg_ev = OmegaConf.to_container(cfg_ev)
        ckpt = ckpt_of(cfg_ev.pop("ckpt_of"))
        dest = os.path.join(out, key)
        cmd = [sys.executable, os.path.join(REPO, "scripts", f"{script}.py"), "--ckpt", ckpt, "--root", args.root,
               "--start", d.test[0], "--end", d.test[1], "--out", dest]
        for k, v in cfg_ev.items():
            flag = f"--{k.replace('_', '-')}"
            if isinstance(v, bool):
                cmd += [flag] if v else []
            else:
                cmd += [flag, *([str(x) for x in v] if isinstance(v, (list, tuple)) else [str(v)])]
        print(f"{key}: {' '.join(cmd)}", flush=True)
        t0 = time.perf_counter()
        res = subprocess.run(cmd)
        report[key] = {"done": res.returncode == 0, "returncode": res.returncode, "out": dest,
                       "time_s": round(time.perf_counter() - t0, 1)}
        save()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
