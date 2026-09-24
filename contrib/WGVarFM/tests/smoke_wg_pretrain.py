"""End-to-end smoke test of the WG pre-training pipeline on synthetic CROSCIM-like files.

Writes fake daily NetCDF files (polar-stereographic xc/yc grid with 2D lat/lon, one
sensor per resolution) in a temporary directory, then instantiates the real xp config
(config/xp/WGVarFM/wg_pretrain_croscim_sic.yaml) with paths, dates and HEALPix level
overridden, and runs a Lightning fast_dev_run (train + val + test).

Runs on GPU or CPU (slow). Run from the repo root:
    python contrib/WGVarFM/tests/smoke_wg_pretrain.py
"""

import tempfile
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import xarray as xr
from omegaconf import OmegaConf

XP = Path(__file__).resolve().parents[3] / "config/xp/WGVarFM/wg_pretrain_croscim_sic.yaml"
DATES = pd.date_range("2022-05-01", periods=4, freq="D")
# domain [-3.85e6, 3.75e6] x [2.47e6, -4.9e6] m, as arctic_croscim
X0, X1, Y0, Y1 = -3.84e6, 3.74e6, 2.46e6, -4.89e6


def polar_stereo_to_latlon(x, y):
    """Inverse north polar stereographic projection on a sphere (lat_ts = 70N)."""
    r_earth, k = 6.371e6, (1 + np.sin(np.deg2rad(70.0))) / 2
    rho = np.hypot(x, y)
    lat = 90.0 - np.rad2deg(2 * np.arctan(rho / (2 * r_earth * k)))
    lon = np.rad2deg(np.arctan2(x, -y)) - 45.0
    return lat, (lon + 180.0) % 360.0 - 180.0


def write_files(root, name, pattern, date_fmt, variables, dx, rng):
    xc = np.arange(X0, X1, dx)
    yc = np.arange(Y0, Y1, -dx)
    lat, lon = polar_stereo_to_latlon(*np.meshgrid(xc, yc))
    (root / name).mkdir()
    for d in DATES:
        data = {}
        for v, scale in variables.items():
            field = scale * rng.random((1, len(yc), len(xc))).astype(np.float32)
            field[:, rng.random((len(yc), len(xc))) < 0.3] = np.nan  # gaps
            data[v] = (("time", "yc", "xc"), field)
        ds = xr.Dataset(
            data,
            coords=dict(
                time=[d.to_datetime64()], yc=yc, xc=xc, lat=(("yc", "xc"), lat), lon=(("yc", "xc"), lon)
            ),
        )
        ds.to_netcdf(root / name / pattern.format(d.strftime(date_fmt)))


def main():
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        # three sensors at three resolutions (coarsened here to keep the test small)
        write_files(root, "asip", "asip_L3_{}.nc", "%Y%m%d", {"sic": 100.0}, 40e3, rng)
        write_files(root, "cimr", "CIMR5km_{}.nc", "%Y-%m-%d", {"SIC": 1.0}, 80e3, rng)
        write_files(
            root, "atm", "atm5km_{}.nc", "%Y-%m-%d",
            {"t2m": 10.0, "msl": 1e5, "u10": 5.0, "v10": 5.0}, 80e3, rng,
        )

        cfg = OmegaConf.load(XP)
        cfg.paths = {"src": tmp, "asip": f"{tmp}/asip", "cimr": f"{tmp}/cimr", "covariates": f"{tmp}/atm"}
        cfg.healpix_level = 3
        cfg.datamodule.num_workers = 0
        cfg.datamodule.batch_size = 2
        for split, (a, b) in {"train": (0, 1), "val": (2, 2), "test": (3, 3)}.items():
            cfg.datamodule.domains[split].time._args_ = [str(DATES[a].date()), str(DATES[b].date())]

        dm = hydra.utils.instantiate(cfg.datamodule)
        lit = hydra.utils.instantiate(cfg.model)

        dm.setup()
        batch = next(iter(dm.train_dataloader()))
        print("tokens per stream:", batch.tokens_lens.sum(-1).tolist())
        for name in cfg.streams:
            sd = batch.samples[0].streams_data[name]
            print(f"  {name}: source {tuple(sd.source_tokens_cells[0].shape)}, "
                  f"target {tuple(sd.target_values[0].shape)}")

        trainer = pl.Trainer(
            accelerator="auto",
            devices=1,
            precision=cfg.trainer.precision,
            gradient_clip_val=cfg.trainer.gradient_clip_val,
            fast_dev_run=True,
            logger=False,
        )
        trainer.fit(lit, datamodule=dm)
        trainer.test(lit, datamodule=dm)
        loss = trainer.callback_metrics["test_loss"]
        assert torch.isfinite(loss), loss
        print(f"OK test_loss={loss.item():.4f}")


if __name__ == "__main__":
    main()
