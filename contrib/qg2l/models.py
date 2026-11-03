"""
4DVarNet components for the periodic QG testbed.

* ``periodic(module)``: switches every padded Conv2d of a module to circular
  padding (doubly periodic domain).
* ``LitQG2L``: ``src.models.Lit4dVarNet`` with a test loop adapted to the
  full-domain sliding windows of ``data.QG2LDataModule``: overlapping windows
  are blended in time with the reconstruction weight, the result is written as
  a netCDF with the same layout as the 4DVar output (time, layer, y, x; psi in
  m2/s) so that ``diag.py`` treats both methods identically.
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import src.models

from . import metrics as qg_metrics
from .data import blend_windows, psi_from_state


def periodic(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d) and any(p > 0 for p in np.atleast_1d(m.padding)):
            m.padding_mode = "circular"
    return module


class LitQG2L(src.models.Lit4dVarNet):
    def __init__(self, *args, export_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.export_path = export_path

    # same training objective as the base model; the prior is evaluated on the
    # full (2-layer) state
    def test_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.test_data = []
        out = self(batch=batch)
        self.test_data.append(out.detach().cpu().float())

    def on_test_epoch_end(self):
        dm = self.trainer.datamodule
        ds = dm.test_ds
        m, s = self.norm_stats
        out = torch.cat(self.test_data).numpy() * s + m  # (n_win, L*T, ny, nx)
        L, T = len(dm.layers), dm.n_time
        out = out.reshape(out.shape[0], L, T, *out.shape[-2:])
        w = self.rec_weight.cpu().numpy().reshape(L, T, *out.shape[-2:])[:, :, 0, 0]  # (L, T)

        t0, t1 = dm.splits["test"]
        rec = blend_windows(out, ds.starts, t0, t1 - t0, w)  # (n_days, L, ny, nx)

        truth = dm.state[t0:t1][:, [l - 1 for l in dm.layers]]
        scores = qg_metrics.all_scores(rec, truth, dx=self._dx(dm))
        print("\n" + "\n".join(f"| {k:16s} | {v:10.4f} |" for k, v in scores.items()))

        self.test_data = self._to_dataset(rec, dm, t0, scores)
        paths = []
        if self.logger is not None and getattr(self.logger, "log_dir", None):
            paths.append(Path(self.logger.log_dir) / "test_data.nc")
            self.logger.log_metrics(scores)
        if self.export_path:
            paths.append(Path(self.export_path))
        for p in paths:
            p.parent.mkdir(parents=True, exist_ok=True)
            self.test_data.to_netcdf(p)
            print("wrote", p)

    @staticmethod
    def _dx(dm):
        return dm.model_kw["L"] / dm.model_kw["nx"]

    @staticmethod
    def _to_dataset(rec_state, dm, t0, scores):
        import xarray as xr

        n_days, L = rec_state.shape[:2]
        psi = np.full((n_days, 2) + rec_state.shape[-2:], np.nan, dtype=np.float32)
        for i, l in enumerate(dm.layers):
            psi[:, l - 1] = psi_from_state(rec_state[:, i], dm.model_kw)
        ref = dm.ds.isel(time=slice(t0, t0 + n_days))
        return xr.Dataset(
            dict(psi=(("time", "layer", "y", "x"), psi, dict(units="m2 s-1", long_name="4DVarNet reconstruction"))),
            coords=dict(time=ref.time.values, layer=[1, 2], y=ref.y.values, x=ref.x.values),
            attrs=dict(method="4dvarnet", model_kw=json.dumps(dm.model_kw), scores=json.dumps(scores),
                       layers_estimated=json.dumps(list(dm.layers))),
        )
