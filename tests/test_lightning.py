"""End-to-end smoke test: Lightning fit / validate / test of N0 then N1 with
the real CascadeDataModule on small synthetic Zarr stores (tiny backbones,
CPU). Run from the repo root, e.g.:

    pytest tests/test_lightning.py
"""
import os

import numpy as np
import pytorch_lightning as pl
import torch

from croscim import data as D
from croscim import grid as g
from croscim.transition import LitCascadeN1
from tests.test_data import DAYS, IX0, IY0, _write_stores
from tests.test_n0 import TINY, make_model


class _SyntheticDM(D.CascadeDataModule):
    """CascadeDataModule on the synthetic stores: open ocean everywhere, one tile."""

    def setup(self, stage=None):
        omega_h = np.ones(g.shape("H"), np.float32)
        origins = np.array([[IY0, IX0]])
        for split in ("train", "val", "test"):
            self.ds[split] = D.CascadeDataset(self.root, self._days(split), omega_h, origins,
                                              tiles_per_day=1, train=split == "train", horizon=self.horizon)


def _trainer(path):
    return pl.Trainer(accelerator="cpu", max_epochs=1, limit_train_batches=2, limit_val_batches=1,
                      limit_test_batches=1, logger=pl.loggers.CSVLogger(path), enable_checkpointing=False,
                      enable_progress_bar=False, num_sanity_val_steps=0)


def test_fit_n0_then_n1(tmp_path):
    root = str(tmp_path / "stores")
    _write_stores(root)
    last = str(np.datetime64("1970-01-01") + np.timedelta64(DAYS[-1], "D"))
    first = str(np.datetime64("1970-01-01") + np.timedelta64(DAYS[-3], "D"))

    dm0 = _SyntheticDM(root, None, [last, last], [last, last], [last, last], dl_kw={"batch_size": 1})
    n0 = make_model()
    tr = _trainer(str(tmp_path / "logs0"))
    tr.fit(n0, datamodule=dm0)
    tr.test(n0, datamodule=dm0)
    assert torch.isfinite(torch.tensor(tr.callback_metrics["test_loss"].item()))
    assert any(f.startswith("rank_hist_test") for f in os.listdir(tr.logger.log_dir))

    dm1 = _SyntheticDM(root, None, [first, first], [first, first], [first, first], dl_kw={"batch_size": 1}, horizon=2)
    n1 = LitCascadeN1(n0=n0, swin=TINY, k_max=2, n_steps=1, push_steps=1, n_members=2)
    tr = _trainer(str(tmp_path / "logs1"))
    tr.fit(n1, datamodule=dm1)
    tr.test(n1, datamodule=dm1)
    assert "test_rmse_SIC_k2" in tr.callback_metrics
