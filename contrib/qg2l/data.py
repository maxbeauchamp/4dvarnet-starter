"""
4DVarNet data pipeline for the two-layer QG OSSE (output of ``simulate.generate``).

State = the two layers stacked along the channel dimension, both expressed in
"SSH-equivalent" units  s_i = f0 / g * psi_i  (m), so that one normalisation fits
both layers and layer 1 is exactly the SSH:

    channel c = (layer index) * n_time + (time index in the window)

The input holds the SSH observations on the layer-1 channels and NaN on the
layer-2 channels (never observed), so ``src.models.BaseObsCost`` (which masks
with ``isfinite``) only sees the observations.  Set ``layers=[1]`` to
reconstruct the SSH only (standard 4DVarNet set-up); the Lyapunov diagnostic
needs the full state and then refuses to run on that reconstruction.

The domain is doubly periodic and small (64 x 64 by default): every item is a
full-domain window of ``n_time`` consecutive days, so no spatial stitching is
needed; overlapping windows are blended in time at test time
(see ``models.LitQG2L``).
"""
import json

import numpy as np
import pytorch_lightning as pl
import torch
import torch.utils.data

from src.data import TrainingItem


def load_qg2l(path):
    import xarray as xr

    ds = xr.open_dataset(path).load()
    kw = json.loads(ds.attrs["model_kw"])
    return ds, kw


def state_from_psi(psi, model_kw):
    """psi (..., 2, ny, nx) [m2/s] -> SSH-equivalent state [m]."""
    return model_kw["f0"] / model_kw["g"] * psi


def psi_from_state(state, model_kw):
    return model_kw["g"] / model_kw["f0"] * state


def n_channels(n_time, layers):
    return int(n_time) * len(layers)


def qg_rec_weight(n_time, layers, nx, offset=1, layer_weights=None):
    """Triangular-in-time reconstruction weight (len(layers) * n_time, nx, nx).

    Same time profile as ``src.utils.get_triang_time_wei`` (no spatial crop:
    the domain is periodic)."""
    n_layers = len(layers)
    t = np.arange(n_time)
    w_t = 1 - np.abs(offset + 2 * t - n_time) / n_time
    lw = np.ones(n_layers) if layer_weights is None else np.asarray(layer_weights, float)
    w = lw[:, None] * w_t[None, :]
    return np.broadcast_to(w.reshape(-1, 1, 1), (n_layers * n_time, nx, nx)).astype("float32").copy()


def blend_windows(out, starts, t0, n_days, w):
    """Blend overlapping window predictions in time.

    out (n_win, L, T, ny, nx), starts (n_win,) absolute start days, w (L, T) weights.
    Returns (n_days, L, ny, nx) for days [t0, t0 + n_days) (NaN where never covered)."""
    n_win, L, T = out.shape[:3]
    acc = np.zeros((n_days, L) + out.shape[-2:])
    cnt = np.zeros((n_days, L, 1, 1))
    for i, start in enumerate(starts):
        j = int(start) - t0
        acc[j:j + T] += np.moveaxis(out[i] * w[:, :, None, None], 0, 1)
        cnt[j:j + T] += w.T[:, :, None, None]
    return np.where(cnt > 0, acc / np.maximum(cnt, 1e-12), np.nan)


class QGWindowDataset(torch.utils.data.Dataset):
    """Sliding windows over [t0, t1) days.

    tgt   (n_layers * n_time, ny, nx)  normalised SSH-equivalent state
    input (n_layers * n_time, ny, nx)  normalised SSH obs on layer-1 channels, NaN elsewhere

    aug_factor > 0 appends ``aug_factor * n`` items whose observations are
    re-simulated from the truth with the sampling mask (and noise level) of
    another random window: same idea as ``src.data.AugmentedDataset``.
    """

    def __init__(self, state, obs, obs_sigma, t0, t1, n_time, stride=1, layers=(1, 2),
                 norm=(0.0, 1.0), aug_factor=0, aug_only=False, seed=0):
        self.state = state  # (T, 2, ny, nx) float32, SSH-equivalent (m)
        self.obs = obs  # (T, ny, nx)
        self.obs_sigma = obs_sigma
        self.n_time = n_time
        self.layers = [l - 1 for l in layers]
        self.starts = np.arange(t0, t1 - n_time + 1, stride)
        self.norm = norm
        self.aug_factor = int(aug_factor)
        self.aug_only = bool(aug_only)
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        n = len(self.starts)
        return n * (1 + self.aug_factor - int(self.aug_only and self.aug_factor > 0))

    def _window(self, s):
        sl = slice(s, s + self.n_time)
        tgt = self.state[sl][:, self.layers]  # (T, L, ny, nx)
        return np.moveaxis(tgt, 1, 0)  # (L, T, ny, nx)

    def _input(self, obs_win):
        L = len(self.layers)
        inp = np.full((L,) + obs_win.shape, np.nan, dtype=np.float32)
        if 0 in self.layers:
            inp[self.layers.index(0)] = obs_win
        return inp

    def __getitem__(self, idx):
        n = len(self.starts)
        if self.aug_only and self.aug_factor > 0:
            idx = idx + n
        s = self.starts[idx % n]
        tgt = self._window(s)
        if idx < n:
            obs_win = self.obs[s:s + self.n_time]
        else:
            # re-sample: sampling pattern of another window applied to this truth
            s2 = self.starts[self.rng.integers(n)]
            msk = np.isfinite(self.obs[s2:s2 + self.n_time])
            sig = np.nan_to_num(self.obs_sigma[s2:s2 + self.n_time])
            ssh = self.state[s:s + self.n_time, 0]
            noisy = ssh + sig * self.rng.standard_normal(ssh.shape).astype(np.float32)
            obs_win = np.where(msk, noisy, np.nan).astype(np.float32)
        inp = self._input(obs_win)
        m, sd = self.norm
        L, T = tgt.shape[:2]
        shp = (L * T,) + tgt.shape[2:]
        return TrainingItem(
            input=((inp - m) / sd).reshape(shp).astype(np.float32),
            tgt=((tgt - m) / sd).reshape(shp).astype(np.float32),
        )


class QG2LDataModule(pl.LightningDataModule):
    """
    path:      netCDF written by ``simulate.generate``
    n_time:    window length (days)
    layers:    [1, 2] (full state) or [1] (SSH only)
    splits:    dict(train=(t0, t1), val=..., test=...) in days; default = file attrs
    """

    def __init__(self, path, n_time=15, layers=(1, 2), splits=None, dl_kw=None, aug_factor=1,
                 aug_only=False, test_stride=1, norm_stats=None):
        super().__init__()
        self.path = path
        self.n_time = int(n_time)
        self.layers = list(layers)
        self._splits = splits
        self.dl_kw = dict(dl_kw or dict(batch_size=16, num_workers=1))
        self.aug_factor = aug_factor
        self.aug_only = aug_only
        self.test_stride = test_stride
        self._norm_stats = None if norm_stats is None else tuple(norm_stats)
        self._loaded = False

    # -- data -----------------------------------------------------------------
    def _load(self):
        if self._loaded:
            return
        ds, kw = load_qg2l(self.path)
        self.ds = ds
        self.model_kw = kw
        self.state = state_from_psi(ds.psi.values.astype(np.float32), kw).astype(np.float32)
        self.obs = ds.obs_ssh.values.astype(np.float32)
        self.obs_sigma = ds.obs_sigma.values.astype(np.float32)
        self.splits = {k: tuple(v) for k, v in (self._splits or json.loads(ds.attrs["splits"])).items()
                       if k in ("train", "val", "test")}
        self._loaded = True

    def norm_stats(self):
        if self._norm_stats is None:
            self._load()
            t0, t1 = self.splits["train"]
            x = self.state[t0:t1][:, [l - 1 for l in self.layers]]
            self._norm_stats = (float(x.mean()), float(x.std()))
            print("Norm stats", self._norm_stats)
        return self._norm_stats

    def _ds(self, split, **kw):
        t0, t1 = self.splits[split]
        return QGWindowDataset(self.state, self.obs, self.obs_sigma, t0, t1, self.n_time,
                               layers=self.layers, norm=self.norm_stats(), **kw)

    def setup(self, stage=None):
        self._load()
        self.train_ds = self._ds("train", aug_factor=self.aug_factor, aug_only=self.aug_only)
        self.val_ds = self._ds("val")
        self.test_ds = self._ds("test", stride=self.test_stride)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_ds, shuffle=True, **self.dl_kw)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_ds, shuffle=False, **self.dl_kw)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_ds, shuffle=False, **self.dl_kw)
