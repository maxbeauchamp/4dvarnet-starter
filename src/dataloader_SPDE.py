"""
DataModule for the SPDE diffusion dataset.

NetCDF layout (single file):
    x   (time, lat, lon)  float32  – ground truth field
    y   (time, lat, lon)  float32  – sparse observations (NaN where unobserved)
    OI  (time, lat, lon)  float64  – Optimal Interpolation baseline (for benchmarking only)

The whole dataset is a single spatio-temporal window of shape
(T=15, H=100, W=100).  There is no natural time split, so we expose
OI as a fixed numpy array and generate a synthetic dataset of
*random patches* extracted from (x, y) for training.

DataModule
----------
SPDEDataModule
    input : y  (sparse obs, NaN outside tracks)
    tgt   : x  (ground truth)
    oi    : OI array exposed as attribute for downstream benchmarking

The full 15×100×100 window is used directly (no lon/lat patching needed
given the small size).  A random 80/10/10 split over "virtual samples"
(produced by small temporal jitter / random sub-windows) is performed.

If the dataset is used as-is (one single 15-step window) we fall back
to exposing that unique window for train/val/test so the pipeline still
runs end-to-end.
"""

from __future__ import annotations

import functools as ft
from collections import namedtuple
from typing import Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.utils.data
import xarray as xr

# Shared named-tuple – same interface as dataloader_SSH.py
TrainingItem = namedtuple("TrainingItem", ["input", "tgt"])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SPDEDataset(torch.utils.data.Dataset):
    """
    Wraps arrays (y, x) of shape (T, H, W).

    Each item is a TrainingItem(input=y[...], tgt=x[...]) where
    NaN values in *input* are kept as NaN (the Lightning module is
    responsible for creating the mask / nan_to_num).

    Parameters
    ----------
    y         : (T, H, W) float32 – observations (NaN where missing)
    x         : (T, H, W) float32 – ground truth
    indices   : list of int        – which time indices to expose as samples
                (each index i gives a window starting at i of length window_size)
    window_size : int              – temporal window length (default: full T)
    postpro_fn  : optional callable applied to each TrainingItem
    """

    def __init__(
        self,
        y: np.ndarray,
        x: np.ndarray,
        indices: list[int],
        window_size: int,
        postpro_fn=None,
    ) -> None:
        super().__init__()
        self.y = y.astype(np.float32)
        self.x = x.astype(np.float32)
        self.indices = indices
        self.window_size = window_size
        self.postpro_fn = postpro_fn

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> TrainingItem:
        i = self.indices[item]
        inp = self.y[i : i + self.window_size]   # (W, H, W_spatial)
        tgt = self.x[i : i + self.window_size]
        batch = TrainingItem(input=inp, tgt=tgt)
        if self.postpro_fn is not None:
            batch = self.postpro_fn(batch)
        return batch


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------

class SPDEDataModule(pl.LightningDataModule):
    """
    LightningDataModule for the SPDE diffusion NetCDF dataset.

    Parameters
    ----------
    path          : path to SPDE_diffusion_dataset.nc
    window_size   : number of time steps per sample (default: full T=15)
    stride        : stride between consecutive windows (default: 1)
    train_ratio   : fraction of windows used for training
    val_ratio     : fraction of windows used for validation
                    (remainder goes to test)
    dl_kw         : kwargs forwarded to DataLoader (batch_size, num_workers…)
    norm_stats    : optional (mean, std) tuple; computed from training data
                    if None
    seed          : random seed for reproducible train/val/test split

    Attributes
    ----------
    oi  : np.ndarray (T, H, W) – OI baseline array (float32)
          Available after calling setup().
    x   : np.ndarray (T, H, W) – full ground truth
    y   : np.ndarray (T, H, W) – full sparse observations
    ds  : xr.Dataset            – full xarray dataset
    """

    def __init__(
        self,
        path: str,
        window_size: Optional[int] = None,
        stride: int = 1,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        dl_kw: Optional[dict] = None,
        norm_stats: Optional[Tuple[float, float]] = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.path = path
        self.stride = stride
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.dl_kw = dl_kw or {"batch_size": 4, "num_workers": 2}
        self._norm_stats = norm_stats
        self.seed = seed

        # populated in setup()
        self.ds: Optional[xr.Dataset] = None
        self.x: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None
        self.oi: Optional[np.ndarray] = None
        self.window_size: Optional[int] = window_size

        self.train_ds: Optional[SPDEDataset] = None
        self.val_ds:   Optional[SPDEDataset] = None
        self.test_ds:  Optional[SPDEDataset] = None

    # ------------------------------------------------------------------
    # Norm stats
    # ------------------------------------------------------------------
    def norm_stats(self) -> Tuple[float, float]:
        if self._norm_stats is None:
            self._norm_stats = self._compute_norm_stats()
            print(f"[SPDEDataModule] Norm stats (mean, std): {self._norm_stats}")
        return self._norm_stats

    def _compute_norm_stats(self) -> Tuple[float, float]:
        """Compute mean/std of ground truth x on training windows."""
        rng = np.random.default_rng(self.seed)
        T = self.x.shape[0]
        ws = self.window_size
        all_starts = list(range(0, T - ws + 1, self.stride))
        rng.shuffle(all_starts)
        n_train = max(1, int(len(all_starts) * self.train_ratio))
        train_idx = all_starts[:n_train]
        samples = np.stack([self.x[i : i + ws] for i in train_idx])
        mean = float(np.nanmean(samples))
        std  = float(np.nanstd(samples))
        std  = std if std > 1e-8 else 1.0
        return mean, std

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def setup(self, stage: str = "test") -> None:
        self.ds = xr.open_dataset(self.path)

        # Load arrays
        self.x  = self.ds["x"].values.astype(np.float32)   # (T, H, W)
        self.y  = self.ds["y"].values.astype(np.float32)   # (T, H, W) – may contain NaN
        self.oi = self.ds["OI"].values.astype(np.float32)  # (T, H, W)

        T = self.x.shape[0]

        # Default window size = full temporal extent
        if self.window_size is None:
            self.window_size = T

        ws = self.window_size

        # Build list of all valid start indices
        all_starts = list(range(0, T - ws + 1, self.stride))

        if len(all_starts) == 0:
            raise ValueError(
                f"window_size={ws} > T={T}: no valid window. "
                "Set window_size <= T."
            )

        # Split indices
        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(len(all_starts)).tolist()

        n_train = max(1, int(len(perm) * self.train_ratio))
        n_val   = max(1, int(len(perm) * self.val_ratio))
        # ensure at least 1 sample in test
        n_train = min(n_train, len(perm) - 2)
        n_val   = min(n_val,   len(perm) - n_train - 1)
        n_test  = len(perm) - n_train - n_val

        train_starts = [all_starts[i] for i in perm[:n_train]]
        val_starts   = [all_starts[i] for i in perm[n_train:n_train + n_val]]
        test_starts  = [all_starts[i] for i in perm[n_train + n_val:]]

        print(
            f"[SPDEDataModule] T={T}, window={ws}, stride={self.stride} → "
            f"{len(all_starts)} windows: "
            f"train={len(train_starts)}, val={len(val_starts)}, test={len(test_starts)}"
        )

        post_fn = self._make_post_fn()

        self.train_ds = SPDEDataset(self.y, self.x, train_starts, ws, postpro_fn=post_fn)
        self.val_ds   = SPDEDataset(self.y, self.x, val_starts,   ws, postpro_fn=post_fn)
        self.test_ds  = SPDEDataset(self.y, self.x, test_starts,  ws, postpro_fn=post_fn)

    # ------------------------------------------------------------------
    # Post-processing (normalisation)
    # ------------------------------------------------------------------
    def _make_post_fn(self):
        m, s = self.norm_stats()

        def normalize(arr: np.ndarray) -> np.ndarray:
            return (arr - m) / s

        def fn(item: TrainingItem) -> TrainingItem:
            return TrainingItem(
                input=normalize(item.input),
                tgt=normalize(item.tgt),
            )

        return fn

    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------
    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.train_ds, shuffle=True, **self.dl_kw
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.val_ds, shuffle=False, **self.dl_kw
        )

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.test_ds, shuffle=False, **self.dl_kw
        )

    # ------------------------------------------------------------------
    # Convenience: full-domain tensors for evaluation
    # ------------------------------------------------------------------
    def get_full_tensors(
        self, device="cpu", dtype=torch.float32
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (y_full, x_full, oi_full) as tensors of shape (1, T, H, W),
        ready to be passed to the solver in eval mode.
        y_full contains NaN where observations are missing.
        """
        m, s = self.norm_stats()
        to_t = lambda arr: torch.tensor((arr - m) / s, dtype=dtype, device=device).unsqueeze(0)
        return to_t(self.y), to_t(self.x), to_t(self.oi)
