"""
data.py
=======
On-the-fly dataloader of the CROSCIM cascade (formulation sections 4 and 8).

Workers only slice native values out of Zarr stores (written once by
scripts/convert_sources.py: crop + format change, no regridding),
next to the NetCDF directories of each source under <root>
(e.g. /Odyssey/public/CROSCIM_dataset):

    ASIP_L3_zarr/{sic,std,acq,days}     500 m, uint8 % (255 = missing), acq in
                                         minutes since 00:00 (int16, -32768 = missing)
    ASIP_L3_zarr/{m_mean,m_w}           4 km sigma^-2-weighted means / weights of each day
    {data_noise,out_CRISTAL,atm_data}_zarr/{<var>,xc,yc,days}
                                         CIMR / CRISTAL / ERA5, 5 km, float32, NaN = missing

Everything else (Y^in / Y^tgt partition, gridding G_l, coverage masks, detail
targets, statistics S^_C) is computed per batch on the GPU, in
`CascadeDataModule.on_after_batch_transfer` -> `build_batch`.

A sample is a 64 x 64 C tile (2048 km) at analysis day t, with a window of
W days [t-W+1, t] (N0); for the transition N1 the item spans [t-W+1, t+K]
and `build_batch` returns one sample per analysis day t, ..., t+K.

ASIP enters at H (1 km) for the last W_H days and at M (4 km) for all W days:
the M maps of days < t are stored by the conversion (sigma^-2-weighted 8 x 8
sums of the 500 m pixels, croscim/asip.py; no partition applies to them),
the one of day t is computed on the GPU after the partition. The partition
only applies to day t; earlier days are inputs.
"""

from __future__ import annotations

import datetime as dt
import os

import numpy as np
import pytorch_lightning as pl
import torch
import zarr
from torch import Tensor

from . import grid as g
from . import obs
from .asip import ASIP_SIGMA_FLOOR, asip_to_m  # noqa: F401  (re-exported)
from .layout import EPOCH, SOURCES, STORES  # noqa: F401  (re-exported)

N_C = 64          # tile size in C cells
W = 15            # N0 window (days)
W_H = 3           # ASIP window at H (days); ASIP at M covers all W days
DX5 = 5_000.0     # 5 km source grid
N5 = 411          # 5 km cells read per tile side (2048 km / 5 km, rounded up + 1)
MIN_TILE_OCEAN = 0.2
P_TGT = 0.3       # probability that a partition unit is held out as target
BLOCK_C = 8       # partition blocks of the 5 km sources: 8 x 8 C cells (256 km)
ASIP_PASS_GAP_MIN = 10  # ASIP partition units: acquisitions less than 10 min apart form one pass
OBS_SIGMA = {"cimr": (0.05, 0.10), "cristal": (0.30, 0.05)}  # SIC [-], SIT [m], HS [m]
GAMMA = {"SIC": 0.5, "SIT": 0.05}  # min. observed ocean fraction of a parent for a detail target


def to_day(date: str) -> int:
    return (dt.date.fromisoformat(date) - EPOCH).days


def valid_tile_origins(omega_c: np.ndarray, stride: int) -> np.ndarray:
    """C-tile origins (iy0, ix0) on a `stride` lattice (the last origin is
    clamped to the domain edge) whose ocean fraction is >= MIN_TILE_OCEAN."""
    n = omega_c.shape[0]
    starts = sorted(set(list(range(0, n - N_C, stride)) + [n - N_C]))
    ii = np.array([(i, j) for i in starts for j in starts])
    frac = np.array([omega_c[i:i + N_C, j:j + N_C].mean() for i, j in ii])
    return ii[frac >= MIN_TILE_OCEAN]


class CascadeDataset(torch.utils.data.Dataset):

    def __init__(self, root, days, omega_h, origins, tiles_per_day=None, train=True, horizon=0,
                 asip_weighted=False):
        self.root, self.days, self.omega_h, self.origins = root, list(days), omega_h, origins
        self.train, self.horizon, self.asip_weighted = train, horizon, asip_weighted
        self.tiles_per_day = tiles_per_day if train else len(origins)
        self._arr = None

    def _open(self):
        arr = {}
        for src, names in {"asip": ("sic", "std", "acq", "m_mean", "m_w"), **SOURCES}.items():
            path = os.path.join(self.root, STORES[src])
            arr[src] = {k: zarr.open_array(os.path.join(path, k), mode="r") for k in names}
            arr[src]["day_index"] = {int(d): i for i, d in enumerate(zarr.open_array(os.path.join(path, "days"), mode="r")[:])}
            if src != "asip":
                arr[src]["xc"] = zarr.open_array(os.path.join(path, "xc"), mode="r")[:]
                arr[src]["yc"] = zarr.open_array(os.path.join(path, "yc"), mode="r")[:]
        self._arr = arr

    def __len__(self):
        return len(self.days) * self.tiles_per_day

    def _draw_origin(self, t: int) -> int:
        """Training tile: uniform over the ocean tiles, or (asip_weighted) 80 % of
        the draws in proportion to the ASIP coverage of day t (the fine levels
        learn only where ASIP observes: most uniform tiles carry no M / H
        target), 20 % uniform."""
        n = len(self.origins)
        i = self._arr["asip"]["day_index"].get(t)
        if not self.asip_weighted or i is None or torch.rand(()) < 0.2:
            return int(torch.randint(n, ()))
        fm = g.factor("M", "C")
        nc = g.shape("C")[0]
        cov = (self._arr["asip"]["m_w"][i] > 0).reshape(nc, fm, nc, fm).mean((1, 3))
        integ = np.pad(cov.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
        y, x = self.origins[:, 0], self.origins[:, 1]
        score = integ[y + N_C, x + N_C] - integ[y, x + N_C] - integ[y + N_C, x] + integ[y, x]
        if score.sum() <= 0:
            return int(torch.randint(n, ()))
        return int(torch.multinomial(torch.as_tensor(score, dtype=torch.float64), 1))

    def _read_asip(self, days, iy0, ix0):
        """500 m sic / std of the last W_H + K days, acq of the analysis days
        t..t+K, and the stored 4 km (mean, weight) maps of all days but the
        last (no partition applies to them, see croscim/asip.py)."""
        a = self._arr["asip"]
        f = int(g.DX_M["C"] // 500)
        sl = (slice(iy0 * f, (iy0 + N_C) * f), slice(ix0 * f, (ix0 + N_C) * f))
        n, nm, nd, nh = N_C * f, N_C * g.factor("M", "C"), len(days), W_H + self.horizon
        sic = np.full((nh, n, n), 255, np.uint8)
        std = np.full((nh, n, n), 255, np.uint8)
        acq = np.full((self.horizon + 1, n, n), -32768, np.int16)
        m_mean = np.zeros((nd - 1, nm, nm), np.float32)
        m_w = np.zeros((nd - 1, nm, nm), np.float32)
        for k, d in enumerate(days):
            i = a["day_index"].get(d)
            if i is None:
                continue
            if k >= nd - nh:
                sic[k - nd + nh], std[k - nd + nh] = a["sic"][(i,) + sl], a["std"][(i,) + sl]
            if k >= nd - 1 - self.horizon:
                acq[k - nd + 1 + self.horizon] = a["acq"][(i,) + sl]
        fm = g.factor("M", "C")
        sl_m = (slice(iy0 * fm, (iy0 + N_C) * fm), slice(ix0 * fm, (ix0 + N_C) * fm))
        have = [(k, a["day_index"][d]) for k, d in enumerate(days[:-1]) if d in a["day_index"]]
        if have:
            i0, i1 = have[0][1], have[-1][1] + 1
            block_mean, block_w = a["m_mean"][(slice(i0, i1),) + sl_m], a["m_w"][(slice(i0, i1),) + sl_m]
            for k, i in have:
                m_mean[k], m_w[k] = block_mean[i - i0], block_w[i - i0]
        return sic, std, acq, m_mean, m_w

    def _read_5km(self, src, days, iy0, ix0):
        a = self._arr[src]
        xc, yc = a["xc"], a["yc"]
        y_edges, x_edges = g.tile_edges("C", iy0, ix0, N_C)
        j0 = int(np.floor((x_edges[0].item() - (xc[0] - DX5 / 2)) / DX5))
        i0 = int(np.floor(((yc[0] + DX5 / 2) - y_edges[0].item()) / DX5))
        x5, y5 = xc[0] + DX5 * np.arange(j0, j0 + N5), yc[0] - DX5 * np.arange(i0, i0 + N5)
        # clip the window to the store, pad with NaN outside
        ri, rj = slice(max(i0, 0), min(i0 + N5, len(yc))), slice(max(j0, 0), min(j0 + N5, len(xc)))
        oi, oj = slice(ri.start - i0, ri.stop - i0), slice(rj.start - j0, rj.stop - j0)
        out = np.full((len(days), len(SOURCES[src]), N5, N5), np.nan, np.float32)
        have = [(k, a["day_index"][d]) for k, d in enumerate(days) if d in a["day_index"]]
        if have:
            # one read of the whole window per variable: chunks span several days
            i0, i1 = have[0][1], have[-1][1] + 1
            for v, name in enumerate(SOURCES[src]):
                block = a[name][i0:i1, ri, rj]
                for k, i in have:
                    out[k, v, oi, oj] = block[i - i0]
        return out, x5, y5

    def __getitem__(self, idx):
        if self._arr is None:
            self._open()
        t = self.days[idx // self.tiles_per_day]
        if self.train:
            # torch RNG: reseeded per worker by the DataLoader (numpy is not)
            iy0, ix0 = self.origins[self._draw_origin(t)]
            seed = torch.randint(2 ** 31, ()).item()
        else:
            iy0, ix0 = self.origins[idx % self.tiles_per_day]
            seed = idx
        days = list(range(t - W + 1, t + self.horizon + 1))
        sic, std, acq, m_mean, m_w = self._read_asip(days, iy0, ix0)
        sl_h = g.tile_slices(iy0, ix0, N_C)["H"]
        item = {"day": t, "iy0": iy0, "ix0": ix0, "seed": seed,
                "asip_sic": sic, "asip_std": std, "asip_acq": acq, "asip_m": m_mean, "asip_m_w": m_w,
                "omega_h": self.omega_h[sl_h]}
        for src in SOURCES:
            item[src], item["x5"], item["y5"] = self._read_5km(src, days, iy0, ix0)
        return item


def _points(v, keep, x5, y5):
    """Valid 5 km cells of v[T, V, ny, nx] as points (treated as L2)."""
    t, c, i, j = torch.nonzero(keep & torch.isfinite(v), as_tuple=True)
    return x5[j], y5[i], v[t, c, i, j], t * v.shape[1] + c


def window(it: dict, k: int) -> dict:
    """The N0 item of analysis day t+k out of an item spanning [t-W+1, t+K]."""
    out = {**it, "asip_sic": it["asip_sic"][k:k + W_H], "asip_std": it["asip_std"][k:k + W_H],
           "asip_acq": it["asip_acq"][k], "asip_m": it["asip_m"][k:k + W - 1],
           "asip_m_w": it["asip_m_w"][k:k + W - 1], "seed": it["seed"] + k}
    out.update({src: it[src][k:k + W] for src in SOURCES})
    return out


def partition(it: dict) -> dict:
    """Y^in / Y^tgt partition at day t: ASIP by pass (acquisition times
    chained by gaps < ASIP_PASS_GAP_MIN), 5 km sources by blocks of
    BLOCK_C x BLOCK_C C cells.
    Returns the target masks {"asip": [ny, nx] (500 m), "blk": [N5, N5]}."""
    dev = it["omega_h"].device
    gen = torch.Generator(dev).manual_seed(int(it["seed"]))
    y_edges, x_edges = g.tile_edges("C", int(it["iy0"]), int(it["ix0"]), N_C)
    nb = N_C // BLOCK_C
    tgt_blk = torch.rand(nb * nb, generator=gen, device=dev) < P_TGT
    by = torch.floor((y_edges[0].item() - it["y5"].double()) / (BLOCK_C * g.DX_M["C"])).long().clamp(0, nb - 1)
    bx = torch.floor((it["x5"].double() - x_edges[0].item()) / (BLOCK_C * g.DX_M["C"])).long().clamp(0, nb - 1)
    acq = it["asip_acq"]
    ok_t = (it["asip_sic"][-1] != 255) & (acq != -32768)
    return {"asip": obs.unit_partition(asip_passes(acq, ok_t), ok_t, P_TGT, gen),
            "blk": tgt_blk[by[:, None] * nb + bx[None]]}


def asip_passes(acq: Tensor, valid: Tensor) -> Tensor:
    """Pass id of each pixel: the distinct acquisition times (minutes) of the
    tile, sorted, are split wherever two successive ones are more than
    ASIP_PASS_GAP_MIN apart (scenes of one pass are ~1 min apart, passes
    ~100 min). A fixed time binning would cut a pass in two at a bin edge."""
    times = torch.unique(acq[valid].long())
    if len(times) == 0:
        return torch.zeros_like(acq, dtype=torch.long)
    label = torch.cat([times.new_zeros(1), torch.cumsum(torch.diff(times) > ASIP_PASS_GAP_MIN, 0)])
    return label[torch.searchsorted(times, acq.long()).clamp(max=len(times) - 1)]


def build_sample(it: dict, part: dict | None = None) -> dict:
    """Partition, gridding and targets for one sample (tensors on device)."""
    part = partition(it) if part is None else part
    dev = it["omega_h"].device
    iy0, ix0 = int(it["iy0"]), int(it["ix0"])
    f64 = dict(dtype=torch.float64, device=dev)
    omega_h = it["omega_h"].double()
    omega_m = g.restrict(omega_h, torch.ones_like(omega_h), 4)[0]
    omega_c = g.restrict(omega_m, torch.ones_like(omega_m), 8)[0]
    edges = {lvl: [e.to(dev) for e in g.tile_edges(lvl, iy0, ix0, N_C)] for lvl in g.LEVELS}
    x5, y5 = it["x5"].to(**f64), it["y5"].to(**f64)
    ay, ax = obs.overlap_matrix(y5, DX5, edges["C"][0]), obs.overlap_matrix(x5, DX5, edges["C"][1])

    sic, std = it["asip_sic"], it["asip_std"]
    tgt_asip, tgt_blk = part["asip"], part["blk"]
    nw = it["cimr"].shape[0]
    is_t = torch.zeros(nw, 1, 1, 1, dtype=torch.bool, device=dev)
    is_t[-1] = True
    keep_in, keep_tgt = ~(is_t & tgt_blk), is_t & tgt_blk

    out = {"omega_h": omega_h.float(), "omega_m": omega_m.float(), "omega_c": omega_c.float()}

    # ---- inputs: G_l(Y^in) per day ----
    asip_in = torch.ones_like(sic, dtype=torch.bool)
    asip_in[-1] = ~tgt_asip
    out["in_H"], out["in_H_w"] = obs.grid_asip(sic, std, asip_in, ASIP_SIGMA_FLOOR)
    m_t, w_t = g.restrict(out["in_H"][-1], out["in_H_w"][-1], 4)
    out["in_M_asip"] = torch.cat([it["asip_m"], m_t[None].float()])
    out["in_M_asip_w"] = torch.cat([it["asip_m_w"], w_t[None].float()])
    cimr_sigma = torch.tensor(OBS_SIGMA["cimr"], **f64)[:, None, None]
    # CIMR is a prior, never a target: all of it enters Y^in, day t included
    out["in_C_cimr"], out["in_C_cimr_w"] = obs.grid_dense(it["cimr"].double(), cimr_sigma, ay, ax)
    cr_sigma = torch.tensor(OBS_SIGMA["cristal"], **f64)
    # CRISTAL Y^in as a point table (x, y, value, variable 0 = SIT / 1 = HS, day offset <= 0)
    px, py, pv, pg = _points(it["cristal"].double(), keep_in, x5, y5)
    out["pts"] = torch.stack([px, py, pv, (pg % 2).double(), (pg // 2 - (nw - 1)).double()], -1)
    out["f_C"] = obs.grid_dense(it["era5"].double(), torch.ones((), **f64), ay, ax)[0]

    # ---- targets: G_l(Y^tgt) at day t ----
    sic_h, w_h = obs.grid_asip(sic[-1], std[-1], tgt_asip, ASIP_SIGMA_FLOOR)
    sic_m, w_m_sic = g.restrict(sic_h, w_h, 4)
    px, py, pv, pg = _points(it["cristal"][-1:].double(), keep_tgt[-1:], x5, y5)
    pw = cr_sigma[pg % 2] ** -2
    cr_c, cr_cw = obs.grid_points(px, py, pv, pw, pg, 2, *edges["C"])
    cr_m, cr_mw = obs.grid_points(px, py, pv, pw, pg, 2, *edges["M"])
    # one target per variable: SIC from ASIP (C cells observed over >= GAMMA_S of
    # their ocean), SIT and HS from CRISTAL
    sic_c, _ = g.restrict(sic_m, w_m_sic, g.factor("M", "C"))
    cov_c = g.restrict((w_h > 0).double(), omega_h, g.factor("H", "C"))[0]
    out["tgt_C"] = torch.stack([sic_c, cr_c[0], cr_c[1]])  # SIC, SIT, HS
    out["tgt_C_m"] = torch.stack([cov_c >= obs.GAMMA_S, cr_cw[0] > 0, cr_cw[1] > 0]) & (omega_c > 0)
    out["tgt_M"] = torch.stack([sic_m, cr_m[0]])
    out["tgt_M_m"] = torch.stack([(w_m_sic > 0) & (omega_m > 0), (cr_mw[0] > 0) & (omega_m > 0)])
    out["tgt_H"], out["tgt_H_m"] = sic_h[None], ((w_h > 0) & (omega_h > 0))[None]
    d_sic_m, m_sic_m = obs.detail_target(sic_m, w_m_sic, omega_m, 8, GAMMA["SIC"])
    d_sit_m, m_sit_m = obs.detail_target(cr_m[0], cr_mw[0], omega_m, 8, GAMMA["SIT"])
    out["tgt_dM"], out["tgt_dM_m"] = torch.stack([d_sic_m, d_sit_m]), torch.stack([m_sic_m, m_sit_m])
    d_sic_h, m_sic_h = obs.detail_target(sic_h, w_h, omega_h, 4, GAMMA["SIC"])
    out["tgt_dH"], out["tgt_dH_m"] = d_sic_h[None], m_sic_h[None]
    out["tgt_S"], out["tgt_S_m"], out["tgt_S_sigma"] = obs.subgrid_stats(sic_h, w_h, omega_h, cr_m[0], cr_mw[0], omega_m)
    return {k: v.float() if v.is_floating_point() else v for k, v in out.items()}


def _stack(samples: list[dict]) -> dict:
    out = {k: torch.stack([s[k] for s in samples]) for k in samples[0] if k != "pts"}
    n_pts = max(len(s["pts"]) for s in samples)
    pts = samples[0]["pts"].new_zeros(len(samples), n_pts, 5)
    valid = torch.zeros(len(samples), n_pts, dtype=torch.bool, device=pts.device)
    for b, s in enumerate(samples):
        pts[b, :len(s["pts"])], valid[b, :len(s["pts"])] = s["pts"], True
    out["pts"], out["pts_valid"] = pts, valid
    return out


def build_batch(batch: dict) -> list[dict]:
    """One batch per analysis day t, ..., t+K (a single one for N0)."""
    n_days = batch["asip_acq"].shape[1]
    out = []
    for k in range(n_days):
        samples = [build_sample(window({key: v[b] for key, v in batch.items()}, k))
                   for b in range(len(batch["seed"]))]
        day = _stack(samples)
        day.update({"day": batch["day"] + k, "iy0": batch["iy0"], "ix0": batch["ix0"]})
        out.append(day)
    return out


class CascadeDataModule(pl.LightningDataModule):

    def __init__(self, root, natural_earth_zip, train_dates, val_dates, test_dates,
                 tiles_per_day=8, dl_kw=None, horizon=0, asip_weighted_tiles=False):
        super().__init__()
        self.asip_weighted_tiles = asip_weighted_tiles  # training tiles drawn by ASIP coverage (fine levels)
        self.horizon = horizon  # 0: N0 batches; K: N1 sequences of K + 1 analysis days
        self.root, self.natural_earth_zip = root, natural_earth_zip
        self.dates = {"train": train_dates, "val": val_dates, "test": test_dates}
        self.tiles_per_day, self.dl_kw = tiles_per_day, dl_kw or {}
        self.ds = {}

    def _days(self, split):
        """Analysis days t of the split such that t..t+K are in every store."""
        start, end = (to_day(d) for d in self.dates[split])
        avail = set(range(start, end + 1 + self.horizon))
        for src in ("asip", *SOURCES):
            avail &= {int(d) for d in zarr.open_array(os.path.join(self.root, STORES[src], "days"), mode="r")[:]}
        return [t for t in range(start, end + 1) if all(t + k in avail for k in range(self.horizon + 1))]

    def setup(self, stage=None):
        if self.ds:
            return
        from .ocean import ocean_fraction_h
        omega_h = ocean_fraction_h(self.natural_earth_zip)
        omega_t = torch.from_numpy(omega_h)
        omega_c = g.restrict(omega_t, torch.ones_like(omega_t), g.factor("H", "C"))[0].numpy()
        self.ds["train"] = CascadeDataset(self.root, self._days("train"), omega_h,
                                          valid_tile_origins(omega_c, 1), self.tiles_per_day, train=True,
                                          horizon=self.horizon, asip_weighted=self.asip_weighted_tiles)
        for split in ("val", "test"):
            self.ds[split] = CascadeDataset(self.root, self._days(split), omega_h,
                                            valid_tile_origins(omega_c, N_C), train=False, horizon=self.horizon)

    def _loader(self, split, **kw):
        kw = {**self.dl_kw, **kw}
        if not kw.get("num_workers"):
            kw.pop("prefetch_factor", None)  # only valid with worker processes
        return torch.utils.data.DataLoader(self.ds[split], **kw)

    def train_dataloader(self):
        return self._loader("train")

    def val_dataloader(self):
        return self._loader("val", shuffle=False)

    def test_dataloader(self):
        return self._loader("test", shuffle=False)

    def on_after_batch_transfer(self, batch, dataloader_idx):
        days = build_batch(batch)
        return days if self.horizon else days[0]
