"""
encoders.py
===========
Observation encoders and multi-scale operators of the CROSCIM cascade
(formulation section 3a-b):

    PointCellEncoder  irregular observations -> tokens -> cross-attention per
                      cell (segmented softmax), a_l(k) on Omega_l + mask
    DenseEncoder      gridded observations on their own level (masked
                      "CNN a pas")
    AsipHEncoder      ASIP at H fused with the upward pooling to M: a strided
                      CNN applied per C cell, on non-empty cells only, with
                      gradient checkpointing
    Up                Pi^up: K learned queries cross-attending to the children
                      of each parent (masked children excluded)
    Down              Pi^down: (Pi^down g)(i) = W g(par(i)) + p_i
    Fuse              fusion block U_l
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from . import grid as g


def child_offsets(f: int, device=None) -> Tensor:
    """Position of each child in its parent, in [-1, 1]: [f*f, 2] (row-major)."""
    o = (torch.arange(f, device=device, dtype=torch.float32) + 0.5) / f * 2 - 1
    yy, xx = torch.meshgrid(o, o, indexing="ij")
    return torch.stack([yy.flatten(), xx.flatten()], -1)


def _pos_mlp(d: int) -> nn.Module:
    return nn.Sequential(nn.Linear(2, d), nn.GELU(), nn.Linear(d, d))


class PointCellEncoder(nn.Module):
    """e_n = MLP[y_n, type_n, x_n - x_k(n), t_n - t]; a(k) = sum_{n in k}
    softmax_{n in k}(q . W_K e_n / sqrt d) W_V e_n (empty cell -> 0, mask 0)."""

    def __init__(self, d: int, n_types: int, level: str, window: int):
        super().__init__()
        self.level, self.window, self.n_types = level, window, n_types
        self.mlp = nn.Sequential(nn.Linear(1 + n_types + 3, d), nn.GELU(), nn.Linear(d, d))
        self.q = nn.Parameter(torch.randn(d) / math.sqrt(d))
        self.wk, self.wv = nn.Linear(d, d, bias=False), nn.Linear(d, d)

    def forward(self, pts: Tensor, valid: Tensor, iy0: Tensor, ix0: Tensor, n_c: int) -> tuple[Tensor, Tensor]:
        """pts [B, P, 5] = (x, y, normalised value, type, day offset); returns
        a [B, d, n, n] and the mask [B, n, n] of the level-`level` tile."""
        bsz, d = pts.shape[0], self.q.shape[0]
        f, dx = g.factor(self.level, "C"), g.DX_M[self.level]
        n = n_c * f
        top = g.HALF_EXTENT_M - iy0.to(pts) * g.DX_M["C"]
        left = -g.HALF_EXTENT_M + ix0.to(pts) * g.DX_M["C"]
        cx = (pts[..., 0] - left[:, None]) / dx
        cy = (top[:, None] - pts[..., 1]) / dx
        ix, iy = torch.floor(cx).long(), torch.floor(cy).long()
        ok = valid & (ix >= 0) & (ix < n) & (iy >= 0) & (iy < n)
        seg = ((torch.arange(bsz, device=pts.device)[:, None] * n + iy) * n + ix)[ok]
        p = pts[ok]
        feat = torch.cat([p[:, 2:3], F.one_hot(p[:, 3].long(), self.n_types).to(p),
                          (cx - ix)[ok][:, None] - 0.5, (cy - iy)[ok][:, None] - 0.5, p[:, 4:5] / self.window], -1)
        e = self.mlp(feat)
        # segmented softmax accumulated in float32 (under bf16 autocast the
        # projections come out in bf16 and exp in float32)
        s = (self.wk(e) @ self.q).float() / math.sqrt(d)
        v = self.wv(e).float()
        smax = s.new_full((bsz * n * n,), -float("inf")).scatter_reduce(0, seg, s, "amax", include_self=True)
        ex = torch.exp(s - smax[seg])
        den = s.new_zeros(bsz * n * n).index_add(0, seg, ex)
        num = s.new_zeros(bsz * n * n, d).index_add(0, seg, ex[:, None] * v)
        mask = den > 0
        a = torch.where(mask[:, None], num / den.clamp_min(1e-30)[:, None], 0.0)
        return a.view(bsz, n, n, d).permute(0, 3, 1, 2), mask.view(bsz, n, n)


class DenseEncoder(nn.Module):
    """Gridded observations on their own grid -> d features, 0 where nothing
    is observed."""

    def __init__(self, n_in: int, d: int):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(n_in, d, 1), nn.GELU(), nn.Conv2d(d, d, 3, padding=1))

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        return self.net(x) * mask[:, None]


class AsipHEncoder(nn.Module):
    """E_H fused with Pi^up_{H->M}: per C cell, a strided CNN maps the 32 x 32
    H pixels to the 8 x 8 M children. Only C cells with observations are
    processed; the CNN is checkpointed."""

    def __init__(self, n_in: int, d: int):
        super().__init__()
        f = g.factor("H", "M")
        self.net = nn.Sequential(nn.Conv2d(n_in, d, f, stride=f), nn.GELU(), nn.Conv2d(d, d, 1))

    def forward(self, x: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """x [B, c, NH, NH], mask [B, NH, NH] -> ([B, d, NM, NM], mask [B, NM, NM])."""
        bsz, c, nh, _ = x.shape
        fc, fm = g.factor("H", "C"), g.factor("H", "M")
        nc, k = nh // fc, fc // fm
        patches = x.view(bsz, c, nc, fc, nc, fc).permute(0, 2, 4, 1, 3, 5).reshape(-1, c, fc, fc)
        nonempty = mask.view(bsz, nc, fc, nc, fc).any(4).any(2).flatten()
        d = self.net[-1].out_channels
        out = x.new_zeros(bsz * nc * nc, d, k, k)
        if nonempty.any():
            sel = patches[nonempty]
            res = checkpoint(self.net, sel, use_reentrant=False) if torch.is_grad_enabled() else self.net(sel)
            out[nonempty] = res.to(out.dtype)  # bf16 under autocast
        out = out.view(bsz, nc, nc, d, k, k).permute(0, 3, 1, 4, 2, 5).reshape(bsz, d, nc * k, nc * k)
        mask_m = mask.view(bsz, nc * k, fm, nc * k, fm).any(4).any(2)
        return out * mask_m[:, None], mask_m


class Up(nn.Module):
    """(Pi^up h)(k) = W_O [sum_{i in ch(k)} softmax_i(q_j . W_K(h_i + p_i)/sqrt d) W_V(h_i + p_i)]_{j=1..K}."""

    def __init__(self, d: int, f: int, n_queries: int = 4):
        super().__init__()
        self.f = f
        self.q = nn.Parameter(torch.randn(n_queries, d) / math.sqrt(d))
        self.wk, self.wv = nn.Linear(d, d, bias=False), nn.Linear(d, d)
        self.wo = nn.Linear(n_queries * d, d)
        self.pos = _pos_mlp(d)

    def forward(self, h: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """h [B, d, N, N] (children), mask [B, N, N] -> ([B, d, N/f, N/f], parent mask)."""
        bsz, d, n, _ = h.shape
        f, npar = self.f, n // self.f
        x = h.permute(0, 2, 3, 1).reshape(bsz, npar, f, npar, f, d).permute(0, 1, 3, 2, 4, 5).reshape(-1, f * f, d)
        m = mask.reshape(bsz, npar, f, npar, f).permute(0, 1, 3, 2, 4).reshape(-1, f * f)
        x = x + self.pos(child_offsets(f, h.device).to(h))
        s = torch.einsum("kd,npd->nkp", self.q, self.wk(x)) / math.sqrt(d)
        s = s.masked_fill(~m[:, None], -float("inf"))
        a = torch.nan_to_num(torch.softmax(s, -1))  # parents without children -> 0
        out = self.wo(torch.einsum("nkp,npd->nkd", a, self.wv(x)).flatten(1))
        pm = m.any(-1)
        out = out * pm[:, None]
        return out.view(bsz, npar, npar, d).permute(0, 3, 1, 2), pm.view(bsz, npar, npar)


class Down(nn.Module):
    """(Pi^down g)(i) = W g(par(i)) + p_i, from a parent grid to its f x f children."""

    def __init__(self, n_in: int, d: int, f: int):
        super().__init__()
        self.f = f
        self.lin = nn.Conv2d(n_in, d, 1)
        self.pos = _pos_mlp(d)

    def forward(self, gp: Tensor) -> Tensor:
        f = self.f
        p = self.pos(child_offsets(f, gp.device).to(gp)).T.reshape(-1, f, f)  # [d, f, f]
        out = g.prolong(self.lin(gp), f)
        n = out.shape[-1] // f
        return out + p.repeat(1, out.shape[-2] // f, n)[None]


class Fuse(nn.Module):
    """Fusion block U_l of the upward pyramid."""

    def __init__(self, n_in: int, d: int):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(n_in, d, 1), nn.GELU(), nn.Conv2d(d, d, 3, padding=1))

    def forward(self, *parts: Tensor) -> Tensor:
        return self.net(torch.cat(parts, 1))
