"""(rebuttal) Observation costs and data helpers for the sea-ice (SIC) experiment.

SIC observations come in two kinds:
  * `input`  : sparse high-resolution SAR retrievals (gappy ASIP), NaN where unobserved;
  * `osisaf` : the gap-free coarse passive-microwave product (12.5 km), NaN on land.
The training target `asip` is itself only partially observed (NaN gaps): every data term is masked.

Conventions used by the SIC notebooks of the rebuttal:
  * variational models (VarCM, DynVarCM): y = cat(input, osisaf) along channels (2C), split by the costs below:
        J_o(x) = 1/2 ||M_sar (x - y_sar)||^2  +  w_osi * k^2/2 ||M_c (P_k x - P_k y_osi)||^2
    with P_k the k x k block average (masked on land) and M_c its validity mask; the k^2 factor gives one coarse cell
    the weight of the k^2 pixels it averages.
  * flow/consistency models with a frame-indexed network (DynFM, DynCM): one C-channel field, `fuse_obs`
    (SAR where observed, OSI SAF elsewhere), so that the networks and frame logic of the SSH notebooks are unchanged.
"""
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def fuse_obs(sar: Tensor, osi: Tensor) -> Tensor:
    """SAR where observed, OSI SAF elsewhere (NaN only where both are missing, i.e. land)."""
    return torch.where(torch.isnan(sar), osi, sar)


def masked_pool(a: Tensor, k: int):
    """k x k block average ignoring NaN. Returns (pooled values with 0 where empty, validity mask)."""
    m = (~torch.isnan(a)).to(a.dtype if a.is_floating_point() else torch.float32)
    s = F.avg_pool2d(torch.nan_to_num(a) * m, k, ceil_mode=True)
    c = F.avg_pool2d(m, k, ceil_mode=True)
    return s / c.clamp(min=1e-6), (c > 0.5).to(s.dtype)


class SICObsCost(nn.Module):
    """Static SIC observation cost (VarCM). x: (B, C, H, W); y = cat(sar, osisaf): (B, 2C, H, W). Scalar."""

    def __init__(self, n_frames: int, pool: int = 16, w_osi: float = 1.0, sigma_obs: float = 1.0) -> None:
        super().__init__()
        self.n_frames, self.pool, self.w_osi, self.sigma_obs = n_frames, pool, w_osi, sigma_obs

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        C = self.n_frames
        sar, osi = y[:, :C], y[:, C:2 * C]
        m = (~torch.isnan(sar)).to(x.dtype)
        j_sar = (m * (x - torch.nan_to_num(sar))).pow(2).sum()
        if osi.shape[1] == 0:                       # y without the OSI SAF channels: SAR term only
            return j_sar / (2.0 * self.sigma_obs ** 2)
        px, _ = masked_pool(x, self.pool)
        po, mc = masked_pool(osi.to(x.dtype), self.pool)
        j_osi = (mc * (px - po)).pow(2).sum() * self.pool ** 2
        return (j_sar + self.w_osi * j_osi) / (2.0 * self.sigma_obs ** 2)


class SICFrameObsCost(nn.Module):
    """Frame-wise SIC observation cost (DynVarCM), same interface as VarDynCMObsCost:
    x: (B, 1, H, W) state of the current frame, y = cat(sar, osisaf): (B, 2C, H, W), t: (B,) solver time.
    The frame weights w_k(t) are those of VarDynCMObsCost (frame 0 at t = sb, frame C-1 at t = 0)."""

    def __init__(self, n_frames: int, spinup_boundary: float = 0.7, pool: int = 16, w_osi: float = 1.0,
                 sigma_obs: float = 1.0) -> None:
        super().__init__()
        self.n_frames, self.spinup_boundary = n_frames, spinup_boundary
        self.pool, self.w_osi, self.sigma_obs = pool, w_osi, sigma_obs

    def frame_weights(self, t: Tensor, dtype) -> Tensor:
        C, sb = self.n_frames, self.spinup_boundary
        B, dev = t.shape[0], t.device
        phys_step = sb / max(C - 1, 1)
        phys_steps = torch.linspace(0.0, sb, C, device=dev, dtype=dtype)
        tc = t.to(dtype).clamp(0.0, sb)
        idx = torch.searchsorted(phys_steps.unsqueeze(0).expand(B, -1).contiguous(), tc.unsqueeze(1)
                                 ).squeeze(1).clamp(1, C - 1)
        a = ((phys_steps[idx] - tc) / phys_step).clamp(0.0, 1.0)
        w = torch.zeros(B, C, device=dev, dtype=dtype)
        b = torch.arange(B, device=dev)
        w[b, C - idx] = a
        w[b, C - 1 - idx] = 1.0 - a
        return w

    def forward(self, x: Tensor, y: Tensor, t: Tensor) -> Tensor:
        C = self.n_frames
        sar, osi = y[:, :C].to(x.dtype), y[:, C:2 * C].to(x.dtype)
        w = self.frame_weights(t, x.dtype)                                   # (B, C)
        cost = x.new_zeros(x.shape[0])
        px, _ = masked_pool(x, self.pool)
        for k in range(C):
            wk = w[:, k]
            if not bool((wk > 0).any()):
                continue
            mk = (~torch.isnan(sar[:, [k]])).to(x.dtype)
            j_sar = (mk * (x - torch.nan_to_num(sar[:, [k]]))).pow(2).sum(dim=(-1, -2, -3))
            po, mc = masked_pool(osi[:, [k]], self.pool)
            j_osi = (mc * (px - po)).pow(2).sum(dim=(-1, -2, -3)) * self.pool ** 2
            cost = cost + wk * (j_sar + self.w_osi * j_osi)
        n = (~torch.isnan(sar)).to(x.dtype).sum(dim=(-1, -2, -3)).clamp(min=1.0)
        return (cost / n / (2.0 * self.sigma_obs ** 2)).mean()
