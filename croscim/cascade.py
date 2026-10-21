"""
cascade.py
==========
N0, the initialisation network of the CROSCIM cascade (formulation
sections 3-6; brief phases 3-5):

    p(Z | Y, F) = p(Z_C | c_C) p(Z_M | Z_C, c_M) p(dX_H | Z_C, Z_M, c_H)

    G_C  Z_C = (SIC, SIT, HS, S_C)  on the 64 x 64 C tile,  c_C = (h_C, f)
    G_M  dX_M = (dSIC, dSIT)        on 128 x 128 M crops,   c_M = (X_C^eta, h_M, Pi^down[f, Phi_C])
    G_H  dX_H = dSIC                on 256 x 256 H crops,   c_H = (X_C^eta, dX_M^eta, h_H, Pi^down[f, Phi_C])

Crops are aligned on C cells, so Pi0 (noise and velocity of the details) is
exact per crop. At training time the coarse conditioning comes from a sample
of the EMA cascade, without gradient, plus a small noise eta. Each context
group (h, Phi, f) is replaced by "null" with probability p_null, so v(.|h)
and v(.|null) are learned together (guidance). Phi_C = (S_C, psi_C) with
psi_C = Psi_C(h_C) a small latent field.

    L = sum_l (L_X^l + lambda_S L_S^l) + lambda_c L_cyc + lambda_psi L_psi

`levels` selects the trained generators (phase 3: ["C"] with pyramid=False).

corrector=True (formulation section 8): G_M and G_H generate corrections of
the prolonged coarser sample, dX_M = X_M - P' X_C, dX_H = X_H - P' X_M, with
targets Y - P' X^sample and without the constraint R dX = 0 (no Pi0
projection), censored at the physical bounds of X. They can then correct the
coarse errors (level, ice edge) and remove the C-cell steps; X_C stays the G_C
sample (R X_M is the consistent coarse field). corrector=False: exact
details (R dX = 0) with the targets Y - P' R Y, for the former checkpoints.
"""

from __future__ import annotations

import os

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim.swa_utils import AveragedModel

from . import grid as g
from . import obs
from .data import N_C, W, W_H
from .encoders import AsipHEncoder, DenseEncoder, Down, Fuse, PointCellEncoder, Up
from .generators import Generator, channel_mean

X_NAMES = ("SIC", "SIT", "HS")
Z_NAMES = X_NAMES + obs.S_NAMES
N_X, N_Z, N_S = len(X_NAMES), len(Z_NAMES), len(obs.S_NAMES)
D_NAMES = {"M": ("dSIC_M", "dSIT_M"), "H": ("dSIC_H",)}
FORCING = ("t2m", "msl", "u10", "v10")
# physical bounds of each Z channel (None: unbounded)
Z_BOUNDS = {"SIC": (0, 1), "SIT": (0, None), "HS": (0, None),
            "h_ow": (0, 1), "h_thin": (0, 1), "h_mid": (0, 1), "h_pack": (0, 1),
            "sic_std": (0, None), "edge_density": (0, 1), "aniso_c": (-1, 1), "aniso_s": (-1, 1),
            "sit_std": (0, None)}
CROP_C = {"M": 16, "H": 8}  # crop sizes in C cells (M: 128 x 128, H: 256 x 256)
GROUPS = ("h", "phi", "f")


def make_ema(module: nn.Module, decay: float) -> AveragedModel:
    """EMA wrapper with a fixed decay rate (same as the former CROSCIM FM model)."""
    ema = AveragedModel(module, avg_fn=lambda avg, cur, n: decay * avg + (1.0 - decay) * cur)
    ema.eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    return ema


def strict_mask(b: dict) -> Tensor:
    """Mask of the strict Z_C targets (scores): without the weak CIMR SIC targets."""
    return torch.cat([b["tgt_C_m"] & (b["tgt_C_sigma"] == 0), b["tgt_S_m"]], 1)


def crop(x: Tensor, oy: Tensor, ox: Tensor, size: int, f: int) -> Tensor:
    """Per-sample crop of the last two dims; origins (oy, ox) in C cells, f
    cells of x per C cell."""
    return torch.stack([x[b, ..., int(oy[b]) * f:(int(oy[b]) + size) * f, int(ox[b]) * f:(int(ox[b]) + size) * f]
                        for b in range(x.shape[0])])


def tile_pos(b: dict, oy=None, ox=None, size=N_C) -> Tensor:
    """Normalised projected position (x, y) of the C cells of each (crop of a) tile."""
    x_c, y_c = g.centers("C")
    out = []
    for i in range(len(b["iy0"])):
        iy0 = int(b["iy0"][i]) + (0 if oy is None else int(oy[i]))
        ix0 = int(b["ix0"][i]) + (0 if ox is None else int(ox[i]))
        yy = y_c[iy0:iy0 + size][:, None].expand(size, size)
        xx = x_c[ix0:ix0 + size][None, :].expand(size, size)
        out.append(torch.stack([xx, yy]) / g.HALF_EXTENT_M)
    return torch.stack(out).to(b["omega_c"])


class Norm(nn.Module):
    """Normalisation of the state, observations and forcing: (x - mean) / std."""

    def __init__(self, norm: dict):
        super().__init__()
        self.norm = norm
        for name, names in (("z", Z_NAMES), ("dM", D_NAMES["M"]), ("dH", D_NAMES["H"])):
            mean = torch.tensor([0.0 if name != "z" else norm[k][0] for k in names])
            std = torch.tensor([norm[k][1] for k in names])
            self.register_buffer(f"{name}_mean", mean.view(1, -1, 1, 1), persistent=False)
            self.register_buffer(f"{name}_std", std.view(1, -1, 1, 1), persistent=False)
        lo = [-float("inf") if Z_BOUNDS[k][0] is None else (Z_BOUNDS[k][0] - norm[k][0]) / norm[k][1] for k in Z_NAMES]
        hi = [float("inf") if Z_BOUNDS[k][1] is None else (Z_BOUNDS[k][1] - norm[k][0]) / norm[k][1] for k in Z_NAMES]
        self.register_buffer("z_lo", torch.tensor(lo).view(1, -1, 1, 1), persistent=False)
        self.register_buffer("z_hi", torch.tensor(hi).view(1, -1, 1, 1), persistent=False)

    def __call__(self, x: Tensor, names, dim: int) -> Tensor:
        shape = [1] * x.ndim
        shape[dim] = len(names)
        mean = torch.tensor([self.norm[k][0] for k in names], device=x.device, dtype=x.dtype).view(shape)
        std = torch.tensor([self.norm[k][1] for k in names], device=x.device, dtype=x.dtype).view(shape)
        return (x - mean) / std

    def z(self, z: Tensor) -> Tensor:
        return (z - self.z_mean) / self.z_std

    def z_inv(self, z: Tensor) -> Tensor:
        return z * self.z_std + self.z_mean


class Nets(nn.Module):
    """All N0 networks (one EMA copy for the whole set)."""

    def __init__(self, d: int, d_psi: int, swin: dict, n_up_queries: int = 4, raw_days: int = 0,
                 smooth_cond: bool = False):
        super().__init__()
        self.d, self.d_psi, self.raw_days = d, d_psi, raw_days
        self.e_cimr = DenseEncoder(W * 4, d)
        self.e_pts_c = PointCellEncoder(d, 2, "C", W)   # CRISTAL HS (Y_C)
        self.e_pts_m = PointCellEncoder(d, 2, "M", W)   # CRISTAL SIT (Y_M)
        self.e_asip_m = DenseEncoder(W * 2, d)          # ASIP aggregated at M, per day
        self.e_asip_h = AsipHEncoder(W_H * 2, d)        # ASIP at H, fused with Pi^up_{H->M}
        self.e_f = DenseEncoder(W * 5, d)               # ERA5 + validity, per day
        self.up_cm = Up(d, g.factor("M", "C"), n_up_queries)
        self.fuse_m = Fuse(3 * d + 3, d)
        self.fuse_c = Fuse(4 * d + 3, d)
        self.psi = nn.Conv2d(d, d_psi, 1)
        n_phi = d + N_S + d_psi
        self.down_m = Down(n_phi, d, g.factor("M", "C"), smooth_cond)
        self.down_h = Down(n_phi, d, g.factor("H", "C"), smooth_cond)
        self.gen = nn.ModuleDict({
            "C": Generator(N_Z, self.n_cond("C"), (N_C, N_C), 32., swin["C"]),
            "M": Generator(2, self.n_cond("M"), (128, 128), 4., swin["M"]),
            "H": Generator(1, self.n_cond("H"), (256, 256), 1., swin["H"]),
        })

    def n_cond(self, level: str) -> int:
        d = self.d
        return {"C": 2 * d + 2 + 3 + 4 * self.raw_days, "M": N_X + 2 * d + 3 + 1,
                "H": N_X + d + 2 * W_H + 3 + 1}[level]


class LitCascadeN0(pl.LightningModule):
    """
    Parameters
    ----------
    norm : dict name -> [mean, std] for the Z_C channels, the forcing, and
        the detail scales dSIC_M, dSIT_M, dSIC_H (mean ignored).
    swin : {"C": ..., "M": ..., "H": ...} kwargs of FMSwinWrapper.
    levels : generators trained, subset of ["C", "M", "H"].
    pyramid : use the upward pyramid (Pi^up) in h_M and h_C.
    """

    def __init__(self, norm, swin, levels=("C", "M", "H"), pyramid=True, d=32, d_psi=4,
                 n_up_queries=4, raw_days=0, smooth_cond=False, corrector=False, p_null=0.1, eta=0.05, lambda_s=1.0, lambda_c=0.1, lambda_psi=0.01,
                 lambda_perp=1.0, cyc_tau=0.05, lr=1e-4, betas=(0.9, 0.995), lr_scheduler_start_factor=1e-5,
                 lr_scheduler_iters=20_000, ema_decay=0.999, n_steps=20, cond_steps=6, n_members=8,
                 score_every=5, n_score_batches=20):
        super().__init__()
        self.save_hyperparameters()
        self.levels, self.pyramid = tuple(levels), pyramid
        self.nrm = Norm(norm)
        self.nets = Nets(d, d_psi, swin, n_up_queries, raw_days, smooth_cond)
        self.ema = make_ema(self.nets, ema_decay)
        self._ranks = {}

    def train(self, mode: bool = True):
        """The EMA copy always stays in eval mode (no drop-path in the samples
        used as conditioning)."""
        super().train(mode)
        self.ema.eval()
        return self

    # ----------------------------------------------------------- contexts

    def encode(self, nets: Nets, b: dict) -> dict:
        """Upward pyramid: h_M = U_M(a_M, Pi^up h_H), h_C = U_C(a_C, Pi^up h_M, f), psi_C."""
        nrm, bsz = self.nrm, b["omega_c"].shape[0]
        m = b["in_C_cimr_w"] > 0
        x = torch.cat([torch.where(m, nrm(b["in_C_cimr"], ("SIC", "SIT"), 2), 0.0), m.to(b["in_C_cimr"])], 2)
        m_cimr = m.any(2).any(1)
        a_cimr = nets.e_cimr(x.flatten(1, 2), m_cimr)

        pts = b["pts"].clone()
        sit, hs = pts[..., 3] == 0, pts[..., 3] == 1
        pts[..., 2] = torch.where(sit, (pts[..., 2] - nrm.norm["SIT"][0]) / nrm.norm["SIT"][1],
                                  (pts[..., 2] - nrm.norm["HS"][0]) / nrm.norm["HS"][1])
        a_pc, m_pc = nets.e_pts_c(pts, b["pts_valid"] & hs, b["iy0"], b["ix0"], N_C)
        a_pm, m_pm = nets.e_pts_m(pts, b["pts_valid"] & sit, b["iy0"], b["ix0"], N_C)

        m = b["in_M_asip_w"] > 0
        x = torch.cat([torch.where(m, nrm(b["in_M_asip"], ("SIC",), 1), 0.0), m.to(b["in_M_asip"])], 1)
        m_am = m.any(1)
        a_am = nets.e_asip_m(x, m_am)

        m = b["in_H_w"] > 0
        if self.pyramid:
            x = torch.cat([torch.where(m, nrm(b["in_H"], ("SIC",), 1), 0.0), m.to(b["in_H"])], 1)
            up_h, m_uh = nets.e_asip_h(x, m.any(1))
        else:
            up_h, m_uh = torch.zeros_like(a_am), torch.zeros_like(m_am)
        h_m = nets.fuse_m(a_pm, a_am, up_h, m_pm[:, None].to(a_am), m_am[:, None].to(a_am), m_uh[:, None].to(a_am))
        mask_m = m_pm | m_am | m_uh
        if self.pyramid:
            up_m, m_um = nets.up_cm(h_m, mask_m)
        else:
            up_m, m_um = torch.zeros_like(a_cimr), torch.zeros_like(m_cimr)

        f = b["f_C"]
        ok = torch.isfinite(f).all(2)
        x = torch.cat([torch.nan_to_num(nrm(f, FORCING, 2)) * ok[:, :, None], ok[:, :, None].to(f)], 2)
        f_enc = nets.e_f(x.flatten(1, 2), torch.ones_like(m_cimr))
        h_c = nets.fuse_c(a_cimr, a_pc, up_m, f_enc, m_cimr[:, None].to(f), m_pc[:, None].to(f), m_um[:, None].to(f))
        return {"h_C": h_c, "m_C": m_cimr | m_pc | m_um, "f": f_enc, "h_M": h_m, "m_M": mask_m,
                "psi": nets.psi(h_c)}

    @staticmethod
    def _flags(keep: dict, like: Tensor) -> list[Tensor]:
        return [keep[k].to(like).view(-1, 1, 1, 1).expand(-1, 1, *like.shape[-2:]) for k in GROUPS]

    def cond_c(self, ctx: dict, b: dict, keep: dict) -> tuple[Tensor, Tensor]:
        kh, kf = (keep[k].to(ctx["f"]).view(-1, 1, 1, 1) for k in ("h", "f"))
        flags = self._flags(keep, ctx["f"])
        parts = [ctx["h_C"] * kh, ctx["f"] * kf, flags[0], flags[2], b["omega_c"][:, None], tile_pos(b)]
        r = self.hparams.raw_days
        if r:
            # the last r days of gridded CIMR (Y^in), uncompressed: the encoder squeezes the 15 days
            # into d channels, and G_C did not even reach the persistence of CIMR
            m = b["in_C_cimr_w"][:, -r:] > 0
            raw = torch.cat([torch.where(m, self.nrm(b["in_C_cimr"][:, -r:], ("SIC", "SIT"), 2), 0.0), m.to(kh)], 2)
            parts.append(raw.flatten(1, 2) * kh)
        cond = torch.cat(parts, 1)
        return cond, (b["omega_c"] > 0)[:, None].to(cond)

    def _phi(self, ctx: dict, z_c: Tensor, psi: Tensor, keep: dict) -> Tensor:
        """[f, S_C, psi_C] (normalised) on the C tile, with dropped groups at 0."""
        kp, kf = (keep[k].to(z_c).view(-1, 1, 1, 1) for k in ("phi", "f"))
        return torch.cat([ctx["f"] * kf, self.nrm.z(z_c)[:, N_X:] * kp, psi * kp], 1)

    def cond_m(self, nets: Nets, ctx: dict, b: dict, z_c: Tensor, psi: Tensor, oy, ox, keep: dict):
        """c_M on the M crop (origin oy, ox in C cells; z_c physical)."""
        n, f = CROP_C["M"], g.factor("M", "C")
        x_c = crop(self.nrm(self.coarse_m(z_c, b), X_NAMES, 1), oy, ox, n, f)  # P' X_C at M
        phi = crop(self._phi(ctx, z_c, psi, keep), oy, ox, n, 1)
        h = crop(ctx["h_M"], oy, ox, n, f) * keep["h"].to(z_c).view(-1, 1, 1, 1)
        omega = crop(b["omega_m"], oy, ox, n, f)[:, None]
        flags = self._flags(keep, omega)
        cond = torch.cat([x_c, nets.down_m(phi), h, *flags, omega], 1)
        return cond, (omega > 0).to(cond), omega[:, 0]

    def cond_h(self, nets: Nets, ctx: dict, b: dict, z_c: Tensor, x_mh: Tensor, psi: Tensor, oy, ox, keep: dict):
        """c_H on the H crop (origin in C cells); x_mh = P' X_M = (SIC, SIT)
        brought to H on the same crop (physical, see coarse_h)."""
        n, fh = CROP_C["H"], g.factor("H", "C")
        nrm = self.nrm
        xm = nrm(x_mh, ("SIC", "SIT"), 1)
        if self.hparams.smooth_cond:  # snow brought to H without steps (P' at M, then bilinear)
            hs_m = crop(nrm(self.coarse_m(z_c, b)[:, 2:3], ("HS",), 1), oy, ox, n, g.factor("M", "C"))
            hs = F.interpolate(hs_m, scale_factor=g.factor("H", "M"), mode="bilinear", align_corners=False)
        else:
            hs = g.prolong(crop(nrm.z(z_c)[:, 2:3], oy, ox, n, 1), fh)
        phi = crop(self._phi(ctx, z_c, psi, keep), oy, ox, n, 1)
        m = crop(b["in_H_w"], oy, ox, n, fh) > 0
        hh = torch.cat([torch.where(m, nrm(crop(b["in_H"], oy, ox, n, fh), ("SIC",), 1), 0.0), m.to(xm)], 1)
        hh = hh * keep["h"].to(xm).view(-1, 1, 1, 1)
        omega = crop(b["omega_h"], oy, ox, n, fh)[:, None]
        flags = self._flags(keep, omega)
        cond = torch.cat([xm, hs, nets.down_h(phi), hh, *flags, omega], 1)
        return cond, (omega > 0).to(cond), omega[:, 0]

    def keep_groups(self, bsz: int, device, drop: bool) -> dict:
        p = self.hparams.p_null if drop else 0.0
        return {k: torch.rand(bsz, device=device) >= p for k in GROUPS}

    # ------------------------------------------------------------ targets

    def target_c(self, b: dict):
        z = torch.cat([b["tgt_C"], b["tgt_S"]], 1)
        m = torch.cat([b["tgt_C_m"], b["tgt_S_m"]], 1)
        floor = torch.cat([b["tgt_C_sigma"], b["tgt_S_sigma"]], 1) / self.nrm.z_std
        return torch.where(m, self.nrm.z(z), 0.0), m, floor

    def choose_crops(self, b: dict) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """M crop (16 C cells) with the most dX_M targets among 8 random
        candidates; H crop (8 C cells) inside it with the most dX_H targets."""
        bsz, dev = b["omega_c"].shape[0], b["omega_c"].device
        nm, nh = CROP_C["M"], CROP_C["H"]
        km, kh = ("tgt_M_m", "tgt_H_m") if self.hparams.corrector else ("tgt_dM_m", "tgt_dH_m")
        cov_m = g.block_sum(b[km].any(1).float(), g.factor("M", "C"))
        cov_h = g.block_sum(b[kh][:, 0].float(), g.factor("H", "C"))
        oy = torch.zeros(bsz, dtype=torch.long, device=dev)
        ox, hy, hx = oy.clone(), oy.clone(), oy.clone()
        for i in range(bsz):
            cand = torch.randint(0, N_C - nm + 1, (8, 2), device=dev)
            score = torch.stack([cov_m[i, y:y + nm, x:x + nm].sum() for y, x in cand])
            oy[i], ox[i] = cand[score.argmax()]
            sub = [(oy[i] + dy, ox[i] + dx) for dy in (0, 4, 8) for dx in (0, 4, 8)]
            score = torch.stack([cov_h[i, y:y + nh, x:x + nh].sum() for y, x in sub])
            hy[i], hx[i] = sub[int(score.argmax())]
        return oy, ox, hy, hx

    # ------------------------------------------------------------ cascade

    def pi0(self, omega: Tensor, f: int):
        w = omega[:, None]
        return lambda x: g.pi0(x, w, f)

    def fine_kw(self, coarse: Tensor, omega: Tensor, f: int, std: Tensor) -> dict:
        """Projection (exact details) or bounds of the correction (corrector):
        coarse + dX in [0, 1] for SIC, >= 0 for SIT; coarse = (SIC, SIT) or SIC."""
        if not self.hparams.corrector:
            return {"proj": self.pi0(omega, f)}
        lo = -coarse / std
        hi = torch.cat([(1 - coarse[:, :1]) / std[:, :1], torch.full_like(coarse[:, 1:], float("inf"))], 1)
        return {"lo": lo, "hi": hi}

    @torch.no_grad()
    def sample_c(self, nets, ctx, b, keep, n_steps, generator=None, w=1.0):
        cond, valid = self.cond_c(ctx, b, keep)
        cond_null = None
        if w != 1.0:
            null = {**keep, "h": torch.zeros_like(keep["h"])}
            cond_null = self.cond_c(ctx, b, null)[0]
        z = nets.gen["C"].sample(cond, valid, n_steps, lo=self.nrm.z_lo, hi=self.nrm.z_hi,
                                 cond_null=cond_null, w=w, generator=generator)
        return self.nrm.z_inv(z)

    @torch.no_grad()
    def sample_m(self, nets, ctx, b, z_c, psi, oy, ox, keep, n_steps, generator=None):
        cond, valid, omega = self.cond_m(nets, ctx, b, z_c, psi, oy, ox, keep)
        coarse = crop(self.coarse_m(z_c, b)[:, :2], oy, ox, CROP_C["M"], g.factor("M", "C"))
        kw = self.fine_kw(coarse, omega, g.factor("M", "C"), self.nrm.dM_std)
        d = nets.gen["M"].sample(cond, valid, n_steps, generator=generator, **kw)
        return d * self.nrm.dM_std

    @torch.no_grad()
    def sample_h(self, nets, ctx, b, z_c, x_m, psi, oy, ox, keep, n_steps, generator=None):
        cond, valid, omega = self.cond_h(nets, ctx, b, z_c, x_m, psi, oy, ox, keep)
        kw = self.fine_kw(x_m[:, :1], omega, g.factor("H", "M"), self.nrm.dH_std)
        d = nets.gen["H"].sample(cond, valid, n_steps, generator=generator, **kw)
        return d * self.nrm.dH_std

    @staticmethod
    def coarse_m(z_c: Tensor, b: dict) -> Tensor:
        """P' X_C = (SIC, SIT, HS) brought to M on the whole tile (smooth
        prolongation, computed before any crop so that crops have their true
        neighbours)."""
        return g.prolong_s(z_c[:, :N_X], b["omega_m"][:, None], g.factor("M", "C"))

    def x_m(self, z_c: Tensor, d_m: Tensor, b: dict, oy, ox) -> Tensor:
        """(SIC, SIT) at M on the M crop: P' X_C + dX_M."""
        return crop(self.coarse_m(z_c, b)[:, :2], oy, ox, CROP_C["M"], g.factor("M", "C")) + d_m

    @staticmethod
    def coarse_h(x_m: Tensor, b: dict, oy, ox) -> Tensor:
        """P' X_M brought to H on the M crop (origin oy, ox) that x_m covers."""
        fh = g.factor("H", "M")
        w = crop(b["omega_h"], oy, ox, CROP_C["M"], g.factor("H", "C"))[:, None]
        return g.prolong_s(x_m, w, fh)

    @torch.no_grad()
    def sample_cascade(self, b: dict, keep: dict | None = None, n_steps: int | None = None, use_ema=True,
                       z_c: Tensor | None = None, psi: Tensor | None = None, generator=None) -> dict:
        """One member of the full cascade on the whole tile (fine levels on
        non-overlapping crops aligned on C cells). Given z_c / psi (e.g. mode
        A rollout), only the fine levels are generated."""
        nets = self.ema.module if use_ema else self.nets
        n_steps = n_steps or self.hparams.n_steps
        bsz, dev = b["omega_c"].shape[0], b["omega_c"].device
        keep = keep or self.keep_groups(bsz, dev, drop=False)
        ctx = self.encode(nets, b)
        psi = ctx["psi"] if psi is None else psi
        if z_c is None:
            z_c = self.sample_c(nets, ctx, b, keep, n_steps, generator)
        fm, fh = g.factor("M", "C"), g.factor("H", "M")
        d_m = torch.zeros(bsz, 2, N_C * fm, N_C * fm, device=dev)
        d_h = torch.zeros(bsz, 1, N_C * fm * fh, N_C * fm * fh, device=dev)
        for cy in range(0, N_C, CROP_C["M"]):
            for cx in range(0, N_C, CROP_C["M"]):
                o = torch.full((bsz,), cy, device=dev), torch.full((bsz,), cx, device=dev)
                sl = (slice(cy * fm, (cy + CROP_C["M"]) * fm), slice(cx * fm, (cx + CROP_C["M"]) * fm))
                d_m[..., sl[0], sl[1]] = self.sample_m(nets, ctx, b, z_c, psi, *o, keep, n_steps, generator)
        x_m = self.coarse_m(z_c, b)[:, :2] + d_m
        x_mh = g.prolong_s(x_m, b["omega_h"][:, None], fh)  # P' X_M on the whole tile
        for cy in range(0, N_C, CROP_C["H"]):
            for cx in range(0, N_C, CROP_C["H"]):
                o = torch.full((bsz,), cy, device=dev), torch.full((bsz,), cx, device=dev)
                n = CROP_C["H"]
                sl = (slice(cy * fm * fh, (cy + n) * fm * fh), slice(cx * fm * fh, (cx + n) * fm * fh))
                d_h[..., sl[0], sl[1]] = self.sample_h(nets, ctx, b, z_c, crop(x_mh, *o, n, fm * fh), psi, *o,
                                                       keep, n_steps, generator)
        x_h = x_mh[:, :1] + d_h
        return {"z_c": z_c, "psi": psi, "d_m": d_m, "d_h": d_h, "x_m": x_m, "x_h": x_h}

    # --------------------------------------------------------------- loss

    def losses(self, b: dict, drop: bool = True) -> dict:
        nrm, hp = self.nrm, self.hparams
        bsz, dev = b["omega_c"].shape[0], b["omega_c"].device
        keep = self.keep_groups(bsz, dev, drop)
        ctx = self.encode(self.nets, b)
        if "C" not in self.levels:
            # G_C frozen (phase 4): the fine levels must not train the C side of the encoder
            # (h_C, f), which conditions G_C; without pyramid h_M does not reach h_C. The psi head
            # (not used by G_C) is trained on the frozen h_C
            h_c = ctx["h_C"].detach()
            ctx = {**ctx, "h_C": h_c, "f": ctx["f"].detach(), "psi": self.nets.psi(h_c)}
        out = {}
        if "C" in self.levels:
            z1, m, floor = self.target_c(b)
            cond, valid = self.cond_c(ctx, b, keep)
            per_ch, has, _, _ = self.nets.gen["C"].loss(z1, m, cond, valid, floor, nrm.z_lo, nrm.z_hi)
            out["X_C"], out["S_C"] = channel_mean(per_ch[:N_X], has[:N_X]), channel_mean(per_ch[N_X:], has[N_X:])
        z_c = None
        if "M" in self.levels or "H" in self.levels:
            ema = self.ema.module
            with torch.no_grad():
                ctx_e = self.encode(ema, b)
                all_in = self.keep_groups(bsz, dev, drop=False)
                z_c = self.sample_c(ema, ctx_e, b, all_in, hp.cond_steps)
                z_c = nrm.z_inv(nrm.z(z_c) + hp.eta * torch.randn_like(z_c))
            oy, ox, hy, hx = self.choose_crops(b)
            fm, fh = g.factor("M", "C"), g.factor("H", "M")
            if "M" in self.levels:
                cond, valid, omega = self.cond_m(self.nets, ctx, b, z_c, ctx["psi"], oy, ox, keep)
                coarse = crop(self.coarse_m(z_c, b)[:, :2], oy, ox, CROP_C["M"], fm)
                kw = self.fine_kw(coarse, omega, fm, nrm.dM_std)
                if hp.corrector:  # correction of the prolonged sample: Y_M - P' X_C
                    z1 = (crop(b["tgt_M"], oy, ox, CROP_C["M"], fm) - coarse) / nrm.dM_std
                    m = crop(b["tgt_M_m"], oy, ox, CROP_C["M"], fm)
                else:
                    z1 = crop(b["tgt_dM"], oy, ox, CROP_C["M"], fm) / nrm.dM_std
                    m = crop(b["tgt_dM_m"], oy, ox, CROP_C["M"], fm)
                per_ch, has, _, _ = self.nets.gen["M"].loss(torch.where(m, z1, 0.0), m, cond, valid, **kw)
                out["X_M"] = channel_mean(per_ch, has)
            if "H" in self.levels:
                with torch.no_grad():
                    d_m = self.sample_m(ema, ctx_e, b, z_c, ctx_e["psi"], oy, ox, all_in, hp.cond_steps)
                    x_m = self.x_m(z_c, d_m, b, oy, ox)
                    x_m = x_m + hp.eta * torch.randn_like(x_m) * nrm.dM_std
                n = CROP_C["H"]
                x_mh = crop(self.coarse_h(x_m, b, oy, ox), hy - oy, hx - ox, n, fm * fh)
                cond, valid, omega = self.cond_h(self.nets, ctx, b, z_c, x_mh, ctx["psi"], hy, hx, keep)
                kw = self.fine_kw(x_mh[:, :1], omega, fh, nrm.dH_std)
                if hp.corrector:  # Y_H - P' X_M
                    z1 = (crop(b["tgt_H"], hy, hx, n, fm * fh) - x_mh[:, :1]) / nrm.dH_std
                    m = crop(b["tgt_H_m"], hy, hx, n, fm * fh)
                else:
                    z1 = crop(b["tgt_dH"], hy, hx, n, fm * fh) / nrm.dH_std
                    m = crop(b["tgt_dH_m"], hy, hx, n, fm * fh)
                per_ch, has, z_pred, t = self.nets.gen["H"].loss(torch.where(m, z1, 0.0), m, cond, valid, **kw)
                out["X_H"] = channel_mean(per_ch, has)
                # L_cyc: soft S_C of the one-step fine prediction vs the S_C it was conditioned on
                x_h = x_mh[:, :1] + z_pred * nrm.dH_std
                s_pred = obs.sic_stats(x_h[:, 0], omega, hp.cyc_tau)
                s_cond = crop(z_c, hy, hx, n, 1)[:, N_X:N_X + 8]
                ok = (crop(b["omega_c"], hy, hx, n, 1) > 0.5)[:, None]
                err = ((s_pred - s_cond) / nrm.z_std[:, N_X:N_X + 8]) ** 2 * t.view(-1, 1, 1, 1)
                out["cyc"] = (err * ok).sum() / (ok.sum() * 8).clamp_min(1)
        psi = ctx["psi"]
        l_psi = psi.pow(2).mean()
        if z_c is not None:
            p = psi.permute(0, 2, 3, 1).reshape(-1, psi.shape[1])
            s = nrm.z(z_c)[:, N_X:].permute(0, 2, 3, 1).reshape(-1, N_S)
            cov = (p - p.mean(0)).T @ (s - s.mean(0)) / len(p)
            out["psi_cov"] = cov.pow(2).sum()  # ||Cov(psi, S)||_F^2, logged to check the decorrelation
            l_psi = l_psi + hp.lambda_perp * out["psi_cov"]
        out["psi"] = l_psi
        total = sum(out.get(k, 0) for k in ("X_C", "X_M", "X_H")) + hp.lambda_s * out.get("S_C", 0) \
            + hp.lambda_c * out.get("cyc", 0) + hp.lambda_psi * out["psi"]
        out["loss"] = total
        return out

    def training_step(self, batch, batch_idx):
        out = self.losses(batch)
        self.log_dict({f"train_{k}": v for k, v in out.items()}, prog_bar=True, batch_size=batch["omega_c"].shape[0])
        return out["loss"]

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.ema.update_parameters(self.nets)

    # ------------------------------------------------------------- scores

    def _accumulate(self, key: str, ens: Tensor, tgt: Tensor, m: Tensor, names) -> dict:
        """RMSE of the ensemble mean, fair CRPS, spread per channel on the
        targets; ranks of the targets accumulated in self._ranks[key]."""
        n = ens.shape[1]
        mean = ens.mean(1)
        crps = (ens - tgt[:, None]).abs().mean(1) - (ens[:, :, None] - ens[:, None]).abs().sum((1, 2)) / (2 * n * (n - 1))
        # ranks on the members clipped to the physical bounds (the recomposed M / H
        # fields are not bounded), ties broken at random: targets at a bound
        # (SIC 0 or 1, edge_density 0, ...) otherwise all fall in an end bin
        lo = torch.tensor([Z_BOUNDS[k][0] for k in names], dtype=ens.dtype, device=ens.device)
        hi = torch.tensor([1e30 if Z_BOUNDS[k][1] is None else Z_BOUNDS[k][1] for k in names],
                          dtype=ens.dtype, device=ens.device)
        shape = (1, 1, -1) + (1,) * (ens.ndim - 3)
        ens_b = torch.maximum(torch.minimum(ens, hi.view(shape)), lo.view(shape))
        below, equal = (ens_b < tgt[:, None]).sum(1), (ens_b == tgt[:, None]).sum(1)
        rank = below + (torch.rand(below.shape, device=ens.device) * (equal + 1)).long()
        ranks = self._ranks.setdefault(key, torch.zeros(len(names), n + 1, device=ens.device))
        out = {}
        for c, name in enumerate(names):
            mc = m[:, c]
            if not mc.any():
                continue
            out[f"rmse_{name}"] = ((mean[:, c] - tgt[:, c])[mc] ** 2).mean().sqrt()
            out[f"crps_{name}"] = crps[:, c][mc].mean()
            out[f"spread_{name}"] = ens[:, :, c].std(1)[mc].mean()
            ranks[c] += torch.bincount(rank[:, c][mc], minlength=n + 1).to(ranks)
        return out

    @torch.no_grad()
    def scores(self, b: dict) -> dict:
        """Ensemble scores on Y^tgt: Z_C on the tile; X_M = (SIC, SIT) and
        X_H = SIC on one crop per sample (absolute fields, P' X_C + dX)."""
        nets, hp = self.ema.module, self.hparams
        bsz, dev = b["omega_c"].shape[0], b["omega_c"].device
        keep = self.keep_groups(bsz, dev, drop=False)
        ctx = self.encode(nets, b)
        oy, ox, hy, hx = self.choose_crops(b)
        fm, fh = g.factor("M", "C"), g.factor("H", "M")
        zs, xms, xhs = [], [], []
        for _ in range(hp.n_members):
            z_c = self.sample_c(nets, ctx, b, keep, hp.n_steps)
            zs.append(z_c)
            if "M" in self.levels or "H" in self.levels:
                x_m = self.x_m(z_c, self.sample_m(nets, ctx, b, z_c, ctx["psi"], oy, ox, keep, hp.n_steps),
                               b, oy, ox)
                xms.append(x_m)
                x_mh = crop(self.coarse_h(x_m, b, oy, ox), hy - oy, hx - ox, CROP_C["H"], fm * fh)
                d_h = self.sample_h(nets, ctx, b, z_c, x_mh, ctx["psi"], hy, hx, keep, hp.n_steps)
                xhs.append(x_mh[:, :1] + d_h)
        out = {f"C_{k}": v for k, v in self._accumulate(
            "C", torch.stack(zs, 1), torch.cat([b["tgt_C"], b["tgt_S"]], 1), strict_mask(b), Z_NAMES).items()}
        if xms:
            out.update({f"M_{k}": v for k, v in self._accumulate(
                "M", torch.stack(xms, 1), crop(b["tgt_M"], oy, ox, CROP_C["M"], fm),
                crop(b["tgt_M_m"], oy, ox, CROP_C["M"], fm), ("SIC", "SIT")).items()})
            out.update({f"H_{k}": v for k, v in self._accumulate(
                "H", torch.stack(xhs, 1), crop(b["tgt_H"], hy, hx, CROP_C["H"], fm * fh),
                crop(b["tgt_H_m"], hy, hx, CROP_C["H"], fm * fh), ("SIC",)).items()})
        return out

    def _eval_step(self, batch, batch_idx, phase):
        bsz = batch["omega_c"].shape[0]
        out = self.losses(batch, drop=False)
        self.log_dict({f"{phase}_{k}": v for k, v in out.items()}, batch_size=bsz, sync_dist=True)
        if phase == "test" or (self.current_epoch % self.hparams.score_every == 0
                               and batch_idx < self.hparams.n_score_batches):
            self.log_dict({f"{phase}_{k}": v for k, v in self.scores(batch).items()}, batch_size=bsz, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        self._eval_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        self._eval_step(batch, batch_idx, "test")

    def _save_ranks(self, phase):
        """Rank histograms per level, {level: [n_channels, n_members + 1]}."""
        if self._ranks and self.logger is not None and self.logger.log_dir is not None and self.global_rank == 0:
            os.makedirs(self.logger.log_dir, exist_ok=True)
            torch.save({k: v.cpu() for k, v in self._ranks.items()},
                       os.path.join(self.logger.log_dir, f"rank_hist_{phase}_epoch{self.current_epoch:03d}.pt"))
        self._ranks = {}

    def on_validation_epoch_end(self):
        self._save_ranks("val")

    def on_test_epoch_end(self):
        self._save_ranks("test")

    def configure_optimizers(self):
        params = [p for p in self.nets.parameters() if p.requires_grad]
        opt = torch.optim.Adam(params, lr=self.hparams.lr, betas=tuple(self.hparams.betas))
        sched = torch.optim.lr_scheduler.LinearLR(opt, start_factor=self.hparams.lr_scheduler_start_factor,
                                                  total_iters=self.hparams.lr_scheduler_iters)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step", "frequency": 1}}
