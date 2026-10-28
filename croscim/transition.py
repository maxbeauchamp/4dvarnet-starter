"""
transition.py
=============
N1, the transition network of the CROSCIM cascade (formulation section 5;
brief phase 6), trained for mode A: only G_C^1 advances
Z_C = (X_C, S_C, psi_C) by Delta = 1 day,

    Z_C^{t+1} ~ p(Z_C^{t+1} | Z_C^t, mu Y_(t, t+1], F_{t:t+1})

Initialised from N0: the observation encoders are copied from the N0 EMA
(frozen at first), G_C^1 starts from the G_C^0 weights wherever the shapes
match. The previous state enters through a state encoder E^s with three
groups (X, S, psi) that can each be dropped (diagnostics I^up / I^down).

Training: the input state at t is a sample of N0 (no dense truth exists);
obs availability mu ~ Bernoulli(p_obs); pushforward over K steps (the input of
step k is a sample of step k-1, without gradient), with a curriculum
K = 1 .. k_max; targets at t+k are X^, S^ (Y^tgt of day t+k) and
psi^ = Psi_bar(h_C^{t+k}) from the N0 EMA encoder, where observed:

    L = sum_k gamma^{k-1} [L_X + lambda_S L_S + lambda_psi L_psi^dyn]_{t+k}

Options (v6, N1b; all False for the former checkpoints):
  increment    G_C^1 generates the increment Z^{t+1} - Z^t (normalised; psi
               included), censored so that Z^{t+1} stays within the physical
               bounds: doing nothing is persistence (the v6 N1 regressed to a
               mean state from the first step, worse than persistence);
  state_input  the normalised previous state (and psi) enters as such, besides
               its encoding E^s (same group dropout);
  raw_obs      the gridded CIMR (SIC, SIT, masks) of the day enters
               uncompressed, as raw_days for G_C^0 (the encoder compresses it:
               the v6 N1 barely used the observations, w = 1 ~ w = 0).
  bg_sigma     background term (N1c): where a channel has no target at t+k, its
               target is its own previous state, with the scale floor bg_sigma
               (normalised units), as the background term of 4D-Var. Without it
               S_C (targets only where the held-out ASIP passes cover) was
               resampled at random each step, and accumulated as a random walk
               with the increment (the N1b rollouts became noise).
  analysis_steps  Heun steps of the initial N0 analysis at training (default:
               push_steps; inference uses n_steps: noisier training states).
  weak_targets False (N1d): no weak CIMR targets (SIC, SIT) in N1, the
               background term takes over where ASIP / CRISTAL do not observe;
               the open-water zeros stay. CIMR is lower than ASIP: as a target
               everywhere it pulled the free forecast towards CIMR, day after
               day (the N1c tile lost ice at every step).
  inc_scale    (N1f) the increment is generated in units of inc_scale x the
               state scale: with the state normalisation (SIC: identity) the
               flow started from a noise ~10x the daily increments, and the
               residual, cut one-sided at the bounds (SIC 0 / 1), pulled pack
               ice down and open water up at every step (the smoothing).
  bg_advect    (N1e) the increment (and the background) are relative to the
               previous state moved along the free drift of the day,
               alpha R(-theta) u10 (ERA5 along the grid axes), alpha and theta
               learned end to end; the moved state also enters the conditioning.
"""

from __future__ import annotations

import copy

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch import Tensor

import math

from . import grid as g
from .cascade import N_S, N_X, N_Z, LitCascadeN0, _warp, drift_displacement, make_ema, strict_mask, tile_pos
from . import data as D
from .data import N_C
from .encoders import DenseEncoder
from .generators import Generator, channel_mean

STATE_GROUPS = ("x", "s", "psi")


def only_last_day(b: dict) -> dict:
    """Context of one transition step: observations of the last day only,
    forcing of the last two days (F_{t:t+1})."""
    out = dict(b)
    out.pop("_adv", None)  # N0's moved observations: recomputed (day t only: not moved)
    for k in ("in_C_cimr_w", "in_M_asip_w", "in_H_w"):
        w = b[k].clone()
        w[:, :-1] = 0
        out[k] = w
    out["pts_valid"] = b["pts_valid"] & (b["pts"][..., 4] == 0)
    f = b["f_C"].clone()
    f[:, :-2] = float("nan")
    out["f_C"] = f
    return out


def load_matching(dst: nn.Module, src: nn.Module) -> int:
    """Copy the tensors of src into dst wherever names and shapes match."""
    sd, own = src.state_dict(), dst.state_dict()
    match = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
    dst.load_state_dict(match, strict=False)
    return len(match)


def load_n0(path: str) -> LitCascadeN0:
    """N0 from one of our own checkpoints (hparams + state_dict); explicit
    weights_only=False since the hparams hold Hydra configs (torch >= 2.6
    refuses them by default)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    n0 = LitCascadeN0(**ckpt["hyper_parameters"])
    n0.load_state_dict(ckpt["state_dict"])
    return n0


def load_n1(path: str, n0_ckpt: str | None = None) -> "LitCascadeN1":
    """N1 from one of our own checkpoints. The N0 it was built from is rebuilt
    from its checkpoint (hparam n0_ckpt, or `n0_ckpt` if that file moved),
    then every weight, N0 included, is restored from the N1 checkpoint."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    hp = dict(ckpt["hyper_parameters"])
    if n0_ckpt is not None:
        hp["n0_ckpt"] = n0_ckpt
    n1 = LitCascadeN1(**hp)
    n1.load_state_dict(ckpt["state_dict"])
    return n1


class N1Nets(nn.Module):

    def __init__(self, n0: LitCascadeN0, swin: dict, n_extra: int = 0):
        super().__init__()
        d, d_psi = n0.hparams.d, n0.hparams.d_psi
        self.enc = copy.deepcopy(n0.ema.module)
        for name in ("gen", "down_m", "down_h"):
            delattr(self.enc, name)
        for p in self.enc.parameters():
            p.requires_grad_(True)
        self.e_state = DenseEncoder(N_Z + d_psi + len(STATE_GROUPS), d)
        self.gen = Generator(N_Z + d_psi, 3 * d + 5 + n_extra, (N_C, N_C), 32., swin)
        self.n_loaded = load_matching(self.gen, n0.ema.module.gen["C"])


class LitCascadeN1(pl.LightningModule):

    def __init__(self, n0_ckpt: str | None = None, n0: LitCascadeN0 | None = None, swin: dict | None = None,
                 k_max=7, curriculum_epochs=10, p_obs=0.7, p_null=0.1, gamma=0.9, lambda_s=1.0, lambda_psi=1.0,
                 freeze_encoders=True, push_steps=4, n_steps=20, lr=1e-4, betas=(0.9, 0.995),
                 lr_scheduler_start_factor=1e-5, lr_scheduler_iters=20_000, ema_decay=0.999, n_members=8,
                 score_every=5, n_score_batches=10, increment=False, state_input=False, raw_obs=False,
                 bg_sigma=None, analysis_steps=None, weak_targets=True, inc_scale=None, bg_advect=False,
                 drift_init=(0.015, 15.0)):
        super().__init__()
        self.save_hyperparameters(ignore=["n0"])
        n0 = load_n0(n0_ckpt) if n0 is None else n0
        n0.eval()
        for p in n0.parameters():
            p.requires_grad_(False)
        self.n0, self.nrm = n0, n0.nrm
        self.d_psi = n0.hparams.d_psi
        n_extra = (N_Z + self.d_psi if state_input else 0) + (4 if raw_obs else 0) + (N_Z + self.d_psi if bg_advect else 0)
        self.nets = N1Nets(n0, swin if swin is not None else n0.hparams.swin["C"], n_extra)
        if freeze_encoders:
            for p in self.nets.enc.parameters():
                p.requires_grad_(False)
        self.ema = make_ema(self.nets, ema_decay)
        lo = torch.cat([self.nrm.z_lo, torch.full((1, self.d_psi, 1, 1), -float("inf"))], 1)
        hi = torch.cat([self.nrm.z_hi, torch.full((1, self.d_psi, 1, 1), float("inf"))], 1)
        self.register_buffer("lo", lo, persistent=False)
        self.register_buffer("hi", hi, persistent=False)
        if bg_advect:  # free drift: log alpha, theta (rad, to the right)
            self.drift = nn.Parameter(torch.tensor([math.log(drift_init[0]), math.radians(drift_init[1])]))

    def train(self, mode: bool = True):
        """The frozen N0 and the EMA copy always stay in eval mode."""
        super().train(mode)
        self.n0.eval()
        self.ema.eval()
        return self

    # ----------------------------------------------------------- one step

    def k_now(self, n_days: int) -> int:
        return min(self.hparams.k_max, 1 + self.current_epoch // self.hparams.curriculum_epochs, n_days - 1)

    def keep(self, bsz: int, device, train: bool) -> dict:
        p = self.hparams.p_null if train else 0.0
        keep = {k: torch.rand(bsz, device=device) >= p for k in ("h", "f", *STATE_GROUPS)}
        if train:
            keep["h"] &= torch.rand(bsz, device=device) < self.hparams.p_obs  # mu ~ Bernoulli(p_obs)
        return keep

    def cond(self, nets: N1Nets, b: dict, z: Tensor, psi: Tensor, keep: dict, base: Tensor | None = None):
        """c_C^1 = (h_C of the day, f, E^s(Z^t)); z physical, psi latent."""
        ctx = self.n0.encode(nets.enc, only_last_day(b))
        zn = self.nrm.z(z)
        k = {n: keep[n].to(zn).view(-1, 1, 1, 1) for n in keep}
        flags = [k[n].expand(-1, 1, *zn.shape[-2:]) for n in STATE_GROUPS]
        x_s = torch.cat([zn[:, :N_X] * k["x"], zn[:, N_X:] * k["s"], psi * k["psi"], *flags], 1)
        e_s = nets.e_state(x_s, torch.ones_like(b["omega_c"], dtype=torch.bool))
        parts = [ctx["h_C"] * k["h"], ctx["f"] * k["f"], e_s, k["h"].expand(-1, 1, *zn.shape[-2:]),
                 k["f"].expand(-1, 1, *zn.shape[-2:]), b["omega_c"][:, None], tile_pos(b)]
        if self.hparams.state_input:
            parts.append(x_s[:, :N_Z + self.d_psi])
        if self.hparams.raw_obs:
            m = b["in_C_cimr_w"][:, -1] > 0
            raw = torch.cat([torch.where(m, self.nrm(b["in_C_cimr"][:, -1], ("SIC", "SIT"), 1), 0.0), m.to(zn)], 1)
            parts.append(raw * k["h"])
        if self.hparams.bg_advect:  # the moved state, same group dropout as the state
            parts.append(torch.cat([base[:, :N_X] * k["x"], base[:, N_X:N_Z] * k["s"], base[:, N_Z:] * k["psi"]], 1))
        cond = torch.cat(parts, 1)
        return cond, (b["omega_c"] > 0)[:, None].to(cond)

    def state_n(self, z: Tensor, psi: Tensor) -> Tensor:
        """Normalised (Z, psi)."""
        return torch.cat([self.nrm.z(z), psi], 1)

    def base(self, b: dict, z: Tensor, psi: Tensor) -> Tensor:
        """The state the increment is added to (and the background): the normalised
        previous state, moved along the free drift of the day with bg_advect
        (cells whose departure point is not ocean of the tile keep their value)."""
        s0 = self.state_n(z, psi)
        if not self.hparams.bg_advect:
            return s0
        disp = drift_displacement(b["f_C"][:, -2:], self.drift[0].exp(), self.drift[1])[:, :1]  # t+k-1 -> t+k
        w = (b["omega_c"] > 0)[:, None, None].expand(-1, 1, s0.shape[1], -1, -1).to(s0)
        val, den = _warp(s0[:, None], w, disp, g.DX_M["C"])
        return torch.where(den[:, 0] > 0.5, val[:, 0], s0)

    @torch.no_grad()
    def target(self, b: dict) -> tuple[Tensor, Tensor, Tensor]:
        """(X^, S^, psi^) normalised, mask and scale floor; psi^ from the N0 EMA
        encoder on the full window of the day, where it saw observations."""
        z, m, floor = self.n0.target_c(b)
        if not self.hparams.weak_targets:  # drop the weak CIMR targets (SIC, SIT); keep the open-water zeros
            sg = b["tgt_C_sigma"]
            weak = torch.stack([sg[:, 0] == D.CIMR_TGT_SIGMA, sg[:, 1] == D.CIMR_SIT_TGT_SIGMA, torch.zeros_like(sg[:, 2], dtype=torch.bool)], 1)
            m = torch.cat([m[:, :N_X] & ~weak, m[:, N_X:]], 1)
        ctx = self.n0.encode(self.n0.ema.module, b)
        m_psi = ctx["m_C"][:, None].expand(-1, self.d_psi, -1, -1)
        return (torch.cat([z, ctx["psi"]], 1), torch.cat([m, m_psi], 1),
                torch.cat([floor, torch.zeros_like(ctx["psi"])], 1))

    def step_loss(self, b: dict, z: Tensor, psi: Tensor, keep: dict, noise=None, t=None) -> dict:
        hp = self.hparams
        z1, m, floor = self.target(b)
        base = self.base(b, z, psi)
        cond, valid = self.cond(self.nets, b, z, psi, keep, base)
        lo, hi = self.lo, self.hi
        if hp.bg_sigma:  # background: no target -> previous (moved) state, floor bg_sigma
            bg = (b["omega_c"] > 0)[:, None] & ~m
            z1 = torch.where(bg, base, z1)
            floor = torch.where(bg, torch.full_like(floor, hp.bg_sigma), floor)
            m = m | bg
        if hp.increment:  # in units of inc_scale x the state scale
            sc = hp.inc_scale or 1.0
            z1, lo, hi = torch.where(m, (z1 - base) / sc, 0.0), (lo - base) / sc, (hi - base) / sc
            floor = floor / sc
        per_ch, has, _, _ = self.nets.gen.loss(z1, m, cond, valid, floor, lo, hi, noise=noise, t=t)
        out = {"X": channel_mean(per_ch[:N_X], has[:N_X]), "S": channel_mean(per_ch[N_X:N_Z], has[N_X:N_Z]),
               "psi": channel_mean(per_ch[N_Z:], has[N_Z:])}
        out["loss"] = out["X"] + self.hparams.lambda_s * out["S"] + self.hparams.lambda_psi * out["psi"]
        return out

    @torch.no_grad()
    def step(self, nets: N1Nets, b: dict, z: Tensor, psi: Tensor, keep: dict, n_steps: int,
             w: float = 1.0, generator=None) -> tuple[Tensor, Tensor]:
        """One transition step (physical z, latent psi), with guidance weight w
        on the observations (w = 0: free simulation, w = 1: assimilation)."""
        base = self.base(b, z, psi)
        cond, valid = self.cond(nets, b, z, psi, keep, base)
        cond_null = self.cond(nets, b, z, psi, {**keep, "h": torch.zeros_like(keep["h"])}, base)[0] if w != 1.0 else None
        lo, hi = self.lo, self.hi
        sc = self.hparams.inc_scale or 1.0
        if self.hparams.increment:
            lo, hi = (lo - base) / sc, (hi - base) / sc
        zn = nets.gen.sample(cond, valid, n_steps, lo=lo, hi=hi, cond_null=cond_null, w=w, generator=generator)
        if self.hparams.increment:
            zn = base + zn * sc
        return self.nrm.z_inv(zn[:, :N_Z]), zn[:, N_Z:]

    @torch.no_grad()
    def analysis(self, b0: dict, n_steps: int, generator=None) -> tuple[Tensor, Tensor]:
        """Initial state at t: a sample of N0 (EMA) and psi = Psi_bar(h_C^t)."""
        n0 = self.n0
        ctx = n0.encode(n0.ema.module, b0)
        keep = n0.keep_groups(b0["omega_c"].shape[0], b0["omega_c"].device, drop=False)
        return n0.sample_c(n0.ema.module, ctx, b0, keep, n_steps, generator), ctx["psi"]

    # --------------------------------------------------------------- train

    def losses(self, days: list[dict], train: bool = True) -> dict:
        hp = self.hparams
        bsz, dev = days[0]["omega_c"].shape[0], days[0]["omega_c"].device
        z, psi = self.analysis(days[0], hp.analysis_steps or hp.push_steps)
        total, out = 0.0, {}
        for k in range(1, self.k_now(len(days)) + 1):
            keep = self.keep(bsz, dev, train)
            lk = self.step_loss(days[k], z, psi, keep)
            total = total + hp.gamma ** (k - 1) * lk["loss"]
            if k == 1:  # comparable across epochs (the total grows with the curriculum K): checkpoint selection
                out["loss_k1"] = lk["loss"]
            out.update({f"{n}_k{k}": v for n, v in lk.items() if n != "loss"})
            if k < self.k_now(len(days)):
                z, psi = self.step(self.nets, days[k], z, psi, keep, hp.push_steps)
        out["loss"] = total
        if hp.bg_advect:
            out["drift_alpha"], out["drift_theta_deg"] = self.drift[0].exp().detach(), self.drift[1].detach() * 180 / math.pi
        return out

    def training_step(self, batch, batch_idx):
        out = self.losses(batch)
        self.log_dict({f"train_{k}": v for k, v in out.items()}, batch_size=batch[0]["omega_c"].shape[0])
        self.log("k_now", float(self.k_now(len(batch))))
        return out["loss"]

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.ema.update_parameters(self.nets)

    @torch.no_grad()
    def scores(self, days: list[dict]) -> dict:
        """Ensemble assimilation rollout (w = 1) from the N0 analysis: RMSE and
        fair CRPS of X_C on Y^tgt per lead."""
        from .rollout import rollout
        hp, out = self.hparams, {}
        ens = torch.stack([rollout(self, days, len(days) - 1, w=1.0, seed=s)[0] for s in range(hp.n_members)], 1)
        n = hp.n_members
        for k in range(1, len(days)):
            tgt, m = days[k]["tgt_C"], strict_mask(days[k])[:, :N_X]
            e = ens[:, :, k, :N_X]
            crps = (e - tgt[:, None]).abs().mean(1) - (e[:, :, None] - e[:, None]).abs().sum((1, 2)) / (2 * n * (n - 1))
            for c, name in enumerate(("SIC", "SIT", "HS")):
                if m[:, c].any():
                    out[f"rmse_{name}_k{k}"] = ((e.mean(1)[:, c] - tgt[:, c])[m[:, c]] ** 2).mean().sqrt()
                    out[f"crps_{name}_k{k}"] = crps[:, c][m[:, c]].mean()
        return out

    def _eval_step(self, batch, batch_idx, phase):
        bsz = batch[0]["omega_c"].shape[0]
        out = self.losses(batch, train=False)
        self.log_dict({f"{phase}_{k}": v for k, v in out.items()}, batch_size=bsz, sync_dist=True)
        if phase == "test" or (self.current_epoch % self.hparams.score_every == 0
                               and batch_idx < self.hparams.n_score_batches):
            self.log_dict({f"{phase}_{k}": v for k, v in self.scores(batch).items()}, batch_size=bsz, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        self._eval_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        self._eval_step(batch, batch_idx, "test")

    def configure_optimizers(self):
        params = [p for p in self.nets.parameters() if p.requires_grad]
        if self.hparams.bg_advect:
            params.append(self.drift)
        opt = torch.optim.Adam(params, lr=self.hparams.lr, betas=tuple(self.hparams.betas))
        sched = torch.optim.lr_scheduler.LinearLR(opt, start_factor=self.hparams.lr_scheduler_start_factor,
                                                  total_iters=self.hparams.lr_scheduler_iters)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step", "frequency": 1}}
