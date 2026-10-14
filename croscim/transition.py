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
"""

from __future__ import annotations

import copy

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch import Tensor

from .cascade import N_S, N_X, N_Z, LitCascadeN0, make_ema, tile_pos
from .data import N_C
from .encoders import DenseEncoder
from .generators import Generator, channel_mean

STATE_GROUPS = ("x", "s", "psi")


def only_last_day(b: dict) -> dict:
    """Context of one transition step: observations of the last day only,
    forcing of the last two days (F_{t:t+1})."""
    out = dict(b)
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

    def __init__(self, n0: LitCascadeN0, swin: dict):
        super().__init__()
        d, d_psi = n0.hparams.d, n0.hparams.d_psi
        self.enc = copy.deepcopy(n0.ema.module)
        for name in ("gen", "down_m", "down_h"):
            delattr(self.enc, name)
        for p in self.enc.parameters():
            p.requires_grad_(True)
        self.e_state = DenseEncoder(N_Z + d_psi + len(STATE_GROUPS), d)
        self.gen = Generator(N_Z + d_psi, 3 * d + 5, (N_C, N_C), 32., swin)
        self.n_loaded = load_matching(self.gen, n0.ema.module.gen["C"])


class LitCascadeN1(pl.LightningModule):

    def __init__(self, n0_ckpt: str | None = None, n0: LitCascadeN0 | None = None, swin: dict | None = None,
                 k_max=7, curriculum_epochs=10, p_obs=0.7, p_null=0.1, gamma=0.9, lambda_s=1.0, lambda_psi=1.0,
                 freeze_encoders=True, push_steps=4, n_steps=20, lr=1e-4, betas=(0.9, 0.995),
                 lr_scheduler_start_factor=1e-5, lr_scheduler_iters=20_000, ema_decay=0.999, n_members=8,
                 score_every=5, n_score_batches=10):
        super().__init__()
        self.save_hyperparameters(ignore=["n0"])
        n0 = load_n0(n0_ckpt) if n0 is None else n0
        n0.eval()
        for p in n0.parameters():
            p.requires_grad_(False)
        self.n0, self.nrm = n0, n0.nrm
        self.d_psi = n0.hparams.d_psi
        self.nets = N1Nets(n0, swin if swin is not None else n0.hparams.swin["C"])
        if freeze_encoders:
            for p in self.nets.enc.parameters():
                p.requires_grad_(False)
        self.ema = make_ema(self.nets, ema_decay)
        lo = torch.cat([self.nrm.z_lo, torch.full((1, self.d_psi, 1, 1), -float("inf"))], 1)
        hi = torch.cat([self.nrm.z_hi, torch.full((1, self.d_psi, 1, 1), float("inf"))], 1)
        self.register_buffer("lo", lo, persistent=False)
        self.register_buffer("hi", hi, persistent=False)

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

    def cond(self, nets: N1Nets, b: dict, z: Tensor, psi: Tensor, keep: dict) -> tuple[Tensor, Tensor]:
        """c_C^1 = (h_C of the day, f, E^s(Z^t)); z physical, psi latent."""
        ctx = self.n0.encode(nets.enc, only_last_day(b))
        zn = self.nrm.z(z)
        k = {n: keep[n].to(zn).view(-1, 1, 1, 1) for n in keep}
        flags = [k[n].expand(-1, 1, *zn.shape[-2:]) for n in STATE_GROUPS]
        x_s = torch.cat([zn[:, :N_X] * k["x"], zn[:, N_X:] * k["s"], psi * k["psi"], *flags], 1)
        e_s = nets.e_state(x_s, torch.ones_like(b["omega_c"], dtype=torch.bool))
        cond = torch.cat([ctx["h_C"] * k["h"], ctx["f"] * k["f"], e_s, k["h"].expand(-1, 1, *zn.shape[-2:]),
                          k["f"].expand(-1, 1, *zn.shape[-2:]), b["omega_c"][:, None], tile_pos(b)], 1)
        return cond, (b["omega_c"] > 0)[:, None].to(cond)

    @torch.no_grad()
    def target(self, b: dict) -> tuple[Tensor, Tensor, Tensor]:
        """(X^, S^, psi^) normalised, mask and scale floor; psi^ from the N0 EMA
        encoder on the full window of the day, where it saw observations."""
        z, m, floor = self.n0.target_c(b)
        ctx = self.n0.encode(self.n0.ema.module, b)
        m_psi = ctx["m_C"][:, None].expand(-1, self.d_psi, -1, -1)
        return (torch.cat([z, ctx["psi"]], 1), torch.cat([m, m_psi], 1),
                torch.cat([floor, torch.zeros_like(ctx["psi"])], 1))

    def step_loss(self, b: dict, z: Tensor, psi: Tensor, keep: dict, noise=None, t=None) -> dict:
        z1, m, floor = self.target(b)
        cond, valid = self.cond(self.nets, b, z, psi, keep)
        per_ch, has, _, _ = self.nets.gen.loss(z1, m, cond, valid, floor, self.lo, self.hi, noise=noise, t=t)
        out = {"X": channel_mean(per_ch[:N_X], has[:N_X]), "S": channel_mean(per_ch[N_X:N_Z], has[N_X:N_Z]),
               "psi": channel_mean(per_ch[N_Z:], has[N_Z:])}
        out["loss"] = out["X"] + self.hparams.lambda_s * out["S"] + self.hparams.lambda_psi * out["psi"]
        return out

    @torch.no_grad()
    def step(self, nets: N1Nets, b: dict, z: Tensor, psi: Tensor, keep: dict, n_steps: int,
             w: float = 1.0, generator=None) -> tuple[Tensor, Tensor]:
        """One transition step (physical z, latent psi), with guidance weight w
        on the observations (w = 0: free simulation, w = 1: assimilation)."""
        cond, valid = self.cond(nets, b, z, psi, keep)
        cond_null = self.cond(nets, b, z, psi, {**keep, "h": torch.zeros_like(keep["h"])})[0] if w != 1.0 else None
        zn = nets.gen.sample(cond, valid, n_steps, lo=self.lo, hi=self.hi, cond_null=cond_null, w=w,
                             generator=generator)
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
        z, psi = self.analysis(days[0], hp.push_steps)
        total, out = 0.0, {}
        for k in range(1, self.k_now(len(days)) + 1):
            keep = self.keep(bsz, dev, train)
            lk = self.step_loss(days[k], z, psi, keep)
            total = total + hp.gamma ** (k - 1) * lk["loss"]
            out.update({f"{n}_k{k}": v for n, v in lk.items() if n != "loss"})
            if k < self.k_now(len(days)):
                z, psi = self.step(self.nets, days[k], z, psi, keep, hp.push_steps)
        out["loss"] = total
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
            tgt, m = days[k]["tgt_C"], days[k]["tgt_C_m"]
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
        opt = torch.optim.Adam(params, lr=self.hparams.lr, betas=tuple(self.hparams.betas))
        sched = torch.optim.lr_scheduler.LinearLR(opt, start_factor=self.hparams.lr_scheduler_start_factor,
                                                  total_iters=self.hparams.lr_scheduler_iters)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step", "frequency": 1}}
