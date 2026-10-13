"""
rollout.py
==========
Scenarios and diagnostics of the CROSCIM cascade (formulation section 7;
brief phase 7).

Rollouts (`days`: list of per-day batches t, t+1, ..., from a datamodule
with horizon >= n_steps):
    mode A  only G_C^1 advances Z_C = (X_C, S_C, psi_C); fine levels on demand
            with `downscale` (G_M^0, G_H^0, h = null)
    mode B  full cascade at every step, then S_C <- S_C(X_H)
Each step k uses its own noise generator seeded from (seed, k): two rollouts
with the same seed share their noise (paired noise).

Interventions are callables (k, z) -> z applied to the physical Z_C after
step k (k = 0: initial condition), always followed by the projection Pi_S.

Diagnostics: I^up / I^down (loss gaps when a group is replaced by null),
response function R_{X<-S}(tau) with paired noise, controllability slope
dS_C(X_H) / dS*, comparison of modes A and B over the SAR coverage.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Union

import torch
from torch import Tensor

from . import grid as g
from . import obs
from .cascade import N_X, N_Z, Z_BOUNDS, Z_NAMES

Intervention = Callable[[int, Tensor], Tensor]
H_OW, H_PACK = Z_NAMES.index("h_ow"), Z_NAMES.index("h_pack")


def _gen(seed: int, k: int, device) -> torch.Generator:
    return torch.Generator(device).manual_seed(seed * 10_007 + k)


def project_s(z: Tensor) -> Tensor:
    """Pi_S: every channel within its physical bounds, SIC histogram summing to
    1 (an empty histogram becomes the class containing SIC_C)."""
    z = z.clone()
    for c, name in enumerate(Z_NAMES):
        lo, hi = Z_BOUNDS[name]
        z[:, c] = z[:, c].clamp(min=lo, max=hi)
    h = z[:, H_OW:H_PACK + 1]
    tot = h.sum(1, keepdim=True)
    k = torch.bucketize(z[:, :1], torch.tensor(obs.SIC_BIN_EDGES, device=z.device), right=True)
    one_hot = (torch.arange(h.shape[1], device=z.device).view(1, -1, 1, 1) == k).to(h)
    z[:, H_OW:H_PACK + 1] = torch.where(tot > 1e-6, h / tot.clamp_min(1e-6), one_hot)
    return z


# ---------------------------------------------------------------- rollouts

@torch.no_grad()
def downscale(n0, b: dict, z_c: Tensor, psi: Tensor, seed: int = 0, with_obs: bool = False) -> dict:
    """Fine realisations of a given (X_C, S_C, psi_C): G_M^0 then G_H^0 on the
    whole tile, conditioned by Pi^down Phi and Pi^down f (h = null unless
    with_obs)."""
    bsz, dev = z_c.shape[0], z_c.device
    keep = n0.keep_groups(bsz, dev, drop=False)
    keep["h"] = torch.full_like(keep["h"], with_obs)
    return n0.sample_cascade(b, keep=keep, z_c=z_c, psi=psi, generator=_gen(seed, 10_006, dev))


def s_from_fine(fine: dict, b: dict) -> Tensor:
    """S_C(X_H) (8 SIC channels) and sit_std from X_M, on full fields."""
    s = obs.sic_stats(fine["x_h"][:, 0], b["omega_h"])
    sit_std, _ = obs.sit_std_stat(fine["x_m"][:, 1], b["omega_m"])
    return torch.cat([s, sit_std[:, None]], 1)


@torch.no_grad()
def rollout(n1, days: Sequence[dict], n_steps: int, w: Union[float, Callable[[int], float]] = 1.0, seed: int = 0,
            intervention: Optional[Intervention] = None, z0: Optional[Tensor] = None, psi0: Optional[Tensor] = None,
            mode: str = "A", fine_with_obs: bool = False, keep: Optional[dict] = None,
            sampler_steps: Optional[int] = None) -> tuple[Tensor, Tensor]:
    """Rollout of n_steps days from the N0 analysis of days[0] (or z0, psi0).
    Returns the trajectory [B, n_steps + 1, 12, 64, 64] (physical) and psi."""
    nets = n1.ema.module
    sampler_steps = sampler_steps or n1.hparams.n_steps
    dev = days[0]["omega_c"].device
    if z0 is None:
        z, psi = n1.analysis(days[0], sampler_steps, _gen(seed, 0, dev))
    else:
        z, psi = z0, psi0
    z = project_s(intervention(0, z) if intervention is not None else z)
    zs, psis = [z], [psi]
    for k in range(1, n_steps + 1):
        wk = w(k) if callable(w) else w
        kp = keep or n1.keep(z.shape[0], dev, train=False)
        z, psi = n1.step(nets, days[k], z, psi, kp, sampler_steps, w=wk, generator=_gen(seed, k, dev))
        if mode == "B":
            fine = downscale(n1.n0, days[k], z, psi, seed * 10_007 + k, with_obs=fine_with_obs)
            z = torch.cat([z[:, :N_X], s_from_fine(fine, days[k])], 1)
        if intervention is not None:
            z = intervention(k, z)
        z = project_s(z)
        zs.append(z)
        psis.append(psi)
    return torch.stack(zs, 1), torch.stack(psis, 1)


# ----------------------------------------------------------- interventions

def channels(names: Sequence[str]) -> list[int]:
    return [Z_NAMES.index(n) for n in names]


def perturb_ic(delta_s: Tensor, names: Sequence[str]) -> Intervention:
    """(1) S_C^0 <- S_C^0 + delta S (shape modification), then free."""
    idx = channels(names)

    def f(k, z):
        if k == 0:
            z = z.clone()
            z[:, idx] = z[:, idx] + delta_s
        return z
    return f


def prescribe(s_star: Callable[[int], Tensor], names: Sequence[str]) -> Intervention:
    """(3) do(S_C^t = s*(t)) after every step."""
    idx = channels(names)

    def f(k, z):
        z = z.clone()
        z[:, idx] = s_star(k)
        return z
    return f


def nudge(s_star: Callable[[int], Tensor], names: Sequence[str], dt_over_tau: float) -> Intervention:
    """(4) relaxation S_C <- S_C + dt / tau (s* - S_C) after every step."""
    idx = channels(names)

    def f(k, z):
        if k == 0:
            return z
        z = z.clone()
        z[:, idx] = z[:, idx] + dt_over_tau * (s_star(k) - z[:, idx])
        return z
    return f


def permute_s(seed: int = 0) -> Intervention:
    """Placebo: spatial permutation of S_C over the tile at the initial condition."""
    def f(k, z):
        if k != 0:
            return z
        z = z.clone()
        perm = torch.randperm(z.shape[-2] * z.shape[-1], generator=torch.Generator().manual_seed(seed)).to(z.device)
        s = z[:, N_X:N_Z].flatten(2)[..., perm]
        z[:, N_X:N_Z] = s.view_as(z[:, N_X:N_Z])
        return z
    return f


def modify_forcing(days: Sequence[dict], d_t2m: float = 0.0, rotate_deg: float = 0.0) -> list[dict]:
    """(2) forcing scenario: +d_t2m on T2m and winds rotated by rotate_deg."""
    out, a = [], torch.deg2rad(torch.tensor(rotate_deg))
    for b in days:
        f = b["f_C"].clone()
        u, v = f[:, :, 2].clone(), f[:, :, 3].clone()
        f[:, :, 0] += d_t2m
        f[:, :, 2], f[:, :, 3] = torch.cos(a) * u - torch.sin(a) * v, torch.sin(a) * u + torch.cos(a) * v
        out.append({**b, "f_C": f})
    return out


# ------------------------------------------------------------- diagnostics

@torch.no_grad()
def loss_gaps(n1, days: Sequence[dict], seed: int = 0) -> dict:
    """I^up = L_X(S -> null) - L_X; I^down_f = L_S(f -> null) - L_S;
    I^down_X = L_S(X -> null) - L_S, on the first transition step, with the
    same noise and pseudo-time in every evaluation."""
    b0, b1 = days[0], days[1]
    bsz, dev = b0["omega_c"].shape[0], b0["omega_c"].device
    z, psi = n1.analysis(b0, n1.hparams.n_steps, _gen(seed, 0, dev))
    gen = _gen(seed, 1, dev)
    noise = torch.randn(bsz, N_Z + n1.d_psi, *b0["omega_c"].shape[-2:], generator=gen, device=dev)
    t = torch.rand(bsz, 1, 1, 1, generator=gen, device=dev)
    full = n1.keep(bsz, dev, train=False)

    def run(drop):
        return n1.step_loss(b1, z, psi, {**full, drop: torch.zeros_like(full[drop])} if drop else full, noise, t)

    ref, no_s, no_f, no_x = run(None), run("s"), run("f"), run("x")
    return {"I_up": no_s["X"] - ref["X"], "I_down_f": no_f["S"] - ref["S"], "I_down_X": no_x["S"] - ref["S"]}


@torch.no_grad()
def response(n1, days: Sequence[dict], delta_s: Tensor, names: Sequence[str], n_steps: int,
             n_members: int = 8, w: float = 0.0) -> Tensor:
    """R_{X<-S}(tau) = E_eps[X_C^{t+tau} | do(S^t + dS)] - E_eps[X_C^{t+tau} | do(S^t)],
    with the same noise in both rollouts: [B, n_steps + 1, 3, 64, 64]."""
    diffs = []
    for s in range(n_members):
        base, psi = rollout(n1, days, n_steps, w=w, seed=s)
        pert, _ = rollout(n1, days, n_steps, w=w, seed=s, intervention=perturb_ic(delta_s, names),
                          z0=base[:, 0], psi0=psi[:, 0])
        diffs.append(pert[:, :, :N_X] - base[:, :, :N_X])
    return torch.stack(diffs).mean(0)


@torch.no_grad()
def controllability(n0, b: dict, z_c: Tensor, psi: Tensor, name: str, delta: float, seed: int = 0) -> Tensor:
    """Slope dS_C(X_H) / dS* for one SIC statistic: fine fields generated from
    S and from S + delta (paired noise), mean over ocean C cells (~1 if S
    controls the fine field)."""
    c = Z_NAMES.index(name)
    z2 = z_c.clone()
    z2[:, c] += delta
    s1 = s_from_fine(downscale(n0, b, z_c, psi, seed), b)[:, c - N_X]
    s2 = s_from_fine(downscale(n0, b, project_s(z2), psi, seed), b)[:, c - N_X]
    ocean = b["omega_c"] > 0.5
    return ((s2 - s1)[ocean] / delta).mean()


@torch.no_grad()
def compare_modes(n1, days: Sequence[dict], n_steps: int, seed: int = 0, gamma: float = 0.5) -> list[dict]:
    """Mode A vs mode B (same noise) on the C cells covered by the SAR on each
    day (observed ocean fraction >= gamma): RMSE of X_C and S_C differences."""
    za, _ = rollout(n1, days, n_steps, seed=seed, mode="A")
    zb, _ = rollout(n1, days, n_steps, seed=seed, mode="B")
    out = []
    for k in range(1, n_steps + 1):
        b = days[k]
        cov = g.restrict((b["in_H_w"][:, -1] > 0).float(), b["omega_h"], g.factor("H", "C"))[0] >= gamma
        diff = (za[:, k] - zb[:, k]) ** 2
        out.append({"k": k, "n_cells": int(cov.sum()),
                    "rmse_X": diff[:, :N_X].permute(1, 0, 2, 3)[:, cov].mean().sqrt() if cov.any() else torch.tensor(float("nan")),
                    "rmse_S": diff[:, N_X:].permute(1, 0, 2, 3)[:, cov].mean().sqrt() if cov.any() else torch.tensor(float("nan"))})
    return out
