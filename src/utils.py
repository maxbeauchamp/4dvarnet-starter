"""Generic 4DVarNet helpers (optimisers, reconstruction weights) used by the QG testbed."""
import numpy as np
import kornia
import torch

def pipe(inp, fns):
    for f in fns:
        inp = f(inp)
    return inp

def kwgetattr(obj, name):
    return getattr(obj, name)

def callmap(inp, fns):
    return [fn(inp) for fn in fns]

def half_lr_adam(lit_mod, lr):
    return torch.optim.Adam(
        [
            {"params": lit_mod.solver.grad_mod.parameters(), "lr": lr},
            {"params": lit_mod.solver.obs_cost.parameters(), "lr": lr},
            {"params": lit_mod.solver.prior_cost.parameters(), "lr": lr / 2},
        ],
    )


def cosanneal_lr_adam(lit_mod, lr, T_max=100, weight_decay=0.):
    opt = torch.optim.Adam(
        [
            {"params": lit_mod.solver.grad_mod.parameters(), "lr": lr},
            {"params": lit_mod.solver.obs_cost.parameters(), "lr": lr},
            {"params": lit_mod.solver.prior_cost.parameters(), "lr": lr / 2},
        ], weight_decay=weight_decay
    )
    return {
        "optimizer": opt,
        "lr_scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=T_max),
    }

def triang_lr_adam(lit_mod, lr_min=5e-5, lr_max=3e-3, nsteps=200):
    opt = torch.optim.Adam(
        [
            {"params": lit_mod.solver.grad_mod.parameters(), "lr": lr_max},
            {"params": lit_mod.solver.prior_cost.parameters(), "lr": lr_max / 2},
        ],
    )
    return {
        "optimizer": opt,
        "lr_scheduler": torch.optim.lr_scheduler.CyclicLR(
            opt,
            base_lr=lr_min,
            max_lr=lr_max,
            step_size_up=nsteps,
            step_size_down=nsteps,
            gamma=0.95,
            cycle_momentum=False,
            mode="exp_range",
        ),
    }


def get_constant_crop(patch_dims, crop, dim_order=["time", "lat", "lon"]):
    patch_weight = np.zeros([patch_dims[d] for d in dim_order], dtype="float32")
    mask = tuple(
        slice(crop[d], -crop[d]) if crop.get(d, 0) > 0 else slice(None, None)
        for d in dim_order
    )
    patch_weight[mask] = 1.0
    return patch_weight


def get_cropped_hanning_mask(patch_dims, crop, **kwargs):
    pw = get_constant_crop(patch_dims, crop)

    t_msk = kornia.filters.get_hanning_kernel1d(patch_dims["time"])

    patch_weight = t_msk[:, None, None] * pw
    return patch_weight.cpu().numpy()


def get_triang_time_wei(patch_dims, offset=0, **crop_kw):
    pw = get_constant_crop(patch_dims, **crop_kw)
    return np.fromfunction(
        lambda t, *a: (
            (1 - np.abs(offset + 2 * t - patch_dims["time"]) / patch_dims["time"]) * pw
        ),
        patch_dims.values(),
    )
