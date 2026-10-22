"""Sign conventions of scripts/drift_feasibility.py on a known displacement:
semi-Lagrangian advection and block matching must agree with the grid axes
(+x along the columns, +y along -rows)."""
import os
import sys

import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import drift_feasibility as DF  # noqa: E402


def _field(n=64, shift=(0, 0)):
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    cy, cx = 30 + shift[0], 28 + shift[1]
    return torch.tensor(np.exp(-((ii - cy) ** 2 + (jj - cx) ** 2) / 40.0), dtype=torch.float32)


def test_advection_moves_along_the_grid_axes():
    s0, valid = _field(), torch.ones(64, 64, dtype=torch.bool)
    # +x by 3 cells and +y by 2 cells (= 2 rows up) in one day
    ux = torch.full((64, 64), 3 * DF.DXM / 86400.0)
    uy = torch.full((64, 64), 2 * DF.DXM / 86400.0)
    pred, ok = DF.advect(s0, valid, ux, uy, torch.full((64, 64), 86400.0))
    s1 = _field(shift=(-2, 3))
    assert ((pred - s1)[ok] ** 2).mean() < 1e-6 * (s1 ** 2).mean() + 1e-8


def test_block_match_finds_the_shift():
    s0, s1 = _field(), _field(shift=(-2, 3))
    v = torch.ones(64, 64, dtype=torch.bool)
    mse, cnt = DF.block_match(s0, v, s1, v, 4, 32)
    mse = torch.where(cnt >= 0.5 * 32 * 32, mse, float("inf"))
    k = mse[:, :, 0, 0].flatten().argmin()
    dy, dx = int(k // 9) - 4, int(k % 9) - 4
    assert (dy, dx) == (-2, 3)  # rows up 2 (= +y), columns +3 (= +x)


def test_free_drift_turns_right_and_rotates_east_north():
    u, v = torch.ones(1), torch.zeros(1)
    # grid frame, no turn: along +x
    ux, uy = DF.free_drift(u, v, torch.zeros(1), "grid", 0.02, 0.0)
    assert torch.allclose(ux, torch.tensor([0.02])) and torch.allclose(uy, torch.zeros(1), atol=1e-9)
    # turn 90 deg to the right of +x: -y
    ux, uy = DF.free_drift(u, v, torch.zeros(1), "grid", 0.02, 90.0)
    assert torch.allclose(uy, torch.tensor([-0.02]), atol=1e-7)
    # east wind at lon = lon_0 + 90 deg (delta = pi / 2): along +y of the grid
    ux, uy = DF.free_drift(u, v, torch.full((1,), np.pi / 2), "geo", 0.02, 0.0)
    assert torch.allclose(uy, torch.tensor([0.02]), atol=1e-7)
