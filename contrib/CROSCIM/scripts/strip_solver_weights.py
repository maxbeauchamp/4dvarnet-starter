#!/usr/bin/env python3
"""Strip one resolution's solver weights from a CROSCIM checkpoint.

Produces a weights-only file (state_dict, no optimizer/epoch state) with
every tensor belonging to a given resolution's solver, EMA network, and
log-scale model removed. Loading it via `src.train.base_training`'s
`init_weights_ckpt` (strict=False) lets that resolution restart training
from scratch while every other resolution keeps its pretrained weights.

Why: for the multi-res CROSCIM Flow Matching setup (x50 coarse -> x10
fine anomaly correction), x50 and x10 are independent solver instances
(`solver.solvers.solver_x50`, `solver.solvers.solver_x10`, plus their own
entries in `ema_networks`/`log_scale_models`) -- there is no cross-talk in
the checkpoint's weights, so dropping one resolution's tensors and loading
with strict=False leaves the others untouched and re-initialises only the
dropped one to its constructor's random init.

------------------------------------------------------------------------
How to run
------------------------------------------------------------------------
    python strip_solver_weights.py IN.ckpt OUT.ckpt --res 10

Then in the training config's entrypoint:
    entrypoints:
      - _target_: src.train.base_training
        ...
        ckpt: null                    # fresh fit -- no Lightning resume
        init_weights_ckpt: OUT.ckpt   # x50 pretrained, x10 fresh

Also set `model.training_strategy: simultaneous` for that run so x10 gets
gradient updates from epoch 0 instead of waiting through a full progressive
x50-first phase again (x50 keeps fine-tuning from its already-good weights
in parallel, at no extra cost).
"""
from __future__ import annotations

import argparse

import torch


def strip_resolution(in_path: str, out_path: str, res: int) -> None:
    ckpt = torch.load(in_path, map_location="cpu")
    state_dict = ckpt["state_dict"]
    prefixes = (
        f"solver.solvers.solver_x{res}.",
        f"ema_networks.solver_x{res}.",
        f"log_scale_models.solver_x{res}.",
    )
    kept = {k: v for k, v in state_dict.items() if not k.startswith(prefixes)}
    dropped = len(state_dict) - len(kept)
    if dropped == 0:
        raise RuntimeError(
            f"No tensor matched solver_x{res} under {prefixes} -- "
            f"check --res, or that this checkpoint actually has that resolution."
        )
    torch.save({"state_dict": kept}, out_path)
    print(f"Kept {len(kept)} tensors, dropped {dropped} (solver_x{res}) -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("in_path", help="Source .ckpt (full Lightning checkpoint)")
    ap.add_argument("out_path", help="Output path for the stripped weights-only file")
    ap.add_argument("--res", type=int, required=True,
                     help="Resolution to drop and restart from scratch (e.g. 10)")
    args = ap.parse_args()
    strip_resolution(args.in_path, args.out_path, args.res)


if __name__ == "__main__":
    main()
