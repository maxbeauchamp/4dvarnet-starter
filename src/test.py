import glob
import torch
import os
import random
torch.set_float32_matmul_precision('high')
from pytorch_lightning import loggers

def base_test(trainer, dm, lit_mod,
              save_dir="/Odyssey/private/m19beauc/4dvarnet-starter/results", ckpt_path=None):

    '''
    ckpt = torch.load(ckpt_path)["state_dict"]
    lit_mod.load_state_dict(ckpt)

    dm.setup()
    lit_mod._norm_stats = dm.norm_stats()
    dm._norm_stats = dm.norm_stats()
    '''

    version = 'version_' + str(random.randint(0, 100000))
    logger_name = "lightning_logs"
    print(os.path.join(save_dir, logger_name, version))
    tb_logger = loggers.TensorBoardLogger(save_dir=save_dir,
                                                   name=logger_name,
                                                   version=version)
    trainer.logger = tb_logger

    trainer.test(lit_mod, datamodule=dm, ckpt_path=ckpt_path)
    return tb_logger.log_dir


def ensemble_test(trainer, dm, lit_mod, n_members=5,
                   save_dir="/Odyssey/private/m19beauc/4dvarnet-starter/results", ckpt_path=None):
    """Run `base_test` `n_members` times, sequentially (stochastic solvers —
    CM/FM — draw a fresh noise sample each call, so each pass is a different
    member). The checkpoint is only loaded on the first pass — later passes
    reuse the already-loaded `lit_mod` in place, avoiding a reload of the
    checkpoint/datamodule per member.

    Each pass writes its own NetCDF (kept on disk, one per member). After all
    passes, a final NetCDF is written per test window/resolution where every
    `pred_<var>` is replaced by `spread_<var>` = std across the n_members
    per-member values (`tgt_<var>`/obs fields are deterministic across members
    and are taken from the first member unchanged).
    """
    import xarray as xr

    log_dirs = [
        base_test(trainer, dm, lit_mod, save_dir=save_dir,
                  ckpt_path=ckpt_path if m == 0 else None)
        for m in range(n_members)
    ]

    ref_files = sorted(os.path.basename(f) for f in glob.glob(os.path.join(log_dirs[0], "test_data_*.nc")))
    for fname in ref_files:
        member_ds = [xr.open_dataset(os.path.join(log_dir, fname)) for log_dir in log_dirs]

        pred_vars = [v for v in member_ds[0].data_vars if v.startswith("pred_")]
        merged = member_ds[0].copy(deep=True)
        for var in pred_vars:
            spread = xr.concat([ds[var] for ds in member_ds], dim="member").std(dim="member")
            del merged[var]
            merged[f"spread_{var[len('pred_'):]}"] = spread

        out_path = os.path.join(save_dir, fname)
        merged.to_netcdf(out_path)
        print(f"[ensemble_test] wrote {out_path} (spread over {n_members} members)")

        for ds in member_ds:
            ds.close()
