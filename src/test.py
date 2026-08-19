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
    """Run `base_test` `n_members` times (stochastic solvers — CM/FM — draw a
    fresh noise sample each call) and merge the resulting NetCDF files into
    one: `pred_<var>` gains a `member` dimension, plus a new `spread_<var>` =
    std over members. `tgt_<var>`/obs fields are deterministic across members
    and are taken from the first member unchanged.

    Per-member intermediate files (one `base_test` log dir each) are left on
    disk, not deleted.
    """
    import xarray as xr

    log_dirs = [base_test(trainer, dm, lit_mod, save_dir=save_dir, ckpt_path=ckpt_path)
                for _ in range(n_members)]

    ref_files = sorted(os.path.basename(f) for f in glob.glob(os.path.join(log_dirs[0], "test_data_*.nc")))
    for fname in ref_files:
        member_ds = [xr.open_dataset(os.path.join(log_dir, fname)) for log_dir in log_dirs]

        pred_vars = [v for v in member_ds[0].data_vars if v.startswith("pred_")]
        merged = member_ds[0].copy(deep=True)
        for var in pred_vars:
            stacked = xr.concat([ds[var] for ds in member_ds], dim="member")
            merged[var] = stacked
            merged[f"spread_{var[len('pred_'):]}"] = stacked.std(dim="member")

        out_path = os.path.join(save_dir, fname)
        merged.to_netcdf(out_path)
        print(f"[ensemble_test] wrote {out_path} ({n_members} members)")

        for ds in member_ds:
            ds.close()
