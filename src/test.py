import torch
import os
import random
torch.set_float32_matmul_precision('high')
from pytorch_lightning import loggers

def base_test(trainer, dm, lit_mod, 
              save_dir="/dmidata/users/maxb/4dvarnet-starter/results", ckpt_path=None):

    '''
    ckpt = torch.load(ckpt_path)["state_dict"]
    lit_mod.load_state_dict(ckpt)

    dm.setup()
    lit_mod._norm_stats = dm.norm_stats()
    dm._norm_stats = dm.norm_stats()
    '''
    if trainer.logger is not None:
        print()
        print("Logdir:", trainer.logger.log_dir)
        print()

    trainer.test(lit_mod, datamodule=dm, ckpt_path=ckpt_path)

