import torch
import os
import random
torch.set_float32_matmul_precision('high')
from pytorch_lightning import loggers
from omegaconf import OmegaConf
import hydra
from pathlib import Path



def base_training(trainer, dm, lit_mod, 
                  save_dir="/Odyssey/private/m19beauc/DMI/results",
                  ckpt=None,test=True):

    if trainer.logger is not None:
        print()
        print("Logdir:", trainer.logger.log_dir)
        print()

    trainer.fit(lit_mod, datamodule=dm, ckpt_path=ckpt)
    if test:
        trainer.test(lit_mod, datamodule=dm, ckpt_path='best')






def load_from_cfg(cfg_path, key):
    """
    Load configurations from a specified file and instantiate the
    desired node.
    """
    cfg = OmegaConf.load(Path(cfg_path))
    node = OmegaConf.select(cfg, key)
    return hydra.utils.call(node)


def train_from_pretrained_model(trainer, dm, lit_mod, save_dir,
                                 config_path,ckpt_path=None, test=True):
    if trainer.logger is not None:
        print()
        print("Logdir:", trainer.logger.log_dir)
        print()

    if ckpt_path is not None:
        print(" Load pretrained config and model from : ", ckpt_path)
        solver = load_from_cfg(config_path, key="model")
        ckpt = torch.load(ckpt_path, weights_only=True)
        solver.load_state_dict(ckpt["state_dict"])
        lit_mod.solver.load_state_dict(solver.solver.state_dict())
    else:
        print(" Training from scratch (no pretrained model)", flush=True)

    trainer.fit(lit_mod, datamodule=dm)
    if test:
        trainer.test(lit_mod, datamodule=dm, ckpt_path='best')






def multi_dm_training(trainer, dm, lit_mod, test_dm=None, test_fn=None, ckpt=None):
    if trainer.logger is not None:
        print()
        print("Logdir:", trainer.logger.log_dir)
        print()

    trainer.fit(lit_mod, datamodule=dm, ckpt_path=ckpt)

    if test_fn is not None:
        if test_dm is None:
            test_dm = dm
        lit_mod._norm_stats = test_dm.norm_stats()

        best_ckpt_path = trainer.checkpoint_callback.best_model_path
        trainer.callbacks = []
        trainer.test(lit_mod, datamodule=test_dm, ckpt_path=best_ckpt_path)

        print("\nBest ckpt score:")
        print(test_fn(lit_mod).to_markdown())
        print("\n###############")
