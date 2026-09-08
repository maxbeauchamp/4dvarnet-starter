import torch
import os
import random
torch.set_float32_matmul_precision('high')
from pytorch_lightning import loggers

def base_training(trainer, dm, lit_mod,
                  save_dir="/Odyssey/private/m19beauc/DMI/results",
                  ckpt=None, test=True, init_weights_ckpt=None):

    if init_weights_ckpt is not None:
        # Load a partial state_dict (e.g. produced by
        # contrib/CROSCIM/scripts/strip_solver_weights.py) before the fit
        # starts, strict=False so tensors absent from the checkpoint are
        # left at lit_mod's own (random) init -- lets one resolution's
        # solver restart from scratch while others keep pretrained weights.
        # Independent of `ckpt` (Lightning's own resume, which restores
        # optimizer/epoch state too).
        state_dict = torch.load(init_weights_ckpt, map_location='cpu')['state_dict']
        missing, unexpected = lit_mod.load_state_dict(state_dict, strict=False)
        print(f"init_weights_ckpt={init_weights_ckpt}: loaded {len(state_dict)} tensors, "
              f"{len(missing)} left at random init, {len(unexpected)} unused")

    version = 'version_' + str(random.randint(0, 100000))
    logger_name = "lightning_logs"
    print(os.path.join(save_dir, logger_name, version))
    tb_logger = loggers.TensorBoardLogger(save_dir=save_dir,
                                                   name=logger_name,
                                                   version=version)
    trainer.logger = tb_logger

    if trainer.logger is not None:
        print()
        print("Logdir:", trainer.logger.log_dir)
        print()

    trainer.fit(lit_mod, datamodule=dm, ckpt_path=ckpt)
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
