"""LightningModule wrapping the vendored WeatherGenerator model (data-agnostic).

Streams are described by the same config as the data pipeline:
    {name: {vars: [...], token_size, source: bool, target: bool, ...}}
`source` streams get an embedding network, `target` streams a decoder
(coordinate embedding + TargetPredictionEngine + EnsPredictionHead).

Training objective: MSE between the ensemble-mean prediction and the target points
(for WG pre-training, the observations of the masked HEALPix cells).
"""

import math

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

from contrib.WGVarFM.tokenization import TARGET_COORDS_SIZE, source_size
from contrib.WGVarFM.weathergen_ext.model.model import Model, ModelParams


def build_wg_config(wg, streams):
    """WG config = architecture (`wg`) + streams in WG format + fixed training settings."""
    cf = OmegaConf.create(OmegaConf.to_container(OmegaConf.create(wg), resolve=True))
    wg_streams = {}
    for name, sc in streams.items():
        target_channels = list(sc["vars"]) if sc.get("target", False) else []
        wg_streams[name] = {
            "name": name,
            "token_size": sc["token_size"],
            "embed": dict(cf.stream_defaults.embed),
            "embed_target_coords": dict(cf.stream_defaults.embed_target_coords),
            "target_readout": dict(cf.stream_defaults.target_readout),
            "pred_head": dict(cf.stream_defaults.pred_head),
            "train_target_channels": target_channels,
            "val_target_channels": target_channels,
        }
    cf.streams = wg_streams
    cf.fe_num_blocks = 0  # no latent forecast yet: reconstruction at the input time
    cf.training_config = {"losses": {"physical": {"type": "LossPhysical"}}}
    cf.validation_config = {}
    return cf


class LitWGFM(pl.LightningModule):
    def __init__(
        self,
        wg,
        streams,
        lr=5e-5,
        lr_start=1e-6,
        warmup_steps=512,
        betas=(0.975, 0.9875),
        weight_decay=0.1,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.streams = streams
        self.lr = lr
        self.lr_start = lr_start
        self.warmup_steps = warmup_steps
        self.betas = betas
        self.weight_decay = weight_decay

        self.cf = build_wg_config(wg, streams)
        sources_size = [
            source_size(len(sc["vars"])) if sc.get("source", True) else 0
            for sc in streams.values()
        ]
        targets_num_channels = [len(sc["vars"]) for sc in streams.values()]
        targets_coords_size = [TARGET_COORDS_SIZE for _ in streams]

        self.model_params = ModelParams(self.cf).create(self.cf)
        self.model = Model(self.cf, sources_size, targets_num_channels, targets_coords_size).create()
        self.target_streams = [n for n, sc in streams.items() if sc.get("target", False)]

    def forward(self, batch):
        return self.model(self.model_params, batch)

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        return batch.to(device)

    def loss(self, batch, output):
        """Mean over target streams of the per-stream MSE (ensemble mean vs targets)."""
        losses = {}
        for name in self.target_streams:
            preds = output.get_physical_prediction(0, name)
            if preds is None:
                continue
            tgts = [s.streams_data[name].target_values[0] for s in batch.get_samples()]
            pred = torch.cat([p.mean(0) for p in preds]).float()
            tgt = torch.cat(tgts).float()
            if tgt.numel() > 0:
                losses[name] = torch.nn.functional.mse_loss(pred, tgt)
        total = torch.stack(list(losses.values())).mean() if losses else None
        return total, losses

    def _step(self, batch, phase):
        loss, per_stream = self.loss(batch, self(batch))
        if loss is None:  # no masked target point in this batch
            return None
        bs = len(batch)
        self.log(f"{phase}_loss", loss, prog_bar=True, batch_size=bs, sync_dist=phase != "train")
        for name, v in per_stream.items():
            self.log(f"{phase}_loss_{name}", v, batch_size=bs, sync_dist=phase != "train")
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")

    def configure_optimizers(self):
        """AdamW + warmup (lr_start -> lr) then cosine decay to 0, as in WG's default config.

        Without warmup the first Adam step (~lr on every parameter) already degrades the model.
        """
        params = [p for p in self.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(
            params, lr=self.lr, betas=tuple(self.betas), weight_decay=self.weight_decay
        )
        total = max(self.trainer.estimated_stepping_batches, self.warmup_steps + 1)
        start = self.lr_start / self.lr

        def factor(step):
            if step < self.warmup_steps:
                return start + (1 - start) * 0.5 * (1 - math.cos(math.pi * step / self.warmup_steps))
            progress = (step - self.warmup_steps) / (total - self.warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, factor)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}
