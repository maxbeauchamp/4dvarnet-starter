"""Stand-ins for the WeatherGenerator symbols imported by the vendored `model/` code.

Replaces `weathergen.common.config`, `weathergen.common.io`,
`weathergen.datasets.batch`, `weathergen.utils.utils` and
`weathergen.utils.distributed`, so that only `model/` and a few
`datasets/` files need to be vendored. Function bodies are
copied from WeatherGenerator (Apache-2.0, commit 1719b8e).
"""

import dataclasses
from typing import Any, Literal

import torch
import torch.distributed as dist
from omegaconf import DictConfig

# weathergen.common.config
Config = DictConfig


# weathergen.common.io (only the fields read by the vendored tokenizer)
@dataclasses.dataclass
class IOReaderData:
    coords: Any
    geoinfos: Any
    data: Any
    datetimes: Any
    is_spoof: bool = False

# weathergen.datasets.batch: only used as type hints in the vendored code;
# batches are duck-typed and built by the WGVarFM tokenization adapter.
ModelBatch = Any
BatchSamples = Any

# weathergen.train.utils
Stage = Literal["train", "val", "test"]
TRAIN: Stage = "train"


# weathergen.utils.utils
def get_dtype(value: str) -> torch.dtype:
    """
    changes the conf value to a torch dtype
    """
    if value == "bf16":
        return torch.bfloat16
    elif value == "fp16":
        return torch.float16
    elif value == "fp32":
        return torch.float32
    else:
        raise NotImplementedError(
            f"Dtype {value} is not recognized, choose either, bf16, fp16, or fp32"
        )


def is_stream_forcing(stream_cfg: dict, stage: Stage | None = None) -> bool:
    """
    Determine if stream is forcing, i.e. does not produce (physical) predictions
    """
    is_forcing = stream_cfg.get("forcing", False)
    if stage is not None:
        is_forcing = is_forcing or (
            (len(stream_cfg.get("train_target_channels", [])) == 0)
            if stage == TRAIN
            else (len(stream_cfg.get("val_target_channels", [])) == 0)
        )
    else:
        is_forcing = is_forcing or (
            len(stream_cfg.get("train_target_channels", [])) == 0
            and len(stream_cfg.get("val_target_channels", [])) == 0
        )

    return is_forcing


# weathergen.utils.distributed
def is_root(pg: dist.ProcessGroup | None = None) -> bool:
    """
    Check if the current rank is the root rank (rank 0).
    """
    if not (dist.is_available() and dist.is_initialized()):
        return True
    return dist.get_rank(pg) == 0
