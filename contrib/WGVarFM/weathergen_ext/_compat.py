"""Stand-ins for the WeatherGenerator symbols imported by the vendored `model/` code.

Replaces `weathergen.common.config`, `weathergen.common.io`,
`weathergen.datasets.batch`, `weathergen.utils.utils` and
`weathergen.utils.distributed`, so that only `model/` and a few
`datasets/` files need to be vendored. Also provides an SDPA fallback
for `flash_attn` when it is not installed. Function bodies are
copied from WeatherGenerator (Apache-2.0, commit 1719b8e).
"""

import dataclasses
from typing import Any, Literal

import torch
import torch.distributed as dist
import torch.nn.functional as F
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


# flash_attn: use it when installed, else fall back to torch SDPA on jagged nested tensors
# (on GPU, torch dispatches these to its own varlen flash/efficient kernels).
def sdpa_attn_func(q, k, v, dropout_p=0.0, softcap=0.0, **kwargs):
    """(B, L, H, D) layout, as flash_attn.flash_attn_func."""
    assert not softcap, "softcap is not supported by the SDPA fallback"
    assert not kwargs, f"unsupported flash_attn arguments: {list(kwargs)}"
    k, v = k.to(q.dtype), v.to(q.dtype)
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), dropout_p=dropout_p
    )
    return out.transpose(1, 2)


def _sdpa_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, dropout_p):
    lens_q = cu_seqlens_q.long().diff()
    lens_k = cu_seqlens_k.long().diff()
    keep = lens_q > 0
    if not keep.any():
        return torch.zeros_like(q)
    # sequences without queries (e.g. empty HEALPix cells) are dropped with their keys
    if not keep.all():
        key_mask = torch.repeat_interleave(keep, lens_k)
        k, v = k[key_mask], v[key_mask]
        lens_q, lens_k = lens_q[keep], lens_k[keep]
    assert (lens_k > 0).all(), "query sequence without keys"
    offs_q = F.pad(lens_q.cumsum(0), (1, 0))
    offs_k = F.pad(lens_k.cumsum(0), (1, 0))

    def jagged(t, offs):
        return torch.nested.nested_tensor_from_jagged(t, offsets=offs).transpose(1, 2)

    with torch.autocast(q.device.type, enabled=False):
        out = F.scaled_dot_product_attention(
            jagged(q, offs_q), jagged(k, offs_k), jagged(v, offs_k), dropout_p=dropout_p
        )
    return out.transpose(1, 2).values()


class _SDPAVarlen(torch.autograd.Function):
    """Keeps nested tensors out of the autograd graph (torch.utils.checkpoint, used all over
    WG, does not support them): forward without graph, backward recomputes the attention
    with the same RNG state (same dropout mask)."""

    @staticmethod
    def forward(ctx, q, k, v, cu_seqlens_q, cu_seqlens_k, dropout_p):
        ctx.dropout_p = dropout_p
        ctx.cuda = q.is_cuda
        ctx.rng_cpu = torch.get_rng_state()
        ctx.rng_cuda = torch.cuda.get_rng_state(q.device) if q.is_cuda else None
        ctx.save_for_backward(q, k, v, cu_seqlens_q, cu_seqlens_k)
        with torch.no_grad():
            return _sdpa_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, dropout_p)

    @staticmethod
    def backward(ctx, grad_out):
        q, k, v, cu_q, cu_k = ctx.saved_tensors
        q, k, v = (t.detach().requires_grad_() for t in (q, k, v))
        devices = [q.device] if ctx.cuda else []
        with torch.random.fork_rng(devices=devices), torch.enable_grad():
            torch.set_rng_state(ctx.rng_cpu)
            if ctx.cuda:
                torch.cuda.set_rng_state(ctx.rng_cuda, q.device)
            out = _sdpa_varlen(q, k, v, cu_q, cu_k, ctx.dropout_p)
        grads = torch.autograd.grad(out, (q, k, v), grad_out, allow_unused=True)
        dq, dk, dv = (torch.zeros_like(t) if g is None else g for t, g in zip((q, k, v), grads))
        return dq, dk, dv, None, None, None


def sdpa_attn_varlen_func(
    q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
    dropout_p=0.0, softcap=0.0, **kwargs,
):
    """Packed (total, H, D) layout with cumulative sequence lengths, as
    flash_attn.flash_attn_varlen_func."""
    assert not softcap, "softcap is not supported by the SDPA fallback"
    assert not kwargs, f"unsupported flash_attn arguments: {list(kwargs)}"
    k, v = k.to(q.dtype), v.to(q.dtype)
    return _SDPAVarlen.apply(
        q.contiguous(), k.contiguous(), v.contiguous(), cu_seqlens_q, cu_seqlens_k, dropout_p
    )


try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func

    HAS_FLASH_ATTN = True
except ImportError:
    flash_attn_func, flash_attn_varlen_func = sdpa_attn_func, sdpa_attn_varlen_func
    HAS_FLASH_ATTN = False
