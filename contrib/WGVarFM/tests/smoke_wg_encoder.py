"""Smoke test of the vendored WeatherGenerator encoder (step 1 of WGVarFM).

Builds a tiny EncoderModule (one 'linear' stream, HEALPix level 2), feeds it
random sparse tokens through a duck-typed batch and checks the latent shape,
finiteness and the backward pass.

Requires CUDA + flash-attn (the vendored varlen attention asserts flash).
Run from the repo root:
    python contrib/WGVarFM/tests/smoke_wg_encoder.py
"""

import torch
from omegaconf import OmegaConf

from contrib.WGVarFM.weathergen_ext.model.encoder import EncoderModule
from contrib.WGVarFM.weathergen_ext.model.model import ModelParams

HL = 2
NUM_CELLS = 12 * 4**HL
TOKEN_SIZE = 8
NUM_CHANNELS = 1

cf = OmegaConf.create(
    {
        "healpix_level": HL,
        "streams": {
            "sic": {
                "token_size": TOKEN_SIZE,
                "embed": {"net": "linear"},
            }
        },
        "mixed_precision_dtype": "bf16",
        "attention_dtype": "bf16",
        "with_flash_attention": True,
        "norm_type": "LayerNorm",
        "qk_norm_type": None,
        "norm_eps": 1e-4,
        "mlp_norm_eps": 1e-5,
        "embed_dropout_rate": 0.0,
        "embed_unembed_mode": "block",
        "ae_local_dim_embed": 64,
        "ae_local_num_blocks": 1,
        "ae_local_num_heads": 4,
        "ae_local_dropout_rate": 0.0,
        "ae_local_with_qk_lnorm": True,
        "ae_local_num_queries": 1,
        "ae_local_queries_per_cell": False,
        "ae_adapter_embed": 32,
        "ae_adapter_num_heads": 4,
        "ae_adapter_with_residual": True,
        "ae_adapter_with_qk_lnorm": True,
        "ae_adapter_dropout_rate": 0.0,
        "ae_aggregation_num_blocks": 1,
        "ae_aggregation_att_dense_rate": 1.0,
        "ae_aggregation_block_factor": 64,
        "ae_aggregation_num_heads": 4,
        "ae_aggregation_dropout_rate": 0.0,
        "ae_aggregation_with_qk_lnorm": True,
        "ae_aggregation_mlp_hidden_factor": 2,
        "ae_global_dim_embed": 128,
        "ae_global_num_blocks": 2,
        "ae_global_num_heads": 4,
        "ae_global_att_dense_rate": 1.0,
        "ae_global_block_factor": 64,
        "ae_global_dropout_rate": 0.0,
        "ae_global_with_qk_lnorm": True,
        "ae_global_mlp_hidden_factor": 2,
        "num_register_tokens": 0,
        "num_class_tokens": 0,
        "latent_noise_kl_weight": 0.0,
    }
)


class _StreamData:
    def __init__(self, source_tokens_cells):
        self.source_tokens_cells = source_tokens_cells


class _Sample:
    def __init__(self, streams_data):
        self.streams_data = streams_data


class _Batch:
    """Duck-typed stand-in for weathergen ModelBatch (only what the encoder reads)."""

    def __init__(self, samples, tokens_lens):
        self.samples = samples
        # (num_steps_input, num_samples, num_streams, num_cells)
        self.tokens_lens = tokens_lens

    def get_num_source_steps(self):
        return self.tokens_lens.shape[0]

    def get_samples(self):
        return self.samples

    def get_device(self):
        return self.tokens_lens.device

    def __len__(self):
        return len(self.samples)


def make_batch(num_samples, device, obs_fraction=0.3, max_tok_per_cell=4):
    samples, lens = [], []
    for _ in range(num_samples):
        observed = torch.rand(NUM_CELLS) < obs_fraction
        n_tok = torch.randint(1, max_tok_per_cell + 1, (NUM_CELLS,)) * observed
        tokens = torch.randn(int(n_tok.sum()), TOKEN_SIZE, NUM_CHANNELS)
        samples.append(_Sample({"sic": _StreamData([tokens.to(device)])}))
        lens.append(n_tok)
    tokens_lens = torch.stack(lens)[None, :, None, :].to(device=device, dtype=torch.int32)
    return _Batch(samples, tokens_lens)


def main():
    assert torch.cuda.is_available(), "flash-attn requires a CUDA device"
    device = torch.device("cuda")
    torch.manual_seed(0)

    num_samples = 2
    model_params = ModelParams(cf).create(cf).to(device)
    encoder = EncoderModule(
        cf, sources_size=[NUM_CHANNELS], targets_num_channels=None, targets_coords_size=None
    ).to(device)
    batch = make_batch(num_samples, device)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        tokens_global, _ = encoder(model_params, batch)

    expected = (num_samples, NUM_CELLS * cf.ae_local_num_queries, cf.ae_global_dim_embed)
    assert tuple(tokens_global.shape) == expected, (tuple(tokens_global.shape), expected)
    assert torch.isfinite(tokens_global).all()

    tokens_global.float().pow(2).mean().backward()
    n_grad = sum(p.grad is not None for p in encoder.parameters() if p.requires_grad)
    assert n_grad > 0, "no gradient reached the encoder"

    print(f"OK latent {tuple(tokens_global.shape)}, {n_grad} parameter tensors with grad")


if __name__ == "__main__":
    main()
