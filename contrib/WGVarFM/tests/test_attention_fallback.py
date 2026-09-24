"""Equivalence test of the SDPA fallback for flash_attn (weathergen_ext/_compat.py).

Compares sdpa_attn_varlen_func / sdpa_attn_func with a per-sequence reference attention
(including empty query sequences, as produced by empty HEALPix cells), forward and
gradients, on CPU and on GPU when available. If flash_attn is installed, also compares
with it on GPU. Run from the repo root:
    python contrib/WGVarFM/tests/test_attention_fallback.py
"""

import torch
import torch.nn.functional as F
import torch.utils.checkpoint

from contrib.WGVarFM.weathergen_ext import _compat

H, D = 4, 32


def reference_varlen(q, k, v, cu_q, cu_k):
    out = torch.zeros_like(q)
    for i in range(len(cu_q) - 1):
        q0, q1, k0, k1 = int(cu_q[i]), int(cu_q[i + 1]), int(cu_k[i]), int(cu_k[i + 1])
        if q1 > q0:
            att = torch.softmax(
                torch.einsum("qhd,khd->hqk", q[q0:q1], k[k0:k1]) / D**0.5, dim=-1
            )
            out[q0:q1] = torch.einsum("hqk,khd->qhd", att, v[k0:k1])
    return out


def make_inputs(device, dtype):
    torch.manual_seed(0)
    # includes empty query sequences (with and without keys) and 9-key sequences (1-ring)
    lens_q = torch.tensor([3, 0, 7, 1, 0, 120, 5])
    lens_k = torch.tensor([9, 9, 9, 2, 0, 64, 9])
    cu_q = F.pad(lens_q.cumsum(0), (1, 0)).to(device, torch.int32)
    cu_k = F.pad(lens_k.cumsum(0), (1, 0)).to(device, torch.int32)
    q = torch.randn(int(lens_q.sum()), H, D, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(int(lens_k.sum()), H, D, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(int(lens_k.sum()), H, D, device=device, dtype=dtype, requires_grad=True)
    return q, k, v, cu_q, cu_k, int(lens_q.max()), int(lens_k.max())


def grads(fn, q, k, v, *args):
    out = fn(q, k, v, *args)
    g = torch.autograd.grad(out.float().pow(2).sum(), (q, k, v))
    return out, g


def check_close(name, a, b, atol):
    err = (a.float() - b.float()).abs().max().item()
    assert err < atol, f"{name}: max abs error {err:.2e} >= {atol:.0e}"
    print(f"  {name}: max abs error {err:.2e}")


def run(device, dtype, atol):
    print(f"[{device}, {dtype}]")
    q, k, v, cu_q, cu_k, mq, mk = make_inputs(device, dtype)

    out_ref, g_ref = grads(lambda *a: reference_varlen(*a[:5]), q, k, v, cu_q, cu_k)
    out, g = grads(_compat.sdpa_attn_varlen_func, q, k, v, cu_q, cu_k, mq, mk)
    check_close("varlen out", out, out_ref, atol)
    for n, a, b in zip("qkv", g, g_ref):
        check_close(f"varlen grad {n}", a, b, atol * 10)

    # as used in WG: inside activation checkpointing, non-contiguous q/k/v, mixed dtypes
    qkv = torch.randn(len(q), H, 3 * D, device=device, dtype=dtype, requires_grad=True)
    qn, kn, vn = qkv[..., :D], qkv[..., D : 2 * D], qkv[..., 2 * D :].float()
    ck = torch.utils.checkpoint.checkpoint(
        _compat.sdpa_attn_varlen_func, qn, kn, vn, cu_q, cu_q, mq, mq, use_reentrant=False
    )
    ck.float().pow(2).sum().backward()
    check_close("checkpointed out", ck, reference_varlen(qn, kn, vn.to(dtype), cu_q, cu_q), atol)
    assert torch.isfinite(qkv.grad).all()

    # dropout: backward must reuse the forward dropout mask
    # (directional finite difference with the same seed, i.e. the same mask)
    if device == "cpu":
        qd = q.detach().double().requires_grad_()
        kd, vd = k.detach().double(), v.detach().double()

        def loss(a):
            torch.manual_seed(1)
            out = _compat.sdpa_attn_varlen_func(a, kd, vd, cu_q, cu_k, mq, mk, dropout_p=0.3)
            return out.pow(2).sum()

        loss(qd).backward()
        d, eps = torch.randn_like(qd), 1e-6
        with torch.no_grad():
            fd = (loss(qd + eps * d) - loss(qd - eps * d)) / (2 * eps)
        check_close("dropout grad (finite diff.)", (qd.grad * d).sum(), fd, 1e-5 * fd.abs().item())

    # all query sequences empty
    z = _compat.sdpa_attn_varlen_func(
        q[:0], k, v, torch.zeros_like(cu_q), cu_k, 0, mk
    )
    assert z.shape[0] == 0

    if device == "cuda" and _compat.HAS_FLASH_ATTN:
        from flash_attn import flash_attn_varlen_func

        out_fa, _ = grads(flash_attn_varlen_func, q, k, v, cu_q, cu_k, mq, mk)
        check_close("varlen vs flash_attn", out, out_fa, atol)

    x = torch.randn(2, 50, H, D, device=device, dtype=dtype)
    dense_ref = reference_varlen(
        x.flatten(0, 1), x.flatten(0, 1), x.flatten(0, 1),
        torch.tensor([0, 50, 100], device=device), torch.tensor([0, 50, 100], device=device),
    ).reshape(x.shape)
    check_close("dense out", _compat.sdpa_attn_func(x, x, x), dense_ref, atol)


def main():
    print(f"flash_attn installed: {_compat.HAS_FLASH_ATTN}")
    run("cpu", torch.float32, 1e-5)
    if torch.cuda.is_available():
        run("cuda", torch.bfloat16, 3e-2)
    print("OK")


if __name__ == "__main__":
    main()
