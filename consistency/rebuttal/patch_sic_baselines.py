"""SIC baselines (direct UNet, 4DVarNet-LSTM): masked losses on the gappy ASIP reference and OSI SAF in the inputs.

* UNet: input cat(fused, mask_SAR), fused = SAR where observed, else OSI SAF (same conditioning field as FM/CM).
* 4DVarNet: observations packed as y = cat(SAR, OSI SAF) (2C channels, NaN -> 0, masks per source);
  J_o = masked SAR misfit + w * misfit of 16x16 block averages to OSI SAF (the J_o of VarCM/DynVarCM);
  the solver starts from the fused field. Losses (training, validation) restricted to observed ASIP pixels.
Idempotent: python consistency/rebuttal/patch_sic_baselines.py
"""
import json, os, re

NB = os.path.join(os.path.dirname(__file__), "..", "Notebooks", "Notebooks_SIC")
MARK = "# (rebuttal) SIC baseline helpers"
HELPERS = MARK + r'''
import torch as _t
import torch.nn.functional as _F

def sic_pack(sar, osi):
    """y = cat(SAR, OSI SAF) with NaN -> 0, and the matching masks (float)."""
    osi = osi.to(sar)
    m = _t.cat([_t.isfinite(sar), _t.isfinite(osi)], 1).to(sar.dtype)
    return _t.cat([sar.nan_to_num(0.0), osi.nan_to_num(0.0)], 1), m

def sic_fuse(sar, osi):
    """SAR where observed, else OSI SAF (NaN -> 0 on land), and the SAR mask."""
    osi = osi.to(sar)
    m = _t.isfinite(sar).to(sar.dtype)
    return _t.where(m > 0, sar.nan_to_num(0.0), osi.nan_to_num(0.0)), m

def sic_init_state(y, m):
    C = y.shape[1] // 2
    return _t.where(m[:, :C] > 0, y[:, :C], y[:, C:])

def masked_mse(x, tgt):
    v = _t.isfinite(tgt).to(x.dtype)
    return (((x - tgt.nan_to_num(0.0)) ** 2) * v).sum() / v.sum().clamp(min=1.0)

class SICObsCost4DV(_t.nn.Module):
    def __init__(self, pool=16, w_osi=1.0):
        super().__init__(); self.pool, self.w_osi = pool, w_osi
    def forward(self, state, y, m):
        C = state.shape[1]
        sar, osi, ms, mo = y[:, :C], y[:, C:], m[:, :C], m[:, C:]
        j_sar = (((state - sar) ** 2) * ms).sum() / ms.sum().clamp(min=1.0)
        P = lambda z: _F.avg_pool2d(z, self.pool, ceil_mode=True)
        cnt = P(mo)
        xs = P(state * mo) / cnt.clamp(min=1e-6)
        os_ = P(osi * mo) / cnt.clamp(min=1e-6)
        v = (cnt > 0.5).to(state.dtype)
        j_osi = (((xs - os_) ** 2) * v).sum() / v.sum().clamp(min=1.0)
        return j_sar + self.w_osi * j_osi
'''

def patch(name, rules, helper_before):
    p = os.path.join(NB, name); nb = json.load(open(p)); n = 0
    if not any(MARK in "".join(c["source"]) for c in nb["cells"]):
        i = next(i for i, c in enumerate(nb["cells"]) if helper_before in "".join(c["source"]))
        nb["cells"].insert(i, {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                               "source": HELPERS.splitlines(keepends=True)})
    for c in nb["cells"]:
        s = "".join(c["source"])
        for a, b in rules:
            if a in s:
                s = s.replace(a, b); n += 1
        c["source"] = s.splitlines(keepends=True)
    json.dump(nb, open(p, "w"), indent=1, ensure_ascii=False); open(p, "a").write("\n")
    print(name, "replacements:", n)

UNET = [
    ("    def _prepare_inputs(self, y: Tensor) -> Tuple[Tensor, Tensor]:\n        mask     = y.isfinite().to(y.dtype)\n        y_filled = y.nan_to_num(0.0)\n",
     "    def _prepare_inputs(self, batch) -> Tuple[Tensor, Tensor]:\n        y_filled, mask = sic_fuse(batch.input, batch.osisaf)   # (rebuttal) SAR + OSI SAF\n"),
    ("self._prepare_inputs(batch.input)", "self._prepare_inputs(batch)"),
    ("loss  = F.mse_loss(x_hat, batch.tgt)", "loss  = masked_mse(x_hat, batch.tgt)"),
    ("mask     = y.isfinite().to(dtype)\ny_filled = y.nan_to_num(0.0)",
     "y_filled, mask = sic_fuse(y, batch.osisaf.to(device=device, dtype=dtype))"),
    ("    _mask_b     = _y_b.isfinite().to(dtype)\n    _y_filled_b = _y_b.nan_to_num(0.0)",
     "    _y_filled_b, _mask_b = sic_fuse(_y_b, _tb.osisaf.to(device=device, dtype=dtype))"),
]
FDV = [
    ("    @staticmethod\n    def _prepare_inputs(y: Tensor) -> Tuple[Tensor, Tensor]:\n        mask     = y.isfinite().to(y.dtype)\n        y_filled = y.nan_to_num(0.0)\n        return y_filled, mask\n",
     "    @staticmethod\n    def _prepare_inputs(batch) -> Tuple[Tensor, Tensor]:\n        return sic_pack(batch.input, batch.osisaf)   # (rebuttal) cat(SAR, OSI SAF) + masks\n"),
    ("self._prepare_inputs(batch.input)", "self._prepare_inputs(batch)"),
    ("loss = sum(w * F.mse_loss(xk, tgt) for xk, w in zip(iterates, weights))",
     "loss = sum(w * masked_mse(xk, tgt) for xk, w in zip(iterates, weights))"),
    ("loss = F.mse_loss(self.solver(y_filled, mask), tgt)", "loss = masked_mse(self.solver(y_filled, mask), tgt)"),
    ("loss = F.mse_loss(x_final, batch.tgt)", "loss = masked_mse(x_final, batch.tgt)"),
    ("        x = y_filled.clone()\n", "        x = sic_init_state(y_filled, mask)   # (rebuttal) fused SAR/OSI SAF\n"),
    ("obs_cost   = ObsCost(w=1.0)", "obs_cost   = SICObsCost4DV(pool=16, w_osi=1.0)   # (rebuttal) SAR + OSI SAF, as VarCM"),
    ("mask     = y.isfinite().to(dtype)\ny_filled = y.nan_to_num(0.0)",
     "y_filled, mask = sic_pack(y, batch.osisaf.to(device=device, dtype=dtype))"),
    ("    _mask_b     = _y_b.isfinite().to(dtype)\n    _y_filled_b = _y_b.nan_to_num(0.0)",
     "    _y_filled_b, _mask_b = sic_pack(_y_b, _tb.osisaf.to(device=device, dtype=dtype))"),
]

# the SIC items have no `tgt` field: the reference is `asip`
for _r in (UNET, FDV):
    _r += [("batch.tgt", "batch.asip"), ("_tb.tgt", "_tb.asip")]

if __name__ == "__main__":
    patch("Notebook_direct_unet_sic.ipynb", UNET, "class LitDirectUNet")
    patch("Notebook_4dvarnet_unet_sic.ipynb", FDV, "class ObsCost")
