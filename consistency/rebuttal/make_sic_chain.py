"""(rebuttal) SIC notebooks of DynFM, DynCM (DynCMT), VarCM and DynVarCM with the recipes of SSH.

The SIC notebooks of these four methods were generated from the GP ones and had never been run: their code reads
batch.tgt / batch.input, while the SIC data module yields (asip, osisaf, input). This script
  1. applies the same rebuttal builders as for GP/SSH (DynCMT, VarCM CRPS, VarDynCM CRPS; DynFM SELECT_BY_VAL);
  2. adapts them to the SIC data (semi-supervised: the target `asip` has gaps, every data term is masked):
       * batches expose tgt = asip (NaN gaps) and input =
           - DynFM, DynCM : fuse(sar, osisaf) (SAR where observed, OSI SAF elsewhere), C channels, so that the
                            frame-indexed networks and samplers of SSH are unchanged;
           - VarCM, DynVarCM: cat(sar, osisaf), 2C channels, split by the SIC observation costs
                            J_o = SAR misfit + OSI SAF misfit at the coarse scale (consistency_models/sic_costs.py);
       * DynFM: loss masked where the velocity target is unobserved;
       * DynCM: zero-weight data terms no longer turn the loss into NaN (gappy targets);
       * UNetConfig(..., cond_channels=2*C) of the generated notebooks (no such field) -> UNetConfig(channels=C);
       * batch size 1 for VarCM / DynVarCM (as SSH; 240x240 fields x 8 members).
Idempotent: re-running rebuilds from the kept submitted (_v1) notebooks. Usage: python make_sic_chain.py
"""
import importlib.util
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
NB = os.path.join(HERE, "..", "Notebooks", "Notebooks_SIC")
MARK = "(rebuttal) SIC adapter"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, f"{name}.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ADAPTER = '''
# ── {mark}: batches with tgt = asip (gaps) and input = {what} ──
def _to_sic_item(it):
    # keep the TrainingItem_4da type: XrDataset then adds tgt = asip; only `input` changes
    sar, osi = it.input, it.osisaf
    inp = {expr}
    return it._replace(input=inp)
'''
EXPR = {"fused": "np.where(np.isnan(sar), osi, sar).astype(np.float32)",
        "cat": "np.concatenate([sar, osi], axis=0).astype(np.float32)"}
WHAT = {"fused": "fuse(SAR, OSI SAF), C channels", "cat": "cat(SAR, OSI SAF), 2C channels"}
BATCH1 = '''
    # (rebuttal) batch size 1, as SSH (240x240 fields, ensembles of 8 members in training)
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_ds, shuffle=True, batch_size=1, num_workers=1)
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_ds, shuffle=False, batch_size=1, num_workers=1)
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_ds, shuffle=False, batch_size=1, num_workers=1)
'''
PARAMS = ("OSI_POOL = 16            # (rebuttal, SIC) block size of the coarse OSI SAF term of J_o (pixels)\n"
          "OSI_WEIGHT = 1.0         # (rebuttal, SIC) weight of the OSI SAF term of J_o\n")


def sub(s, old, new, what, name, count=1):
    n = s.count(old)
    if n < 1 or (count and n != count):
        raise SystemExit(f"{name}: anchor {what!r} found {n} times: {old[:90]!r}")
    return s.replace(old, new)


def adapt(name, mode, batch1=False, extra=None):
    path = os.path.join(NB, name)
    nb = json.load(open(path))
    if any(MARK in "".join(c["source"]) for c in nb["cells"]):
        print(f"{name}: already adapted"); return
    done = set()
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        s = "".join(c["source"])
        if s.lstrip().startswith("# --- parameters"):
            s = s.rstrip("\n") + "\n" + PARAMS; done.add("params")
        if "class CroppedDataModule(BaseDataModule):" in s and "TrainingItem_4da._make," in s:
            s = sub(s, "class CroppedDataModule(BaseDataModule):",
                    ADAPTER.format(mark=MARK, what=WHAT[mode], expr=EXPR[mode]) + "\nclass CroppedDataModule(BaseDataModule):",
                    "adapter", name)
            s = sub(s, "TrainingItem_4da._make,\n", "TrainingItem_4da._make,\n            _to_sic_item,\n", "build_batch", name)
            if batch1:
                s = sub(s, "    def setup(self, stage='test'):", BATCH1 + "\n    def setup(self, stage='test'):", "batch1", name)
            done.add("data")
        if "cond_channels=2*C" in s:
            s = s.replace("UNetConfig(channels=C, cond_channels=2*C)", "UNetConfig(channels=C)")
        if extra:
            s = extra(s, name, done)
        c["source"] = s.splitlines(keepends=True)
    if {"params", "data"} - done:
        raise SystemExit(f"{name}: anchors not found for {({'params', 'data'} - done)}")
    json.dump(nb, open(path, "w"), indent=1, ensure_ascii=False)
    with open(path, "a") as f:
        f.write("\n")
    print(f"{name}: adapted ({mode})")


def dynfm_extra(s, name, done):
    if "def training_step(self, batch, batch_idx):" in s and "x_full = batch.tgt" in s:
        s = sub(s, "        x_full = batch.tgt\n",
                "        x_raw  = batch.tgt\n        x_full = torch.nan_to_num(x_raw)   # (rebuttal, SIC) gappy target\n",
                "x_full", name)
        s = sub(s, "        x_t, v_target, c_out = self._build_interpolant(x_full, t, noise)\n",
                "        x_t, v_target, c_out = self._build_interpolant(x_full, t, noise)\n"
                "        _valid = torch.isfinite(self._build_interpolant(x_raw, t, noise)[1])   # observed velocity targets\n",
                "interp", name)
        s = sub(s, "            loss = neglogpdf(residual, log_s).mean()\n",
                "            loss = (neglogpdf(residual, log_s) * _valid).sum() / _valid.sum().clamp(min=1)\n", "loss", name)
        s = sub(s, "            loss = F.mse_loss(v_pred, v_target)\n",
                "            loss = ((v_pred - v_target).pow(2) * _valid).sum() / _valid.sum().clamp(min=1)\n", "mse", name)
        done.add("loss")
    return s


def dyncm_extra(s, name, done):
    old = "        loss = loss_cm + self.config.lambda_obs * loss_obs + self.config.lambda_IC * loss_IC\n"
    if old in s:
        s = sub(s, old,
                "        # (rebuttal, SIC) zero-weight terms are skipped: with gappy targets they may be NaN (0 * NaN = NaN)\n"
                "        loss = (loss_cm + (self.config.lambda_obs * loss_obs if self.config.lambda_obs else 0.0)\n"
                "                + (self.config.lambda_IC * loss_IC if self.config.lambda_IC else 0.0))\n", "loss", name)
        done.add("loss")
    return s


def varcm_extra(s, name, done):
    if "BaseObsCost(sigma_obs=1.0)" in s:
        s = s.replace("BaseObsCost(sigma_obs=1.0)",
                      "SICObsCost(n_frames=C, pool=OSI_POOL, w_osi=OSI_WEIGHT)")
        s = "from consistency_models.sic_costs import SICObsCost\n" + s
        done.add("cost")
    if "_oc = BaseObsCost();" in s:          # sanity-check cell of the GP notebook: SIC cost, gradient conditioning
        s = s.replace("_oc = BaseObsCost();", "_oc = SICObsCost(n_frames=C, pool=OSI_POOL);")
        s = s.replace('sigma_noise=80.0, conditioning_mode="obs",', 'sigma_noise=80.0, conditioning_mode="grad",')
        s = "from consistency_models.sic_costs import SICObsCost\n" + s
    return s


def vardyncm_extra(s, name, done):
    if "VarDynCMObsCost(sigma_obs=1.0, spinup_boundary=SPINUP_BOUNDARY)" in s:
        s = s.replace("VarDynCMObsCost(sigma_obs=1.0, spinup_boundary=SPINUP_BOUNDARY)",
                      "SICFrameObsCost(n_frames=C, spinup_boundary=SPINUP_BOUNDARY, pool=OSI_POOL, w_osi=OSI_WEIGHT)")
        s = ("from consistency_models.sic_costs import SICFrameObsCost\n"
             "from consistency_models.consistency_models_VarDynCM import VarDynCMSamplingAndEditing as _VDS\n"
             "_VDS.n_frames = C   # (rebuttal, SIC) y carries 2C channels: frame count for the sampler\n") + s
        done.add("cost")
    if "        B, C, H, W = y.shape\n        mask_y = (~torch.isnan(y))" in s and "grad_cond" in s:
        s = sub(s, "        B, C, H, W = y.shape\n        mask_y = (~torch.isnan(y))",
                "        if y.shape[1] == 2 * self.config.channels:   # (rebuttal, SIC) cat(sar, osisaf) -> fused field\n"
                "            _C = self.config.channels\n"
                "            y = torch.where(torch.isnan(y[:, :_C]), y[:, _C:], y[:, :_C])\n"
                "        B, C, H, W = y.shape\n        mask_y = (~torch.isnan(y))", "unet fuse", name)
        done.add("unet")
    return s


def fix_cmt_norm(name="Notebook_consistency_model_CMT_sic.ipynb"):
    """CM SIC must use the normalisation of its FM teacher (raw SIC in [0, 1]): the z-score of the generated CM
    notebook fed the teacher states on another scale (wrong ODE steps -> collapsed CM, SSR 0.08)."""
    path = os.path.join(NB, name)
    nb = json.load(open(path))
    anchor = "SIC_STD  = float(np.sqrt(_sqsum / _count - SIC_MEAN ** 2))\n"
    new = (anchor + "if not SIC_ZSCORE:   # (rebuttal) same normalisation as the FM teacher: raw SIC in [0, 1]\n"
           "    print(f'(z-score disabled; train stats were mean={SIC_MEAN:.4f} std={SIC_STD:.4f})')\n"
           "    SIC_MEAN, SIC_STD = 0.0, 1.0\n")
    done = set()
    for c in nb["cells"]:
        s = "".join(c["source"])
        if s.lstrip().startswith("# --- parameters") and "SIC_ZSCORE" not in s:
            s = s.rstrip("\n") + "\nSIC_ZSCORE = False       # (rebuttal) z-score SIC (False: raw [0, 1], as the FM teacher)\n"
            done.add("param")
        if anchor in s and "SIC_ZSCORE" not in s:
            s = s.replace(anchor, new); done.add("norm")
        c["source"] = s.splitlines(keepends=True)
    if done:
        json.dump(nb, open(path, "w"), indent=1, ensure_ascii=False)
        with open(path, "a") as f:
            f.write("\n")
    print(f"{name}: normalisation fix {sorted(done) or 'already applied'}")


if __name__ == "__main__":
    mt = _load("make_teacher_notebooks")
    mt.build_dyncmt("Notebooks_SIC", "sic", "logs_DynFM_sic_v2", False)
    mt.build_varcm_crps("Notebooks_SIC", "sic")
    mt.build_vardyncm_crps("Notebooks_SIC", "sic")
    pt = _load("patch_ssh_teachers")
    pt.patch(("Notebooks_SIC", "Notebook_flowmatching_model_DynFM_sic.ipynb"), dict(
        log="logs_DynFM_sic", tag="DynFM_sic", lit="LitDynFM", obj="lit_dynfm", steps=19,
        sample=pt.SPECS[("Notebooks_SSH_GF", "Notebook_flowmatching_model_DynFM_ssh_gf.ipynb")]["sample"]))
    adapt("Notebook_flowmatching_model_DynFM_sic.ipynb", "fused", extra=dynfm_extra)
    adapt("Notebook_consistency_model_DynCMT_sic.ipynb", "fused", extra=dyncm_extra)
    adapt("Notebook_consistency_model_VarCM_sic.ipynb", "cat", batch1=True, extra=varcm_extra)
    adapt("Notebook_consistency_model_VarDynCM_sic.ipynb", "cat", batch1=True, extra=vardyncm_extra)
    fix_cmt_norm()
    subprocess.run([sys.executable, os.path.join(HERE, "patch_max_time.py")], cwd=os.path.join(HERE, ".."), check=True)
