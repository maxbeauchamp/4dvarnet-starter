"""Rebuttal: SSH teachers (FM, DynFM) with checkpoint selection on the validation CRPS of the ensemble.

The SSH training set is small (one 128x128 patch, Feb-Sep 2013): the submitted teachers were trained for
~2000 epochs on the training loss only, and longer training makes them over-confident on the test period
(FM: SSR 0.27 at ~200 epochs, 0.15 at ~1970). With SELECT_BY_VAL=True the notebook
  * samples a small ensemble (VAL_MEMBERS members, VAL_STEPS Heun steps, EMA network) on VAL_BATCHES validation
    windows every VAL_EVERY epochs and logs the same score as the Var* models (fair CRPS of the pixels and of the
    spatial increments, `crps_field_loss`) as `val_crps`;
  * keeps the 3 best checkpoints on `val_crps` and saves the best one as best_model (training and SKIP_TRAINING).
RUN_TAG gives the run its own LOG_DIR / figure folder. Defaults reproduce the submitted notebooks.
Idempotent (marker below). Usage: python patch_ssh_teachers.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
XP = os.path.join(HERE, "..", "Notebooks", "Notebooks_SSH_GF")
MARK = "(rebuttal) SELECT_BY_VAL"

PARAMS = ("RUN_TAG = ''             # suffix of LOG_DIR / figure folder for a variant run\n"
          "SELECT_BY_VAL = False    # (rebuttal) SELECT_BY_VAL: keep the checkpoint with the best validation CRPS\n"
          "VAL_EVERY = 10           # epochs between two validation-CRPS evaluations\n"
          "VAL_BATCHES = 8          # validation windows per evaluation\n"
          "VAL_MEMBERS = 8          # ensemble members per window\n"
          "VAL_STEPS = {steps}           # Heun steps of the validation sampler\n")

VAL = '''
# ── (rebuttal) validation CRPS of the ensemble, used to select the checkpoint ──
from consistency_models.consistency_models_VarCM import crps_field_loss

def _val_crps_step(self, batch, batch_idx):
    if isinstance(batch, list):
        batch = batch[0]
    M = VAL_MEMBERS
    dtype = next(self.network.parameters()).dtype
    x0, y = batch.tgt.to(dtype=dtype), batch.input.to(dtype=dtype)
    net = self.ema_model.module.to(device=self.device, dtype=dtype).eval()
    _devs = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=_devs), torch.no_grad():
        torch.manual_seed(1234 + batch_idx)
        xr, yr = x0.repeat_interleave(M, 0), y.repeat_interleave(M, 0)
{SAMPLE}
        score = crps_field_loss(ens.float(), x0.float(), M, (1, 2, 4, 8), 1.0)
    self.log("val_crps", score, on_epoch=True, on_step=False, prog_bar=True, batch_size=x0.shape[0])

if SELECT_BY_VAL:
    {LIT}.validation_step = _val_crps_step

'''

SELECT = '''
# (rebuttal) best validation-CRPS checkpoint -> best_model
if SELECT_BY_VAL:
    _vc = [p for p in glob.glob(os.path.join(CKPT_DIR, '*val_crps=*.ckpt')) if is_valid_checkpoint(p)]
    if _vc:
        _best_val = min(_vc, key=lambda p: float(p.split('val_crps=')[1].split('-')[0].replace('.ckpt', '')))
        {OBJ}.load_state_dict(torch.load(_best_val, map_location='cpu')['state_dict'])
        print(f'[SELECT] best validation CRPS checkpoint: {_best_val}', flush=True)
    else:
        print('[SELECT] no val_crps checkpoint found, keeping the current weights', flush=True)

'''

SPECS = {
    "Notebook_flowmatching_model_FM_ssh_gf.ipynb": dict(
        log="logs_FM_ssh_gf", tag="FM_ssh_gf", lit="LitFlowMatchingModel", obj="lit_fm", steps=10,
        sample="        ens, _ = heun_sample(net, torch.randn_like(xr), yr, n_steps=VAL_STEPS)"),
    "Notebook_flowmatching_model_DynFM_ssh_gf.ipynb": dict(
        log="logs_DynFM_ssh_gf", tag="DynFM_ssh_gf", lit="LitDynFM", obj="lit_dynfm", steps=19,
        sample=("        _nz = torch.randn(xr.shape[0], 1, *xr.shape[-2:], device=xr.device, dtype=xr.dtype)\n"
                "        _, _tr, _tt = dynfm_heun_sample(net, _nz, yr, n_steps=VAL_STEPS,\n"
                "                                      spinup_frac=self.sf, precondition=self.precondition)\n"
                "        ens = torch.cat(extract_dynfm_frames(_tr, _tt, x0.shape[1], self.sf), dim=1)")),
}


def sub(s, old, new, what, name):
    if old not in s:
        raise SystemExit(f"{name}: anchor not found ({what}): {old!r}")
    return s.replace(old, new)


def patch(name, sp):
    path = os.path.join(XP, name)
    nb = json.load(open(path))
    if any(MARK in "".join(c["source"]) for c in nb["cells"]):
        print(f"{name}: already patched"); return
    done = set()
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        s = "".join(c["source"])
        if s.lstrip().startswith("# --- parameters"):
            s += PARAMS.format(steps=sp["steps"]); done.add("params")
        if f'LOG_DIR     = "{sp["log"]}"' in s and "trainer = Trainer(" in s:
            s = sub(s, f'LOG_DIR     = "{sp["log"]}"', f'LOG_DIR     = "{sp["log"]}" + RUN_TAG', "log", name)
            s = sub(s, "trainer = Trainer(", VAL.replace("{SAMPLE}", sp["sample"]).replace("{LIT}", sp["lit"])
                    + "trainer = Trainer(", "val", name)
            s = sub(s, "    log_every_n_steps=1,\n",
                    "    log_every_n_steps=1,\n"
                    "    check_val_every_n_epoch=VAL_EVERY if SELECT_BY_VAL else 1,\n"
                    "    limit_val_batches=VAL_BATCHES if SELECT_BY_VAL else 1.0,\n", "trainer", name)
            s = sub(s, '            monitor="train_loss",\n',
                    '            monitor="val_crps" if SELECT_BY_VAL else "train_loss",\n', "monitor", name)
            s = sub(s, '            filename="{epoch:03d}-{step}-{train_loss:.4f}",\n',
                    '            filename="{val_crps:.5f}-{epoch:03d}" if SELECT_BY_VAL else "{epoch:03d}-{step}-{train_loss:.4f}",\n',
                    "filename", name)
            s = sub(s, f"\nema_net = {sp['obj']}.ema_model.module\n",
                    "\n" + SELECT.replace("{OBJ}", sp["obj"]) + f"ema_net = {sp['obj']}.ema_model.module\n", "select", name)
            done.add("training")
        if f'MODEL_PATH = os.path.join("{sp["log"]}", "best_model")' in s:
            s = s.replace(f'MODEL_PATH = os.path.join("{sp["log"]}", "best_model")',
                          f'MODEL_PATH = os.path.join("{sp["log"]}" + RUN_TAG, "best_model")'); done.add("eval")
        if f"FIG_TAG = '{sp['tag']}'" in s:
            s = s.replace(f"FIG_TAG = '{sp['tag']}'", f"FIG_TAG = '{sp['tag']}' + RUN_TAG"); done.add("fig")
        c["source"] = s.splitlines(keepends=True)
    missing = {"params", "training", "eval", "fig"} - done
    if missing:
        raise SystemExit(f"{name}: anchors not found for {missing}")
    json.dump(nb, open(path, "w"), indent=1, ensure_ascii=False)
    print(f"{name}: patched")


if __name__ == "__main__":
    for n, sp in SPECS.items():
        patch(n, sp)
