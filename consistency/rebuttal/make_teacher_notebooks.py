#!/usr/bin/env python3
"""Build the corrected (ODE-teacher target) notebooks from the submitted ones, by text substitution.

    python consistency/rebuttal/make_teacher_notebooks.py

CM -> CMT for GP and SSH_GF (method `CMT` in methods.yaml), VarCM -> VarCMT and DynCM -> DynCMT for GP.
Changes, and nothing else:
  * training object: TeacherPairwiseConsistencyTraining with the trained FM (logs_FM_spde/best_model) as teacher;
  * EMA of the target network 0.999 (CTM-style) instead of the CT schedule's 0.95;
  * new LOG_DIR (logs_CMT_spde), so the submitted checkpoints are never overwritten;
  * WARM_START (parameters cell, default True): initialise from the submitted CM checkpoint when no CMT checkpoint
    exists yet, so that the fine-tuning budget (MAX_EPOCHS in methods.yaml) can be small.
The sampler, the network and the metrics cells are those of the submitted notebook.
"""
import json
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "Notebooks"

SPECS = [
    dict(src=NB / "Notebooks_GP" / "Notebook_consistency_model_CM_spde.ipynb",
         dst=NB / "Notebooks_GP" / "Notebook_consistency_model_CMT_spde.ipynb",
         old_log="logs_CT_CM_spde", new_log="logs_CMT_spde", fm_log="logs_FM_spde",
         ct_line="ct = ConsistencyTrainingFewSteps_TimeEmbedding(final_timesteps=17)",
         lit_cfg="LitConsistencyModelConfig(lr_scheduler_iters=1000),\n)",
         lit_name="lit_cm", label="CM_spde", new_label="CMT_spde"),
    dict(src=NB / "Notebooks_SSH_GF" / "Notebook_consistency_model_CM_ssh_gf.ipynb",
         dst=NB / "Notebooks_SSH_GF" / "Notebook_consistency_model_CMT_ssh_gf.ipynb",
         old_log="logs_CT_CM_ssh_gf", new_log="logs_CMT_ssh_gf", fm_log="logs_FM_ssh_gf",
         ct_line="ct = ConsistencyTrainingFewSteps_TimeEmbedding(\n    final_timesteps=17, sigma_max=10, pairwise=PAIRWISE,\n)",
         lit_cfg="LitConsistencyModelConfig(lr_scheduler_iters=1000),\n)",
         lit_name="lit_cm", label="CM_ssh_gf", new_label="CMT_ssh_gf"),
]


def sub(src, old, new, what):
    if src.count(old) != 1:
        raise SystemExit(f"anchor for {what!r} found {src.count(old)} times")
    return src.replace(old, new)


def build(spec):
    nb = json.load(open(spec["src"]))
    cells = nb["cells"]
    done = set()
    for c in cells:
        if c["cell_type"] != "code":
            continue
        s = "".join(c["source"])
        if s.lstrip().startswith("# --- parameters"):
            s += ("WARM_START = True        # CMT: start from the submitted checkpoint when no CMT checkpoint exists\n"
                  "TEACHER_HEUN_SUBSTEPS = 1\n"
                  "P_LOCAL = 0.25          # probability of the local-anchor pair (k = n-1)\n")
            done.add("params")
        if "from consistency_models.utils import update_ema_model_" in s:
            s = sub(s, "from consistency_models.utils import update_ema_model_",
                    "from consistency_models.utils import update_ema_model_\n"
                    "from consistency_models.teacher_targets import FMTeacherVE, TeacherPairwiseConsistencyTraining\n"
                    "from flowmatching_models.flowmatching_models_FM import UNetFM", "imports")
            done.add("imports")
        if spec["ct_line"] in s:
            s = sub(s, f"'{spec['old_log']}'", f"'{spec['new_log']}'", "LOG_DIR")
            s = sub(s, spec["ct_line"],
                    f"_fm_teacher = UNetFM.from_pretrained(os.path.join('{spec['fm_log']}', 'best_model')).cuda().eval()\n"
                    "ct = TeacherPairwiseConsistencyTraining(\n"
                    "    FMTeacherVE(_fm_teacher, n_sub=TEACHER_HEUN_SUBSTEPS), num_timesteps=18, p_local=P_LOCAL)\n"
                    "print('[CMT] ODE-teacher targets from', os.path.join(" f"'{spec['fm_log']}'" ", 'best_model'))",
                    "ct")
            s = sub(s, spec["lit_cfg"],
                    "LitConsistencyModelConfig(lr_scheduler_iters=1000, initial_ema_decay_rate=0.999),\n)\n"
                    f"if WARM_START and resume_ckpt is None and not SKIP_TRAINING:\n"
                    f"    _old = find_best_valid_checkpoint(os.path.join('{spec['old_log']}', 'checkpoints'))\n"
                    f"    if _old is None:\n"
                    f"        raise RuntimeError('WARM_START=True but no submitted checkpoint in {spec['old_log']}')\n"
                    f"    {spec['lit_name']}.load_state_dict(torch.load(_old, map_location='cpu')['state_dict'])\n"
                    f"    print(f'[CMT] warm start from {{_old}}', flush=True)",
                    "lit config")
            done.add("training")
        if "write_calibration_csv(_calib_csv," in s:
            s = s.replace(f"write_calibration_csv(_calib_csv, '{spec['label']}'",
                          f"write_calibration_csv(_calib_csv, '{spec['new_label']}'")
            s = s.replace("'CM (ens. mean, full test set)'", "'CMT (ens. mean, full test set)'")
        c["source"] = s.splitlines(keepends=True)
    missing = {"params", "imports", "training"} - done
    if missing:
        raise SystemExit(f"{spec['src'].name}: anchors not found for {missing}")
    with open(spec["dst"], "w") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(f"wrote {spec['dst'].relative_to(NB.parent)}")


def build_varcmt():
    """GP VarCM -> VarCMT: same training plus the tangent condition (lambda_tan), warm-started from VarCM."""
    src = NB / "Notebooks_GP" / "Notebook_consistency_model_VarCM_spde.ipynb"
    dst = NB / "Notebooks_GP" / "Notebook_consistency_model_VarCMT_spde.ipynb"
    nb = json.load(open(src))
    done = set()
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        s = "".join(c["source"])
        if s.lstrip().startswith("# --- parameters"):
            s += ("WARM_START = True        # VarCMT: start from the submitted VarCM checkpoint\n"
                  "LAMBDA_TAN = 1.0         # weight of the tangent condition (0 = submitted VarCM)\n")
            done.add("params")
        if "\nct = VarCMTraining(" in s:
            s = sub(s, 'LOG_DIR     = f"logs_CT_VarCM_spde_{COND_MODE.replace(\'+\', \'_\')}"',
                    'LOG_DIR     = "logs_VarCMT_spde_grad"', "LOG_DIR")
            s = sub(s, "    conditioning_mode = COND_MODE,\n)",
                    "    conditioning_mode = COND_MODE,\n    lambda_tan        = LAMBDA_TAN,\n)", "lambda_tan")
            s = sub(s, "    LitVarCMConfig(lr_scheduler_iters=1000),\n)",
                    "    LitVarCMConfig(lr_scheduler_iters=1000),\n)\n"
                    "if WARM_START and resume_ckpt is None and not SKIP_TRAINING:\n"
                    "    _old = find_best_valid_checkpoint(os.path.join('logs_CT_VarCM_spde_grad', 'checkpoints'))\n"
                    "    if _old is None:\n"
                    "        raise RuntimeError('WARM_START=True but no submitted checkpoint in logs_CT_VarCM_spde_grad')\n"
                    "    lit_cm.load_state_dict(torch.load(_old, map_location='cpu')['state_dict'])\n"
                    "    print(f'[VarCMT] warm start from {_old}', flush=True)", "warm start")
            done.add("training")
        if "write_calibration_csv(_calib_csv," in s:
            s = s.replace("write_calibration_csv(_calib_csv, 'VarCM_spde'", "write_calibration_csv(_calib_csv, 'VarCMT_spde'")
        c["source"] = s.splitlines(keepends=True)
    if {"params", "training"} - done:
        raise SystemExit(f"VarCMT: anchors not found for {({'params', 'training'} - done)}")
    with open(dst, "w") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(f"wrote {dst.relative_to(NB.parent)}")


def build_dyncmt():
    """GP DynCM -> DynCMT: DynFM-teacher targets (TeacherDynamicalConsistencyTraining), no data anchors,
    EMA 0.999, warm start from the submitted DynCM."""
    src = NB / "Notebooks_GP" / "Notebook_consistency_model_DynCM_spde.ipynb"
    dst = NB / "Notebooks_GP" / "Notebook_consistency_model_DynCMT_spde.ipynb"
    nb = json.load(open(src))
    done = set()
    old_ct = """ct = ConsistencyTrainingDynamicalSystems(
    sigma_min=0.002,
    sigma_max=10.0,
    rho=7.0,
    initial_timesteps=5,
    final_timesteps=50,
)"""
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        s = "".join(c["source"])
        if s.lstrip().startswith("# --- parameters"):
            s += ("WARM_START = True        # DynCMT: start from the submitted DynCM checkpoint\n"
                  "P_LOCAL = 0.25\n"
                  "N_GRID = 64             # fixed training grid (sampler grid is NSTEPS)\n")
            done.add("params")
        if "from consistency_models.utils import update_ema_model_" in s:
            s = sub(s, "from consistency_models.utils import update_ema_model_",
                    "from consistency_models.utils import update_ema_model_\n"
                    "from consistency_models.teacher_targets import DynFMTeacher, TeacherDynamicalConsistencyTraining\n"
                    "from flowmatching_models.flowmatching_models_FM import UNetFM", "imports")
            done.add("imports")
        if old_ct in s:
            s = sub(s, 'LOG_DIR     = "logs_CT_DynCM_spde"', 'LOG_DIR     = "logs_DynCMT_spde"', "LOG_DIR")
            s = sub(s, old_ct,
                    "_dynfm = UNetFM.from_pretrained(os.path.join('logs_DynFM_spde', 'best_model')).cuda().eval()\n"
                    "ct = TeacherDynamicalConsistencyTraining(\n"
                    "    DynFMTeacher(_dynfm, spinup_boundary=0.7, sigma_min=0.002, sigma_max=10.0),\n"
                    "    num_timesteps=N_GRID, p_local=P_LOCAL, sigma_min=0.002, sigma_max=10.0, rho=7.0)\n"
                    "print('[DynCMT] DynFM-teacher targets from logs_DynFM_spde/best_model')", "ct")
            s = sub(s, "        lambda_obs=0.1,\n    ),\n)",
                    "        lambda_obs=0.0,              # DynCMT: no L2 data anchors (they pull towards the mean)\n"
                    "        lambda_IC=0.0,\n"
                    "        initial_ema_decay_rate=0.999,\n    ),\n)\n"
                    "if WARM_START and resume_ckpt is None and not SKIP_TRAINING:\n"
                    "    _old = find_resume_ckpt(os.path.join('logs_CT_DynCM_spde', 'checkpoints'))\n"
                    "    if _old is None:\n"
                    "        raise RuntimeError('WARM_START=True but no submitted checkpoint in logs_CT_DynCM_spde')\n"
                    "    lit_cm.load_state_dict(torch.load(_old, map_location='cpu')['state_dict'])\n"
                    "    print(f'[DynCMT] warm start from {_old}', flush=True)", "lit config")
            done.add("training")
        if "write_calibration_csv(_calib_csv," in s:
            s = s.replace("write_calibration_csv(_calib_csv, 'DynCM_spde'", "write_calibration_csv(_calib_csv, 'DynCMT_spde'")
        c["source"] = s.splitlines(keepends=True)
    if {"params", "imports", "training"} - done:
        raise SystemExit(f"DynCMT: anchors not found for {({'params', 'imports', 'training'} - done)}")
    with open(dst, "w") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(f"wrote {dst.relative_to(NB.parent)}")


def build_varcrps(xp_dir="Notebooks_GP", suffix="spde", old_log="logs_CT_VarCM_spde_grad"):
    """VarCM -> VarCRPS: teacher-free posterior sampler. Same transport and self-consistency terms as VarCM, but
    the data anchors (L_long, L_obs) are the fair CRPS of CRPS_MEMBERS draws of x_T per (x0, y) instead of a
    pseudo-Huber regression on x0 (whose optimum is the conditional mean). Fixed time grid (no curriculum
    restart), warm start from the submitted VarCM."""
    src = NB / xp_dir / f"Notebook_consistency_model_VarCM_{suffix}.ipynb"
    dst = NB / xp_dir / f"Notebook_consistency_model_VarCRPS_{suffix}.ipynb"
    nb = json.load(open(src))
    done = set()
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        s = "".join(c["source"])
        if s.lstrip().startswith("# --- parameters"):
            s += ("WARM_START = True        # VarCRPS: start from the submitted VarCM checkpoint\n"
                  "CRPS_MEMBERS = 8         # draws of x_T per (x0, y) in the fair-CRPS anchor\n")
            done.add("params")
        if "\nct = VarCMTraining(" in s:
            i = s.index("LOG_DIR     = f")
            j = s.index("\n", i)
            s = s[:i] + f'LOG_DIR     = "logs_VarCRPS_{suffix}_grad"' + s[j:]
            s = sub(s, "    initial_timesteps = 5,\n", "    initial_timesteps = 50,   # VarCRPS: fixed grid\n", "grid")
            s = sub(s, "    conditioning_mode = COND_MODE,\n)",
                    "    conditioning_mode = COND_MODE,\n    crps_members      = CRPS_MEMBERS,\n)", "crps")
            s = sub(s, "    LitVarCMConfig(lr_scheduler_iters=1000),\n)",
                    "    LitVarCMConfig(lr_scheduler_iters=1000),\n)\n"
                    "if WARM_START and resume_ckpt is None and not SKIP_TRAINING:\n"
                    f"    _old = find_best_valid_checkpoint(os.path.join('{old_log}', 'checkpoints'))\n"
                    "    if _old is None:\n"
                    f"        raise RuntimeError('WARM_START=True but no submitted checkpoint in {old_log}')\n"
                    "    lit_cm.load_state_dict(torch.load(_old, map_location='cpu')['state_dict'])\n"
                    "    print(f'[VarCRPS] warm start from {_old}', flush=True)", "warm start")
            done.add("training")
        if "write_calibration_csv(_calib_csv," in s:
            s = s.replace(f"write_calibration_csv(_calib_csv, 'VarCM_{suffix}'",
                          f"write_calibration_csv(_calib_csv, 'VarCRPS_{suffix}'")
        c["source"] = s.splitlines(keepends=True)
    if {"params", "training"} - done:
        raise SystemExit(f"VarCRPS: anchors not found for {({'params', 'training'} - done)}")
    with open(dst, "w") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(f"wrote {dst.relative_to(NB.parent)}")


if __name__ == "__main__":
    for spec in SPECS:
        build(spec)
    build_varcmt()
    build_dyncmt()
    build_varcrps()
