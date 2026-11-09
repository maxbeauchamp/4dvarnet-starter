#!/usr/bin/env python3
"""Build the corrected (ODE-teacher target) notebooks from the submitted ones, by text substitution.

    python consistency/rebuttal/make_teacher_notebooks.py

Currently: GP CM -> Notebook_consistency_model_CMT_spde.ipynb (method `CMT` in methods.yaml).
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
         lit_name="lit_cm"),
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
            s = s.replace("write_calibration_csv(_calib_csv, 'CM_spde'", "write_calibration_csv(_calib_csv, 'CMT_spde'")
            s = s.replace("'CM (ens. mean, full test set)'", "'CMT (ens. mean, full test set)'")
        c["source"] = s.splitlines(keepends=True)
    missing = {"params", "imports", "training"} - done
    if missing:
        raise SystemExit(f"{spec['src'].name}: anchors not found for {missing}")
    with open(spec["dst"], "w") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(f"wrote {spec['dst'].relative_to(NB.parent)}")


if __name__ == "__main__":
    for spec in SPECS:
        build(spec)
