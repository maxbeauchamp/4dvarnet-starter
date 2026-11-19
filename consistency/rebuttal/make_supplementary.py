"""Build the anonymised supplementary-material archive of the paper.

    python consistency/rebuttal/make_supplementary.py [OUT_DIR]      (default: ../supplementary)

It copies only what reproduces the tables and figures of the paper, with the same layout as this repository so
that the relative data paths of the notebooks keep working:

    data/                        the three NetCDF files (not copied: see README, DATA below)
    src/                         data loaders (GP, SSH, SIC)
    consistency/consistency_models, flowmatching_models, calibration.py, spectral_utils.py
    consistency/Notebooks/...    the notebooks of the methods in METHODS, outputs stripped
    consistency/sbatch_submission/run_training.py, run_metrics.py, methods.yaml (pruned to METHODS)
    consistency/rebuttal/*_fig.py  figure scripts
    environment.yaml, README.md

Edit METHODS when the final runs change. The script fails if an identifying string is left in a copied file.
"""
import json, os, re, shutil, sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT.parent / "supplementary"

# paper row -> methods.yaml key, per benchmark (keep in sync with Tables 2-3)
METHODS = {
    "GP": {"FM": "FM", "CM": "CMT", "VarCM": "VarCM_scratch2_student", "DynFM": "DynFM",
           "DynCM": "DynCMT", "DynVarCM": "VarDynCM_grid", "UNet": "UNet", "4DVarNet": "4dvarnet_lstm"},
    "SSH_GF": {"FM": "FM_aug", "CM": "CMT_aug", "VarCM": "VarCM_aug_student", "DynFM": "DynFM_v2",
               "DynCM": "DynCMT", "DynVarCM": "VarDynCM_scratch", "UNet": "UNet", "4DVarNet": "4dvarnet_lstm"},
    "SIC": {"FM": "FM_v3", "CM": "CMT_v3", "VarCM": "VarCM_aug_student", "DynFM": "DynFM_v3",
            "DynCM": "DynCMT_v3", "DynVarCM": "VarDynCM_grid_sic", "UNet": "UNet", "4DVarNet": "4dvarnet_lstm"},
}
# methods needed to build another one (teachers, warm starts)
DEPENDS = {"VarDynCM_grid": ["VarDynCM_scratch"], "VarDynCM_grid_sic": ["VarDynCM_scratch"]}
DATA = {
    "GP": "SPDE_diffusion_dataset_1.nc",
    "SSH_GF": "natl_gf_w_5nadirs_swot.nc",
    "SIC": "asip_database_daw15_sparse.nc",
}
FORBIDDEN = re.compile(r"conversation|reviewers? [A-Za-z]{4}\b|\b(XETn|qyTj|aayn)\b|m19beauc|[Bb]eauchamp|[Ff]ablet|Odyssey|imt-atlantique|cmcc\.it|/Users/|/home/\w+", re.I)
COPY_TREES = ["src", "consistency/consistency_models", "consistency/flowmatching_models"]
COPY_FILES = ["environment.yaml", "consistency/calibration.py", "consistency/spectral_utils.py",
              "consistency/sbatch_submission/run_training.py", "consistency/sbatch_submission/run_metrics.py",
              "consistency/rebuttal/sic_costs.py" if (ROOT / "consistency/rebuttal/sic_costs.py").exists() else None,
              "consistency/rebuttal/sic_fig.py", "consistency/rebuttal/ssh_transport_fig.py",
              "consistency/rebuttal/toy_gaussian_targets.py", "consistency/sbatch_submission/run_nfe_sweep.py",
              "consistency/rebuttal/gp_nfe_fig.py", "consistency/rebuttal/ssh_nfe_fig.py",
              "consistency/rebuttal/gp_exact_posterior.py"]


def strip_notebook(src, dst):
    nb = json.load(open(src))
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            c["outputs"], c["execution_count"] = [], None
        c.get("metadata", {}).pop("execution", None)
    nb.get("metadata", {}).pop("papermill", None)
    dst.parent.mkdir(parents=True, exist_ok=True)
    json.dump(nb, open(dst, "w"), indent=1, ensure_ascii=False)


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    ign = shutil.ignore_patterns("__pycache__", "*.pyc", "*.ckpt", "logs_*", "lightning_logs")
    for d in COPY_TREES:
        shutil.copytree(ROOT / d, OUT / d, ignore=ign)
    for f in filter(None, COPY_FILES):
        if (ROOT / f).exists():
            (OUT / f).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / f, OUT / f)
    if (OUT / "consistency/rebuttal").exists():
        (OUT / "consistency/rebuttal").rename(OUT / "consistency/scripts")
    cfg = yaml.safe_load(open(ROOT / "consistency/sbatch_submission/methods.yaml"))
    keys = set()
    for xp, rows in METHODS.items():
        for k in rows.values():
            keys.add(k); keys.update(DEPENDS.get(k, []))
    pruned, table = {}, []
    for k in sorted(keys):
        m = cfg["methods"][k] if "methods" in cfg else cfg[k]
        m = dict(m); m["xp"] = {xp: v for xp, v in (m.get("xp") or {}).items()
                               if xp in METHODS and (k in METHODS[xp].values() or
                                                     any(k in DEPENDS.get(x, []) for x in METHODS[xp].values()))}
        if not m["xp"]:
            continue
        pruned[k] = m
        for xp, v in m["xp"].items():
            nb = ROOT / "consistency/Notebooks" / {"GP": "Notebooks_GP", "SSH_GF": "Notebooks_SSH_GF",
                                                   "SIC": "Notebooks_SIC"}[xp] / v["notebook"]
            strip_notebook(nb, OUT / nb.relative_to(ROOT))
    out_cfg = {"methods": pruned} if "methods" in cfg else pruned
    for k, v in cfg.items():
        if k != "methods" and "methods" in cfg:
            out_cfg[k] = v
    (OUT / "consistency/sbatch_submission").mkdir(parents=True, exist_ok=True)
    yaml.safe_dump(out_cfg, open(OUT / "consistency/sbatch_submission/methods.yaml", "w"), sort_keys=False)
    for xp, rows in METHODS.items():
        for row, k in rows.items():
            if k in pruned:
                table.append(f"| {xp.replace('_GF', '')} | {row} | `{k}` | `{pruned[k]['xp'][xp]['notebook']}` |")
    (OUT / "data").mkdir(exist_ok=True)
    readme = README.format(table="\n".join(table),
                           data="\n".join(f"* `data/{f}` ({xp.replace('_GF', '')})" for xp, f in DATA.items()))
    (OUT / "README.md").write_text(readme)
    for p in OUT.rglob("*"):   # neutral wording for cluster-specific comments
        if p.is_file() and p.suffix in {".py", ".ipynb", ".yaml", ".md"}:
            s = p.read_text(errors="ignore")
            s2 = re.sub(r"odyssey/plan\.json", "the run plan", s)
            s2 = re.sub(r"\bOdyssey\b", "the cluster", s2)
            s2 = re.sub(r" ?added for the ICLR rebuttal \(reviewers[^)]*\)", "", s2)
            s2 = re.sub(r" ?\(ICLR rebuttal[^)]*\)", "", s2)
            s2 = re.sub(r" ?\(rebuttal run [^)]*\)", "", s2)
            s2 = s2.replace(" of the rebuttal", "")
            s2 = re.sub(r"\(rebuttal, ?", "(", s2)
            s2 = re.sub(r" ?\(rebuttal\) ?", " ", s2)
            s2 = s2.replace("in this\\n\",\n    \"        # conversation)", "in this\\n\",\n    \"        # version)")
            s2 = s2.replace("this conversation", "this version")
            s2 = re.sub(r"ICLR rebuttal \([A-Za-z]{4}[^)]*\):? ?", "", s2)
            s2 = re.sub(r"ICLR rebuttal:? ?", "", s2)
            s2 = s2.replace("consistency/rebuttal/", "consistency/scripts/")
            s2 = re.sub(r" ?\((?:the reviewers' diagnosis )?\(?(?:XETn|qyTj|aayn)[^)]*\)", "", s2)
            s2 = s2.replace("of the reviewers' diagnosis", "of the collapse of an L2 one-step map")
            if s2 != s:
                p.write_text(s2)
    bad = []
    for p in OUT.rglob("*"):
        if p.is_file() and p.suffix in {".py", ".ipynb", ".yaml", ".md", ".txt", ".json"}:
            for i, l in enumerate(p.read_text(errors="ignore").splitlines()):
                if FORBIDDEN.search(l):
                    bad.append(f"{p.relative_to(OUT)}:{i + 1}: {l.strip()[:100]}")
    if bad:
        print("IDENTIFYING STRINGS LEFT:\n  " + "\n  ".join(bad[:40]))
        sys.exit(1)
    n = sum(1 for _ in OUT.rglob("*.ipynb"))
    print(f"supplementary written to {OUT}: {len(pruned)} methods, {n} notebooks")


README = """# Pairwise Consistency Models for Bayesian Transport and Variational Data Assimilation — code

Anonymised code to reproduce the tables and figures of the paper.

## Setup

    conda env create -f environment.yaml
    conda activate 4dvarnet-starter

## Data

The notebooks read the three NetCDF files from `data/`:

{data}

* GP (79 MB): included in `data/`. The fields are simulated from the SPDE of Appendix H (implicit scheme, noise-free
  observations along tracks); `consistency/scripts/gp_exact_posterior.py` recovers the generating operator from the
  file and checks it on the data.
* SSH (0.9 GB) and SIC (13.6 GB) exceed the size limit of the supplementary material and will be released with the
  code. The SSH data are the NATL60 Gulf Stream OSSE with nadir observations of the public 2020a SSH mapping data
  challenge (https://github.com/ocean-data-challenges/2020a_SSH_mapping_NATL60), on a 128x128 grid at 0.1 degree. The
  SIC file holds the ASIP SIC fields, the SAR-derived observations and the OSI SAF field on a common grid
  (Appendix D.3), all from public products.

## Layout

* `src/`: data loaders of the three benchmarks.
* `consistency/consistency_models/`: pairwise consistency models (CM, DynCM, VarCM, DynVarCM), ODE-teacher targets,
  variational costs (`sic_costs.py` for SIC), samplers.
* `consistency/flowmatching_models/`: flow-matching teachers (FM, DynFM).
* `consistency/Notebooks/Notebooks_{{GP,SSH_GF,SIC}}/`: one notebook per method and benchmark (training, sampling,
  figures, full test-set metrics). Their first cell holds the parameters; `methods.yaml` overrides them per run.
* `consistency/sbatch_submission/`: `run_training.py` and `run_metrics.py` execute a notebook with the parameters of
  a method in `methods.yaml`; metrics are written to `consistency/sbatch_submission/results/<XP>/`.

## Reproducing the tables

Train, then evaluate (20 members, full test set):

    cd consistency/sbatch_submission
    python run_training.py --xp SSH_GF --method FM_aug --reset
    python run_metrics.py  --xp SSH_GF --method FM_aug

Teachers must be trained before the models distilled from them (FM before CM, DynFM before DynCM), and
`VarDynCM_scratch` before its fine-tuned variants. Methods of each table row:

| Benchmark | Row | `--method` | Notebook |
|---|---|---|---|
{table}

`results/<XP>/<method>_metrics.csv` gives RMSE, effective resolution and CRPS of the ensemble;
`<method>_metrics_calibration.csv` the spread-skill ratio and rank histogram (Tables 2-3, rank histograms);
`*_calibration_frames.csv` the same per day (SSR per day).

## Figures

* `consistency/scripts/sic_fig.py`, `consistency/scripts/ssh_transport_fig.py`: SIC example and transport figures, from
  the ensembles dumped by the notebooks with `--set FIG_DUMP=True`.
* `consistency/scripts/toy_gaussian_targets.py`: the Gaussian example of Appendix E.
* Number of network evaluations. GP: `python run_nfe_sweep.py --method CMT --grid 2,3,5,9,16` (and the other
  methods), then `consistency/scripts/gp_nfe_fig.py`. SSH: `python run_metrics.py --xp SSH_GF --method CMT_aug
  --set NSTEPS_EVAL=5 --out-tag _nfe5` for each model and step count (DynVarCM: `--set FRAME_GRID=True
  --set FRAME_REFINE_SWEEP=True --set NSTEPS_EVAL=K --out-tag _nfe_frK`), then `consistency/scripts/ssh_nfe_fig.py`.

## Exact GP posterior

`python consistency/scripts/gp_exact_posterior.py` computes the exact Gaussian posterior of the GP benchmark on its
test windows (the implicit SPDE scheme gives a block-tridiagonal window precision; exact conditioning on the
noise-free observations, banded Cholesky) and scores its mean and 20 exact posterior samples with the same code as
the models (row "Exact posterior" of Table 2). CPU only, a few minutes.

All runs use one GPU (A100 or H100, 40-80 GB).
"""

if __name__ == "__main__":
    main()
