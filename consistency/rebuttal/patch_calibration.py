#!/usr/bin/env python3
"""Add the calibration metrics (consistency/calibration.py) to the full-test-set cell of every
generative notebook. Idempotent: a cell that already calls calib_stats is left alone.

For each notebook the metrics cell already has, per (sample, t): the de-normalised ensemble `_ens_p`
(M, H, W), the truth `_gt_p` (H, W) and the mask `_valid_t`. The patch
  1. creates `_calib = []` before the test loop,
  2. appends `calib_stats(_ens_p, _gt_p, _valid_t)` next to the existing CRPS append,
  3. after the metrics table, writes <METRICS_CSV stem>_calibration.csv (spread, RMSE, SSR, CRPS over all
     valid pixels, rank-histogram deviation, rank counts).

    python consistency/rebuttal/patch_calibration.py            # patch
    python consistency/rebuttal/patch_calibration.py --check    # report only
"""
import glob, json, re, sys
from pathlib import Path

NB_ROOT = Path(__file__).resolve().parents[1] / "Notebooks"
CRPS_APPEND = re.compile(r"^(\s*)(\w+)\['crps'\]\.append\(_crps_val\)\s*$", re.M)
LOOP = "for _tb in datamodule.test_dataloader():"
KEEP = "df_metrics = df_metrics[_keep]"

TAIL = """
# ── Calibration (ICLR rebuttal: spread-skill ratio, CRPS on all valid pixels, rank histogram) ──
_cs = calib_summary(_calib)
_calib_csv = (METRICS_CSV[:-4] if METRICS_CSV.endswith('.csv') else METRICS_CSV) + '_calibration.csv'
write_calibration_csv(_calib_csv, {label!r}, _cs, N_SAMPLES)
print(f"Calibration ({{N_SAMPLES}} members): SSR {{_cs['ssr']:.3f}}  spread {{_cs['spread']:.4f}}  "
      f"RMSE {{_cs['rmse']:.4f}}  CRPS(all px) {{_cs['crps_all']:.4f}}  rank dev {{_cs['rank_dev']:.2f}} -> {{_calib_csv}}")
"""


def label_of(path):
    stem = Path(path).stem.replace("Notebook_", "")
    return re.sub(r"^(consistency_model_|flowmatching_model_)", "", stem)


def patch_source(src, label):
    if "calib_stats" in src or LOOP not in src or KEEP not in src:
        return None
    m = CRPS_APPEND.search(src)
    if not m:
        return None
    indent = m.group(1)
    src = src.replace(m.group(0), m.group(0) + f"\n{indent}_calib.append(calib_stats(_ens_p, _gt_p, _valid_t))", 1)
    i = src.index(LOOP)
    line_start = src.rfind("\n", 0, i) + 1
    lead = src[line_start:i]
    src = (src[:line_start] + f"{lead}from calibration import calib_stats, calib_summary, write_calibration_csv\n"
           f"{lead}_calib = []\n" + src[line_start:])
    k = src.index(KEEP) + len(KEEP)
    src = src[:k] + "\n" + TAIL.format(label=label) + src[k:]
    return src


def main():
    check = "--check" in sys.argv
    for f in sorted(glob.glob(str(NB_ROOT / "Notebooks_*" / "Notebook_*.ipynb"))):
        nb = json.load(open(f))
        done = False
        for c in nb["cells"]:
            if c["cell_type"] != "code":
                continue
            src = "".join(c["source"])
            if "calib_stats" in src:
                done = "already"
                break
            new = patch_source(src, label_of(f))
            if new:
                done = True
                if not check:
                    c["source"] = new.splitlines(keepends=True)
                break
        print(f"{'patched' if done is True else (done or 'skipped'):8s} {Path(f).relative_to(NB_ROOT)}")
        if done is True and not check:
            with open(f, "w") as fh:
                json.dump(nb, fh, indent=1, ensure_ascii=False)
                fh.write("\n")


if __name__ == "__main__":
    main()
