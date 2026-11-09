# Paper: where does end-to-end learned data assimilation win?

Short conference-style paper (LaTeX, ~6-10 pages) presenting the two-layer QG testbed of this branch, the
Lyapunov-subspace error diagnostic and the comparison 4DVarNet / 4DVar / hybrid.

## Link with the code

Every table, figure and number quoted in the text is generated from the cached results of the sweep
(`contrib/qg2l`), never typed by hand:

```bash
# on the cluster, after the sweep (also run automatically at the end of the Slurm job)
python paper/scripts/make_paper_assets.py collect --root data/qg2l_scenarios_small --out paper/results/results_small.json
python paper/scripts/make_paper_assets.py collect --root data/qg2l_scenarios_full  --out paper/results/results_full.json
python paper/scripts/make_paper_assets.py render --results paper/results/results_small.json --full paper/results/results_full.json
```

- `results/*.json`: collected numbers (scores, error energy per Lyapunov subspace, bootstrap intervals, cost,
  Desroziers / oracle R, tuning, seeds).
- `tables/macros.tex`: in-text numbers, e.g. `\ratio{Net}{Unst}{Real}` = E_ref / E_4DVarNet in the unstable
  subspace for the realistic scenario with its 90 % interval; `\ssh{Net}{Base}`, `\nunst{Base}`, ...
- `tables/{scores,cost,robustness}.tex`, `figures/ratios.pdf`: generated; `figures/*.png`: testbed figures.
- Orange `[...]` in the PDF (`\pend`) marks what still awaits the final sweep (tuned 4DVar, seeds, full size).

The current `results/results_small.json` holds the small-size sweep before tuning / seeds (reference: 4DVar with
nominal R); the final run replaces it and the reference becomes the tuned 4DVar automatically.

## Overleaf

- Simplest: upload `paper/` as a zip (New Project > Upload Project), main document `main.tex`.
- Two-way link with this branch (Overleaf Git integration; may require a paid Overleaf plan): create the project,
  copy its git URL (Menu > Git), then `OVERLEAF=<url> bash paper/sync_overleaf.sh push` to send `paper/`, and
  `... pull` to bring Overleaf edits back into the branch. Overleaf's GitHub synchronisation links a whole
  repository (default branch) to a project, so the subtree route is the one that keeps the paper next to the code
  of this branch.

Build locally: `cd paper && latexmk -pdf main.tex`.
