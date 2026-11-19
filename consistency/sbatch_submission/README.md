# Running the notebooks

* `methods.yaml`: the methods of the paper, per benchmark (notebook, checkpoint folder `log_dir`, papermill
  parameters). The header maps the rows of Tables 2-3 to the method keys.
* `run_training.py --xp XP --method M [--reset]`: executes the notebook of M with its parameters (training).
* `run_metrics.py --xp XP --method M [--set NAME=VALUE ...] [--out-tag TAG]`: reloads the checkpoint and evaluates
  the full test set (20 members for the generative methods); writes `results/<XP>/<M><TAG>_metrics*.csv`
  (RMSE / resolved scale / CRPS; spread-skill ratio and rank histogram; the same per frame).
* `run_nfe_sweep.py --method M [--grid ...]` (GP): RMSE, SSR and composition gap versus the number of network
  evaluations, on three test batches; `results/GP/<M>_nfe_sweep.csv`.

`results/` and `figures/` are not tracked; the CSVs of the paper are in `../paper_results/`.
