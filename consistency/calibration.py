"""Ensemble calibration metrics added for the ICLR rebuttal (reviewers qyTj W3, XETn W4, aayn W3).

The notebooks call ``calib_stats`` once per (test sample, timestep) on the de-normalised ensemble and
``calib_summary`` once at the end. Numpy only.

* spread-skill ratio  SSR = sqrt(mean ensemble variance) / RMSE(ensemble mean) * sqrt((M + 1) / M)
  (the finite-ensemble factor makes SSR = 1 for a perfectly calibrated M-member ensemble);
* CRPS over ALL valid pixels (the original notebooks only score observed pixels, where the
  posterior is tight), computed exactly from the sorted ensemble;
* rank histogram of the truth among the members, summarised by its normalised deviation from flat
  (0 = flat; > 0 = under/over-dispersion or bias).
"""
import numpy as np

MAX_PIXELS = 4000  # pixels drawn per field for CRPS and ranks (keeps the full test set affordable)


def _crps_sorted(ens_sorted, obs):
    """Exact ensemble CRPS, ens_sorted: (M, P) sorted along axis 0, obs: (P,). Returns (P,)."""
    m = ens_sorted.shape[0]
    t1 = np.abs(ens_sorted - obs[None]).mean(0)
    w = (2 * np.arange(1, m + 1) - m - 1)[:, None]
    t2 = (w * ens_sorted).sum(0) / (m * m)
    return t1 - t2


def calib_stats(ens, gt, valid, rng=None):
    """ens: (M, H, W), gt: (H, W), valid: (H, W) bool, all de-normalised. Returns a small dict."""
    ens = np.asarray(ens, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(gt) & np.all(np.isfinite(ens), axis=0)
    m = ens.shape[0]
    e, g = ens[:, valid], gt[valid]
    if g.size == 0 or m < 2:
        return None
    mean = e.mean(0)
    err = mean - g
    out = dict(m=m, var=float(e.var(0, ddof=1).mean()), mse=float((err ** 2).mean()),
               bias=float(err.mean()), mse_db=float(err.var()))  # mse_db: field-mean error removed
    rng = rng or np.random.default_rng(0)
    idx = rng.choice(g.size, size=min(MAX_PIXELS, g.size), replace=False)
    es = np.sort(e[:, idx], axis=0)
    out["crps_all"] = float(_crps_sorted(es, g[idx]).mean())
    ranks = (es < g[idx][None]).sum(0)
    out["ranks"] = np.bincount(ranks, minlength=m + 1).astype(np.int64)
    return out


def calib_summary(stats):
    """Aggregate a list of calib_stats outputs. Returns dict of floats + the rank histogram."""
    stats = [s for s in stats if s]
    if not stats:
        return dict(spread=np.nan, rmse=np.nan, ssr=np.nan, crps_all=np.nan, rank_dev=np.nan, ranks=[],
                    bias=np.nan, ssr_debiased=np.nan)
    m = stats[0]["m"]
    var = float(np.mean([s["var"] for s in stats]))
    mse = float(np.mean([s["mse"] for s in stats]))
    ranks = np.sum([s["ranks"] for s in stats if len(s["ranks"]) == m + 1], axis=0)
    p = ranks / max(ranks.sum(), 1)
    rank_dev = float(np.sqrt(((p - 1.0 / (m + 1)) ** 2).sum()) * (m + 1) / np.sqrt(m))
    bias = float(np.mean([s.get("bias", np.nan) for s in stats]))
    mse_db = float(np.mean([s.get("mse_db", np.nan) for s in stats]))
    return dict(spread=float(np.sqrt(var)), rmse=float(np.sqrt(mse)),
                bias=bias, ssr_debiased=float(np.sqrt(var / mse_db) * np.sqrt((m + 1) / m)) if mse_db > 0 else np.nan,
                ssr=float(np.sqrt(var / mse) * np.sqrt((m + 1) / m)) if mse > 0 else np.nan,
                crps_all=float(np.mean([s["crps_all"] for s in stats])), rank_dev=rank_dev,
                ranks=ranks.tolist())


def write_calibration_csv(path, method, summary, n_members):
    """One-row CSV next to METRICS_CSV: method, members, spread, rmse, ssr, crps_all, rank_dev, ranks."""
    import csv, os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    row = dict(method=method, members=n_members, **{k: summary[k] for k in
               ("spread", "rmse", "ssr", "crps_all", "rank_dev")}, ranks=" ".join(map(str, summary["ranks"])),
               bias=summary.get("bias", np.nan), ssr_debiased=summary.get("ssr_debiased", np.nan))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        w.writeheader(); w.writerow(row)
    return row
