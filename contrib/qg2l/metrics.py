"""
Reconstruction scores for the periodic QG testbed (numpy only).

Same spirit as the SSH data-challenge scores used in src/utils.py, adapted to a
doubly periodic box in km:
    mu        = 1 - RMSE / RMS(truth)                       (per layer)
    lambda_x  = effective resolution [km]: shortest wavelength at which the
                isotropic PSD score 1 - PSD(err) / PSD(truth) is still >= 0.5
"""
import numpy as np

from .qgmodel import isotropic_spectrum


def rmse(rec, ref):
    return float(np.sqrt(np.nanmean((rec - ref) ** 2)))


def mu_score(rec, ref):
    return float(1.0 - rmse(rec, ref) / np.sqrt(np.nanmean(ref**2)))


def psd_score(rec, ref, dx):
    """Returns wavelengths (km), score(k) and the effective resolution lambda_x (km)."""
    k, p_err = isotropic_spectrum(rec - ref, dx)
    _, p_ref = isotropic_spectrum(ref, dx)
    score = 1.0 - p_err / np.maximum(p_ref, 1e-30)
    wl = 2 * np.pi / k / 1e3
    order = np.argsort(wl)[::-1]  # large -> small scales
    wl, score = wl[order], score[order]
    lam = np.nan
    below = np.where(score < 0.5)[0]
    if len(below) == 0:
        lam = float(wl[-1])
    elif below[0] > 0:
        i = below[0]
        # linear interpolation between the last good and the first bad wavelength
        s0, s1 = score[i - 1], score[i]
        lam = float(wl[i - 1] + (0.5 - s0) * (wl[i] - wl[i - 1]) / (s1 - s0))
    return wl, score, lam


def all_scores(rec_state, ref_state, dx):
    """rec_state, ref_state: (T, L, ny, nx) in SSH-equivalent units (m)."""
    out = {}
    for l in range(rec_state.shape[1]):
        r, t = rec_state[:, l], ref_state[:, l]
        name = "ssh" if l == 0 else f"layer{l+1}"
        out[f"rmse_{name}"] = rmse(r, t)
        out[f"mu_{name}"] = mu_score(r, t)
        out[f"lx_{name}_km"] = psd_score(r, t, dx)[2]
    return out
