"""
Spectral metrics for 2D spatial fields.

Inspired by the 2020a SSH mapping NATL60 challenge evaluation framework:
  ocean-data-challenges/2020a_SSH_mapping_NATL60

Key metric:  PSD_score(k) = 1 - PSD(error)(k) / PSD(signal)(k)
Resolved scale: smallest wavelength where PSD_score >= 0.5.

Units: dimensionless pixels by default; set dx (km) for physical units.
"""

from __future__ import annotations

import numpy as np


def _hann2d(H: int, W: int) -> np.ndarray:
    return np.outer(np.hanning(H), np.hanning(W))


def radial_psd_2d(
    field2d: np.ndarray,
    dx: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Radially-averaged 2D PSD of a single spatial field.

    Parameters
    ----------
    field2d : (H, W) — NaN replaced by 0 internally
    dx      : pixel spacing (km or any unit)

    Returns
    -------
    wavelengths : 1-D array [dx units], sorted *descending* (large → small scale)
    psd         : radially-averaged PSD at those wavelengths
    """
    H, W = field2d.shape
    f = field2d.astype(float).copy()
    f -= np.nanmean(f)       # constant detrend
    f[np.isnan(f)] = 0.0
    f *= _hann2d(H, W)       # 2-D Hann window

    F = np.fft.fft2(f)
    psd2d = (np.abs(F) ** 2) * (dx ** 2) / (H * W)   # [unit^2 · px^2]

    kx = np.fft.fftfreq(W, d=dx)
    ky = np.fft.fftfreq(H, d=dx)
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX ** 2 + KY ** 2)

    dk = min(abs(kx[1] - kx[0]), abs(ky[1] - ky[0]))
    k_centers = np.arange(0.5 * dk, K.max() + dk, dk)

    psd_r = np.full(len(k_centers), np.nan)
    for i, k0 in enumerate(k_centers):
        mask = (K >= k0 - 0.5 * dk) & (K < k0 + 0.5 * dk)
        if mask.any():
            psd_r[i] = np.mean(psd2d[mask])

    with np.errstate(divide='ignore', invalid='ignore'):
        wavelengths = np.where(k_centers > 0, 1.0 / k_centers, np.nan)

    # Sort descending by wavelength (large scale first)
    order = np.argsort(wavelengths)[::-1]
    return wavelengths[order], psd_r[order]


def psd_spectral_score(
    pred2d: np.ndarray,
    gt2d: np.ndarray,
    dx: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Spectral score following the SSH mapping challenge convention.

    score(k) = 1 - PSD(pred - gt)(k) / PSD(gt)(k)

    Score = 1  → perfect reconstruction at wavenumber k
    Score = 0  → error energy equals signal energy (no skill)
    Score < 0  → reconstruction is worse than nothing

    Parameters
    ----------
    pred2d, gt2d : (H, W)  — physical-unit fields
    dx           : pixel spacing

    Returns
    -------
    wavelengths : 1-D [dx units], descending
    psd_gt      : PSD of GT
    psd_err     : PSD of (pred - gt)
    score       : spectral score ∈ (-∞, 1]
    """
    wl, psd_gt = radial_psd_2d(gt2d, dx=dx)
    err = np.where(np.isnan(pred2d) | np.isnan(gt2d), 0.0, pred2d - gt2d)
    _, psd_err = radial_psd_2d(err, dx=dx)

    with np.errstate(divide='ignore', invalid='ignore'):
        score = 1.0 - psd_err / psd_gt

    score[~np.isfinite(psd_gt) | (psd_gt <= 0)] = np.nan
    return wl, psd_gt, psd_err, score


def resolved_scale(
    wavelengths: np.ndarray,
    score: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """Smallest spatial wavelength where spectral score >= threshold.

    Wavelengths must be sorted *descending* (as returned by psd_spectral_score).
    Uses linear interpolation at the 0.5 crossing.

    Returns nan if no crossing found.
    """
    valid = np.isfinite(wavelengths) & np.isfinite(score) & (wavelengths > 0)
    wl = wavelengths[valid]
    sc = score[valid]
    if len(sc) < 2:
        return np.nan

    above = sc >= threshold
    crossings = np.where(np.diff(above.astype(int)) < 0)[0]
    if len(crossings) == 0:
        return np.nan

    i = crossings[0]
    w0, s0 = wl[i],     sc[i]
    w1, s1 = wl[i + 1], sc[i + 1]
    if s0 == s1:
        return float(0.5 * (w0 + w1))
    return float(w0 + (threshold - s0) * (w1 - w0) / (s1 - s0))
