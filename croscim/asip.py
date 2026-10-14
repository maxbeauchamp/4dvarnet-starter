"""
asip.py
=======
ASIP 500 m -> M (4 km) aggregation, numpy only (no torch): used by the
conversion script, which stores the 4 km maps of every day, and by the tests.

For the days before the analysis day t no input / target partition applies,
so their 4 km maps are the same in every batch and can be stored once; the
500 m fields are only read for the last days (H inputs, partition of day t).
"""

import numpy as np

ASIP_SIGMA_FLOOR = 0.05  # floor of the ASIP SIC uncertainty (fraction)
M_PER_500M = 8           # 500 m pixels per 4 km cell (one side)

_V_LUT = np.where(np.arange(256) == 255, 0, np.arange(256) / 100).astype(np.float32)
_W_LUT = (np.maximum(np.arange(256) / 100, ASIP_SIGMA_FLOOR) ** -2).astype(np.float32)


def asip_to_m(sic: np.ndarray, std: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ASIP 500 m (uint8 %, 255 = missing) -> M (4 km): sigma^-2-weighted
    8 x 8 means and summed weights. Same weighting as obs.grid_asip, so it
    equals R_{M<-H} of the H gridding."""
    f = M_PER_500M
    w = _W_LUT[std] * (sic != 255)  # byte lookups: much cheaper than float ops on 16 M pixels
    ny, nx = sic.shape[-2] // f, sic.shape[-1] // f
    num = (w * _V_LUT[sic]).reshape(ny, f, nx, f).sum((1, 3))
    den = w.reshape(ny, f, nx, f).sum((1, 3))
    return np.where(den > 0, num / np.where(den > 0, den, 1), 0).astype(np.float32), den
