"""Synthetic but realistic multiscale SST (Gulf Stream region) + land mask."""
import json
import numpy as np
from PIL import Image, ImageDraw

R_EARTH = 6371.0
NE_DIR = "/home/claude/data/ne/geojson"


# ---------------------------------------------------------------- land mask
class LandMask:
    def __init__(self, path, lon0, lon1, lat0, lat1, res):
        self.lon0, self.lat1, self.res = lon0, lat1, res
        W = int(round((lon1 - lon0) / res))
        H = int(round((lat1 - lat0) / res))
        img = Image.new("1", (W, H), 0)
        d = ImageDraw.Draw(img)
        for ring in iter_rings(path, outer_only=False):
            # holes are rare (lakes not in land file); fill outer rings only
            if ring["hole"]:
                continue
            xy = [((x - lon0) / res, (lat1 - y) / res) for x, y in ring["xy"]]
            if len(xy) > 2:
                d.polygon(xy, fill=1)
        self.m = np.array(img, dtype=bool)

    def __call__(self, lon, lat):
        i = ((self.lat1 - lat) / self.res).astype(int)
        j = ((lon - self.lon0) / self.res).astype(int)
        H, W = self.m.shape
        ok = (i >= 0) & (i < H) & (j >= 0) & (j < W)
        out = np.zeros(np.shape(lon), bool)
        out[ok] = self.m[i[ok], j[ok]]
        return out


def iter_rings(path, outer_only=True):
    with open(path) as fh:
        gj = json.load(fh)
    for feat in gj["features"]:
        g = feat["geometry"]
        polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
        for poly in polys:
            for k, ring in enumerate(poly):
                if outer_only and k > 0:
                    continue
                yield {"xy": ring, "hole": k > 0}


_LAND_G = None
_LAND_R = None


def is_land(lon, lat):
    global _LAND_G, _LAND_R
    if _LAND_G is None:
        _LAND_G = LandMask(f"{NE_DIR}/ne_50m_land.geojson", -180, 180, -90, 90, 0.05)
        _LAND_R = LandMask(f"{NE_DIR}/ne_10m_land.geojson", -100, -20, 15, 65, 1 / 240)
    lon = np.asarray(lon, float)
    lat = np.asarray(lat, float)
    inreg = (lon > -100) & (lon < -20) & (lat > 15) & (lat < 65)
    out = _LAND_G(lon, lat)
    if inreg.any():
        out[inreg] = _LAND_R(lon[inreg], lat[inreg])
    return out


# ---------------------------------------------------------------- SST field
def _modes(seed, lam_min, lam_max, n_scales, n_dir, slope):
    rng = np.random.default_rng(seed)
    lams = np.geomspace(lam_max, lam_min, n_scales)
    U, K, A, P = [], [], [], []
    for lam in lams:
        for _ in range(n_dir):
            u = rng.normal(size=3)
            u /= np.linalg.norm(u)
            U.append(u)
            K.append(2 * np.pi * R_EARTH / lam)
            A.append((lam / lam_max) ** slope)
            P.append(rng.uniform(0, 2 * np.pi))
    A = np.array(A)
    return np.array(U), np.array(K), A / np.sqrt((A ** 2).sum()), np.array(P)


MODES_DLAT = _modes(1, 4.0, 1200.0, 40, 6, 0.8)
MODES_DLON = _modes(2, 4.0, 1200.0, 40, 6, 0.8)
MODES_FINE = _modes(3, 1.5, 60.0, 30, 6, 0.6)


def _sum_modes(r, modes, chunk=60):
    U, K, A, P = modes
    out = np.zeros(r.shape[0])
    for s in range(0, len(K), chunk):
        ph = (r @ U[s:s + chunk].T) * K[s:s + chunk] + P[s:s + chunk]
        out += np.cos(ph) @ A[s:s + chunk]
    return out


def _unit(lon, lat):
    lo, la = np.radians(lon), np.radians(lat)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)


def _smoothstep(x, a, b):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


EDDIES = [  # lon, lat, radius(deg), amplitude (degC)
    (-66.5, 39.6, 0.55, 3.2), (-60.5, 40.9, 0.65, 2.8), (-70.0, 39.4, 0.45, 2.4),
    (-63.5, 35.4, 0.6, -3.4), (-57.0, 37.0, 0.7, -2.6), (-69.5, 35.9, 0.5, -2.2),
    (-52.5, 39.4, 0.8, 2.2), (-48.5, 43.4, 0.7, -2.0),
]


def sst(lon, lat, chunk=400_000):
    lon = np.asarray(lon, float).ravel()
    lat = np.asarray(lat, float).ravel()
    out = np.empty_like(lon)
    for s in range(0, lon.size, chunk):
        out[s:s + chunk] = _sst(lon[s:s + chunk], lat[s:s + chunk])
    return out


def _sst(lon, lat):
    r = _unit(lon, lat)
    reg = _smoothstep(lat, 18, 26) * (1 - _smoothstep(lat, 52, 60)) \
        * _smoothstep(lon, -82, -76) * (1 - _smoothstep(lon, -38, -28))
    dlat = 1.25 * _sum_modes(r, MODES_DLAT) * (0.35 + 0.65 * reg)
    dlon = 1.6 * _sum_modes(r, MODES_DLON) * (0.35 + 0.65 * reg)
    lw, aw = lon + dlon, lat + dlat
    # climatology (cooler subpolar gyre, warm subtropics)
    s2 = np.sin(np.radians(aw)) ** 2
    lin = 37.0 - 48.0 * s2
    clim = 28.5 - np.logaddexp(0, (28.5 - lin) * 1.5) / 1.5
    clim = np.maximum(clim, -1.5)
    # Gulf Stream path + meanders
    x = lw + 75.0
    path = 35.4 + 0.30 * x - 0.0028 * x * x
    grow = _smoothstep(x, 2, 14)
    path += grow * (0.85 * np.sin(2 * np.pi * x / 8.5 + 0.6) + 0.35 * np.sin(2 * np.pi * x / 3.7 + 1.9))
    n = aw - path
    band = _smoothstep(x, -3.0, 1.0) * (1 - _smoothstep(x, 30, 40))
    front = band * (-3.0 * np.tanh(n / 0.35) - 1.8 * np.tanh(n / 0.05))
    warm_core = band * 1.2 * np.exp(-(n + 0.25) ** 2 / 0.18)
    clim_gs = clim + 0.8 * reg * (1 - band)  # mild adjustment away from jet
    T = clim_gs - 2.2 * band + front + warm_core
    # rings
    for elon, elat, rad, amp in EDDIES:
        d2 = ((lw - elon) * np.cos(np.radians(elat))) ** 2 + (aw - elat) ** 2
        T += 1.4 * amp * np.exp(-d2 / (2 * rad * rad)) * (1 - 0.35 * np.exp(-d2 / (0.5 * rad * rad)))
    # submesoscale texture, strongest near fronts
    act = 0.12 + 0.9 * band * np.exp(-n ** 2 / 0.8) + 0.25 * reg
    T += 0.35 * act * _sum_modes(r, MODES_FINE)
    return T
