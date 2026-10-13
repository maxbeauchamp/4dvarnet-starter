"""Synthetic multiscale Arctic sea-ice concentration on a nested polar-stereographic grid.

Grid: north polar stereographic (central meridian -45 deg, true scale 70N),
domain [-3200, 3200] km. Levels: C = 32 km, M = 4 km, H = 1 km
(nesting x8 then x4 -> 64 and 16 children). All cell edges are aligned,
so R (block mean) and P (injection) are exact.
"""
import json
import numpy as np
from PIL import Image, ImageDraw

R_E = 6371.0
LON0 = -45.0
K70 = (1 + np.sin(np.radians(70))) / 2
X0 = -3200.0
DC, DM, DH = 32.0, 4.0, 1.0
NE = "/home/claude/data/ne/geojson"


def fwd(lon, lat):
    rho = 2 * R_E * K70 * np.tan(np.radians(90 - np.asarray(lat)) / 2)
    th = np.radians(np.asarray(lon) - LON0)
    return rho * np.sin(th), -rho * np.cos(th)


def inv(x, y):
    rho = np.hypot(x, y)
    lat = 90 - 2 * np.degrees(np.arctan(rho / (2 * R_E * K70)))
    lon = np.degrees(np.arctan2(x, -y)) + LON0
    return (lon + 180) % 360 - 180, lat


# ------------------------------------------------------------------ land
def land_raster(x0, y0, n, d, path=f"{NE}/ne_10m_land.geojson"):
    """Boolean land mask on an n x n grid of cell size d starting at (x0, y0) (lower-left)."""
    img = Image.new("1", (n, n), 0)
    dr = ImageDraw.Draw(img)
    with open(path) as fh:
        gj = json.load(fh)
    for f in gj["features"]:
        g = f["geometry"]
        polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
        for poly in polys:
            ring = np.array(poly[0])
            if ring[:, 1].max() < 45:
                continue
            ring = ring[ring[:, 1] > 30] if ring[:, 1].min() < 30 else ring
            if len(ring) < 3:
                continue
            x, y = fwd(ring[:, 0], ring[:, 1])
            px = (x - x0) / d
            py = n - (y - y0) / d
            if px.max() < -5 or px.min() > n + 5 or py.max() < -5 or py.min() > n + 5:
                continue
            dr.polygon(list(zip(px, py)), fill=1)
    return np.array(img, bool)[::-1]          # row 0 = lowest y


# ------------------------------------------------------------------ noise
def _modes(seed, lmin, lmax, nsc, ndir, slope):
    r = np.random.default_rng(seed)
    lam = np.repeat(np.geomspace(lmax, lmin, nsc), ndir)
    th = r.uniform(0, 2 * np.pi, lam.size)
    k = 2 * np.pi / lam
    A = (lam / lmax) ** slope
    return np.stack([k * np.cos(th), k * np.sin(th)], 1), A / np.sqrt((A ** 2).sum()), r.uniform(0, 2 * np.pi, lam.size)


WX = _modes(11, 2.0, 900.0, 34, 6, 0.85)
WY = _modes(12, 2.0, 900.0, 34, 6, 0.85)
FN = _modes(13, 1.0, 40.0, 20, 6, 0.4)


def _sum(x, y, m, chunk=64):
    K, A, P = m
    out = np.zeros(x.shape)
    for s in range(0, len(A), chunk):
        ph = np.multiply.outer(x, K[s:s + chunk, 0]) + np.multiply.outer(y, K[s:s + chunk, 1]) + P[s:s + chunk]
        out += np.cos(ph) @ A[s:s + chunk]
    return out


EDGE = np.array([(-180, 61), (-165, 60.5), (-150, 67), (-120, 66), (-95, 60), (-70, 58), (-55, 54),
                 (-45, 59), (-35, 64), (-22, 69.5), (-10, 73.5), (0, 76.5), (10, 78.8), (20, 78),
                 (35, 77), (50, 74.5), (65, 72), (90, 70), (130, 70), (160, 64), (180, 61)])


def edge_lat(lon):
    return np.interp(lon, EDGE[:, 0], EDGE[:, 1], period=360)


# ------------------------------------------------------------------ leads
def make_leads(seed, bbox, n):
    r = np.random.default_rng(seed)
    segs = []
    for _ in range(n):
        L = 15 + 120 * r.pareto(1.4) % 320
        w = 0.75 + 0.6 * r.pareto(2.5) % 1.6
        a = 0.3 + 0.55 * r.uniform()
        x, y = r.uniform(bbox[0], bbox[1]), r.uniform(bbox[2], bbox[3])
        th = r.uniform(0, np.pi) if r.uniform() < 0.5 else r.normal(0.7, 0.25)
        step = 4.0
        for _ in range(int(L / step)):
            th += r.normal(0, 0.025)
            x1, y1 = x + step * np.cos(th), y + step * np.sin(th)
            segs.append((x, y, x1, y1, w, a))
            x, y = x1, y1
    return np.array(segs)


def lead_factor(x, y, segs, chunk=256):
    """Multiplicative factor in (0, 1]: 1 - max_k a_k exp(-(d_k / w_k)^2)."""
    out = np.zeros(x.shape)
    for s in range(0, len(segs), chunk):
        S = segs[s:s + chunk]
        ax, ay, bx, by, w, a = [S[:, i] for i in range(6)]
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy
        px = x[:, None] - ax
        py = y[:, None] - ay
        t = np.clip((px * dx + py * dy) / L2, 0, 1)
        d2 = (px - t * dx) ** 2 + (py - t * dy) ** 2
        out = np.maximum(out, (a * np.exp(-d2 / (w * w))).max(1))
    return 1 - out


def sic(x, y, segs=None, chunk=200_000):
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    out = np.empty_like(x)
    for s in range(0, x.size, chunk):
        out[s:s + chunk] = _sic(x[s:s + chunk], y[s:s + chunk], segs)
    return out


def _sic(x, y, segs):
    xw = x + 70.0 * _sum(x, y, WX)
    yw = y + 70.0 * _sum(x, y, WY)
    lon, lat = inv(xw, yw)
    d = lat - edge_lat(lon)                       # deg, >0 inside pack
    t = np.clip((d + 0.7) / 1.6, 0, 1)
    c = t * t * (3 - 2 * t)
    floes = 0.55 + 0.45 * np.tanh(3.0 * _sum(x, y, FN))      # MIZ floe texture
    miz = np.exp(-((d - 0.1) / 0.6) ** 2)
    c = c * (1 - 0.65 * miz * (1 - floes))
    c = np.where(c > 0.02, np.minimum(c, 0.985), c)
    if segs is not None and len(segs):
        c = c * np.where(c > 0.6, lead_factor(x, y, segs), 1.0)
    return np.clip(c, 0, 1)
