import numpy as np, time
from arctic import *

CENTER = fwd(np.array([-2.0]), np.array([78.5]))
out = {}
t0 = time.time()
# ---------------- overview: C field from 4-km samples, ocean-weighted
nM = int(-2 * X0 / DM)                        # 1600
c = X0 + DM * (np.arange(nM) + 0.5)
X, Y = np.meshgrid(c, c)
S4 = sic(X, Y).reshape(X.shape)
L4 = land_raster(X0, X0, nM, DM)
w = (~L4).astype(float)
k = int(DC / DM)
W = w.reshape(nM // k, k, nM // k, k).sum((1, 3))
V = (w * S4).reshape(nM // k, k, nM // k, k).sum((1, 3))
XC_all = np.where(W > 0.5 * k * k, V / np.maximum(W, 1e-9), np.nan)
out["XC_all"] = XC_all
print("overview", time.time() - t0)

# ---------------- decomposition region: 8x8 C cells aligned
cx, cy = float(CENTER[0][0]) + 80.0, float(CENTER[1][0])
ic = int(np.floor((cx - X0) / DC)) - 4
jc = int(np.floor((cy - X0) / DC)) - 4
x0 = X0 + ic * DC; y0 = X0 + jc * DC
nH = int(8 * DC / DH)                         # 256
g = DH * (np.arange(nH) + 0.5)
XX, YY = np.meshgrid(x0 + g, y0 + g)
segs = make_leads(5, (x0 - 150, x0 + 8 * DC + 150, y0 - 150, y0 + 8 * DC + 150), 150)
XH = sic(XX, YY, segs).reshape(XX.shape)
LH = land_raster(x0, y0, nH, DH)
wH = (~LH).astype(float)

def R(v, w, f):
    n = v.shape[0] // f
    Wb = w.reshape(n, f, n, f).sum((1, 3))
    Vb = (w * v).reshape(n, f, n, f).sum((1, 3))
    return np.where(Wb > 0, Vb / np.maximum(Wb, 1e-12), 0.0), Wb

def P(v, f):
    return np.repeat(np.repeat(v, f, 0), f, 1)

XM, wM = R(XH, wH, 4)
XC, wC = R(XM, wM, 8)
dM = XM - P(XC, 8)
dH = XH - P(XM, 4)
chk = (np.abs(R(dM, wM, 8)[0]).max(), np.abs(R(dH, wH, 4)[0]).max(),
       np.abs((P(P(XC, 8), 4) + P(dM, 4) + dH - XH)[wH > 0]).max())
print("checks", chk)
mk = lambda a: np.where(wH > 0, a, np.nan)
out.update(dec_XH=mk(XH), dec_PXC=mk(P(P(XC, 8), 4)), dec_PdM=mk(P(dM, 4)), dec_dH=mk(dH),
           dec_origin=np.array([x0, y0]), dec_checks=np.array(chk),
           dec_XM=np.where(wM > 0, XM, np.nan), dec_XC=np.where(wC > 0, XC, np.nan))

# ---------------- zoom choices
best, bij = -1, None
for i in range(0, 7):
    for j in range(0, 7):
        blk = XH[j * 32:(j + 2) * 32, i * 32:(i + 2) * 32]
        if wH[j * 32:(j + 2) * 32, i * 32:(i + 2) * 32].min() < 1 or blk.mean() < 0.75:
            continue
        s = blk.std()
        if s > best:
            best, bij = s, (i, j)
i, j = bij
out["zM_ij"] = np.array([i, j])            # in C-cell units within dec region (2x2 C block)
blkH = XH[j * 32:(j + 2) * 32, i * 32:(i + 2) * 32]
best, bb = -1, None
for a in range(0, 13, 2):
    for b in range(0, 13, 2):
        sub = blkH[b * 4:(b + 4) * 4, a * 4:(a + 4) * 4]
        if sub.std() > best:
            best, bb = sub.std(), (a, b)
out["zH_ab"] = np.array(bb)                # in M-cell units within M zoom (4x4 M block)
print("zoom", bij, bb, time.time() - t0)
np.savez_compressed("cache2.npz", **out)
