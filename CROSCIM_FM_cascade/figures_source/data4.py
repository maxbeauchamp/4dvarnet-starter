"""Synthetic fields for Fig. 4: sub-grid statistics S, controllability, rollout snapshots."""
import numpy as np
from arctic import fwd, X0, DC, DH, sic, lead_factor

cx, cy = [float(v[0]) for v in fwd(np.array([-60.0]), np.array([86.0]))]
ic = int(np.floor((cx - X0) / DC)) - 2; jc = int(np.floor((cy - X0) / DC)) - 2
x0, y0 = X0 + ic * DC, X0 + jc * DC
n = int(4 * DC / DH)                                   # 128 px, 4x4 C cells
g = DH * (np.arange(n) + 0.5)
XX, YY = np.meshgrid(x0 + g, y0 + g)
bbox = (x0 - 120, x0 + 4 * DC + 120, y0 - 120, y0 + 4 * DC + 120)


def leads(seed, nl, theta=None, spread=0.12, amp=1.0, wscale=1.0):
    r = np.random.default_rng(seed)
    segs = []
    for _ in range(nl):
        L = 20 + 90 * r.pareto(1.4) % 220
        w = (0.8 + 0.6 * r.pareto(2.5) % 1.5) * wscale
        a = (0.35 + 0.5 * r.uniform()) * amp
        x, y = r.uniform(bbox[0], bbox[1]), r.uniform(bbox[2], bbox[3])
        th0, nz = r.uniform(0, np.pi), r.normal()
        th = th0 if theta is None else theta + spread * nz
        for _ in range(int(L / 3.0)):
            th += 0.03 * r.normal()
            x1, y1 = x + 3.0 * np.cos(th), y + 3.0 * np.sin(th)
            segs.append((x, y, x1, y1, w, a)); x, y = x1, y1
    return np.array(segs)


def field(segs):
    return sic(XX, YY, segs).reshape(n, n)


def Rb(v, f=32):
    m = v.shape[0] // f
    return v.reshape(m, f, m, f).mean((1, 3))


def Pb(v, f=32):
    return np.repeat(np.repeat(v, f, 0), f, 1)


def stats(c, f=32):
    """Per-C-cell sub-grid statistics S_C computed from the 1-km field c."""
    m = c.shape[0] // f
    soft = 1 / (1 + np.exp(-(0.80 - c) / 0.03))         # soft lead mask
    gy, gx = np.gradient(c)
    my, mx = np.gradient(soft)
    out = {k: np.zeros((m, m)) for k in ["mean", "ow", "lead", "dens", "var", "aniso", "theta"]}
    out["hist"] = np.zeros((m, m, 5))
    for j in range(m):
        for i in range(m):
            sl = np.s_[j * f:(j + 1) * f, i * f:(i + 1) * f]
            cc = c[sl]
            out["mean"][j, i] = cc.mean()
            out["ow"][j, i] = (1 - cc).mean()
            out["lead"][j, i] = soft[sl].mean()
            out["dens"][j, i] = np.hypot(mx[sl], my[sl]).sum() / 2 / f ** 2 * 1e2   # km of lead edge / 100 km^2 ~ length density
            out["var"][j, i] = cc.var()
            Jxx, Jyy, Jxy = (gx[sl] ** 2).mean(), (gy[sl] ** 2).mean(), (gx[sl] * gy[sl]).mean()
            tr = Jxx + Jyy; det = Jxx * Jyy - Jxy ** 2
            l1 = tr / 2 + np.sqrt(max(tr ** 2 / 4 - det, 0)); l2 = tr - l1
            out["aniso"][j, i] = (l1 - l2) / (l1 + l2 + 1e-12)
            out["theta"][j, i] = 0.5 * np.arctan2(2 * Jxy, Jxx - Jyy) + np.pi / 2   # lead direction ⟂ gradient
            out["hist"][j, i] = np.histogram(cc, bins=[0, .2, .4, .6, .8, 1.0001])[0] / cc.size
    return out


out = {}
ref = field(leads(1, 34))
ali = field(leads(1, 34, theta=0.45))     # same leads as ref, re-oriented (shape edit)
den = field(leads(3, 80))
out.update(ref=ref, ali=ali, den=den)
for k, v in [("ref", ref), ("ali", ali), ("den", den)]:
    for s, a in stats(v).items():
        out[f"S_{k}_{s}"] = a
# rollout snapshots (mode A, on-demand downscaling): opening by divergence, then refreezing
t0 = field(leads(7, 22))
t1 = field(np.concatenate([leads(7, 22), leads(8, 50, theta=1.9, spread=0.25)]))
t2 = field(np.concatenate([leads(7, 22, amp=0.45), leads(8, 50, theta=1.9, spread=0.25, amp=0.45)]))
out.update(t0=t0, t1=t1, t2=t2)
for k in ["t0", "t1", "t2"]:
    out[f"S_{k}_lead"] = stats(out[k])["lead"]
    out[f"S_{k}_mean"] = stats(out[k])["mean"]
np.savez_compressed("cache4.npz", **out)
for k in ["ref", "ali", "den"]:
    print(k, "mean %.3f lead %.3f dens %.2f aniso %.2f" % tuple(out[f"S_{k}_{s}"].mean() for s in ["mean", "lead", "dens", "aniso"]))
print("max dXC ali-ref", np.abs(Rb(ali) - Rb(ref)).max(), "den-ref", np.abs(Rb(den) - Rb(ref)).max())
for k in ["t0", "t1", "t2"]:
    print(k, out[f"S_{k}_lead"].mean(), out[f"S_{k}_mean"].mean())
