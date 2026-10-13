import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle, Circle, Wedge, Ellipse
from style import *
from arctic import DC, DM, DH

D = np.load("cache2.npz")
E = np.load("cache4.npz")
W, H = 7.2, 7.55
fig = plt.figure(figsize=(W, H))
LINK, SHAPE, LEARN = "#7a4fa3", "#b07a12", "#7d8696"


def rgba(img, cmap, lim):
    a = cmap(np.clip((img - lim[0]) / (lim[1] - lim[0]), 0, 1))
    a[np.isnan(img)] = mpl.colors.to_rgba(LAND)
    return a


def img_ax(x, y, w, h, col=MUTED, lw=0.8):
    ax = fig.add_axes([x / W, y / H, w / W, h / H])
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(col); s.set_linewidth(lw)
    return ax


def cgrid(ax, n, m=4, col="white", lw=0.35, alpha=0.8):
    s = []
    for k in range(1, m):
        s += [[(k * n / m - .5, -.5), (k * n / m - .5, n - .5)], [(-.5, k * n / m - .5), (n - .5, k * n / m - .5)]]
    ax.add_collection(LineCollection(s, colors=col, lw=lw, alpha=alpha))


def mark(ax, x, y, kind, col, r=0.034, z=6):
    if kind == "full":
        ax.add_patch(Circle((x, y), r, fc=col, ec=col, lw=0.5, zorder=z))
    elif kind == "half":
        ax.add_patch(Wedge((x, y), r, 90, 270, fc=col, ec="none", zorder=z))
        ax.add_patch(Circle((x, y), r, fc="none", ec=col, lw=0.5, zorder=z))
    else:
        ax.add_patch(Circle((x, y), r, fc="white", ec=col, lw=0.5, zorder=z))


def chip(ax, x, y, txt, kind, col, w=None, fs=5.6, dashed=False, tag=None, h=0.17):
    w = w or 0.13 + 0.047 * len(txt.replace("$", "").replace("\\", "").replace("_", "")) * fs / 5.6
    rbox(ax, x, y - h / 2, w, h, fc="white", ec=col, lw=0.55, r=0.035, z=4, ls=(0, (2, 1.2)) if dashed else "-")
    mark(ax, x + 0.085, y, kind, col)
    ax.text(x + 0.15, y - 0.004, txt, fontsize=fs, va="center", ha="left", color=INK, zorder=7)
    if tag:
        ax.add_patch(Rectangle((x + w - 0.045, y - h / 2 + 0.02), 0.025, h - 0.04, fc=tag, ec="none", zorder=7))
    return x + w


# ======================================================================= (a)
ya = H - 0.22
panel_title(fig, 0.06, ya, "a", "What each level carries: state $X_\\ell$ (generated), sub-grid parameters $S_\\ell$ (explicit), observations $Y_\\ell$", W, H)
axA = inch_axes(fig, 0.06, H - 2.74, 7.08, 2.42, W, H)
axA.set_ylim(-0.12, 2.30)
rows = [("C", r"$\mathcal{C}$", "32 km", 1.80), ("M", r"$\mathcal{M}$", "4 km", 1.09), ("H", r"$\mathcal{H}$", "1 km", 0.38)]
# headers
for x, t in [(0.98, "state $X_\\ell$ (flow matching target)"), (3.02, "sub-grid parameters $S_\\ell$ (computed by $\\mathcal{S}$ from finer obs.)"),
             (6.00, "observations $Y_\\ell$")]:
    axA.text(x, 2.22, t, fontsize=6.0, color=INK, fontweight="bold", va="center")
# thumbnails: X_C, dX_M, dX_H (same 128-km window as Fig. 1)
WI, WJ = 2, 3
dM = D["dec_XM"] - np.repeat(np.repeat(D["dec_XC"], 8, 0), 8, 1)
thumbs = {"C": (D["dec_XC"][WJ:WJ + 4, WI:WI + 4], ICE, (0, 1)),
          "M": (dM[WJ * 8:WJ * 8 + 32, WI * 8:WI * 8 + 32], BAL, (-0.5, 0.5)),
          "H": (D["dec_dH"][WJ * 32:WJ * 32 + 128, WI * 32:WI * 32 + 128], BAL, (-0.3, 0.3))}
th = 0.60
for lv, sym, km, yc in rows:
    rbox(axA, 0.0, yc - 0.33, 7.06, 0.66, fc=TINT[lv], ec="none", r=0.05, z=0, alpha=0.55)
    axA.text(0.07, yc + 0.11, sym, fontsize=10, color=COL[lv], fontweight="bold", va="center")
    axA.text(0.07, yc - 0.10, km, fontsize=5.6, color=DARK[lv], va="center")
    img, cm, lim = thumbs[lv]
    ia = img_ax(0.06 + 0.33, H - 2.62 + yc - th / 2, th, th, COL[lv], 0.8)
    ia.imshow(rgba(img, cm, lim), origin="lower", interpolation="nearest")
    if lv != "C":
        cgrid(ia, img.shape[0], col=COL["C"], lw=0.5)
    ia.text(0.5, -0.03, ["$X_\\mathcal{C}$", "$\\delta X_\\mathcal{M}$", "$\\delta X_\\mathcal{H}$"]["CMH".index(lv)],
            transform=ia.transAxes, fontsize=5.4, ha="center", va="top", color=DARK[lv])
# --- state chips
c, m, h = COL["C"], COL["M"], COL["H"]
x = 0.98
for t, k, w in [("SIC", "full", 0.40), ("$u$", "full", 0.29), ("$v$", "full", 0.29), ("SIT", "half", 0.38), ("snow", "none", 0.44)]:
    x = chip(axA, x, 1.89, t, k, c, w=w) + 0.04
axA.text(0.98, 1.62, "snow: weakly observed, mostly inferred via the joint law", fontsize=5.0, color=MUTED, va="center")
x = 0.98
for t, k, dsh, w in [("$\\delta$SIC", "full", False, 0.47), ("$\\delta u$", "half", False, 0.34), ("$\\delta v$", "half", False, 0.34), ("$\\delta$SIT", "half", True, 0.46)]:
    x = chip(axA, x, 1.18, t, k, m, dashed=dsh, w=w) + 0.04
axA.text(0.98, 0.91, "$\\delta u,\\delta v$: SAR drift (regional);  $\\delta$SIT: along-track only (ablation)", fontsize=5.0, color=MUTED, va="center")
chip(axA, 0.98, 0.47, "$\\delta$SIC", "full", h, w=0.47)
axA.text(0.98, 0.20, "no $\\delta u,\\delta v$, SIT, snow at 1 km", fontsize=5.0, color=MUTED, va="center")
# --- S chips
x = 3.02
for t, k, tg, w in [("open water", "full", LINK, 0.66), ("SIC hist.", "full", LINK, 0.56), ("lead frac.", "full", LINK, 0.58),
                 ("lead density", "full", SHAPE, 0.70)]:
    x = chip(axA, x, 1.95, t, k, c, tag=tg, w=w, fs=5.2) + 0.03
x = 3.02
for t, k, tg, w in [("anisotropy", "full", SHAPE, 0.62), ("deformation", "half", SHAPE, 0.66), ("thin-ice frac.", "half", LINK, 0.70),
                 ("SIT var./ridged", "half", SHAPE, 0.78)]:
    x = chip(axA, x, 1.72, t, k, c, tag=tg, w=w, fs=5.2) + 0.03
axA.text(3.02, 1.52, "+ $\\psi_\\mathcal{C}$ (small learned residual)  ·  damage (not obs.) $\\rightarrow$ fracture state",
         fontsize=5.0, color=MUTED, va="center")
x = 3.02
for t, k, tg, w in [("open water", "full", LINK, 0.66), ("lead frac.", "full", LINK, 0.58), ("SIC var.", "full", SHAPE, 0.54)]:
    x = chip(axA, x, 1.18, t, k, m, tag=tg, dashed=True, w=w, fs=5.2) + 0.03
axA.text(3.02, 0.91, "optional $S_\\mathcal{M}$ (from 500-m ASIP): localises $S_\\mathcal{C}$ in the cell;  additive stats nest: $S_\\mathcal{C}^{\\rm frac}=R\\,S_\\mathcal{M}^{\\rm frac}$", fontsize=5.0, color=MUTED, va="center")
axA.text(3.02, 0.40, "— finest level: nothing below 1 km is represented", fontsize=5.4, color=MUTED, va="center")
# --- obs
obs = {"C": ["PMW SIC (AMSR2, CIMR)", "PMW drift, buoys (IABP)", "CS2SMOS, SMOS, snow"],
       "M": ["SAR drift (Sentinel-1)", "altimetry tracks (CS2, IS2)", "SIC from SAR, aggregated"],
       "H": ["ASIP SIC (SAR, 0.5–1 km)", "inside SAR swaths only", ""]}
for lv, sym, km, yc in rows:
    for k, t in enumerate(obs[lv]):
        axA.text(6.00, yc + 0.22 - 0.14 * k, t, fontsize=5.1, color=INK, va="center")
axA.text(6.00, 1.55, "+ forcing $f$ (ERA5) $\\rightarrow E_F$, $\\Pi^{\\downarrow}\\!f$", fontsize=5.0, color=MUTED, va="center")
axA.plot([5.94, 5.94], [0.05, 2.12], color=RULE, lw=0.5)
axA.plot([2.94, 2.94], [0.05, 2.12], color=RULE, lw=0.5)
# legend + forcing
lg = 0.98
axA.text(lg, 2.06, "", fontsize=5)
yl = -0.06
for xx, k, t in [(0.98, "full", "observed (dense)"), (1.86, "half", "partial / sparse"), (2.66, "none", "inferred")]:
    mark(axA, xx, yl, k, INK, r=0.03); axA.text(xx + 0.06, yl, t, fontsize=5.0, va="center", color=INK)
for xx, col, t in [(3.42, LINK, "linked to $X_\\mathcal{C}$ (edit $\\Rightarrow$ $X_\\mathcal{C}$ changes via $R$)"),
                   (5.35, SHAPE, "shape (editable at fixed $X_\\mathcal{C}$)")]:
    axA.add_patch(Rectangle((xx - 0.012, yl - 0.06), 0.025, 0.12, fc=col, ec="none"))
    axA.text(xx + 0.05, yl, t, fontsize=5.0, va="center", color=INK)
# forcing note (right of title row)

# ======================================================================= (b)
yb = H - 2.95
panel_title(fig, 0.06, yb, "b", "$S$ plays two roles: closure upward, constraint downward (same named quantities)", W, H)
y0b = H - 4.92
axB = inch_axes(fig, 0.06, y0b, 7.08, 1.86, W, H)
rbox(axB, 0.0, 0.0, 2.92, 1.80, fc=PANEL_BG, ec="none", r=0.05, z=0)
rbox(axB, 3.02, 0.0, 4.06, 1.80, fc=PANEL_BG, ec="none", r=0.05, z=0)
axB.text(0.08, 1.70, "$\\uparrow$ closure:  $S_\\mathcal{C}=\\mathcal{S}(X_\\mathcal{H})$  (also the supervision target, from $Y_\\mathcal{H}$)",
         fontsize=5.9, color=INK, va="center")
axB.text(3.10, 1.70, "$\\downarrow$ constraint:  $\\delta X_\\mathcal{H}\\sim p(\\,\\cdot\\mid X_\\mathcal{C},\\,\\delta X_\\mathcal{M},\\,\\Pi^{\\downarrow}S_\\mathcal{C},\\,\\Pi^{\\downarrow}f)$  — same $X_\\mathcal{C}$, three targets",
         fontsize=5.9, color=INK, va="center")
# closure: field with ellipses
fs_ = 1.42
ref = E["ref"]
ia = img_ax(0.14, y0b + 0.12, fs_, fs_, COL["H"], 0.8)
ia.imshow(rgba(ref, ICE, (0.4, 1)), origin="lower", interpolation="nearest")
cgrid(ia, 128, col=COL["C"], lw=0.6)
th_, an_, le_ = E["S_ref_theta"], E["S_ref_aniso"], E["S_ref_lead"]
for j in range(4):
    for i in range(4):
        L = 10 + 140 * le_[j, i]
        ia.add_patch(Ellipse((i * 32 + 15.5, j * 32 + 15.5), L, L * (1 - 0.85 * an_[j, i]),
                             angle=np.degrees(th_[j, i]), fc="none", ec=SHAPE, lw=0.8))
ia.add_patch(Rectangle((2 * 32 - .5, 1 * 32 - .5), 32, 32, fc="none", ec=INK, lw=1.0))
ia.set_xlim(-.5, 127.5); ia.set_ylim(-.5, 127.5)
ia.text(0.0, -0.03, "$X_\\mathcal{H}$, 128 km; ellipses: lead fraction (size),\nanisotropy and orientation per $\\mathcal{C}$ cell",
        transform=ia.transAxes, fontsize=4.9, va="top", color=MUTED)
arrow(axB, (1.62, 0.83), (1.78, 0.83), color=INK, lw=0.8)
axB.text(1.70, 0.92, "$\\mathcal{S}$", fontsize=7, ha="center", color=INK)
# S card for highlighted cell
rbox(axB, 1.82, 0.12, 1.02, 1.44, fc="white", ec=COL["C"], lw=0.6, z=1)
axB.text(1.88, 1.46, "$S_\\mathcal{C}$ of the boxed cell", fontsize=5.4, color=COL["C"], va="center", fontweight="bold")
jj, ii = 1, 2
vals = [("open water", E["S_ref_ow"][jj, ii], 0.25, LINK), ("lead frac.", E["S_ref_lead"][jj, ii], 0.25, LINK),
        ("lead dens.", E["S_ref_dens"][jj, ii], 8.0, SHAPE), ("anisotropy", E["S_ref_aniso"][jj, ii], 1.0, SHAPE)]
for k, (t, v, vm, col) in enumerate(vals):
    yy = 1.30 - 0.16 * k
    axB.text(1.88, yy, t, fontsize=4.9, va="center", color=INK)
    axB.add_patch(Rectangle((2.32, yy - 0.035), 0.46, 0.07, fc="#eceef2", ec="none"))
    axB.add_patch(Rectangle((2.32, yy - 0.035), 0.46 * min(v / vm, 1), 0.07, fc=col, ec="none"))
    axB.text(2.79, yy + 0.06, f"{v:.2f}", fontsize=4.4, ha="right", va="bottom", color=MUTED)
hst = E["S_ref_hist"][jj, ii]
axB.text(1.88, 0.62, "SIC histogram", fontsize=4.9, va="center", color=INK)
for k, v in enumerate(hst):
    hh = 0.30 * np.sqrt(v)
    axB.add_patch(Rectangle((1.92 + k * 0.17, 0.22), 0.13, hh, fc=ICE(0.15 + 0.17 * k), ec=RULE, lw=0.3))
axB.text(1.92, 0.17, "0", fontsize=4.4, va="top", color=MUTED); axB.text(2.76, 0.17, "1", fontsize=4.4, va="top", color=MUTED)
# constraint: three realisations
cases = [("ref", "reference $S_\\mathcal{C}$", ""), ("ali", "edit anisotropy (shape)", "$X_\\mathcal{C}$ ≈ fixed"),
         ("den", "edit lead fraction (linked)", "$X_\\mathcal{C}$ decreases")]
xs = [3.16, 4.22, 5.28]
s1 = 0.98
for (k, t, t2), xx in zip(cases, xs):
    ia = img_ax(0.06 + xx, y0b + 0.50, s1, s1, COL["H"], 0.8)
    ia.imshow(rgba(E[k], ICE, (0.4, 1)), origin="lower", interpolation="nearest")
    cgrid(ia, 128, col=COL["C"], lw=0.45)
    ia.text(0.0, 1.03, t, transform=ia.transAxes, fontsize=5.1, va="bottom", color=INK)
    S = {s: E[f"S_{k}_{s}"].mean() for s in ["mean", "lead", "dens", "aniso"]}
    txt = (f"realised $\\mathcal{{S}}$:  lead {S['lead']:.3f}\n"
           f"density {S['dens']:.2f}  ·  aniso. {S['aniso']:.2f}\n"
           f"$\\bar X_\\mathcal{{C}}$ = {S['mean']:.3f}  {t2}")
    axB.text(xx, 0.42, txt, fontsize=4.8, va="top", color=INK, linespacing=1.35)
# controllability inset
cx0 = 6.42
ci = fig.add_axes([(0.06 + cx0) / W, (y0b + 0.50) / H, 0.58 / W, 0.80 / H])
tt = np.linspace(0, 1, 10)
ci.plot([0, 1], [0, 1], color=RULE, lw=0.6, ls=(0, (2, 1.5)))
rng = np.random.default_rng(3)
ci.scatter(tt, tt + 0.05 * rng.normal(size=10), s=4, color=SHAPE, lw=0, zorder=3)
ci.scatter(tt, 0.45 + 0.04 * rng.normal(size=10), s=4, color=LEARN, lw=0, zorder=3)
ci.text(0.04, 0.96, "slope ≈ 1:\nusable knob", fontsize=4.3, color=SHAPE, ha="left", va="top", transform=ci.transAxes)
ci.text(0.98, 0.32, "flat: ignored", fontsize=4.3, color=LEARN, ha="right", va="top", transform=ci.transAxes)
ci.set_xlim(-0.05, 1.05); ci.set_ylim(-0.05, 1.1); ci.set_xticks([]); ci.set_yticks([])
for s in ci.spines.values():
    s.set_linewidth(0.4); s.set_color(MUTED)
ci.set_xlabel("target $S_\\mathcal{C}$", fontsize=4.8, labelpad=1)
ci.set_ylabel("$\\mathcal{S}(X_\\mathcal{H})$", fontsize=4.8, labelpad=1)
ci.set_title("controllability", fontsize=5.2, pad=2)
axB.text(cx0, 0.30, "cycle consistency:\n$\\|\\mathcal{S}(\\hat x_1^\\mathcal{H})-S_\\mathcal{C}\\|^2$\n(penalty or guidance)",
         fontsize=4.8, va="top", color=INK, linespacing=1.3)

# ======================================================================= (c)
yc = H - 5.22
panel_title(fig, 0.06, yc, "c", "Mode A rollout: $G_\\mathcal{C}$ advances $(X_\\mathcal{C},S_\\mathcal{C})$; fine scales on demand", W, H)
y0c = 0.22
WC = 4.62
axC = inch_axes(fig, 0.06, y0c, WC, 1.90, W, H)
rbox(axC, 0.0, 0.0, WC, 1.90, fc=PANEL_BG, ec="none", r=0.05, z=0)
tx = [0.66, 1.76, 2.86]
bw = 0.82
labs = ["$t$", "$t+\\Delta$", "$t+2\\Delta$"]
ev = ["", "divergence event\n(strong winds)", "cold spell\n(leads refreeze)"]
keys = ["t0", "t1", "t2"]
# row labels
axC.text(0.08, 1.74, "forcing $f$\n(ERA5)", fontsize=5.0, color=MUTED, va="center", linespacing=1.2)
axC.text(0.08, 1.20, "coarse,\nprognostic", fontsize=5.0, color=DARK["C"], va="center", linespacing=1.2, fontweight="bold")
axC.text(0.08, 0.42, "fine,\non demand\n(diagnostic)\n\n$G_\\mathcal{M},G_\\mathcal{H}\\mid$\n$X_\\mathcal{C},S_\\mathcal{C},\\Pi^{\\downarrow}\\!f$",
         fontsize=5.0, color=DARK["H"], va="center", linespacing=1.2)
for k in range(1, 3):
    xm = (tx[k - 1] + bw + tx[k]) / 2
    axC.text(xm, 1.76, ev[k], fontsize=4.8, color=INK, ha="center", va="center", linespacing=1.2)
    arrow(axC, (xm, 1.63), (xm, 1.36), color=MUTED, lw=0.6)
for k in range(3):
    x = tx[k]
    rbox(axC, x, 0.96, bw, 0.46, fc="white", ec=COL["C"], lw=0.7, z=2)
    xc = E[keys[k]].reshape(4, 32, 4, 32).mean((1, 3))
    ia = img_ax(0.06 + x + 0.03, y0c + 1.00, 0.37, 0.37, COL["C"], 0.6)
    ia.imshow(rgba(xc, ICE, (0.75, 1)), origin="lower", interpolation="nearest")
    ib = img_ax(0.06 + x + 0.42, y0c + 1.00, 0.37, 0.37, LINK, 0.6)
    ib.imshow(E[f"S_{keys[k]}_lead"], origin="lower", cmap="Purples", vmin=0, vmax=0.3, interpolation="nearest")
    axC.text(x + 0.215, 0.92, "$X_\\mathcal{C}$", fontsize=5.0, ha="center", va="top", color=COL["C"])
    axC.text(x + 0.605, 0.92, "$S_\\mathcal{C}$ (leads)", fontsize=4.6, ha="center", va="top", color=LINK)
    axC.text(x + bw / 2, 1.45, labs[k], fontsize=5.8, ha="center", va="bottom", color=INK)
    if k < 2:
        arrow(axC, (x + bw + 0.01, 1.19), (tx[k + 1] - 0.01, 1.19), color=COL["C"], lw=1.1, ms=6)
        axC.text((x + bw + tx[k + 1]) / 2, 1.23, "$G_\\mathcal{C}$", fontsize=6.2, ha="center", va="bottom", color=DARK["C"])
    ia = img_ax(0.06 + x + 0.16, y0c + 0.08, 0.50, 0.50, COL["H"], 0.7)
    ia.imshow(rgba(E[keys[k]], ICE, (0.4, 1)), origin="lower", interpolation="nearest")
    arrow(axC, (x + 0.41, 0.80), (x + 0.41, 0.61), color=COL["H"], lw=0.7, ls=(0, (2, 1.2)))
    axC.text(x + 0.70, 0.34, "$X_\\mathcal{H}$", fontsize=5.0, va="center", color=DARK["H"])
arrow(axC, (tx[2] + bw + 0.01, 1.19), (tx[2] + bw + 0.22, 1.19), color=COL["C"], lw=1.1, ms=6, ls=(0, (2, 1.5)))
xn = 3.96
axC.text(xn, 1.26, "no feedback from\ngenerated $X_\\mathcal{H}$\nin mode A:\nsmall→large is\ncarried by the\nlearned dynamics\nof $S_\\mathcal{C}$ in $G_\\mathcal{C}$",
         fontsize=4.7, color=INK, va="top", linespacing=1.25)
axC.text(xn, 0.42, "obs. available:\n$+\\,w\\,[v(\\cdot|h)-v(\\cdot)]$\n= nudging", fontsize=4.7, color=MUTED, va="top", linespacing=1.25)

# ======================================================================= (d)
xd = 4.86
panel_title(fig, xd, yc, "d", "Interventions on $S_\\mathcal{C}$ (schematic)", W, H)
ax = fig.add_axes([(xd + 0.30) / W, (y0c + 1.00) / H, 1.98 / W, 0.88 / H])
t = np.linspace(0, 10, 300)
base = 0.08 + 0.02 * np.sin(t * 0.9)
ax.plot(t, base, color=INK, lw=0.9)
ax.text(9.9, 0.062, "control", fontsize=4.6, ha="right", color=INK)
ax.plot(t, base + 0.10 * np.exp(-t / 2.2), color="#2a7fb8", lw=0.8)
ax.plot(t, base + 0.05 * (1 - np.exp(-np.clip(t - 2, 0, None) / 1.5)) * (t > 2), color="#d4561c", lw=0.8)
ax.plot(t, np.where(t > 3, 0.20, np.nan), color=LINK, lw=0.8)
ax.plot(t, np.where(t > 3, base + (0.20 - base) * (1 - np.exp(-(t - 3) / 1.8)) * 0.8, np.nan), color=SHAPE, lw=0.8, ls=(0, (3, 1.2)))
for xx, yy, n_, col in [(0.3, 0.183, "1", "#2a7fb8"), (6.4, 0.152, "2", "#d4561c"), (8.8, 0.212, "3", LINK), (8.8, 0.17, "4", SHAPE)]:
    ax.text(xx, yy, n_, fontsize=5.0, color="white", ha="center", va="center", fontweight="bold",
            bbox=dict(boxstyle="circle,pad=0.12", fc=col, ec="none"))
ax.set_xlim(0, 10); ax.set_ylim(0.03, 0.24); ax.set_xticks([]); ax.set_yticks([])
for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)
for sp in ["left", "bottom"]:
    ax.spines[sp].set_linewidth(0.4); ax.spines[sp].set_color(MUTED)
ax.set_xlabel("rollout time", fontsize=4.9, labelpad=1); ax.set_ylabel("lead frac. $S_\\mathcal{C}$", fontsize=4.9, labelpad=1)
axD = inch_axes(fig, xd, 0.05, 2.30, 1.10, W, H)
items = [("1", "#2a7fb8", "edit $S_\\mathcal{C}^0$, then free: memory"),
         ("2", "#d4561c", "edit $f$: $S_\\mathcal{C}$ responds"),
         ("3", LINK, "do$(S_\\mathcal{C}=s^*)$: clamped, causal test"),
         ("4", SHAPE, "nudge towards $s^*$ with time $\\tau$")]
for k, (n_, col, tx_) in enumerate(items):
    yy = 0.98 - 0.13 * k
    axD.text(0.38, yy, n_, fontsize=4.8, color="white", ha="center", va="center", fontweight="bold",
             bbox=dict(boxstyle="circle,pad=0.12", fc=col, ec="none"))
    axD.text(0.49, yy, tx_, fontsize=4.9, va="center", color=INK)
axD.text(0.30, 0.36, "after any edit, project $S_\\mathcal{C}$ on the consistent\nset (histogram mean = SIC$_\\mathcal{C}$), like $\\Pi_0$ for $\\delta X$",
         fontsize=4.7, va="center", color=MUTED, linespacing=1.3)
axD.text(0.30, 0.10, "Mode B (validation): $G_\\mathcal{C}\\!\\rightarrow\\! G_\\mathcal{M}\\!\\rightarrow\\! G_\\mathcal{H}\\!\\rightarrow\\!\\mathcal{S}\\!\\rightarrow\\! S_\\mathcal{C}$\n"
         "at each step;  A ≈ B $\\Rightarrow$ the closure $S_\\mathcal{C}$ is sufficient",
         fontsize=4.7, va="center", color=INK, linespacing=1.3)

for ext in ["pdf", "png"]:
    fig.savefig(f"CROSCIM_fig4_states_params.{ext}", dpi=400 if ext == "png" else None)
