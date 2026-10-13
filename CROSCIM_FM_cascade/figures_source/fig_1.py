import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle, ConnectionPatch, Polygon
from style import *
from arctic import fwd, X0, DC, DM, DH, NE

D = np.load("cache2.npz")
W, H = 7.2, 5.32
fig = plt.figure(figsize=(W, H))

def rgba(img, cmap, lim):
    a = cmap(np.clip((img - lim[0]) / (lim[1] - lim[0]), 0, 1))
    a[np.isnan(img)] = mpl.colors.to_rgba(LAND)
    return a

def grid_lines(ax, x0, y0, n, d, **kw):
    s = []
    for k in range(n + 1):
        s.append([(x0 + k * d, y0), (x0 + k * d, y0 + n * d)])
        s.append([(x0, y0 + k * d), (x0 + n * d, y0 + k * d)])
    ax.add_collection(LineCollection(s, **kw))

def frame(ax, col, lw=0.9):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(col); s.set_linewidth(lw)

x0d, y0d = D["dec_origin"]
iM, jM = D["zM_ij"]; aH, bH = D["zH_ab"]
# region origins (km)
zM0 = (x0d + iM * DC, y0d + jM * DC)                  # 2x2 C = 16x16 M
zH0 = (zM0[0] + aH * DM, zM0[1] + bH * DM)            # 4x4 M = 16x16 H
cC = (x0d + 4 * DC, y0d + 4 * DC)
zC0 = (cC[0] - 8 * DC, cC[1] - 8 * DC)                # 16x16 C

# ============================================================ (a)
panel_title(fig, 0.06, H - 0.20, "a",
            "Nested polar-stereographic grid: what each level resolves (nesting ×8, then ×4)", W, H)
# overview
ov = fig.add_axes([0.06 / W, (H - 2.12) / H, 1.70 / W, 1.70 / H])
ext = [X0, -X0, X0, -X0]
ov.imshow(rgba(D["XC_all"], ICE, (0, 1)), origin="lower", extent=ext, interpolation="nearest")
from field import iter_rings
segs = []
for ring in iter_rings(f"{NE}/ne_50m_land.geojson"):
    xy = np.array(ring["xy"])
    if xy[:, 1].max() < 45:
        continue
    x, y = fwd(xy[:, 0], xy[:, 1])
    segs.append(np.stack([x, y], -1))
ov.add_collection(LineCollection(segs, colors=COAST, lw=0.25))
for la in (60, 70, 80):
    lo = np.linspace(-180, 180, 361); x, y = fwd(lo, np.full_like(lo, la))
    ov.plot(x, y, color="white", lw=0.25, alpha=0.5)
ov.add_patch(Rectangle((150, -2250), 1350, 2050, fc="none", ec=COL["H"], lw=0.9, ls=(0, (3, 1.5))))
ov.set_xlim(-3000, 3000); ov.set_ylim(-3000, 3000); ov.set_aspect("equal"); frame(ov, MUTED, 0.5)
ov.text(0.0, 1.03, "Arctic domain, $X_\\mathcal{C}$", transform=ov.transAxes, fontsize=6.4, color=INK, va="bottom")
ov.text(0.0, -0.04, "6400 km × 6400 km  ·  200×200 $\\mathcal{C}$ cells\norange dashed: ROI where $\\mathcal{H}$ is generated\n(SAR coverage, illustrative)", transform=ov.transAxes,
        fontsize=5.2, color=MUTED, va="top", linespacing=1.3)

zs, zy = 1.55, H - 2.10
zx = [1.95, 3.57, 5.19]
WI, WJ = 2, 3                                          # window origin in C cells within (b) region
w0 = (x0d + WI * DC, y0d + WJ * DC)                    # 128-km window = 4x4 C = 32x32 M = 128x128 H
axes = []
specs = [
    ("C", DC, D["dec_XC"], r"Coarse $\mathcal{C}$", "Δ = 32 km", "4×4 cells: ice edge, pack mean"),
    ("M", DM, D["dec_XM"], r"Medium $\mathcal{M}$", "Δ = 4 km", "32×32 cells: MIZ, wide leads"),
    ("H", DH, D["dec_XH"], r"High-res. $\mathcal{H}$", "Δ = 1 km", "128×128 cells: leads, floes"),
]
for k, (lv, d, src, t1, t2, t3) in enumerate(specs):
    ax = fig.add_axes([zx[k] / W, zy / H, zs / W, zs / H])
    n = int(4 * DC / d)
    i0 = int(round((w0[0] - x0d) / d)); j0 = int(round((w0[1] - y0d) / d))
    img = src[j0:j0 + n, i0:i0 + n]
    ext_w = [w0[0], w0[0] + 4 * DC, w0[1], w0[1] + 4 * DC]
    ax.imshow(rgba(img, ICE, (0, 1)), origin="lower", extent=ext_w, interpolation="nearest")
    if lv == "M":
        grid_lines(ax, w0[0], w0[1], 32, DM, colors="white", lw=0.18, alpha=0.45)
    grid_lines(ax, w0[0], w0[1], 4, DC, colors=COL["C"], lw=0.9 if lv != "C" else 0.6,
               alpha=1 if lv != "C" else 0.9)
    ax.set_xlim(ext_w[:2]); ax.set_ylim(ext_w[2:]); ax.set_aspect("equal")
    frame(ax, COL[lv])
    ax.text(0, 1.035, t1, transform=ax.transAxes, color=COL[lv], fontsize=7.0, fontweight="bold", va="bottom")
    ax.text(1, 1.035, t2, transform=ax.transAxes, color=INK, fontsize=6.0, va="bottom", ha="right")
    ax.text(0, -0.04, t3, transform=ax.transAxes, color=MUTED, fontsize=5.4, va="top")
    axes.append(ax)
fig.text(zx[0] / W, (zy - 0.25) / H, "same 128-km window at the three levels ($X_\\mathcal{M}=RX_\\mathcal{H}$, "
         "$X_\\mathcal{C}=RX_\\mathcal{M}$); blue lines: $\\mathcal{C}$ cells", fontsize=5.4, color=INK)
ov.add_patch(Rectangle((x0d, y0d), 8 * DC, 8 * DC, fc="none", ec=COL["C"], lw=1.0))
for p, q in [((x0d + 8 * DC, y0d + 8 * DC), (0, 1)), ((x0d + 8 * DC, y0d), (0, 0))]:
    fig.add_artist(ConnectionPatch(xyA=p, coordsA=ov.transData, xyB=q, coordsB=axes[0].transAxes,
                                   color=MUTED, lw=0.45, ls=(0, (2, 1.4))))
cax = fig.add_axes([6.86 / W, (zy + 0.20) / H, 0.07 / W, 1.15 / H])
cb = fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, 1), cmap=ICE), cax=cax)
cb.outline.set_linewidth(0.4); cb.ax.tick_params(labelsize=5.4, width=0.4, length=2)
cb.set_ticks([0, 0.5, 1]); cax.set_title("SIC", fontsize=5.8, pad=3)

# ============================================================ (b)
yb = H - 2.86
panel_title(fig, 0.06, yb, "b", "Exact scale decomposition (synthetic SIC, Fram Strait)", W, H)
ms = 0.98; my = H - 4.10
mx = [0.12, 1.24, 2.36, 3.48]
ext = [x0d, x0d + 8 * DC, y0d, y0d + 8 * DC]
lm = np.ceil(np.nanpercentile(np.abs(D["dec_PdM"]), 99.5) * 10) / 10
lh = np.ceil(np.nanpercentile(np.abs(D["dec_dH"]), 99.5) * 10) / 10
maps = [("dec_XH", ICE, (0, 1), r"$X_\mathcal{H}$", "target (1 km)", "H"),
        ("dec_PXC", ICE, (0, 1), r"$P_{\mathcal{C}\to\mathcal{H}}X_\mathcal{C}$", "large scale", "C"),
        ("dec_PdM", BAL, (-lm, lm), r"$P_{\mathcal{M}\to\mathcal{H}}\,\delta X_\mathcal{M}$", "MIZ, large leads", "M"),
        ("dec_dH", BAL, (-lh, lh), r"$\delta X_\mathcal{H}$", "leads, floes", "H")]
for k, (key, cm, lim, lab, sub, lv) in enumerate(maps):
    ax = fig.add_axes([mx[k] / W, my / H, ms / W, ms / H])
    ax.imshow(rgba(D[key], cm, lim), origin="lower", extent=ext, interpolation="nearest")
    if key == "dec_XH":
        ax.add_patch(Rectangle(w0, 4 * DC, 4 * DC, fc="none", ec=INK, lw=0.6, ls=(0, (2.5, 1.5))))
    if key in ("dec_PXC", "dec_PdM"):
        grid_lines(ax, x0d, y0d, 8, DC, colors="white" if key == "dec_PXC" else COL["C"], lw=0.25, alpha=0.6)
    ax.set_xlim(ext[:2]); ax.set_ylim(ext[2:]); frame(ax, COL[lv], 0.8)
    ax.text(0.5, 1.12, lab, transform=ax.transAxes, ha="center", va="bottom", fontsize=7.0)
    ax.text(0.5, 1.03, sub, transform=ax.transAxes, ha="center", va="bottom", fontsize=5.3, color=MUTED)
    if k:
        fig.text((mx[k] - 0.07) / W, (my + ms / 2) / H, "=" if k == 1 else "+", fontsize=10.5, ha="center", va="center")

def hcb(x, w, cm, lim, ticks):
    cax = fig.add_axes([x / W, (my - 0.10) / H, w / W, 0.045 / H])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(*lim), cmap=cm), cax=cax, orientation="horizontal")
    cb.outline.set_linewidth(0.3); cb.ax.tick_params(labelsize=4.9, width=0.3, length=1.5, pad=1); cb.set_ticks(ticks)
hcb(0.30, 1.85, ICE, (0, 1), [0, 0.5, 1])
hcb(2.49, 0.72, BAL, (-lm, lm), [-lm, 0, lm])
hcb(3.61, 0.72, BAL, (-lh, lh), [-lh, 0, lh])

axE = inch_axes(fig, 0.06, H - 5.26, 4.42, 0.72, W, H)
rbox(axE, 0.04, 0.02, 4.36, 0.68, fc=PANEL_BG, ec=RULE, lw=0.5)
cols = [
    ("Restriction $R$ (fixed)", r"$X_\mathcal{M}(j)=\Sigma_{i\in\mathrm{ch}(j)}\,\tilde\omega_i X_\mathcal{H}(i)$",
     "ocean-weighted block mean"),
    ("Prolongation $P$ (fixed)", r"$(PX_\mathcal{M})(i)=X_\mathcal{M}(\mathrm{par}(i))$", "injection, $RP=I$"),
    ("Residual DOF", r"$\delta X_\mathcal{M}=X_\mathcal{M}-PX_\mathcal{C}$,  $R\,\delta X=0$",
     "zero mean per parent"),
]
for k, (t, e, n) in enumerate(cols):
    xx = 0.13 + k * 1.45
    axE.text(xx, 0.58, t, fontsize=5.7, fontweight="bold", color=INK, va="center")
    axE.text(xx, 0.36, e, fontsize=5.9, color=INK, va="center")
    axE.text(xx, 0.14, n, fontsize=5.0, color=MUTED, va="center")

# ============================================================ (c)
panel_title(fig, 4.70, yb, "c", "Chain rule, coarse $\\rightarrow$ fine", W, H)
axc = inch_axes(fig, 4.66, H - 5.26, 2.50, 2.34, W, H)
SCOL, OCOL = "#7a4fa3", "#2a7fb8"
nodes = [("C", r"$Z_\mathcal{C}=(X_\mathcal{C},\,S_\mathcal{C})$", r"$c_\mathcal{C}=(h_\mathcal{C},\,f)$", 1.96),
         ("M", r"$Z_\mathcal{M}=(\delta X_\mathcal{M},\,S_\mathcal{M})$", r"$c_\mathcal{M}=(h_\mathcal{M},\Pi^{\downarrow}\!f,\Pi^{\downarrow}\Phi_\mathcal{C})$", 1.34),
         ("H", r"$\delta X_\mathcal{H}$", r"$c_\mathcal{H}=(h_\mathcal{H},\Pi^{\downarrow}\!f,\Pi^{\downarrow}\Phi_\mathcal{M})$", 0.72)]
for lv, tz, tc, yy in nodes:
    axc.text(0.02, yy, f"$G_\\mathcal{{{lv}}}$", fontsize=7.2, color=DARK[lv], va="center", fontweight="bold")
    rbox(axc, 0.30, yy - 0.17, 1.02, 0.34, fc=TINT[lv], ec=COL[lv], lw=0.7, z=2)
    axc.text(0.81, yy, tz, fontsize=5.6, ha="center", va="center", color=INK, zorder=3)
    rbox(axc, 1.50, yy - 0.15, 0.98, 0.30, fc="white", ec=OCOL, lw=0.55, z=2)
    axc.text(1.99, yy, tc, fontsize=4.7, ha="center", va="center", color=INK, zorder=3)
    arrow(axc, (1.50, yy), (1.33, yy), color=OCOL, lw=0.7, ms=4)
for (y0, y1, lab) in [(1.96, 1.34, r"$X_\mathcal{C},\ \Pi^{\downarrow}\Phi_\mathcal{C}$"), (1.34, 0.72, r"$X_\mathcal{C},\delta X_\mathcal{M},\Pi^{\downarrow}\Phi_\mathcal{M}$")]:
    arrow(axc, (0.81, y0 - 0.17), (0.81, y1 + 0.17), color=INK, lw=0.8, ms=5)
    axc.text(0.86, (y0 + y1) / 2, lab, fontsize=4.6, va="center", color=MUTED)
for (y0, y1) in [(0.72, 1.34), (1.34, 1.96)]:
    arrow(axc, (2.40, y0 + 0.15), (2.40, y1 - 0.15), color=COL["H"], lw=0.8, ms=5)
    axc.text(2.36, (y0 + y1) / 2, r"$\Pi^{\uparrow}$", fontsize=5.4, va="center", ha="right", color=COL["H"])
axc.text(0.30, 2.24, "large $\\rightarrow$ small: conditioning on the coarser state", fontsize=4.6, color=INK, va="center")
axc.text(2.48, 2.24, "small $\\rightarrow$ large", fontsize=4.6, color=COL["H"], va="center", ha="right")
axc.text(0.02, 0.33, r"$p_\theta(Z\mid Y,F)=p_\theta(Z_\mathcal{C}\mid c_\mathcal{C})\;p_\theta(Z_\mathcal{M}\mid Z_\mathcal{C},c_\mathcal{M})\;p_\theta(\delta X_\mathcal{H}\mid Z_\mathcal{C},Z_\mathcal{M},c_\mathcal{H})$",
         fontsize=4.8, color=INK, va="center")
axc.text(0.02, 0.10, "$h_\\ell$: obs. info (pyramid $\\Pi^{\\uparrow}$)  ·  $\\Phi_\\ell=(S_\\ell,\\psi_\\ell)$: sub-grid  ·  $f$: forcing",
         fontsize=4.6, color=MUTED, va="center")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "final":
        fig.savefig("CROSCIM_fig1_multiscale.pdf"); fig.savefig("CROSCIM_fig1_multiscale.png", dpi=600)
    else:
        fig.savefig("f1.png", dpi=220)
