"""Fig. 2 -- building the context c_l (formulation section 3)."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle, Polygon, Circle
from style import *

E = np.load("cache4.npz")
W, H = 7.2, 6.45
fig = plt.figure(figsize=(W, H))
SCOL, PCOL, OCOL, FCOL = "#7a4fa3", "#b07a12", "#2a7fb8", "#6b7280"


def rgba(img, cmap, lim):
    return cmap(np.clip((img - lim[0]) / (lim[1] - lim[0]), 0, 1))


def img_ax(x, y, w, h, col=MUTED, lw=0.8):
    ax = fig.add_axes([x / W, y / H, w / W, h / H])
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(col); s.set_linewidth(lw)
    return ax


def box_text(ax, x, y, w, h, txt, ec, fc="white", fs=5.4, col=INK, lw=0.6, ls="-"):
    rbox(ax, x, y, w, h, fc=fc, ec=ec, lw=lw, z=2, ls=ls)
    ax.text(x + w / 2, y + h / 2, txt, fontsize=fs, ha="center", va="center", color=col, zorder=3, linespacing=1.3)


# ===================================================================== (a) encoders by data type
ya = H - 0.22
panel_title(fig, 0.06, ya, "a", "Local encoders $a_\\ell=E_\\ell(Y_\\ell^{\\rm in})$, by data type", W, H)
axA = inch_axes(fig, 0.06, H - 2.92, 2.62, 2.58, W, H)
rbox(axA, 0, 0, 2.62, 2.56, fc=PANEL_BG, ec="none", r=0.05, z=0)
rng = np.random.default_rng(4)
# row 1: irregular tokens
y1 = 2.08
axA.text(0.08, 2.44, "irregular: tracks, buoys, SAR drift, L2", fontsize=5.4, color=INK, fontweight="bold", va="center")
ia = img_ax(0.06 + 0.10, H - 2.92 + y1 - 0.24, 0.48, 0.48, COL["M"], 0.7)
ia.set_xlim(0, 4); ia.set_ylim(0, 4)
for k in range(1, 4):
    ia.axvline(k, color=COL["M"], lw=0.3, alpha=0.6); ia.axhline(k, color=COL["M"], lw=0.3, alpha=0.6)
tt = np.linspace(0, 1, 40)
ia.scatter(0.2 + 3.6 * tt, 0.4 + 3.0 * tt + 0.15 * np.sin(9 * tt), s=1.6, color=COL["H"], lw=0)
px, py = rng.uniform(0.2, 3.8, 9), rng.uniform(0.2, 3.8, 9)
ia.quiver(px, py, 0.4 * np.ones(9), 0.25 * np.ones(9), color=COL["M"], scale=8, width=0.012, headwidth=4)
ia.add_patch(Rectangle((2, 2), 1, 1, fc="none", ec=INK, lw=0.9))
arrow(axA, (0.72, y1), (0.86, y1), lw=0.6, ms=4)
for k in range(5):
    axA.add_patch(Rectangle((0.90, y1 + 0.20 - 0.09 * k), 0.30, 0.06, fc=mpl.cm.Oranges(0.25 + 0.13 * k), ec="none"))
axA.text(1.05, y1 - 0.30, "$e_n$ tokens", fontsize=4.6, ha="center", color=MUTED)
arrow(axA, (1.24, y1), (1.38, y1), lw=0.6, ms=4)
box_text(axA, 1.40, y1 - 0.20, 1.14, 0.40, "cross-attention per cell\nlearned query $q$", COL["M"], fs=4.9)
axA.text(0.10, y1 - 0.38, r"$e_n=\mathrm{MLP}[y_n,\sigma_n,\mathrm{type}_n,x_n-x_k,t_n-t]$", fontsize=4.7, color=INK)
axA.text(0.10, y1 - 0.53, r"$a_\ell(k)=\Sigma_{n\in k}\,\mathrm{softmax}_{n\in k}(q^\top W_K e_n)\,W_V e_n$", fontsize=4.7, color=INK)
# row 2: dense gridded
y2 = 1.04
axA.text(0.08, 1.36, "dense gridded: ASIP SIC, gridded PMW", fontsize=5.4, color=INK, fontweight="bold", va="center")
tile = E["den"][64:128, 0:64].copy()
msk = np.zeros_like(tile, bool); msk[:, 44:] = True; msk[50:, 30:] = True
t2 = rgba(tile, ICE, (0.4, 1)); t2[msk] = mpl.colors.to_rgba("#e9ebef")
ia = img_ax(0.06 + 0.10, H - 2.92 + y2 - 0.22, 0.44, 0.44, COL["H"], 0.7)
ia.imshow(t2, origin="lower", interpolation="nearest")
arrow(axA, (0.72, y2), (0.86, y2), lw=0.6, ms=4)
for k in range(3):
    s = 0.36 - 0.10 * k
    axA.add_patch(Rectangle((0.90 + 0.05 * k, y2 - s / 2), 0.06, s, fc=COL["H"], alpha=0.35 + 0.2 * k, ec="none"))
arrow(axA, (1.24, y2), (1.38, y2), lw=0.6, ms=4)
box_text(axA, 1.40, y2 - 0.20, 1.14, 0.40, "masked strided CNN\n(patch-embed + coverage)", COL["H"], fs=4.9)
axA.text(0.10, y2 - 0.32, "only non-empty tiles; mask channel = coverage", fontsize=4.6, color=MUTED)
# row 3: forcing
y3 = 0.26
axA.text(0.08, 0.58, "forcing: ERA5 $T_{2m},q_{2m},u_{10},v_{10}$", fontsize=5.4, color=INK, fontweight="bold", va="center")
xx, yy = np.meshgrid(np.linspace(0, 1, 40), np.linspace(0, 1, 40))
ia = img_ax(0.06 + 0.10, H - 2.92 + y3 - 0.19, 0.38, 0.38, FCOL, 0.7)
ia.imshow(np.sin(4 * xx + 2 * yy) + np.cos(3 * yy), cmap=THERMAL, origin="lower")
arrow(axA, (0.66, y3), (1.38, y3), lw=0.6, ms=4)
box_text(axA, 1.40, y3 - 0.18, 1.14, 0.36, "CNN + regrid $\\rightarrow f$ on $\\Omega_\\mathcal{C}$", FCOL, fs=4.9)

# ===================================================================== (b) bottom-up pyramid
xb = 2.80
panel_title(fig, xb, ya, "b", "Bottom-up pyramid: $h_\\ell$ and $\\psi_\\ell$", W, H)
axB = inch_axes(fig, xb, H - 2.92, 2.30, 2.58, W, H)


def plane(ax, x0, y0, w, h, sk, n, lv, hl=None):
    P = lambda u, v: (x0 + u * w + v * sk, y0 + v * h)
    ax.add_patch(Polygon([P(0, 0), P(1, 0), P(1, 1), P(0, 1)], fc=TINT[lv], ec=COL[lv], lw=0.7, zorder=2))
    segs = [[P(k / n, 0), P(k / n, 1)] for k in range(1, n)] + [[P(0, k / n), P(1, k / n)] for k in range(1, n)]
    ax.add_collection(LineCollection(segs, colors=COL[lv], lw=0.25, alpha=0.6, zorder=3))
    if hl is not None:
        i, j, m = hl
        ax.add_patch(Polygon([P(i / n, j / n), P((i + m) / n, j / n), P((i + m) / n, (j + m) / n), P(i / n, (j + m) / n)],
                             fc=COL[lv], ec=DARK[lv], lw=0.5, alpha=0.85, zorder=4))
    return P


pw, ph, sk = 1.05, 0.34, 0.30
PH = plane(axB, 0.10, 0.14, pw, ph, sk, 16, "H", (8, 8, 4))
PM = plane(axB, 0.10, 0.98, pw, ph, sk, 8, "M", (4, 4, 2))
PC = plane(axB, 0.10, 1.82, pw, ph, sk, 4, "C", (2, 2, 1))
cx = lambda P, i, n: P((i + 0.5) / n, (i + 0.5) / n)
for (Pa, na, ia_), (Pb, nb, ib) in [((PH, 16, 9.5), (PM, 8, 4.5)), ((PM, 8, 4.5), (PC, 4, 2.0))]:
    a0 = Pa((ia_ + .5) / na, (ia_ + .5) / na); b0 = Pb((ib + .5) / nb, (ib + .5) / nb)
    arrow(axB, (a0[0], a0[1] + 0.03), (b0[0], b0[1] - 0.03), color=COL["H"], lw=0.9, ms=5)
axB.text(0.62, 0.70, "$\\Pi^{\\uparrow}$", fontsize=6.4, color=COL["H"])
axB.text(0.62, 1.54, "$\\Pi^{\\uparrow}$", fontsize=6.4, color=COL["H"])
for P, lv, t, y in [(PH, "H", "$h_\\mathcal{H}=a_\\mathcal{H}$", 0.30), (PM, "M", "$h_\\mathcal{M}=U_\\mathcal{M}(a_\\mathcal{M},\\Pi^{\\uparrow}h_\\mathcal{H})$", 1.14),
                     (PC, "C", "$h_\\mathcal{C}=U_\\mathcal{C}(a_\\mathcal{C},\\Pi^{\\uparrow}h_\\mathcal{M},f)$", 1.98)]:
    axB.text(1.50, y + 0.10, t, fontsize=4.9, color=DARK[lv], va="center")
for lv, y in [("M", 1.02), ("C", 1.86)]:
    arrow(axB, (1.55, y - 0.05), (1.80, y - 0.05), color=PCOL, lw=0.6, ms=4)
    axB.text(1.84, y - 0.05, f"$\\Psi_\\mathcal{{{lv}}}\\rightarrow\\psi_\\mathcal{{{lv}}}$", fontsize=4.9, color=PCOL, va="center")
axB.text(1.55, 2.40, "$f=E_F(F)$ enters at $\\mathcal{C}$", fontsize=4.6, color=FCOL, va="center")
axB.text(0.05, 0.02, "fused per $\\mathcal{C}$ tile for 500-m ASIP; on the fly", fontsize=4.5, color=MUTED)

# ===================================================================== (c) Pi_up in one C cell
xc = 5.18
panel_title(fig, xc, ya, "c", "$\\Pi^{\\uparrow}$ in one $\\mathcal{C}$ cell", W, H)
axCc = inch_axes(fig, xc, H - 2.92, 1.98, 2.58, W, H)
sub = E["den"][32:64, 64:96]
lead = 1 / (1 + np.exp(-(0.8 - sub) / 0.03))
w8 = lead.reshape(8, 4, 8, 4).mean((1, 3)) + 0.02
w8 = np.exp(4 * w8 / w8.max()); w8 /= w8.sum()
ia = img_ax(xc + 0.08, H - 2.92 + 1.20, 0.95, 0.95, COL["C"], 0.9)
ia.imshow(rgba(sub, ICE, (0.4, 1)), origin="lower", extent=[0, 8, 0, 8], interpolation="nearest")
ia.set_xlim(0, 8); ia.set_ylim(0, 8)
for k in range(1, 8):
    ia.axvline(k, color=COL["M"], lw=0.3); ia.axhline(k, color=COL["M"], lw=0.3)
ib = img_ax(xc + 1.02 + 0.06, H - 2.92 + 1.20, 0.86, 0.86, SCOL, 0.7)
ib.imshow(w8, origin="lower", cmap="Purples", interpolation="nearest")
axCc.text(0.55, 2.22, "64 children $i$ ($\\mathcal{M}$)", fontsize=4.8, ha="center", color=INK)
axCc.text(1.50, 2.22, "weights $\\alpha_{ji}$", fontsize=4.8, ha="center", color=SCOL)
axCc.text(0.05, 1.02, r"$(\Pi^{\uparrow}h)(k)=W_O[\Sigma_{i\in\mathrm{ch}(k)}\,\alpha_{ji}\,W_V\tilde h_i]_{j=1..K}$", fontsize=4.9, color=INK)
axCc.text(0.05, 0.80, r"$\alpha_{ji}=\mathrm{softmax}_i\,(q_j^\top W_K\tilde h_i/\sqrt{d})$,  $\tilde h_i=h(i)+p_i$", fontsize=4.9, color=INK)
axCc.text(0.05, 0.55, "$q_j$: $K$ learned queries (“questions”)\n$p_i$: position of the child in the parent\nmasked children excluded; all empty $\\rightarrow\\varnothing$",
          fontsize=4.6, color=MUTED, va="top", linespacing=1.3)

# ===================================================================== (d) explicit vs latent
yd = H - 3.22
panel_title(fig, 0.06, yd, "d", "Sub-grid parametrization $\\Phi_\\ell=(S_\\ell,\\psi_\\ell)$: explicit part vs latent part", W, H)
axD = inch_axes(fig, 0.06, 0.12, 3.62, 2.86, W, H)
rbox(axD, 0, 1.46, 3.60, 1.36, fc="#f3eef8", ec="none", r=0.05, z=0)
rbox(axD, 0, 0.02, 3.60, 1.36, fc="#f8f2e6", ec="none", r=0.05, z=0)
axD.text(0.08, 2.70, "$S_\\ell$: physical, explicit, supervised", fontsize=5.8, color=SCOL, fontweight="bold", va="center")
cell = E["den"][32:64, 64:96]
ia = img_ax(0.06 + 0.12, 0.12 + 1.66, 0.80, 0.80, COL["H"], 0.7)
c2 = rgba(cell, ICE, (0.4, 1)); c2[:6, :] = mpl.colors.to_rgba("#e9ebef")
ia.imshow(c2, origin="lower", interpolation="nearest")
axD.text(0.58, 1.60, "$\\hat X_\\mathcal{H}=\\mathcal{G}(Y_\\mathcal{H}^{\\rm tgt})$, cell of (c)", fontsize=4.5, ha="center", color=MUTED, va="top")
arrow(axD, (1.06, 2.12), (1.30, 2.12), lw=0.8, ms=5)
axD.text(1.18, 2.18, "$\\mathcal{S}_\\mathcal{C}$", fontsize=6, ha="center", color=INK, va="bottom")
sv = [(t, float(E[f"S_den_{k}"][1, 2]), vm) for t, k, vm in [("open water", "ow", 0.3), ("lead frac.", "lead", 0.3), ("lead dens.", "dens", 8), ("anisotropy", "aniso", 1)]]
for k, (t, v, vm) in enumerate(sv):
    yy = 2.48 - 0.15 * k
    axD.text(1.34, yy, t, fontsize=4.6, va="center")
    axD.add_patch(Rectangle((1.84, yy - 0.03), 0.42, 0.06, fc="#e6e1ee", ec="none"))
    axD.add_patch(Rectangle((1.84, yy - 0.03), 0.42 * min(v / vm, 1), 0.06, fc=SCOL, ec="none"))
    axD.text(2.28, yy, f"{v:.2f}", fontsize=4.2, va="center", color=MUTED)
axD.text(1.34, 1.80, "$\\hat S_\\mathcal{C}$: computed on the fly,\nno learnable parameter", fontsize=4.6, color=INK, va="center", linespacing=1.25)
arrow(axD, (2.32, 2.12), (2.50, 2.12), lw=0.8, ms=5)
box_text(axD, 2.52, 1.86, 1.00, 0.52, "target of $S_\\mathcal{C}$\nin $Z_\\mathcal{C}$ (generated\nby $G_\\mathcal{C}$): $\\mathcal{L}_S$", SCOL, fs=4.8)
axD.text(1.34, 1.58, r"$m^S=\mathbb{1}[\mathrm{cov}\geq\gamma]$,  $\sigma_S^2=\sigma_0^2+\sigma^2_{\rm samp}/n_{\rm eff}$", fontsize=4.7, color=INK, va="center")
axD.text(0.08, 1.26, "$\\psi_\\ell$: latent, small, estimated for conditioning", fontsize=5.8, color=PCOL, fontweight="bold", va="center")
for k in range(6):
    axD.add_patch(Rectangle((0.20, 0.30 + 0.12 * k), 0.12, 0.10, fc=OCOL, alpha=0.3 + 0.1 * k, ec="none"))
axD.text(0.26, 0.20, "$h_\\mathcal{C}(k)\\in\\mathbb{R}^{d}$", fontsize=4.7, ha="center", color=OCOL, va="top")
arrow(axD, (0.40, 0.66), (0.62, 0.66), lw=0.8, ms=5)
axD.add_patch(Polygon([(0.64, 0.36), (1.10, 0.52), (1.10, 0.80), (0.64, 0.96)], fc="white", ec=PCOL, lw=0.7))
axD.text(0.87, 0.66, "$\\Psi_\\mathcal{C}$", fontsize=6, ha="center", va="center", color=PCOL)
arrow(axD, (1.12, 0.66), (1.32, 0.66), lw=0.8, ms=5)
for k in range(4):
    axD.add_patch(Rectangle((1.36, 0.48 + 0.10 * k), 0.10, 0.08, fc=PCOL, alpha=0.4 + 0.15 * k, ec="none"))
axD.text(1.41, 0.40, "$\\psi_\\mathcal{C}\\in\\mathbb{R}^{d_\\psi}$, $d_\\psi\\approx4$--$8$", fontsize=4.7, ha="left", color=PCOL, va="top")
axD.text(1.66, 0.98, "no target in $\\mathcal{N}_0$: shaped by the losses of\n$G_\\mathcal{M}, G_\\mathcal{H}$ that it conditions\n"
         "$\\mathcal{L}_\\psi=\\mathbb{E}\\|\\psi\\|^2+\\lambda_\\perp\\|\\mathrm{Cov}(\\psi,S)\\|_F^2$\n"
         "in $\\mathcal{N}_1$: target $\\bar\\Psi(h^{t+\\Delta})$ (EMA)",
         fontsize=4.7, color=INK, va="top", linespacing=1.35)

# ===================================================================== (e) context assembly
xe = 3.86
panel_title(fig, xe, yd, "e", "Assembling the contexts $c_\\ell$", W, H)
axE = inch_axes(fig, xe, 0.12, 3.30, 2.86, W, H)
cols = [("obs. info", OCOL), ("forcing", FCOL), ("sub-grid $\\Phi$\nfrom above", SCOL), ("coarser\nstate", INK), ("generated\ntarget", INK)]
cx_ = [0.62, 1.12, 1.62, 2.18, 2.82]
for (t, c), x in zip(cols, cx_):
    axE.text(x, 2.62, t, fontsize=4.9, ha="center", va="center", color=c, linespacing=1.15)
rows = [("C", [("$h_\\mathcal{C}$", OCOL), ("$f$", FCOL), ("—", RULE), ("—", RULE), ("$X_\\mathcal{C},S_\\mathcal{C}$", COL["C"])], 2.22),
        ("M", [("$h_\\mathcal{M}$", OCOL), ("$\\Pi^{\\downarrow}\\!f$", FCOL), ("$\\Pi^{\\downarrow}(S_\\mathcal{C},\\psi_\\mathcal{C})$", SCOL), ("$X^\\eta_\\mathcal{C}$", COL["C"]), ("$\\delta X_\\mathcal{M},S_\\mathcal{M}$", COL["M"])], 1.78),
        ("H", [("$h_\\mathcal{H}$", OCOL), ("$\\Pi^{\\downarrow}\\!f$", FCOL), ("$\\Pi^{\\downarrow}(S_\\mathcal{M},\\psi_\\mathcal{M})$", SCOL), ("$X^\\eta_\\mathcal{C},\\delta X^\\eta_\\mathcal{M}$", COL["M"]), ("$\\delta X_\\mathcal{H}$", COL["H"])], 1.34)]
bw = [0.42, 0.42, 0.56, 0.56, 0.62]
for lv, cells, y in rows:
    axE.text(0.08, y, f"$G_\\mathcal{{{lv}}}$", fontsize=7.4, color=DARK[lv], va="center", fontweight="bold")
    for (t, c), x, w in zip(cells, cx_, bw):
        last = x == cx_[-1]
        rbox(axE, x - w / 2, y - 0.15, w, 0.30, fc=(TINT[lv] if last else "white"), ec=c, lw=0.9 if last else 0.6, z=2,
             ls="-" if t != "—" else (0, (1, 1.5)))
        axE.text(x, y, t, fontsize=4.6, ha="center", va="center", color=INK if t != "—" else MUTED, zorder=3)
    arrow(axE, (cx_[3] + bw[3] / 2 + 0.01, y), (cx_[4] - bw[4] / 2 - 0.01, y), color=MUTED, lw=0.5, ms=3)
axE.add_patch(Rectangle((cx_[0] - 0.24, 1.15), 0.48, 1.30, fc="none", ec=OCOL, lw=0.5, ls=(0, (2, 1.5))))
notes = ["$h_\\ell=\\varnothing$ without observations ($\\mu=0$): forcing $+\\ \\Phi$ drive a",
         "   free simulation; observations add a nudging term",
         "   $v_w=v(\\cdot\\mid\\varnothing)+w\\,[v(\\cdot\\mid h)-v(\\cdot\\mid\\varnothing)]$",
         "coarse obs. do not descend: they act through $X_\\mathcal{C}$",
         "$X^\\eta=X+\\eta$: generated coarse state, slightly noised",
         "training: each group replaced by $\\varnothing$ with prob. $p_\\varnothing$",
         "everything is computed on the fly, per batch"]
for k, t in enumerate(notes):
    axE.text(0.10, 0.96 - 0.135 * k, t, fontsize=4.8, color=INK if not t.startswith("   ") else INK, va="center")

fig.savefig("CROSCIM_fig2_context.pdf")
fig.savefig("CROSCIM_fig2_context.png", dpi=400)
