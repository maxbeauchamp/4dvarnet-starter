"""Fig. 3 -- training: on-the-fly targets, FM cascade, losses, two networks (formulation sections 4-6)."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from style import *

E = np.load("cache4.npz")
W, H = 7.2, 6.75
fig = plt.figure(figsize=(W, H))
SCOL, PCOL, OCOL, FCOL, GREY = "#7a4fa3", "#b07a12", "#2a7fb8", "#6b7280", "#e9ebef"
TGT = "#d4561c"


def rgba(img, cmap, lim, mask=None, bad=GREY):
    a = cmap(np.clip((img - lim[0]) / (lim[1] - lim[0]), 0, 1))
    if mask is not None:
        a[~mask] = mpl.colors.to_rgba(bad)
    return a


def img_ax(x, y, w, h, col=MUTED, lw=0.8):
    ax = fig.add_axes([x / W, y / H, w / W, h / H])
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(col); s.set_linewidth(lw)
    return ax


def grid(ax, n, m, col, lw=0.3, alpha=0.7):
    for k in range(1, m):
        ax.axvline(k * n / m - .5, color=col, lw=lw, alpha=alpha); ax.axhline(k * n / m - .5, color=col, lw=lw, alpha=alpha)


def bm(v, f):
    m = v.shape[0] // f
    return v.reshape(m, f, m, f).mean((1, 3))


def up(v, f):
    return np.repeat(np.repeat(v, f, 0), f, 1)


# ------------------------------------------------------------ synthetic batch
truth = E["den"]
n = truth.shape[0]
yy, xx = np.mgrid[0:n, 0:n]
swath = np.abs((xx - 0.55 * yy) - 38) < 34                       # SAR swath
rng = np.random.default_rng(2)
obsH = np.where(swath, np.clip(truth + 0.02 * rng.normal(size=truth.shape), 0, 1), np.nan)
blk = rng.uniform(size=(8, 8)) < 0.28                            # withheld 16-km blocks
tgt = swath & up(blk, 16)
inp = swath & ~tgt
covM = bm(swath.astype(float), 4)
XM = np.where(covM >= 0.5, bm(np.nan_to_num(obsH), 4) / np.maximum(covM, 1e-9), np.nan)
okM = covM >= 0.5
dH = np.where(up(okM, 4) & swath, np.nan_to_num(obsH) - up(np.nan_to_num(XM), 4), np.nan)
covC = bm(swath.astype(float), 32)
lead = 1 / (1 + np.exp(-(0.8 - np.nan_to_num(obsH, nan=1.0)) / 0.03))
SC = np.where(covC >= 0.6, bm(lead * swath, 32) / np.maximum(covC, 1e-9), np.nan)
wmap = np.where(tgt, 1.0, np.where(inp, 0.35, 0.0))

# ===================================================================== (a)
ya = H - 0.22
panel_title(fig, 0.06, ya, "a", "One training batch, built on the fly: input / target split and ground truths", W, H)
s = 0.98
xs = [0.10, 1.28, 2.46, 3.64, 4.82, 6.00]
y0 = H - 1.62
specs = [
    ("$Y_\\mathcal{H}$ (one SAR swath)", rgba(np.nan_to_num(obsH), ICE, (0.4, 1), swath), COL["H"], 32),
    ("split: $Y^{\\rm in}$ / $Y^{\\rm tgt}$", None, INK, 32),
    ("$\\hat X_\\mathcal{M}=\\mathcal{G}_\\mathcal{M}(Y)$, cov $\\geq\\gamma$", rgba(np.nan_to_num(XM), ICE, (0.4, 1), okM), COL["M"], 8),
    ("$\\widehat{\\delta X}_\\mathcal{H}=\\hat X_\\mathcal{H}-PR\\hat X_\\mathcal{H}$", rgba(np.nan_to_num(dH), BAL, (-0.4, 0.4), ~np.isnan(dH)), COL["H"], 32),
    ("$\\hat S_\\mathcal{C}=\\mathcal{S}_\\mathcal{C}(\\hat X_\\mathcal{H})$ (leads)", None, SCOL, 1),
    ("loss weight $m$", None, INK, 32),
]
for k, (t, im, col, gm) in enumerate(specs):
    ax = img_ax(xs[k], y0, s, s, col, 0.8)
    if k == 1:
        a = np.ones((n, n, 4)); a[:] = mpl.colors.to_rgba(GREY)
        a[inp] = mpl.colors.to_rgba("#9cc3e6"); a[tgt] = mpl.colors.to_rgba("#f2a37a")
        ax.imshow(a, origin="lower", interpolation="nearest")
    elif k == 4:
        ax.imshow(rgba(np.nan_to_num(SC), plt.get_cmap("Purples"), (0, 0.25), ~np.isnan(SC)), origin="lower", interpolation="nearest")
        for (j, i), v in np.ndenumerate(SC):
            ax.text(i, j, "—" if np.isnan(v) else f"{v:.2f}", fontsize=4.3, ha="center", va="center", color=INK if np.isnan(v) or v < 0.15 else "white")
    elif k == 5:
        ax.imshow(wmap, origin="lower", cmap="Greys", vmin=0, vmax=1.3, interpolation="nearest")
    else:
        ax.imshow(im, origin="lower", interpolation="nearest")
    if k != 4:
        N = im.shape[0] if im is not None else n
        grid(ax, N, 4, COL["C"], lw=0.5, alpha=0.8)
    ax.text(0.0, 1.04, t, transform=ax.transAxes, fontsize=4.9, va="bottom", color=INK)
fig.text(xs[1] / W, (y0 - 0.06) / H, "blue: input $\\rightarrow$ context $h$\norange: withheld $\\rightarrow$ target only", fontsize=4.6, va="top", color=MUTED, linespacing=1.25)
fig.text(xs[2] / W, (y0 - 0.06) / H, "weighted mean of obs per cell\ngrey: masked ($m=0$)", fontsize=4.6, va="top", color=MUTED, linespacing=1.25)
fig.text(xs[3] / W, (y0 - 0.06) / H, "w.r.t. the obs' own block mean\n(not the coarse sample): $R\\,\\delta=0$", fontsize=4.6, va="top", color=MUTED, linespacing=1.25)
fig.text(xs[4] / W, (y0 - 0.06) / H, "cells with cov. $\\geq\\gamma$ only;\n$\\sigma_S^2=\\sigma_0^2+\\sigma^2_{\\rm samp}/n_{\\rm eff}$", fontsize=4.6, va="top", color=MUTED, linespacing=1.25)
fig.text(xs[5] / W, (y0 - 0.06) / H, "$Y^{\\rm tgt}$: 1,  $Y^{\\rm in}$: $\\lambda_{\\rm in}<1$,\nunobserved: 0", fontsize=4.6, va="top", color=MUTED, linespacing=1.25)
fig.text(0.10 / W, (y0 - 0.42) / H, "No precomputed truth: gridding $\\mathcal{G}_\\ell$, details, statistics $\\mathcal{S}$, masks and the split are recomputed for every batch. "
         "Scores are always reported on $Y^{\\rm tgt}$.", fontsize=5.2, color=INK)

# ===================================================================== (b) FM cascade
yb = H - 2.42
panel_title(fig, 0.06, yb, "b", "Conditional flow matching, one generator per level", W, H)
axB = inch_axes(fig, 0.06, H - 4.62, 3.62, 2.10, W, H)
rbox(axB, 0, 0, 3.62, 2.08, fc=PANEL_BG, ec="none", r=0.05, z=0)
# 2-D toy: straight paths from noise to data
ax2 = fig.add_axes([0.20 / W, (H - 4.62 + 0.52) / H, 1.08 / W, 1.40 / H])
r2 = np.random.default_rng(1)
e = r2.normal(size=(24, 2))
th = r2.uniform(0, 2 * np.pi, 24)
d = np.stack([2.6 * np.cos(th), 1.2 * np.sin(th)], 1) + r2.normal(scale=0.12, size=(24, 2)) + [0.4, 0]
for a_, b_ in zip(e, d):
    ax2.plot([a_[0], b_[0]], [a_[1], b_[1]], color=COL["C"], lw=0.35, alpha=0.5)
ax2.scatter(*e.T, s=2.5, color=MUTED, lw=0, zorder=3); ax2.scatter(*d.T, s=2.5, color=COL["H"], lw=0, zorder=3)
ax2.set_xticks([]); ax2.set_yticks([])
for sp in ax2.spines.values():
    sp.set_linewidth(0.4); sp.set_color(RULE)
ax2.text(0.02, 0.98, "$\\varepsilon$ (grey) $\\rightarrow z_1$ (orange)", transform=ax2.transAxes, fontsize=4.4, va="top", color=INK)
axB.text(1.40, 1.92, r"$z_s=s\,z_1+(1-s)\,\varepsilon,\quad u=z_1-\varepsilon$", fontsize=5.6, va="center")
axB.text(1.40, 1.72, r"$\hat z_1=z_s+(1-s)\,v_\theta^{\ell}(z_s,s\mid\cdot)$", fontsize=5.6, va="center")
axB.text(1.40, 1.52, "$\\varepsilon$ projected by $\\Pi_0$ for the details", fontsize=4.8, va="center", color=MUTED)
rowsB = [("C", "$Z_\\mathcal{C}=(X_\\mathcal{C},S_\\mathcal{C})$", "$h_\\mathcal{C},\\,f$"),
         ("M", "$Z_\\mathcal{M}=(\\delta X_\\mathcal{M},S_\\mathcal{M})$", "$X_\\mathcal{C}^{\\eta},\\Pi^{\\downarrow}\\Phi_\\mathcal{C},h_\\mathcal{M},\\Pi^{\\downarrow}\\!f$"),
         ("H", "$\\delta X_\\mathcal{H}$", "$X_\\mathcal{C}^{\\eta},\\delta X_\\mathcal{M}^{\\eta},\\Pi^{\\downarrow}\\Phi_\\mathcal{M},h_\\mathcal{H},\\Pi^{\\downarrow}\\!f$")]
for k, (lv, tz, tc) in enumerate(rowsB):
    y = 1.22 - 0.30 * k
    axB.text(1.40, y, f"$G_\\mathcal{{{lv}}}$", fontsize=6.4, color=DARK[lv], va="center", fontweight="bold")
    rbox(axB, 1.66, y - 0.11, 0.86, 0.22, fc=TINT[lv], ec=COL[lv], lw=0.6, z=2)
    axB.text(2.09, y, tz, fontsize=4.6, ha="center", va="center", zorder=3)
    axB.text(2.56, y, "$\\mid$ " + tc, fontsize=4.4, va="center", color=INK)
axB.text(0.12, 0.24, "training: $X^\\eta$ = sample of the level above (EMA model, no gradient) + noise $\\eta$;\n"
         "each context group dropped with prob. $p_\\varnothing$ $\\Rightarrow$ heads $v(\\cdot\\mid h)$ and $v(\\cdot\\mid\\varnothing)$.  "
         "Inference: Heun, $\\mathcal{C}\\rightarrow\\mathcal{M}\\rightarrow\\mathcal{H}$",
         fontsize=4.6, color=INK, va="center", linespacing=1.35)

# ===================================================================== (c) losses
xc = 3.86
panel_title(fig, xc, yb, "c", "Losses (all masked, targets from (a))", W, H)
axC = inch_axes(fig, xc, H - 4.62, 3.30, 2.10, W, H)
axC.text(0.02, 1.94, r"$\mathcal{L}_{\rm NLL}(u,v;\sigma,m)=\mathbb{E}\,\Sigma\, m\odot\left(\frac{1}{2}\|\frac{u-v}{\sigma}\|^2+\log\sigma\right)$", fontsize=5.4, va="center")
L = [("$\\mathcal{L}_X^{\\ell}$", "state", "$\\hat X_\\ell$ or $\\widehat{\\delta X}_\\ell$", "$m^X_\\ell$ (var., level, pixel)", COL["C"]),
     ("$\\mathcal{L}_S^{\\ell}$", "explicit sub-grid", "$\\hat S_\\ell=\\mathcal{S}_\\ell(\\hat X_{\\rm fine})$", "$m^S$, $\\sigma_S$ floor", SCOL),
     ("$\\mathcal{L}_{\\rm cyc}$", "down/up consistency", "$\\|\\mathcal{S}(\\hat x_1^\\mathcal{H})-S_\\mathcal{C}\\|^2$", "soft $\\mathcal{S}$", SCOL),
     ("$\\mathcal{L}_\\psi$", "latent regularity", "$\\|\\psi\\|^2+\\lambda_\\perp\\|\\mathrm{Cov}(\\psi,S)\\|^2$", "—", PCOL),
     ("$\\mathcal{L}_\\psi^{\\rm dyn}$", "latent in $\\mathcal{N}_1$", "$\\bar\\Psi(h^{t+\\Delta})$ (EMA)", "obs. at $t+\\Delta$", PCOL)]
hx = [0.02, 0.40, 1.22, 2.30]
for x, t in zip(hx, ["", "term", "target", "mask"]):
    axC.text(x, 1.70, t, fontsize=5.0, fontweight="bold", va="center")
axC.plot([0, 3.28], [1.61, 1.61], color=RULE, lw=0.5)
for k, (a_, b_, c_, d_, col) in enumerate(L):
    y = 1.46 - 0.21 * k
    axC.text(hx[0], y, a_, fontsize=5.8, va="center", color=col)
    axC.text(hx[1], y, b_, fontsize=4.8, va="center")
    axC.text(hx[2], y, c_, fontsize=4.7, va="center")
    axC.text(hx[3], y, d_, fontsize=4.7, va="center", color=MUTED)
axC.plot([0, 3.28], [0.36, 0.36], color=RULE, lw=0.5)
axC.text(0.02, 0.24, r"$\mathcal{N}_0:\ \Sigma_\ell(\mathcal{L}_X^\ell+\lambda_S\mathcal{L}_S^\ell)+\lambda_c\mathcal{L}_{\rm cyc}+\lambda_\psi\mathcal{L}_\psi$", fontsize=5.2, va="center")
axC.text(0.02, 0.06, r"$\mathcal{N}_1:\ \Sigma_{k=1}^{K}\gamma^{k-1}[\Sigma_\ell(\mathcal{L}_X^\ell+\lambda_S\mathcal{L}_S^\ell)+\lambda_\psi\mathcal{L}^{\rm dyn}_\psi]_{t+k\Delta}$", fontsize=5.2, va="center")

# ===================================================================== (d) two networks
yd = H - 4.90
panel_title(fig, 0.06, yd, "d", "Two networks trained successively: initialization $\\mathcal{N}_0$, then transition $\\mathcal{N}_1$", W, H)
axD = inch_axes(fig, 0.06, 0.06, 7.08, 1.68, W, H)
rbox(axD, 0.0, 0.32, 2.30, 1.34, fc=TINT["C"], ec=COL["C"], lw=0.6, r=0.05, z=0)
rbox(axD, 2.48, 0.32, 4.58, 1.34, fc=TINT["H"], ec=COL["H"], lw=0.6, r=0.05, z=0)
axD.text(0.10, 1.54, "$\\mathcal{N}_0$ — analysis over $[t-W,t]$", fontsize=6.0, fontweight="bold", color=DARK["C"], va="center")
axD.text(0.10, 1.32, r"$Z^t\sim p_{\theta_0}(Z^t\mid Y^{\rm in}_{t-W:t},F_{t-W:t})$", fontsize=5.2, va="center")
for k, lv in enumerate("CMH"):
    rbox(axD, 0.18, 1.00 - 0.22 * k, 0.70, 0.17, fc=COL[lv], ec="none", r=0.03, z=2)
    axD.text(0.53, 1.085 - 0.22 * k, f"$G^0_\\mathcal{{{lv}}}$", fontsize=5.4, color="white", ha="center", va="center", zorder=3)
arrow(axD, (0.92, 0.86), (1.30, 0.86), lw=0.7, ms=4)
for k in range(3):
    ia = img_ax(0.06 + 1.36 + 0.07 * k, 0.06 + 0.62 - 0.05 * k, 0.40, 0.40, COL["C"], 0.5)
    ia.imshow(rgba(np.clip(truth + 0.03 * (k - 1) * np.sin(xx / 7.0), 0, 1), ICE, (0.4, 1)), origin="lower")
axD.text(1.62, 0.42, "ensemble $\\{Z^{t,(n)}\\}$", fontsize=4.8, ha="center", color=INK)
axD.text(0.10, 0.44, "encoders $E_\\ell,U_\\ell,\\Pi^{\\uparrow},\\Psi_\\ell$", fontsize=4.6, color=MUTED, va="center")
arrow(axD, (2.02, 0.86), (2.62, 0.86), color=INK, lw=0.9, ms=5)
axD.text(2.32, 0.92, "init. state", fontsize=4.6, ha="center", va="bottom")
axD.text(2.58, 1.54, "$\\mathcal{N}_1$ — transition (weights initialized from $\\theta_0$)", fontsize=6.0, fontweight="bold", color=DARK["H"], va="center")
axD.text(2.58, 1.32, r"$Z^{t+\Delta}\sim p_{\theta_1}(Z^{t+\Delta}\mid Z^t,\ \mu\,Y_{(t,t+\Delta]},\ F_{t:t+\Delta})$,   $a_\ell=[E^s_\ell(Z^t_\ell),\ \mu E_\ell(Y_\ell)]$",
         fontsize=5.2, va="center")
sx = [2.70, 3.78, 4.86]
mus = ["$\\mu=1$", "$\\mu=0$", "$\\mu=1$"]
for k, x in enumerate(sx):
    rbox(axD, x, 0.76, 0.66, 0.24, fc=COL["C"], ec="none", r=0.03, z=2)
    axD.text(x + 0.33, 0.88, "$G^1_\\mathcal{C}$ (+$G^1_\\mathcal{M},G^1_\\mathcal{H}$)", fontsize=4.4, color="white", ha="center", va="center", zorder=3)
    arrow(axD, (x + 0.67, 0.88), (x + 1.07, 0.88), color=INK, lw=0.8, ms=4)
    axD.text(x + 0.87, 0.93, f"$Z^{{t+{k + 1}\\Delta}}$" if k else "$Z^{t+\\Delta}$", fontsize=4.6, ha="center", va="bottom")
    axD.text(x + 0.33, 1.15, mus[k] + (": obs." if k != 1 else ": no obs."), fontsize=4.5, ha="center", va="center", color=OCOL if k != 1 else MUTED)
    arrow(axD, (x + 0.33, 1.09), (x + 0.33, 1.01), color=OCOL if k != 1 else RULE, lw=0.6, ms=3)
    axD.text(x + 0.33, 0.62, f"loss vs $Y_{{t+{k + 1}\\Delta}}$" if k else "loss vs $Y_{t+\\Delta}$", fontsize=4.4, ha="center", color=TGT, va="center")
axD.text(5.98, 1.08, "pushforward $K$=2–4 steps:\ninput of step $k$ = output of $k-1$\n$\\mu\\sim$ Bernoulli; group dropout\nstate enters as an extra stream $E^s$",
         fontsize=4.7, va="top", color=INK, linespacing=1.3)
axD.text(2.58, 0.44, "input state = samples of $\\mathcal{N}_0$ (no dense truth exists);  targets: $\\hat X,\\hat S$ at $t+k\\Delta$ and $\\bar\\Psi(h^{t+k\\Delta})$",
         fontsize=4.7, color=MUTED, va="center")
axD.text(0.02, 0.14, "Rollout, mode A: only $G^1_\\mathcal{C}$ advances $(X_\\mathcal{C},S_\\mathcal{C},\\psi_\\mathcal{C})$; fine scales on demand with $G^0_\\mathcal{M},G^0_\\mathcal{H}$ ($h=\\varnothing$, conditioned by $\\Pi^{\\downarrow}\\Phi$, $\\Pi^{\\downarrow}\\!f$).  "
         "Mode B (validation): full cascade at each step, then $S_\\mathcal{C}\\leftarrow\\mathcal{S}_\\mathcal{C}(X_\\mathcal{H})$.",
         fontsize=4.9, color=INK, va="center")

fig.savefig("CROSCIM_fig3_training.pdf")
fig.savefig("CROSCIM_fig3_training.png", dpi=400)
