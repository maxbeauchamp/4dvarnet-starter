import numpy as np
import matplotlib as mpl
from matplotlib import font_manager as fm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon
import matplotlib.pyplot as plt

for f in fm.findSystemFonts(["/usr/share/texmf/fonts/opentype/public/tex-gyre"]):
    if "heros" in f.lower():
        fm.fontManager.addfont(f)

mpl.rcParams.update({
    "font.family": "TeX Gyre Heros",
    "font.size": 6.3,
    "mathtext.fontset": "custom",
    "mathtext.rm": "TeX Gyre Heros",
    "mathtext.it": "TeX Gyre Heros:italic",
    "mathtext.bf": "TeX Gyre Heros:bold",
    "mathtext.cal": "cmsy10",
    "axes.linewidth": 0.5,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "lines.solid_capstyle": "round",
})

INK = "#1d2230"
MUTED = "#5b6475"
RULE = "#c9ced8"
PANEL_BG = "#f6f7f9"
LAND = "#d8d6cf"
COAST = "#8a877d"

COL = {"C": "#1f5aa6", "M": "#1f8a5b", "H": "#d4561c"}
TINT = {"C": "#e3ecf8", "M": "#e0f1e8", "H": "#fbe7dc"}
DARK = {"C": "#123d75", "M": "#14603f", "H": "#9c3a0e"}

# cmocean 'thermal'-like perceptual map
THERMAL = LinearSegmentedColormap.from_list("thermal", [
    "#042333", "#1b3a7a", "#3d4f9a", "#6a5c9e", "#94669a",
    "#c0718a", "#e3836a", "#f6a54e", "#f7cf4f", "#e8fa5b"], N=512)
THERMAL.set_bad(LAND)
BAL = LinearSegmentedColormap.from_list("balance", [
    "#18234a", "#2b5aa8", "#7ea7d6", "#f1f1f1", "#e09a85", "#b33a32", "#4a0f1a"], N=512)
BAL.set_bad(LAND)

SST_LIM = (-1, 30)


def inch_axes(fig, x, y, w, h, W, H, frame=False):
    """Axes placed in inches (x,y from bottom-left); data coords = inches."""
    ax = fig.add_axes([x / W, y / H, w / W, h / H])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    if not frame:
        ax.axis("off")
    return ax


def rbox(ax, x, y, w, h, fc="white", ec=RULE, lw=0.6, r=0.04, z=1, alpha=1, ls="-"):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                       fc=fc, ec=ec, lw=lw, zorder=z, alpha=alpha, ls=ls)
    ax.add_patch(p)
    return p


def arrow(ax, p0, p1, color=INK, lw=0.8, ms=5, style="-|>", z=5, cs="arc3,rad=0", ls="-",
          shrinkA=0, shrinkB=0):
    a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms, color=color, lw=lw,
                        zorder=z, connectionstyle=cs, linestyle=ls,
                        shrinkA=shrinkA, shrinkB=shrinkB)
    ax.add_patch(a)
    return a


def panel_title(fig, x, y, letter, text, W, H, size=7.6):
    fig.text(x / W, y / H, f"({letter})", fontsize=size, fontweight="bold", color=INK,
             va="baseline", ha="left")
    fig.text((x + 0.23) / W, y / H, text, fontsize=size, fontweight="bold", color=INK,
             va="baseline", ha="left")

ICE = LinearSegmentedColormap.from_list("ice", [
    "#050a1c", "#13235a", "#244a8e", "#3a78b5", "#6aaed3", "#a9d8ea", "#e4f4f8", "#ffffff"], N=512)
ICE.set_bad(LAND)
