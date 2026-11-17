"""SIC paper figure from a FIG_DUMP npz of the DynVarCM notebook.

Rows: SAR observations | ASIP target | ensemble mean | one member | ensemble spread | |mean - target|.
Columns: a subset of the 9 days of the window.
Colour scales: the four SIC rows share one fixed 0-1 scale; spread and error share one scale whose top is the 99th
percentile of both over the displayed frames (so they are directly comparable). Land is dark grey, missing data
(gaps of the observations or of the target over the ocean) light grey.
"""
import sys, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

f = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else f.replace('.npz', '.pdf')
frames = [int(x) for x in sys.argv[3].split(',')] if len(sys.argv) > 3 else None
member = int(sys.argv[4]) if len(sys.argv) > 4 else 0

d = np.load(f)
ens = d['members'].astype(np.float32)                   # (N, C, H, W)
gt, sar, osi = d['gt'], d['sar'], d['osi']              # (C, H, W), NaN = no data
N, C, H, W = ens.shape
land = np.all(~np.isfinite(osi), axis=0) & np.all(~np.isfinite(gt), axis=0)
mean, sd = ens.mean(0), ens.std(0, ddof=1)
err = np.abs(mean - gt)
if frames is None:                                      # frames with the most target pixels, in time order
    n_obs = np.isfinite(gt).reshape(C, -1).sum(1)
    frames = sorted(np.argsort(-n_obs)[:5].tolist())

# sea-ice palette: open water dark blue -> ice white
ice = LinearSegmentedColormap.from_list('ice', ['#08214a', '#1f5f99', '#5fa3cf', '#bfe0f0', '#ffffff'])
ice.set_bad('#d9d9d9')
unc = plt.get_cmap('magma_r').copy(); unc.set_bad('#d9d9d9')
LAND = np.array([0.45, 0.45, 0.45, 1.0])

vals = np.concatenate([sd[frames][np.isfinite(gt[frames])], err[frames][np.isfinite(err[frames])]])
umax = float(np.nanpercentile(vals, 99)) if vals.size else 0.2
umax = max(0.05, np.ceil(umax * 20) / 20)              # round up to 0.05

rows = [('SAR obs.', sar, 'sic'), ('ASIP target', gt, 'sic'), ('ens. mean', mean, 'sic'),
        (f'member {member + 1}', ens[member], 'sic'), ('ens. spread', sd, 'unc'), ('|mean - target|', err, 'unc')]
plt.rcParams.update({'font.size': 8, 'font.family': 'serif', 'mathtext.fontset': 'cm'})
fig, axs = plt.subplots(len(rows), len(frames), figsize=(1.15 * len(frames) + 0.9, 1.15 * len(rows) + 0.3),
                        squeeze=False, gridspec_kw=dict(wspace=0.04, hspace=0.06))
ims = {}
for i, (lab, a, kind) in enumerate(rows):
    for j, t in enumerate(frames):
        ax = axs[i, j]
        z = np.array(a[t], dtype=float)
        if kind == 'unc':
            z[~np.isfinite(gt[t]) if lab.startswith('|') else land] = np.nan
        z[land] = np.nan
        ims[kind] = ax.imshow(z, cmap=ice if kind == 'sic' else unc, vmin=0, vmax=1 if kind == 'sic' else umax,
                              origin='upper', interpolation='nearest')
        ax.imshow(np.where(land[..., None], LAND, 0.0), origin='upper', interpolation='nearest')
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.4)
        if i == 0:
            ax.set_title(f'day {t + 1}', fontsize=8, pad=2)
        if j == 0:
            ax.set_ylabel(lab, fontsize=8)
        if lab.startswith('|') and np.isfinite(gt[t]).sum() > 10:      # frame SSR on the target pixels
            m = np.isfinite(gt[t])
            ssr = np.sqrt(np.mean(sd[t][m] ** 2) / np.mean((mean[t][m] - gt[t][m]) ** 2)) * np.sqrt((N + 1) / N)
            ax.text(0.03, 0.04, f'SSR {ssr:.2f}', transform=ax.transAxes, fontsize=6.5, color='k',
                    bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.8))
fig.canvas.draw()
def bar(kind, r0, r1, label, ticks):
    top, bot = axs[r0, -1].get_position(), axs[r1, -1].get_position()
    g = 0.012 * (top.y1 - bot.y0)
    cax = fig.add_axes([top.x1 + 0.012, bot.y0 + g, 0.012, top.y1 - bot.y0 - 2 * g])
    cb = fig.colorbar(ims[kind], cax=cax, ticks=ticks, format='%.2f')
    cb.set_label(label, fontsize=8); cb.ax.tick_params(labelsize=7, length=2); cb.outline.set_linewidth(0.4)
bar('sic', 0, 3, 'sea-ice concentration', [0, 0.25, 0.5, 0.75, 1])
bar('unc', 4, 5, 'spread, |error| (SIC)', np.round(np.linspace(0, umax, 3), 3))
fig.savefig(out, bbox_inches='tight', dpi=300)
print('saved', out, 'frames', frames, 'umax', umax, 'window', int(d['window']) if 'window' in d else '?')
