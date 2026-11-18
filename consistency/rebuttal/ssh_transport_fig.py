"""DynVarCM transport on the unified timeline (SSH), from a FIG_DUMP npz of the VarDynCM SSH notebook.

Columns: solver states, paper convention (t = 1 noise -> t = 0); the spin-up states (t > t_IC) then the 9 frames
t_k = t_IC (N-k)/N, labelled by solver time and by day. Rows:
  member 1 | ensemble spread | ensemble mean | reference | |mean - reference| | observations
Colour scales: one diverging SSH scale (member, mean, reference, observations); one logarithmic scale shared by
the spread and the error, so that the decay of the noise in spin-up and the spread of the frames can be read on
the same axis and compared with the error.
"""
import sys, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

f, out = sys.argv[1], sys.argv[2]
n_spin = int(sys.argv[3]) if len(sys.argv) > 3 else 3
d = np.load(f)
P = d['process'].astype(np.float32)            # (M, S, H, W) solver states, physical units
times = np.asarray(d['times'], float)          # (S,) solver times, 1 -> 0
gt, obs = d['gt'], d['obs']                    # (C, H, W)
M, S, H, W = P.shape
C = gt.shape[0]
sb = times[np.argmin(np.abs(times - 0.7))] if np.any(np.isclose(times, 0.7, atol=1e-3)) else 0.7
frame_t = np.array([sb * (C - 1 - k) / (C - 1) for k in range(C)])
frame_idx = [int(np.argmin(np.abs(times - tk))) for tk in frame_t]
spin = [i for i in range(S) if times[i] > sb + 1e-6]
spin = [spin[int(round(j))] for j in np.linspace(0, len(spin) - 1, min(n_spin, len(spin)))]
cols = spin + frame_idx
nsp = len(spin)

mean_f = P[:, frame_idx].mean(0)
sd = P.std(0, ddof=1)                          # (S, H, W)
err = np.abs(mean_f - gt)
vlo, vhi = np.nanpercentile(gt, [1, 99])
pos = sd[cols]; lo = max(np.nanpercentile(np.concatenate([sd[frame_idx].ravel(), err[np.isfinite(err)]]), 2), 1e-4)
hi = np.nanpercentile(np.concatenate([sd[frame_idx].ravel(), err[np.isfinite(err)]]), 99.5)   # noise at t=1 saturates
div = plt.get_cmap('RdBu_r').copy(); div.set_bad('#d9d9d9')
seq = plt.get_cmap('magma').copy(); seq.set_bad('#d9d9d9')

rows = [('member 1', 'div', 'all'), ('ens. spread', 'seq', 'all'), ('ens. mean', 'div', 'frames'),
        ('reference', 'div', 'frames'), ('|mean $-$ ref.|', 'seq', 'frames'), ('observations', 'div', 'frames')]
plt.rcParams.update({'font.size': 7.5, 'font.family': 'serif', 'mathtext.fontset': 'cm'})
ncol = len(cols)
fig, axs = plt.subplots(len(rows), ncol, figsize=(0.62 * ncol + 1.0, 0.66 * len(rows) + 0.5), squeeze=False,
                        gridspec_kw=dict(wspace=0.04, hspace=0.06))
ims = {}
for i, (lab, kind, which) in enumerate(rows):
    for j, si in enumerate(cols):
        ax = axs[i, j]; ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.4)
        k = j - nsp
        if which == 'frames' and k < 0:
            ax.axis('off'); continue
        if lab == 'member 1':
            z = P[0, si]
        elif lab == 'ens. spread':
            z = sd[si]
        elif lab == 'ens. mean':
            z = mean_f[k]
        elif lab == 'reference':
            z = gt[k]
        elif lab.startswith('|'):
            z = err[k]
        else:
            z = obs[k]
        if kind == 'div':
            im = ax.imshow(z, cmap=div, vmin=vlo, vmax=vhi, origin='lower', interpolation='nearest')
        else:
            im = ax.imshow(np.clip(z, lo, None), cmap=seq, norm=LogNorm(lo, hi), origin='lower', interpolation='nearest')
        ims.setdefault(kind, im)
        if i == 0:
            ttl = f'$t={times[si]:.2f}$' if k < 0 else f'$t={times[si]:.2f}$\nday {k + 1}'
            ax.set_title(ttl, fontsize=7, pad=2, color='#3b6ea5' if k < 0 else '#b03a2e')
fig.canvas.draw()
x0 = axs[0, 0].get_position().x0 - 0.008
for i, (lab, kind, which) in enumerate(rows):
    pp = axs[i, -1].get_position()
    fig.text(x0, (pp.y0 + pp.y1) / 2, lab, ha='right', va='center', fontsize=7.5)
# spin-up / physical brackets above the first row
def bracket(j0, j1, text, col):
    a, b = axs[0, j0].get_position(), axs[0, j1].get_position()
    y = a.y1 + 0.075
    fig.add_artist(plt.Line2D([a.x0, b.x1], [y, y], color=col, lw=0.9, transform=fig.transFigure))
    for x in (a.x0, b.x1):
        fig.add_artist(plt.Line2D([x, x], [y, y - 0.012], color=col, lw=0.9, transform=fig.transFigure))
    fig.text((a.x0 + b.x1) / 2, y + 0.006, text, ha='center', va='bottom', fontsize=7.5, color=col)
if nsp:
    bracket(0, nsp - 1, 'spin-up' if nsp < 3 else f'spin-up ($t>t_{{\\mathrm{{IC}}}}={sb:.1f}$): noise $\\to$ initial condition', '#3b6ea5')
bracket(nsp, ncol - 1, 'physical regime: one jump per day', '#b03a2e')
def bar(kind, r0, r1, label, **kw):
    top, bot = axs[r0, -1].get_position(), axs[r1, -1].get_position()
    cax = fig.add_axes([top.x1 + 0.01, bot.y0, 0.009, top.y1 - bot.y0])
    cb = fig.colorbar(ims[kind], cax=cax, **kw); cb.set_label(label, fontsize=7)
    cb.ax.tick_params(labelsize=6.5, length=2); cb.outline.set_linewidth(0.4)
bar('div', 2, 3, 'SSH (m)')
bar('seq', 1, 1, 'std (m)')
bar('seq', 4, 4, '|error| (m)')
fig.savefig(out, bbox_inches='tight', dpi=300)
print('saved', out, 'cols', cols, 'times', np.round(times[cols], 3), 'spread range', lo, hi)
