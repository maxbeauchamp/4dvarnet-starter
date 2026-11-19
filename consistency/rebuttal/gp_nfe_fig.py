"""GP NFE figure: RMSE, SSR, composition gap vs NFE (from run_nfe_sweep.py); three stacked panels.

    python gp_nfe_fig.py [RESULTS_DIR] [OUT.pdf]
"""
import sys, pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
D = sys.argv[1] if len(sys.argv) > 1 else 'consistency/sbatch_submission/results/GP'
out = sys.argv[2] if len(sys.argv) > 2 else 'gp_nfe.pdf'
files = [('FM', 'FM_nfe_sweep.csv'), ('DynFM', 'DynFM_nfe_sweep.csv'), ('CM', 'CMT_nfe_sweep.csv'),
         ('DynCM', 'DynCMT_nfe_sweep.csv'), ('VarCM', 'VarCM_scratch2_nfe_sweep.csv'),
         ('DynVarCM', 'VarDynCM_grid_fr_nfe_sweep.csv')]
col = {'FM': '#555555', 'CM': '#2a6fb0', 'VarCM': '#2e8b57'}
plt.rcParams.update({'font.size': 9, 'font.family': 'serif', 'mathtext.fontset': 'cm'})
fig, axs = plt.subplots(3, 1, figsize=(3.5, 5.6), sharex=True)
handles = []
for name, f in files:
    d = pd.read_csv(f'{D}/{f}').sort_values('nfe')
    base = name.replace('Dyn', '') or 'FM'
    dyn = name.startswith('Dyn')
    kw = dict(color=col[base], ls='--' if dyn else '-', marker='s' if dyn else 'o', ms=3.5, lw=1.4, label=name)
    h, = axs[0].plot(d.nfe, d.rmse, **kw)
    handles.append(h)
    if 'ssr' in d:
        axs[1].plot(d.nfe, d.ssr, **kw)
    if 'gap_rel_spread' in d:
        g = d[d.gap_rel_spread > 0]           # the reference grid has gap 0 by definition
        axs[2].plot(g.nfe, g.gap_rel_spread, **kw)
axs[0].set_ylabel('RMSE')
axs[1].set_ylabel('spread--skill ratio'.replace('--', '–'))
axs[1].axhline(1.0, color='k', lw=0.8, ls=':')
axs[2].set_ylabel('composition gap\n(units of spread)')
axs[2].set_xlabel('network evaluations (NFE)')
axs[2].set_xscale('log')
for a in axs:
    a.grid(alpha=0.3, lw=0.5)
    for s in ('top', 'right'):
        a.spines[s].set_visible(False)
fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0), fontsize=8.5, columnspacing=1.0, handlelength=2.2)
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig(out, bbox_inches='tight')
print('saved', out)
