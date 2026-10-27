"""Figure: reliability-structure frontiers in all six model-survey settings (party-crosstab calibration). Each curve = one persona
type under the post-processing grid (temperature 3..1.5, none, sharpening 0.5..0.05; all followed by calibration); x = test-retest
reliability relative to humans, y = correlation error. Shaded band = human-level reliability (>= 0.9). Curves that end before the band
cannot reach human-level reliability under any post-processing. Writes paper/fig_frontier.pdf."""
import os, sys
from paths import DATA, RES, PAPER, HERE
import json, re, numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'Liberation Sans', 'mathtext.fontset': 'custom', 'mathtext.rm': 'Liberation Sans', 'mathtext.it': 'Liberation Sans:italic', 'pdf.fonttype': 42})
R = RES; SEL = json.load(open(R + 'selected_cfg.json')); DB = SEL.get('diverse_best', 'DIVERSE').split('+')[0]
TYPES = [('BASE', 'Demographic', '#8c8c8c'), ('DIVERSE', 'Diversity instr.', '#1f77b4'), (DB, 'Diversity, best wording', '#17becf'), ('ARGYLE', 'Argyle backstory', '#ff7f0e'),
         ('BACKSTORY', 'Life-story backstory', '#2ca02c'), ('ANTHOLOGY', 'Anthology', '#9467bd'), (SEL['persona'], 'LPP (ours)', '#d62728')]
def postT(meth, v):
    s = meth[len(v):]
    if s == '+OT': return 1.0
    m = re.match(r'^\+(temp|sharp)([0-9.]+)\+OT$', s); return float(m.group(2)) if m else None
COLS = [('cg', 'CEK'), ('gss', 'GSS'), ('hm', 'HM')]; ROWS = [('qwen3.8-27b', 'Qwen3.8-27B'), ('gemma4-26b', 'Gemma-4-26B-A4B')]
fig, axes = plt.subplots(2, 3, figsize=(7.0, 3.9))
XM = 1.2  # common x-range: up to the most reliable plotted point
for M, _ in ROWS:
    for ds, _ in COLS:
        G0 = pd.read_csv(R + f'final_{ds}_{M}_otp_reps.csv').groupby('method')['retest_ratio'].mean()
        xs = [G0[m] for m in G0.index for v, _, _ in TYPES if m.startswith(v + '+') and postT(m, v) is not None and np.isfinite(G0[m])]
        XM = max(XM, max(xs) + 0.04)
for i, (M, mn) in enumerate(ROWS):
    for j, (ds, dn) in enumerate(COLS):
        ax = axes[i, j]; D = pd.read_csv(R + f'final_{ds}_{M}_otp_reps.csv'); G = D.groupby('method')[['retest_ratio', 'corrF']].mean()
        ax.axvspan(0.9, XM, color='#e8f4ec', zorder=0, lw=0)
        for v, nm, col in TYPES:
            pts = sorted([(postT(m, v), *G.loc[m]) for m in G.index if m.startswith(v + '+') and postT(m, v) is not None and np.isfinite(G.loc[m, 'retest_ratio'])], reverse=True)
            if not pts: continue
            T, x, y = map(np.array, zip(*pts)); ours = nm.startswith('LPP')
            ax.plot(x, y, '--' if nm.startswith('Diversity, best') else '-', color=col, lw=1.8 if ours else 0.9, marker='o', ms=2.6 if not ours else 3.4, label=nm, zorder=3 if ours else 2)
            if ours: ax.plot(*G.loc[SEL['method']][['retest_ratio', 'corrF']], '*', color=col, ms=10, zorder=4, mec='k', mew=.5)
        ax.plot(1.0, G.loc['HUMAN_FLOOR', 'corrF'], 'D', color='k', ms=5, zorder=4)
        ax.set_xlim(0.1, XM); ax.tick_params(labelsize=6.5, length=2)
        if i == 0: ax.set_title(dn, fontsize=8.5)
        if j == 0: ax.set_ylabel(f'{mn}\ncorrelation error', fontsize=7.5)
        if i == 1: ax.set_xlabel('reliability / human', fontsize=7.5)
h, l = axes[0, 0].get_legend_handles_labels()
h += [plt.Line2D([], [], color='k', marker='D', ls='', ms=4.5), plt.Rectangle((0, 0), 1, 1, color='#e8f4ec')]; l += ['human floor', 'human-level reliability']
fig.legend(h, l, fontsize=6.6, frameon=False, loc='lower center', ncol=5, handlelength=1.5, columnspacing=1.0)
plt.tight_layout(rect=(0, 0.1, 1, 1), h_pad=0.6, w_pad=0.8); plt.savefig(PAPER + 'fig_frontier.pdf'); plt.savefig(R + 'fig_frontier.png', dpi=170); print('ok')
