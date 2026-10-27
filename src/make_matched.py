"""Correlation error at matched reliability: for every persona type, interpolate its reliability-structure frontier (the same
post-processing grid as Figure 3, all followed by party-crosstab calibration) at a target reliability r* and report corr_F there.
Persona types whose frontier never reaches r* are reported as unreachable together with their maximum reliability.
Writes paper/tab_matched.tex and appends macros to stdout. Usage: python make_matched.py [r*]"""
import os, sys
from paths import DATA, RES, PAPER, HERE
import sys, os, re, json, numpy as np, pandas as pd
R = RES; P = PAPER; SEL = json.load(open(R + 'selected_cfg.json'))
RT = float(sys.argv[1]) if len(sys.argv) > 1 else 0.9
DB = SEL.get('diverse_best', 'DIVERSE').split('+')[0]
TYPES = [('BASE', 'Demographic'), ('DIVERSE', 'Diversity instruction')] + ([(DB, 'Diversity, best of 5 wordings')] if DB != 'DIVERSE' else []) + [ ('ARGYLE', 'Argyle backstory'), ('BACKSTORY', 'Life-story backstory'), ('ANTHOLOGY', 'Anthology'), (SEL['persona'], '\\lpp\\ (ours)')]
VARS = [('VALUES_LPP_k1.6', 'uniform deviations (no coupling)'), ('CONV_B_novalues', 'without value priorities'), ('CONV_B_seed2', 'second latent seed'), ('CONV_B_engLow', 'engagement prior .15/.35/.50'), ('CONV_B_engHigh', 'engagement prior .40/.40/.20')]
SETS = [('cg', 'qwen3.8-27b', 'CEK', 'Qwen'), ('cg', 'gemma4-26b', 'CEK', 'Gemma'), ('gss', 'qwen3.8-27b', 'GSS', 'Qwen'), ('gss', 'gemma4-26b', 'GSS', 'Gemma'), ('hm', 'qwen3.8-27b', 'HM', 'Qwen'), ('hm', 'gemma4-26b', 'HM', 'Gemma')]
def post(m, v):
    s = m[len(v):]
    if s == '+OT': return True
    return re.match(r'^\+(temp|sharp)[0-9.]+\+OT$', s) is not None
cells, ok_sets, pwin, pvals, dci = {}, [], {}, {}, {}
def interp(x, y):
    ok = np.isfinite(x) & np.isfinite(y); x, y = x[ok], y[ok]
    if len(x) == 0: return None
    o = np.argsort(x); x, y = x[o], y[o]
    if x.max() < RT: return None
    i = int(np.argmax(x >= RT)); return y[0] if i == 0 else y[i - 1] + (y[i] - y[i - 1]) * (RT - x[i - 1]) / (x[i] - x[i - 1])
for ds, m, dn, mn in SETS:
    f = R + f'final_{ds}_{m}_otp_reps.csv'
    if not os.path.exists(f): continue
    D = pd.read_csv(f); G = D.groupby('method')[['retest_ratio', 'corrF']].mean()
    if G.retest_ratio.isna().all(): continue
    ok_sets.append((ds, m, dn, mn))
    for v, _ in TYPES + VARS:
        pts = sorted((r.retest_ratio, r.corrF) for k, r in G.iterrows() if k.startswith(v + '+') and post(k, v) and np.isfinite(r.retest_ratio))
        if not pts: continue
        x, y = np.array(pts).T
        if x.max() < RT: cells[(ds, m, v)] = ('unr', x.max())
        else:  # first crossing of r* along increasing reliability, linear interpolation between neighbouring grid points
            i = int(np.argmax(x >= RT)); cells[(ds, m, v)] = ('val', y[0] if i == 0 else y[i - 1] + (y[i] - y[i - 1]) * (RT - x[i - 1]) / (x[i] - x[i - 1]))
    # paired evidence: in each half-sample, interpolate every frontier at r*; LPP vs. each baseline that reaches r* on average
    reps = sorted(D.rep.unique()); wins = []; lose = {v: 0 for v, _ in TYPES[:-1]}; diffs = []
    rv = [v for v, _ in TYPES[:-1] if cells.get((ds, m, v), ('',))[0] == 'val']
    for rp in reps:
        Gr = D[D.rep == rp].set_index('method')
        def at(v):
            ks = [k for k in Gr.index if k.startswith(v + '+') and post(k, v)]
            return interp(Gr.loc[ks, 'retest_ratio'].to_numpy(), Gr.loc[ks, 'corrF'].to_numpy()) if ks else None
        l = at(SEL['persona']); bs = {v: at(v) for v, _ in TYPES[:-1]}
        if l is not None: wins.append(all(l < b for b in bs.values() if b is not None))
        if l is not None and rv and all(bs[v] is not None for v in rv): diffs.append(l - min(bs[v] for v in rv))
        for v, b in bs.items():
            if b is not None and (l is None or l >= b): lose[v] += 1
    pwin[(ds, m)] = np.mean(wins) if wins else np.nan
    dci[(ds, m)] = (np.mean(diffs), *np.percentile(diffs, [2.5, 97.5])) if diffs else None
    for v, _ in TYPES[:-1]:
        if cells.get((ds, m, v), ('',))[0] == 'val': pvals[(ds, m, v)] = (lose[v] + 1) / (len(reps) + 1)
# Holm over all LPP-vs-baseline pairs where the baseline reaches r* (one-sided paired half-sample p-values)
ks = sorted(pvals, key=pvals.get); holm = {}; mx = 0
for i, k in enumerate(ks): mx = max(mx, min(1, (len(ks) - i) * pvals[k])); holm[k] = mx
SHORT = {'BASE': 'Demographic', 'DIVERSE': 'Diversity', DB: 'Best diversity', 'ARGYLE': 'Argyle', 'BACKSTORY': 'Life story', 'ANTHOLOGY': 'Anthology'}
cellrows = {}
for ds, m, dn, mn in ok_sets:
    l = cells[(ds, m, SEL['persona'])]; reach = [v for v, _ in TYPES[:-1] if cells.get((ds, m, v), ('',))[0] == 'val']
    if reach:
        bv = min(reach, key=lambda v: cells[(ds, m, v)][1]); bb = f"{cells[(ds, m, bv)][1]:.2f} ({SHORT[bv]})"
    else: bb = 'none'
    ls = f"{l[1]:.2f}" if l[0] == 'val' else 'n.r.'
    if reach and l[0] == "val" and l[1] < cells[(ds, m, bv)][1]: ls = '\\textbf{' + ls + '}'
    elif reach: bb = '\\textbf{' + bb.split(' ')[0] + '} ' + ' '.join(bb.split(' ')[1:])
    u = cells.get((ds, m, 'VALUES_LPP_k1.6')); us = '--' if u is None else (f"{u[1]:.2f}" if u[0] == 'val' else 'not reached')
    dc = dci.get((ds, m)); dcs = '--' if dc is None else f"{dc[0]:+.2f} [{dc[1]:+.2f}, {dc[2]:+.2f}]".replace('-', '$-$')
    cellrows[f'{ds}|{m}'] = f"{ls} & {us} & {bb} & {len(reach)}/{len(TYPES) - 1} & {dcs} & {100 * pwin[(ds, m)]:.0f}\\%"
json.dump(cellrows, open(R + f'matched_{RT}.json', 'w'))
if not os.environ.get('NOWRITE'):
    import subprocess; subprocess.run([sys.executable, __file__, '0.8'], env=dict(os.environ, NOWRITE='1'), stdout=subprocess.DEVNULL, check=True)
    lo = json.load(open(R + 'matched_0.8.json'))
s = ['\\begin{table*}[t]\\centering\\small\\setlength{\\tabcolsep}{4pt}', '\\resizebox{\\textwidth}{!}{\\begin{tabular}{ll cccccc cccccc}', '\\toprule',
     ' & & \\multicolumn{6}{c}{Reliability 0.8 of human} & \\multicolumn{6}{c}{Reliability 0.9 of human} \\\\', '\\cmidrule(lr){3-8}\\cmidrule(lr){9-14}',
     'Survey & Model & \\lpp & Uncoupled & Best baseline & Reach & $\\Delta$ [95\\% CI] & P(lowest) & \\lpp & Uncoupled & Best baseline & Reach & $\\Delta$ [95\\% CI] & P(lowest) \\\\ \\midrule']
for ds, m, dn, mn in ok_sets:
    if not os.environ.get('NOWRITE'): s.append(f"{dn} & {mn} & {lo[f'{ds}|{m}']} & {cellrows[f'{ds}|{m}']} \\\\")
s += ['\\bottomrule', '\\end{tabular}}', '\\caption{Correlation error at matched reliability, interpolated on the frontier of each persona type (Figure~\\ref{fig:frontier}; party-crosstab calibration, post-processing down to $\\tau=0.01$). Best baseline: lowest among the six baseline persona types that reach the target reliability; Reach: how many do (none: no baseline does). $\\Delta$: \\lpp\\ minus the lowest reaching baseline per half-sample, mean and 2.5/97.5 percentiles. P(lowest): share of the 200 paired half-samples in which \\lpp\\ is below every baseline that reaches the target. Uncoupled: \\lpp\\ with one deviation scale for all engagement levels (not selected; first latent draw); not reached: its frontier ends below the target. A frontier lying entirely above the target is evaluated at its least reliable point. GSS reliability is relative to a two-year panel.}', '\\label{tab:matched}', '\\end{table*}']
print('\n'.join(s))
if not os.environ.get('NOWRITE'): open(P + 'tab_matched.tex', 'w').write('\n'.join(s))
nsig = sum(holm[k] < 0.05 for k in holm if cells[(k[0], k[1], SEL['persona'])][1] < cells[k][1])
nwin = sum(1 for k in holm if cells[(k[0], k[1], SEL['persona'])][1] < cells[k][1])
print('pairs', len(holm), 'LPP lower', nwin, 'Holm-significant', nsig, {k: round(v, 4) for k, v in holm.items()})
winp = [pwin[(ds, m)] for ds, m, *_ in ok_sets if pwin[(ds, m)] >= 0.5]; minp = 100 * min(winp) if winp else 0
nb = [(ds, m, v) for ds, m, *_ in ok_sets for v, _ in TYPES[:-1] if (ds, m, v) in cells]; nun = sum(cells[k][0] == 'unr' for k in nb)
None if os.environ.get('NOWRITE') else open(P + 'macros_matched.tex', 'w').write(f"\\newcommand{{\\MATCHrt}}{{{RT:.1f}}}\n\\newcommand{{\\MATCHnun}}{{{nun}}}\n\\newcommand{{\\MATCHnb}}{{{len(nb)}}}\n\\newcommand{{\\MATCHminp}}{{{minp:.0f}\\%}}\n\\newcommand{{\\MATCHnpair}}{{{len(holm)}}}\n\\newcommand{{\\MATCHnwin}}{{{nwin}}}\n\\newcommand{{\\MATCHnholm}}{{{nsig}}}\n")
