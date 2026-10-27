"""Compact cross-setting summary table (party-crosstab calibration). Usage: python make_transfer.py [otp|pool]"""
import os, sys
from paths import DATA, RES, PAPER, HERE
import sys, json, os, numpy as np, pandas as pd
R = RES; OUT = PAPER; SEL = json.load(open(R + 'selected_cfg.json'))
REG = sys.argv[1] if len(sys.argv) > 1 else 'otp'; TAG = '_otp' if REG == 'otp' else ''; OURS = SEL['method']; BL = SEL['baselines']
FIT = ['BASE+OT+reweight', 'DIVBEST'] + [BL[k] for k in ('BASE', 'DIVERSE', 'ARGYLE', 'BACKSTORY', 'ANTHOLOGY')]
SHP = [f'{k}+sharp0.05+OT' for k in ('BASE', 'DIVERSE', 'ARGYLE', 'BACKSTORY')]
SET = [('CEK', 'cg', 'qwen3.8-27b', 'Qwen'), ('GSS', 'gss', 'qwen3.8-27b', 'Qwen'), ('HM', 'hm', 'qwen3.8-27b', 'Qwen'),
       ('CEK', 'cg', 'gemma4-26b', 'Gemma'), ('GSS', 'gss', 'gemma4-26b', 'Gemma'), ('HM', 'hm', 'gemma4-26b', 'Gemma')]
def rng(v): v = [x for x in v if np.isfinite(x)]; return '--' if not v else (f'{min(v):.2f}' if len(v) == 1 or f'{min(v):.2f}' == f'{max(v):.2f}' else f'{min(v):.2f}--{max(v):.2f}')
DB = SEL.get('diverse_best', 'DIVERSE').split('+')[0]; TYPES = ['BASE', 'DIVERSE', DB, 'ARGYLE', 'BACKSTORY', 'ANTHOLOGY']
import re
def matched(M, v, rt=0.9):  # persona type's frontier interpolated at reliability rt (as in make_matched.py); None if unreachable
    ks = [k for k in M.index if k.startswith(v + '+') and re.match(r'^\+((temp|sharp)[0-9.]+\+)?OT$', k[len(v):])]
    x, y = M.loc[ks, 'retest_ratio'].to_numpy(), M.loc[ks, 'corrF'].to_numpy(); ok = np.isfinite(x); x, y = x[ok], y[ok]; o = np.argsort(x); x, y = x[o], y[o]
    if len(x) == 0 or x.max() < rt: return None
    i = int(np.argmax(x >= rt)); return y[0] if i == 0 else y[i - 1] + (y[i] - y[i - 1]) * (rt - x[i - 1]) / (x[i] - x[i - 1])
rows = []
for nm, ds, m, mn in SET:
    f = R + f'final_{ds}_{m}{TAG}_reps.csv'
    if not os.path.exists(f): continue
    D = pd.read_csv(f); M = D.groupby('method').mean(numeric_only=True); have = set(M.index)
    if OURS not in have: continue
    fit, shp = [x for x in FIT if x in have], [x for x in SHP if x in have]
    a = D[D.method == OURS].set_index('rep').corrF
    allbetter = lambda L: all(((a - D[D.method == x].set_index('rep').corrF) < 0).mean() >= .975 for x in L)
    rr = lambda L: rng([M.loc[x, 'retest_ratio'] for x in L]) if 'retest_ratio' in M and M.retest_ratio.notna().any() else 'n/a'
    rows.append(f"{nm} & {mn} & {M.loc['HUMAN_FLOOR','corrF']:.2f} & {rng([M.loc[x,'corrF'] for x in fit])}{'$^\\dagger$' if fit and allbetter(fit) else ''} & {rr(fit)} & "
                f"{rng([M.loc[x,'corrF'] for x in shp])}{'$^\\dagger$' if shp and allbetter(shp) else ''} & {rr(shp)} & {('\\textbf{' if M.loc[OURS,'corrF'] <= min(M.loc[x,'corrF'] for x in fit + shp) else '{')}{M.loc[OURS,'corrF']:.2f}}} & {rr([OURS])} & {M.loc[OURS,'cross_ratio']:.2f}" + (lambda l, b: f" & {'\\textbf{%.2f}' % l if (b and l < min(b)) else '%.2f' % l} & {(('\\textbf{%.2f}' if min(b) < l else '%.2f') % min(b)) if b else 'none'}")(matched(M, SEL['persona']), [x for x in (matched(M, v) for v in TYPES) if x is not None]) * (REG == 'otp') + " \\\\")
s = ['\\begin{table*}[t]\\centering\\small' + ('\\setlength{\\tabcolsep}{5pt}' if REG == 'otp' else ''), '\\begin{tabular}{llc cc cc ccc' + (' cc' if REG == 'otp' else '') + '}', '\\toprule',
     ' & & Floor & \\multicolumn{2}{c}{Selected baselines} & \\multicolumn{2}{c}{Sharpened baselines} & \\multicolumn{3}{c}{\\lpp}' + (' & \\multicolumn{2}{c}{At Rel.\\ 0.9}' if REG == 'otp' else '') + ' \\\\',
     '\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-10}' + ('\\cmidrule(lr){11-12}' if REG == 'otp' else ''),
     'Survey & Model & corr$_F$ & corr$_F$ & Rel. & corr$_F$ & Rel. & corr$_F$ & Rel. & Cross' + (' & \\lpp & Best bl.' if REG == 'otp' else '') + ' \\\\ \\midrule'] + rows + ['\\bottomrule', '\\end{tabular}',
     ('\\caption{All settings, test halves, ' + ('party-crosstab' if REG == 'otp' else 'topline') + ' calibration; configuration and post-processing frozen from CEK/Qwen. Ranges over the seven selected and four sharpened baselines. GSS Rel.\\ is relative to a two-year panel. $^\\dagger$: \\lpp\\ lower than every baseline of the group in $\\geq$97.5\\% of paired half-samples.' + (' At Rel.\\ 0.9: corr$_F$ interpolated on each frontier (Figure~\\ref{fig:frontier}) for \\lpp\\ and the best baseline reaching it (none: no baseline does).' if REG == 'otp' else '') + '}'),
     '\\label{tab:transfer' + TAG.replace('_', '-') + '}', '\\end{table*}']
open(OUT + f'tab_transfer{TAG}.tex', 'w').write('\n'.join(s)); print('\n'.join(rows))
