"""Appendix robustness table: one line per check, compact. Reads the result files of the individual checks.
Writes paper/tab_robust.tex."""
import os, sys
from paths import DATA, RES, PAPER, HERE
import os, re, json, numpy as np, pandas as pd
R = RES; P = PAPER; SEL = json.load(open(R + 'selected_cfg.json')); O = SEL['method']; TS = SEL['sharp_T']
BL = [v for v in SEL['baselines'].values()] + ['BASE+OT+reweight', 'DIVBEST']
def M(f): return pd.read_csv(R + f).groupby('method').mean(numeric_only=True) if os.path.exists(R + f) else None
rows = []; BESTB = {}; QB = {}
def matched(G, v, rt=0.9):  # persona type's frontier interpolated at reliability rt (as in the matched-reliability analysis); None if unreachable
    ks = [k for k in G.index if k.startswith(v + '+') and re.match(r'^\+((temp|sharp)[0-9.]+\+)?OT$', k[len(v):])]
    x, y = G.loc[ks, 'retest_ratio'].to_numpy(), G.loc[ks, 'corrF'].to_numpy(); ok = np.isfinite(x); x, y = x[ok], y[ok]; o = np.argsort(x); x, y = x[o], y[o]
    if len(x) == 0 or x.max() < rt: return None
    i = int(np.argmax(x >= rt)); return y[0] if i == 0 else y[i - 1] + (y[i] - y[i - 1]) * (rt - x[i - 1]) / (x[i] - x[i - 1])
# 1. external calibration (GSS 2022 tabulations)
cells = []
for m, nm in (('qwen3.8-27b', 'Qwen'), ('gemma4-26b', 'Gemma')):
    D = M(f'final_gss_{m}_otp_cal2022_reps.csv')
    if D is None: continue
    b = D.loc[[x for x in BL if x in D.index], 'corrF']
    bm = [x for x in (matched(D, v) for v in ('BASE', 'DIVERSE', SEL['diverse_best'].split('+')[0], 'ARGYLE', 'BACKSTORY', 'ANTHOLOGY')) if x is not None]
    cells.append(f"{nm}: selected {D.loc[O, 'corrF']:.2f} vs.\\ {b.min():.2f}--{b.max():.2f}; at Rel.\\ 0.9 {matched(D, SEL['persona']):.2f} vs.\\ {min(bm):.2f}")
rows.append(('GSS calibrated with 2022 tabulations', 'corr$_F$, \\lpp\\ vs.\\ baselines', '; '.join(cells)))
# 2. latent seeds
cells = []
for m, nm in (('qwen3.8-27b', 'Qwen'), ('gemma4-26b', 'Gemma')):
    D = M(f'final_cg_{m}_otp_reps.csv'); ks = [f'CONV_B_seed{k}+sharp{TS}+OT' for k in range(1, 7)]
    v = [D.loc[k, 'corrF'] for k in ks if k in D.index]; G = D.groupby(level=0).first()
    bm = [matched(G, b) for b in ('BASE', 'DIVERSE', SEL['diverse_best'].split('+')[0], 'ARGYLE', 'BACKSTORY', 'ANTHOLOGY')]; bm = [b for b in bm if b is not None]
    ref = f"best baseline at Rel.\\ 0.9: {min(bm):.2f}" if bm else "no baseline reaches Rel.\\ 0.9"
    if bm: BESTB[nm.lower()] = f'{min(bm):.2f}'
    if len(v) > 1: cells.append(f"{nm}: {np.mean(v):.2f} $\\pm$ {np.std(v, ddof=1):.2f}, range {min(v):.2f}--{max(v):.2f} ({ref})")
rows.append((f'Redrawn latents, {len(v)} seeds (CEK)', 'corr$_F$ mean $\\pm$ SD', '; '.join(cells)))
# 3. polychoric correlations
cells = []
for m, nm in (('qwen3.8-27b', 'Qwen'), ('gemma4-26b', 'Gemma')):
    f = R + f'polychoric_{m}.csv'
    if os.path.exists(f):
        D = pd.read_csv(f, index_col=0); D = D.drop(index=[i for i in D.index if i.startswith('human')])
        cells.append(f"{nm}: $\\rho$ = {D.corrF_polychoric.corr(D.corrF_pearson, method='spearman'):.2f}")
rows.append(('Polychoric instead of Pearson (CEK)', 'rank correlation of methods', '; '.join(cells)))
# 5. human retest by partisan strength
f = R + 'retest_by_strength.json'
if os.path.exists(f):
    d = json.load(open(f)); rows.append(('Human retest by partisan strength (CEK)', 'independent / leaner / weak / strong', ' / '.join(f'{d[k]:.2f}' for k in ('independent', 'leaner', 'weak partisan', 'strong partisan'))))
# 6. engagement-graded sharpening (development half)
f = R + 'dev_engsharp_qwen3.8-27b.csv'
if os.path.exists(f):
    D = pd.read_csv(f, index_col=0); g = D[[i.startswith('graded') and len(set(i.split()[1].split('/'))) > 1 for i in D.index]]
    rows.append(('Engagement-dependent sharpening (dev.)', 'selection criterion (lower is better)', f"best graded {g.score.min():.2f} vs.\\ uniform {D.loc['uniform T=0.15', 'score']:.2f}"))
# 7. one-dimensional latent
cells = []
for m, nm in (('qwen3.8-27b', 'Qwen'), ('gemma4-26b', 'Gemma')):
    D = M(f'final_cg_{m}_otp_reps.csv'); spec = SEL.get('ideo1d_best')
    if spec and spec in D.index: cells.append(f"{nm}: selected {D.loc[spec, 'corrF']:.2f} at Rel.\\ {D.loc[spec, 'retest_ratio']:.2f}, {matched(D, 'IDEO1D'):.2f} at Rel.\\ 0.9 (\\lpp: {D.loc[O, 'corrF']:.2f} at {D.loc[O, 'retest_ratio']:.2f}, {matched(D, SEL['persona']):.2f} at 0.9)")
IDEO = {nm.lower(): f"{M(f'final_cg_{m}_otp_reps.csv').loc[SEL['ideo1d_best'], 'retest_ratio']:.2f}" for m, nm in (('qwen3.8-27b', 'Qwen'), ('gemma4-26b', 'Gemma')) if SEL['ideo1d_best'] in M(f'final_cg_{m}_otp_reps.csv').index}
rows.append(('One continuous ideology latent (CEK)', 'corr$_F$ and Rel.', '; '.join(cells)))
# 8. non-LLM offsets: the same latents applied directly to the demographic distributions (with / without engagement coupling)
mac = {}
for NMO, label, tag in (('OFFSET', 'Latents as direct offsets, no prompting (CEK)', 'off'), ('OFFSETU', 'Uncoupled persistent offsets, no prompting (CEK)', 'offu')):
    cells = []
    for m, nm in (('qwen3.8-27b', 'Qwen'), ('gemma4-26b', 'Gemma')):
        D = M(f'final_cg_{m}_otp_reps.csv')
        if NMO + '+OT' not in D.index: continue
        om = matched(D, NMO); mx = D.loc[[k for k in D.index if k.startswith(NMO + '+')], 'retest_ratio'].max()
        om_s = f"{om:.2f}" if om is not None else f"not reached (max.\\ {mx:.2f})"; mb = matched(D, 'BASE')
        cells.append(f"{nm}: selected {D.loc[NMO + '+OT', 'corrF']:.2f} at Rel.\\ {D.loc[NMO + '+OT', 'retest_ratio']:.2f}; at Rel.\\ 0.9 {om_s} (\\lpp\\ {matched(D, SEL['persona']):.2f}, demographic {('%.2f' % mb) if mb is not None else 'not reached'})")
        k = nm.lower(); mac[f'{tag}-{k}'] = om_s.split(' (')[0]; mac[f'{tag}-{k}-maxrel'] = f'{mx:.2f}'
        mac[f'off-{k}-lpp'] = f"{matched(D, SEL['persona']):.2f}"; mac[f'off-{k}-base'] = ('%.2f' % mb) if mb is not None else 'not reached'
    rows.append((label, 'corr$_F$ and Rel.', '; '.join(cells)))
seedv = {}
for m, nm in (('qwen3.8-27b', 'qwen'), ('gemma4-26b', 'gemma')):
    D = M(f'final_cg_{m}_otp_reps.csv'); v = [D.loc[k, 'corrF'] for k in [f'CONV_B_seed{k}+sharp{TS}+OT' for k in range(1, 7)] if k in D.index]
    mac[f'seed-{nm}-mean'] = f'{np.mean(v):.2f}'; mac[f'seed-{nm}-sd'] = f'{np.std(v, ddof=1):.2f}'; mac[f'seed-{nm}-max'] = f'{max(v):.2f}'; mac[f'seed-{nm}-min'] = f'{min(v):.2f}'
# 9. perturbed-prompt retest: second wave with reworded questions (para_eval.py)
cells = []
for m, nm in (('qwen3.8-27b', 'Qwen'), ('gemma4-26b', 'Gemma')):
    f = R + f'para_rel_{m}.csv'
    if not os.path.exists(f): continue
    D = pd.read_csv(f).set_index('method'); mx = D.groupby('persona').rel_para.max()
    sel = {SEL['persona']: O, **{b: spec for b, spec in SEL['baselines'].items()}}
    bl = [p for p in mx.index if p != SEL['persona']]
    if SEL['persona'] not in mx.index or not bl: continue
    selb = [D.loc[sel[p], 'rel_para'] for p in bl if p in sel and sel[p] in D.index]
    cells.append(f"{nm}: selected {D.loc[O, 'rel_para']:.2f} vs.\\ {min(selb):.2f}--{max(selb):.2f}; highest reachable {mx[SEL['persona']]:.2f} vs.\\ {mx[bl].min():.2f}--{mx[bl].max():.2f}")
    k = nm.lower(); mac[f'para-{k}-lpp'] = f"{D.loc[O, 'rel_para']:.2f}"; mac[f'para-{k}-lppmax'] = f"{mx[SEL['persona']]:.2f}"; mac[f'para-{k}-blmax'] = f"{mx[bl].max():.2f}"
    mac[f'para-{k}-same'] = f"{D.loc[O, 'rel_same']:.2f}"; mac[f'para-{k}-nbl'] = str(len(bl))
rows.append(('Retest with reworded questions (CEK)', 'Rel., \\lpp\\ vs.\\ baselines', '; '.join(cells)))
# 10. no calibration (nocal_check.py): frontiers of the uncalibrated distributions, interpolated at Rel. 0.9
cells = []
SN = {'BASE': 'demographic', 'DIVERSE': 'diversity', 'ARGYLE': 'Argyle', 'BACKSTORY': 'life story', 'ANTHOLOGY': 'Anthology'}; SN[SEL['diverse_best'].split('+')[0]] = 'best diversity'
for m, nm in (('qwen3.8-27b', 'Qwen'), ('gemma4-26b', 'Gemma')):
    f = R + f'nocal_{m}.csv'
    if not os.path.exists(f): continue
    D = pd.read_csv(f)
    def at(v, rt=0.9):
        d = D[(D.persona == v) & D.rel.notna()].sort_values('rel'); x, y = d.rel.to_numpy(), d.corrF.to_numpy()
        if len(x) == 0 or x.max() < rt: return None
        i = int(np.argmax(x >= rt)); return y[0] if i == 0 else y[i - 1] + (y[i] - y[i - 1]) * (rt - x[i - 1]) / (x[i] - x[i - 1])
    b = {v: at(v) for v in SN if at(v) is not None}; bv = min(b, key=b.get); rest = [x for k, x in b.items() if k != bv]
    cells.append(f"{nm}: \\lpp\\ {at(SEL['persona']):.2f} vs.\\ {SN[bv]} {b[bv]:.2f}, other baselines {min(rest):.2f}--{max(rest):.2f}")
rows.append(('No calibration (CEK)', 'corr$_F$ at Rel.\\ 0.9', '; '.join(cells)))
rows = [r for r in rows if r[2]]
s = ['\\begin{table*}[t]\\centering\\small', '\\begin{tabular}{@{}>{\\raggedright\\arraybackslash}p{3.6cm}>{\\raggedright\\arraybackslash}p{2.7cm}>{\\raggedright\\arraybackslash}p{8.85cm}@{}}', '\\toprule', 'Check & Quantity & Result \\\\ \\midrule'] + [f'{a} & {b} & {c} \\\\' for a, b, c in rows] + \
    ['\\bottomrule', '\\end{tabular}', '\\caption{Robustness checks (party-crosstab calibration unless noted). Correlation errors are not comparable across rows that use different samples.}', '\\label{tab:robust}', '\\end{table*}']
open(P + 'tab_robust.tex', 'w').write('\n'.join(s)); print('\n'.join(s))
mac.update({f'{k}-bestbase': v for k, v in BESTB.items()}); mac.update({f'ideo-{k}-rel': v for k, v in IDEO.items()})
open(P + 'macros_robust.tex', 'w').write(''.join(f'\\expandafter\\def\\csname R@{k}\\endcsname{{{v}}}\n' for k, v in mac.items()) + '\\providecommand{\\RB}[1]{\\ifcsname R@#1\\endcsname\\csname R@#1\\endcsname\\else\\textbf{??}\\fi}\n')
