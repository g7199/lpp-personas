"""LaTeX tables + macros for the paper from final_eval summaries / reps. Usage: python make_tables.py [otp|pool]"""
import os, sys
from paths import DATA, RES, PAPER, HERE
import sys, json, os, numpy as np, pandas as pd
R = RES; OUT = PAPER; SEL = json.load(open(R + 'selected_cfg.json'))
REG = sys.argv[1] if len(sys.argv) > 1 else 'otp'; TAG = '_otp' if REG == 'otp' else ''; TS = SEL['sharp_T']; OURS = SEL['method']
BL = SEL['baselines']
ROWS = [('Human floor', 'HUMAN_FLOOR'), ('Aggregate sampler (no LLM)', 'AGG_SAMPLER'), ('Demographic', 'BASE+OT'), ('#', 'Selected baselines'),
        ('\\quad Demographic + reweighting', 'BASE+OT+reweight'), ('\\quad Demographic + temperature', BL['BASE']), ('\\quad Diversity instruction', BL['DIVERSE']), ('\\quad Diversity, best of 5 wordings', 'DIVBEST'),
        ('\\quad Argyle backstory', BL['ARGYLE']), ('\\quad Life-story backstory', BL['BACKSTORY']), ('\\quad Anthology', BL['ANTHOLOGY']),
        ('#', 'Sharpened baselines ($\\tau{=}0.05$)'), ('\\quad Demographic', 'BASE+sharp0.05+OT'), ('\\quad Diversity instruction', 'DIVERSE+sharp0.05+OT'),
        ('\\quad Argyle backstory', 'ARGYLE+sharp0.05+OT'), ('\\quad Life-story backstory', 'BACKSTORY+sharp0.05+OT'), ('|', ''), ('\\lpp\\ (ours)', OURS)]
def load(ds, m):
    f = R + f'final_{ds}_{m}{TAG}_reps.csv'
    return pd.read_csv(f) if os.path.exists(f) else None
def cell(D, meth, col, fmt='{:.2f}', ci=False):
    if D is None or meth not in set(D.method): return '--'
    x = D[D.method == meth][col]
    if x.isna().all(): return '--'
    return fmt.format(x.mean()) + (f"\\,{{\\scriptsize[{fmt.format(x.quantile(.025))},{fmt.format(x.quantile(.975))}]}}" if ci else '')
def sigmark(D, meth, col):  # paired test vs ours: dagger if ours better in >=97.5% of half-samples
    if D is None or meth == OURS or meth == 'HUMAN_FLOOR' or meth not in set(D.method) or OURS not in set(D.method): return ''
    a, b = D[D.method == OURS].set_index('rep')[col], D[D.method == meth].set_index('rep')[col]
    return '$^\\dagger$' if ((a - b) < 0).mean() >= .975 else ''
def main_table(ds, models, label, caption, rel=True, edp=True, gap=False):
    cols = ([('ED$_{\\text{party}}$', 'ED_party', '{:.3f}')] if edp else []) + [('corr$_F$', 'corrF', '{:.2f}')] + ([('Rel.', 'retest_ratio', '{:.2f}')] if rel else []) + [('Cross', 'cross_ratio', '{:.2f}')] + ([('Gap', 'gap_ratio', '{:.2f}')] if gap else [])
    rows = ROWS + ([('\\quad \\lpp\\ without coupling ($k{=}1.6$)', f'VALUES_LPP_k1.6+sharp{TS}+OT')] if ds != 'cg' else [])
    models = [(m, nm) for m, nm in models if load(ds, m) is not None]
    Ds = {m: load(ds, m) for m, _ in models}
    for D in Ds.values():
        if D is not None and 'rel' not in D: D['rel'] = np.abs(1 - D.retest_ratio)
    wide = len(models) > 1 and len(cols) * len(models) > 6
    s = [('\\begin{table*}[t]' if wide else '\\begin{table}[t]') + '\\centering\\small\\setlength{\\tabcolsep}{%s}' % ('7pt' if wide else '3.5pt'), ('\\resizebox{\\textwidth}{!}{' if wide else '\\resizebox{\\columnwidth}{!}{') + '\\begin{tabular}{l' + 'c' * len(cols) * len(models) + '}', '\\toprule']
    s.append(' & ' + ' & '.join(f'\\multicolumn{{{len(cols)}}}{{c}}{{{nm}}}' for _, nm in models) + ' \\\\')
    s.append(' & ' + ' & '.join(c[0] for _ in models for c in cols) + ' \\\\ \\midrule')
    body = []
    for nm, meth in rows:
        if nm == '#': body.append(f'\\multicolumn{{{1 + len(cols) * len(models)}}}{{l}}{{\\emph{{{meth}}}}} \\\\'); continue
        if nm == '|': body.append('\\midrule'); continue
        r = [nm]
        for m, _ in models:
            D = Ds[m]
            for _, col, fmt in cols:
                v = cell(D, meth, col, fmt)
                if col in ('corrF', 'ED_party'): v += sigmark(D, meth, col)
                if col == 'retest_ratio' and meth != 'HUMAN_FLOOR': v += sigmark(D, meth, 'rel')
                if meth == 'HUMAN_FLOOR' and col == 'retest_ratio': v = '1.00'
                r.append(v)
        if all(c.startswith('--') for c in r[1:]): continue
        body.append(' & '.join(r) + ' \\\\' + (' \\midrule' if meth == 'HUMAN_FLOOR' else ''))
    body = [x for i, x in enumerate(body) if not (x.startswith('\\multicolumn') and (i + 1 == len(body) or body[i + 1].startswith(('\\multicolumn', '\\midrule'))))]
    s += body
    s += ['\\bottomrule', '\\end{tabular}}', f'\\caption{{{caption}}}', f'\\label{{{label}}}', ('\\end{table*}' if wide else '\\end{table}')]
    return '\n'.join(s)
def abl_table(models):
    A = [('\\lpp\\ (full)', OURS), ('\\quad $-$ coupling ($k{=}1.6$)', f'VALUES_LPP_k1.6+sharp{TS}+OT'), ('\\quad $-$ value priorities', f'CONV_B_novalues+sharp{TS}+OT'),
         ('\\quad $-$ issue deviations', f'ENG_VALUES+sharp{TS}+OT'), ('\\quad $-$ sharpening', f"{SEL['persona']}+OT"), ('\\quad party-relative $\\to$ absolute dev.', f'CONV_B_abs+sharp{TS}+OT'), ('\\quad latents $\\to$ placebo attributes', f'CONV_B_placebo+sharp{TS}+OT'), ('\\quad second latent seed', f'CONV_B_seed2+sharp{TS}+OT'), ('\\quad engagement prior .15/.35/.50', f'CONV_B_engLow+sharp{TS}+OT'), ('\\quad engagement prior .40/.40/.20', f'CONV_B_engHigh+sharp{TS}+OT'), ('Demographic + sharpening ($\\tau{=}0.15$)', f'BASE+sharp{TS}+OT'), ('Demographic', 'BASE+OT')]
    cols = [('corr$_F$', 'corrF'), ('Rel.', 'retest_ratio'), ('Cross', 'cross_ratio'), ('Gap$^{\\text{top}}$', 'gap_ratio')]
    Ds = {m: load('cg', m) for m, _ in models}
    f0 = lambda m: R + f'final_cg_{m}_reps.csv'; Dt = {m: (pd.read_csv(f0(m)) if os.path.exists(f0(m)) else None) for m, _ in models}  # topline calibration, for the party gap
    s = ['\\begin{table}[t]\\centering\\small\\setlength{\\tabcolsep}{3pt}', '\\resizebox{\\columnwidth}{!}{\\begin{tabular}{l' + 'cccc' * len(models) + '}', '\\toprule',
         ' & ' + ' & '.join(f'\\multicolumn{{4}}{{c}}{{{nm}}}' for _, nm in models) + ' \\\\', ' & ' + ' & '.join(c[0] for _ in models for c in cols) + ' \\\\ \\midrule']
    for nm, meth in A:
        r = [cell(Dt[m] if col == 'gap_ratio' else Ds[m], meth, col) for m, _ in models for _, col in cols]
        if not all(c == '--' for c in r): s.append(' & '.join([nm] + r) + ' \\\\')
    s += ['\\bottomrule', '\\end{tabular}}', '\\caption{Ablations on the CEK test half (party-crosstab calibration; every row is calibrated, and sharpened with $\\tau{=}0.15$ as in \\lpp\\ unless it removes sharpening; the sharpened baselines of Table~\\ref{tab:main-otp} use $\\tau{=}0.05$). Rel., Cross and Gap$^{\\text{top}}$ (Democrat--Republican gap under topline calibration) are ratios to the human value (ideal 1; human floor for Cross ' + cell(Ds[models[0][0]], 'HUMAN_FLOOR', 'cross_ratio') + ' Qwen, ' + cell(Ds[models[1][0]], 'HUMAN_FLOOR', 'cross_ratio') + ' Gemma).}', '\\label{tab:abl}', '\\end{table}']
    return '\n'.join(s)
QG = [('qwen3.8-27b', 'Qwen3.8-27B'), ('gemma4-26b', 'Gemma-4-26B-A4B')]
Q2 = QG[:2]
cap = ('CEK test half, ' + ('party-crosstab' if REG == 'otp' else 'topline') + ' calibration. ' + ('' if REG == 'otp' else 'ED$_{\\text{party}}$: party-conditional energy distance (bias-corrected, can be $<0$); ') + 'corr$_F$: correlation-matrix error; Rel.: test-retest reliability / human; Cross: cross-domain $|r|$ / human (ideal 1). Means over 200 half-samples; $^\\dagger$: \\lpp\\ better in $\\geq$97.5\\% of paired half-samples (Rel.: closer to 1).')
open(OUT + f'tab_main{TAG}.tex', 'w').write(main_table('cg', QG if REG != 'otp' else Q2, 'tab:main' + TAG.replace('_', '-'), cap, edp=REG != 'otp'))
open(OUT + f'tab_abl{TAG}.tex', 'w').write(abl_table(Q2))
for ds, rel in [('gss', True), ('hm', True)]:
    cp = (f'{ds.upper()} test half, ' + ('party-crosstab' if REG == 'otp' else 'topline') + ' calibration; persona configuration and post-processing frozen from CEK. ED$_{\\text{party}}$: party-conditional energy distance (bias-corrected, can be $<0$); corr$_F$, Rel., Cross as in Table~\\ref{tab:main-otp}; Gap: Democrat--Republican gap / human. $^\\dagger$ as in Table~\\ref{tab:main-otp}.'
          + (' Human reliability reference: HM follow-up wave of the control arm.' if ds == 'hm' else ' Human reliability reference: GSS 2006--08 panel (two-year interval).'))
    open(OUT + f'tab_{ds}{TAG}.tex', 'w').write(main_table(ds, QG, f'tab:{ds}' + TAG.replace('_', '-'), cp, rel, gap=True))
print('written', REG)
