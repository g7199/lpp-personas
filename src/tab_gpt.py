"""Table: GPT-6.1-Sol on the CEK test half under both calibration regimes, plus empirical two-draw reliability."""
import os, sys
from paths import DATA, RES, PAPER, HERE
import json, pandas as pd
R = RES; SEL = json.load(open(R + 'selected_cfg.json')); L = SEL['gpt']['persona']
P = pd.read_csv(R + 'final_cg_gpt-6.1-sol-low_reps.csv').groupby('method').mean(numeric_only=True)
X = pd.read_csv(R + 'final_cg_gpt-6.1-sol-low_otp_reps.csv').groupby('method').mean(numeric_only=True)
T = pd.read_csv(R + 'retest_gpt-6.1-sol-low-low_eval.csv') if False else pd.read_csv(R + 'retest_gpt-6.1-sol-low_eval.csv')
T = T.set_index('variant')
rows = [('Human floor', 'HUMAN_FLOOR', 'HUMAN'), ('Demographic', 'BASE', 'BASE'), ('Diversity instruction', 'DIVERSE', 'DIVERSE'), ('\\lpp', L, L)]
s = ['\\begin{table}[t]\\centering\\small\\setlength{\\tabcolsep}{2.6pt}', '\\resizebox{\\columnwidth}{!}{\\begin{tabular}{lcccc@{\\hspace{6pt}}ccc}', '\\toprule',
     ' & \\multicolumn{4}{c}{Toplines} & \\multicolumn{3}{c}{+ party crosstabs} \\\\', '\\cmidrule(lr){2-5}\\cmidrule(lr){6-8}',
     ' & ED$_{\\text{p}}$ & corr$_F$ & wpSD & Rel. & corr$_F$ & Cross & Within \\\\ \\midrule']
for nm, v, t in rows:
    k = v if v == 'HUMAN_FLOOR' else v + '+OT'
    rel = '1.00' if t == 'HUMAN' else f"{T.loc[t, 'ratio_ot']:.2f}"
    s.append(f"{nm} & {P.loc[k, 'ED_party']:.2f} & {P.loc[k, 'corrF']:.2f} & {P.loc[k, 'wpsd_ratio']:.2f} & {rel} & {X.loc[k, 'corrF']:.2f} & {X.loc[k, 'cross_ratio']:.2f} & {X.loc[k, 'within_ratio']:.2f} \\\\" + (' \\midrule' if t == 'HUMAN' else ''))
s += ['\\bottomrule', '\\end{tabular}}', '\\caption{GPT-6.1-Sol, CEK test half, after calibration with toplines only (left) or toplines and party crosstabs (right). wpSD, Cross, Within: within-party SD, cross- and within-domain $|r|$ relative to humans. Rel.: empirical reliability from two independent calls per persona after topline calibration, relative to humans (1.38 for demographic personas before calibration).}', '\\label{tab:gpt}', '\\end{table}']
open(PAPER + 'tab_gpt.tex', 'w').write('\n'.join(s)); print('\n'.join(s))
