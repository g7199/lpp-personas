"""Diagnosis table (Demographic personas, no post-processing) + macros for the paper."""
import os, sys
from paths import DATA, RES, PAPER, HERE
import os, sys, pandas as pd
V = sys.argv[1] if len(sys.argv) > 1 else 'BASE'; SUF = '' if V == 'BASE' else '_div'
R = RES; OUT = PAPER
COLS = [('qwen3.8-27b', 'cg', 'Qwen'), ('qwen3.8-27b', 'gss', 'Qwen'), ('qwen3.8-27b', 'hm', 'Qwen'), ('gemma4-26b', 'cg', 'Gemma'), ('gemma4-26b', 'gss', 'Gemma'),
        ('gemma4-26b', 'hm', 'Gemma'), ('gpt-6.1-sol-low', 'cg', 'GPT')]
STATS = [('cross_ratio', 'Cross-domain $|r|$'), ('gap_ratio', 'Party gap'), ('wpsd_ratio', 'Within-party SD'), ('pidr2_ratio', 'Party $R^2$')]
G, H, P = {}, {}, {}
for m, ds, _ in COLS:
    f = R + f'final_{ds}_{m}_otp_reps.csv'
    if not os.path.exists(f): continue
    D = pd.read_csv(f).groupby('method').mean(numeric_only=True); G[(m, ds)] = D.loc[V]; H[(m, ds)] = D.loc['HUMAN_FLOOR']
    P[(m, ds)] = pd.read_csv(R + f'final_{ds}_{m}_reps.csv').groupby('method').mean(numeric_only=True).loc['BASE+OT'] if os.path.exists(R + f'final_{ds}_{m}_reps.csv') else None
cols = [c for c in COLS if (c[0], c[1]) in G]
DSN = {'cg': 'CEK', 'gss': 'GSS', 'hm': 'HM'}
s = ['\\begin{table}[t]\\centering\\small\\setlength{\\tabcolsep}{3pt}', '\\resizebox{\\columnwidth}{!}{\\begin{tabular}{l' + 'c' * len(cols) + '}', '\\toprule',
     ' & ' + ' & '.join(n for _, _, n in cols) + ' \\\\', ' & ' + ' & '.join(DSN[d] for _, d, _ in cols) + ' \\\\ \\midrule']
for k, nm in STATS:
    fl = [H[(m, d)][k] for m, d, _ in cols]
    s.append(nm + ' & ' + ' & '.join('n/a' if pd.isna(G[(m, d)][k]) else f"{G[(m, d)][k]:.2f}" for m, d, _ in cols) + ' \\\\')
fl = {k: (min(H[c][k] for c in G if not pd.isna(H[c][k])), max(H[c][k] for c in G if not pd.isna(H[c][k]))) for k, _ in STATS}
s += ['\\bottomrule', '\\end{tabular}}', '\\caption{' + ('Demographic personas' if V == 'BASE' else 'Diversity-instruction personas') + ', no post-processing: statistics of the simulated public relative to the human test half (1 = human). Human floor (two disjoint human samples): ' +
      '; '.join(f"{nm} {fl[k][0]:.2f}--{fl[k][1]:.2f}" for k, nm in STATS) + '.' + (' Gemma/GSS: one item (allowing an atheist to teach) receives the same answer from every diversity-instruction persona; correlations are computed over the remaining items.' if V != 'BASE' else '') + '}', '\\label{tab:diag' + SUF.replace('_', '-') + '}', '\\end{table}']
open(OUT + f'tab_diag{SUF}.tex', 'w').write('\n'.join(s))
if V != 'BASE': print('\n'.join(s)); sys.exit()
cr = [G[c]['cross_ratio'] for c in G]; wi = [G[c]['within_ratio'] for c in G if not pd.isna(G[c]['within_ratio'])]; pr = [G[c]['pidr2_ratio'] for c in G]; po = [P[c]['cross_ratio'] for c in G if P[c] is not None]
mac = [f"\\newcommand{{\\DIAGwithin}}{{{min(wi):.1f}--{max(wi):.1f}$\\times$}}", f"\\newcommand{{\\DIAGcross}}{{{min(cr):.1f}--{max(cr):.1f}$\\times$}}", f"\\newcommand{{\\DIAGpid}}{{{min(pr):.1f}--{max(pr):.1f}$\\times$}}",
       f"\\newcommand{{\\DIAGotpool}}{{{min(po):.1f}--{max(po):.1f}$\\times$}}"]
open(OUT + 'macros_diag.tex', 'w').write('\n'.join(mac) + '\n'); print('\n'.join(s)); print('\n'.join(mac))
