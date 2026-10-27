"""Appendix table for the HM framing experiment (Section 7.5): on-topic effect, its ratio to the human effect, cell-wise correlation
with the human 4x5 effect matrix and RMSE, for demographic personas and LPP under each post-processing. Writes paper/tab_hmdyn.tex."""
import os, sys
from paths import DATA, RES, PAPER, HERE
import json, pandas as pd
R = RES; SEL = json.load(open(R + 'selected_cfg.json')); TS, TT = SEL['sharp_T'], SEL['temp_T']
POST = [('raw', 'none'), ('OT', 'calibration'), (f'temp{TT}+OT', f'temperature {TT:g} + cal.'), (f'sharp{TS}+OT', f'sharpening {TS:g} + cal.')]
s = ['\\begin{table}[t]\\centering\\small\\setlength{\\tabcolsep}{2.5pt}', '\\resizebox{\\columnwidth}{!}{\\begin{tabular}{ll cccc@{\\hspace{6pt}}cccc}', '\\toprule',
     ' & & \\multicolumn{4}{c}{Qwen3.8-27B} & \\multicolumn{4}{c}{Gemma-4-26B-A4B} \\\\', '\\cmidrule(lr){3-6}\\cmidrule(lr){7-10}',
     'Persona & Post-proc. & On-topic & $\\times$hum. & $r_{\\text{cells}}$ & RMSE & On-topic & $\\times$hum. & $r_{\\text{cells}}$ & RMSE \\\\ \\midrule']
D = {}
for m in ('qwen3.8-27b', 'gemma4-26b'):
    d = pd.read_csv(R + f'hmdyn_{m}.csv').set_index(['variant', 'post']); D[m] = d
    hum = d.loc[('BASE', 'raw'), 'direct'] / d.loc[('BASE', 'raw'), 'direct_gain']
SHORT = {'none': 'none', 'calibration': 'cal.', f'temperature {TT:g} + cal.': f'temp.\\ {TT:g} + cal.', f'sharpening {TS:g} + cal.': f'sharp.\\ {TS:g} + cal.'}
for v, vn in (('BASE', 'Demographic'), (SEL['persona'], '\\lpp')):
    for i, (p, pn) in enumerate(POST):
        cells = []
        for m in ('qwen3.8-27b', 'gemma4-26b'):
            d = D[m]; r = d.loc[(v, p)]; best = min(d.loc[(v2, p), 'rmse'] for v2 in ('BASE', SEL['persona']))
            cells += [f'{r.direct:.2f}', f'{r.direct_gain:.1f}', f'{r.r_cells:.2f}', ('\\textbf{%.2f}' if r.rmse == best else '%.2f') % r.rmse]
        s.append(f"{vn if i == 0 else ''} & {SHORT[pn]} & " + ' & '.join(cells) + ' \\\\')
    if v == 'BASE': s.append('\\midrule')
s += ['\\bottomrule', '\\end{tabular}}', f'\\caption{{HM framing experiment (\\S\\ref{{sec:dynamic}}): mean on-topic effect (framed minus control, in human control-arm SD; human {hum:.2f}), its ratio to the human effect, cell-wise correlation with the human $4\\times5$ effect matrix and its RMSE. Calibration is fit on the simulated control arm only. Bold: lower RMSE of the two persona types.}}', '\\label{tab:hmdyn}', '\\end{table}']
open(PAPER + 'tab_hmdyn.tex', 'w').write('\n'.join(s)); print('\n'.join(s))
