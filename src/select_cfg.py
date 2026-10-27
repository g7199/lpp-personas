"""Apply the selection rule (paper, Appendix B) to the development-half summary and write results/selected_cfg.json.
Usage: python select_cfg.py MODEL [--write]"""
import os, sys
from paths import DATA, RES, PAPER, HERE
import sys, json, pandas as pd
from variants import VARIANTS
MODEL = sys.argv[1]; R = RES
S = pd.read_csv(R + f'var_summary_{MODEL}_cal.csv').set_index('variant'); FLOOR = 1.073  # human corrF floor on CAL (var_analyze)
b = S.loc['BASE+OT']; S['score'] = S.ED_party / b.ED_party + (S.corrF - FLOOR) / (b.corrF - FLOOR)
FAMILY = [v for v in VARIANTS if v.split('_')[0] in ('ENG', 'DEV', 'LPP', 'VALUES', 'MORAL') and 'LPPH' not in v and v != 'LPP_DIVERSE']
cand = [v + '+OT' for v in FAMILY if v + '+OT' in S.index]
C = S.loc[cand].sort_values('score'); pd.set_option('display.width', 200)
print(C[['score', 'ED_party', 'corrF', 'cross_ratio', 'within_ratio', 'gap_ratio', 'pidr2']].round(3).to_string())
T = S.loc[[v for v in S.index if v.startswith('BASE+temp')]].sort_values('score'); print(T[['score', 'ED_party', 'corrF', 'ED']].round(3).to_string())
best = C.index[0][:-3]; tb = T.index[0]; Tstar = float(tb.split('temp')[1].replace('+OT', ''))
out = dict(method=best + '+OT', persona=best, cfg=VARIANTS[best], temp_T=Tstar, temp_with_ot=tb.endswith('+OT'), dev_k=VARIANTS[best].get('dev_k', 1.0),
           p_eng=VARIANTS[best].get('p_eng', {'high': 0.25, 'mid': 0.35, 'low': 0.40}), score=float(C.score.iloc[0]), runner_up=C.index[1])
print(out)
if '--write' in sys.argv: json.dump(out, open(R + 'selected_cfg.json', 'w'), indent=1); print('written')
