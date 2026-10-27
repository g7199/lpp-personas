"""Dev-half selection of each baseline's post-processing with the same criterion as the LPP configuration:
mean of T2 (party-crosstab calibration) and S2 (topline calibration), both normalized by the calibrated demographic persona (BASE+OT).
Usage: python select_baselines.py MODEL"""
import os, sys
from paths import DATA, RES, PAPER, HERE
import sys, re, numpy as np, pandas as pd
MODEL = sys.argv[1] if len(sys.argv) > 1 else 'qwen3.8-27b'; R = RES
FL = float(re.search(r'corrF ([0-9.]+)', [l for l in open(R + 'va_cal_otp.log') if 'floors (human' in l][0]).group(1))
O, P = [pd.read_csv(R + f'var_summary_{MODEL}_cal{t}.csv').set_index('variant') for t in ('_otp', '')]
def T2(D, m): b = D.loc['BASE+OT']; return (D.loc[m, 'corrF'] - FL) / (b.corrF - FL) + abs(1 - D.loc[m, 'retest_ratio']) / abs(1 - b.retest_ratio)
def S2(D, m): b = D.loc['BASE+OT']; return D.loc[m, 'ED_party'] / b.ED_party + T2(D, m)
out = {}
for v in ('CONV_B', 'BASE', 'DIVERSE', 'ARGYLE', 'BACKSTORY', 'ANTHOLOGY'):
    cand = [m for m in O.index if m in P.index and re.match(rf'^{v}\+((temp|sharp)[0-9.]+\+)?OT$', m)]
    if not cand: print(v, 'no dev runs'); continue
    sc = pd.Series({m: (T2(O, m) + S2(P, m)) / 2 for m in cand}).sort_values()
    out[v] = sc.index[0]; print(f'{v:10s} best {sc.index[0]:28s} {sc.iloc[0]:.3f} | ' + '  '.join(f'{m.split("+",1)[1]}:{s:.2f}' for m, s in sc.iloc[1:5].items()))
    shp = {m: O.loc[m, 'retest_ratio'] for m in cand if 'sharp' in m}
    if shp: print(f'{"":10s} matched-reliability (dev rel closest to 1, party OT): {min(shp, key=lambda m: abs(1 - shp[m]))}')
print(out)
