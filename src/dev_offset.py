"""Development-half selection of the non-LLM offset control (offsets.py): beta x post-processing grid, same criterion as every
other method (mean of T2 under party-crosstab and S2 under topline calibration, relative to BASE+OT). Usage: python dev_offset.py MODEL"""
import os, sys
from paths import DATA, RES, PAPER, HERE
import os, sys, re, json, numpy as np, pandas as pd
MODEL = sys.argv[1] if len(sys.argv) > 1 else 'qwen3.8-27b'; CPL = (sys.argv[2] if len(sys.argv) > 2 else 'coupled') == 'coupled'; NM = 'OFFSET' if CPL else 'OFFSETU'
src = open(os.path.join(HERE, 'var_analyze.py')).read()
pre = src[:src.index('rows = []')]; post_defs = src[src.index('def ot_map'):src.index('for v in list(loaded):')]
sys.argv = ['x', MODEL, 'cal', '100']; g = {}; exec(pre, g)
from offsets import tilt
R = RES; f = R + f'var_{MODEL}_cal_n100_BASE.csv'
P0, pid0 = g['load'](f); sids = sorted(pd.read_csv(f).sid.unique())
def temp(P, T): Q = (P / np.maximum(P.max(-1, keepdims=True), 1e-300)) ** (1 / T); return Q / Q.sum(-1, keepdims=True)
GRID = [('OT', None)] + [(f'temp{T}+OT', T) for T in (3.0, 2.0, 1.5)] + [(f'sharp{T}+OT', T) for T in (0.5, 0.35, 0.25, 0.15, 0.1, 0.05)]
res = {}
for otp in ('1', '0'):
    os.environ['OTP'] = otp; g2 = dict(g); g2['rng'] = np.random.default_rng(0); exec(post_defs, g2)
    ev = lambda name, P: g2['evaluate'](name, g2['ot'](P), pid0)
    rows = [ev('BASE+OT', P0)]
    for beta in (0.5, 1.0, 2.0, 3.0, 5.0):
        Pb = tilt(P0, sids, pid0, beta, CPL)
        for nm, T in GRID: rows.append(ev(f'{NM}_b{beta}+{nm}', Pb if T is None else temp(Pb, T)))
    res[otp] = pd.DataFrame(rows).set_index('variant')
FL = float(re.search(r'corrF ([0-9.]+)', [l for l in open(R + 'va_cal_otp.log') if 'floors (human' in l][0]).group(1))
def T2(D, m): b = D.loc['BASE+OT']; return (D.loc[m, 'corrF'] - FL) / (b.corrF - FL) + abs(1 - D.loc[m, 'retest_ratio']) / abs(1 - b.retest_ratio)
def S2(D, m): b = D.loc['BASE+OT']; return D.loc[m, 'ED_party'] / b.ED_party + T2(D, m)
O, Pp = res['1'], res['0']
S = pd.DataFrame({m: dict(score=(T2(O, m) + S2(Pp, m)) / 2, corrF_otp=O.loc[m, 'corrF'], rel=O.loc[m, 'retest_ratio']) for m in O.index if m != 'BASE+OT'}).T.sort_values('score')
print(S.round(3).head(12).to_string()); S.to_csv(R + f'dev_{NM.lower()}_{MODEL}.csv')
