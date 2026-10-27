"""Test-retest reliability. Human: corr(w1, w2) per item among CEK control respondents with both waves.
Simulated: corr between two independent draws from each persona's answer distribution (expected exactly: between-persona variance of
E[x] / total variance, computed analytically). Usage: python retest.py MODEL SPLIT N"""
import os, sys
from paths import DATA, RES, PAPER, HERE
import sys, glob, json, numpy as np, pandas as pd, pyreadr
from variants import ITEMS
MODEL, SPLIT, N = sys.argv[1], sys.argv[2], sys.argv[3]; R = RES; KEYS = [k for k, *_ in ITEMS]
src = open(os.path.join(HERE, 'cek_items.py')).read(); g = {}; exec(src[src.index('COL = '):src.index('PID = ')], g); COL = g['COL']
m = pyreadr.read_r(DATA + 'cg/study_1_mturk_cleaned.rds')[None].set_index('subject_id'); ids = json.load(open(RES + 'exp2_split.json'))[SPLIT]
hr = [m.loc[ids, [COL[k], COL[k].replace('_w1', '_w2')]].astype(float).dropna().corr().iloc[0, 1] for k in KEYS]
print(f'HUMAN {SPLIT} retest mean {np.mean(hr):.3f}  (n w/ w2 = {m.loc[ids, "dv_climate_main_w2"].notna().sum()})')
x = np.arange(1, 8); out = [dict(method='HUMAN', retest=np.mean(hr))]
def rt(P):  # P: (n, J, 7) -> analytic reliability of two independent draws
    E = P @ x; V = P @ x ** 2 - E ** 2; return np.mean([E[:, j].var() / (E[:, j].var() + V[:, j].mean()) for j in range(P.shape[1])])
def ot_dummy(P): return P
for f in sorted(glob.glob(R + f'var_{MODEL}_{SPLIT}_n{N}_*.csv')):
    v = f.split(f'_n{N}_')[1][:-4]; d = pd.read_csv(f); sids = sorted(d.sid.unique()); Pm = {(r.sid, r.key): np.array(json.loads(r.p)) for r in d.itertuples()}
    P = np.stack([np.stack([Pm[(s, k)] / Pm[(s, k)].sum() for k in KEYS]) for s in sids]); out.append(dict(method=v, retest=rt(P)))
    if v == 'BASE':
        for T in (1.5, 2.0, 3.0): Q = P ** (1 / T); out.append(dict(method=f'BASE+temp{T}', retest=rt(Q / Q.sum(-1, keepdims=True))))
D = pd.DataFrame(out); D['ratio'] = D.retest / D.retest[0]; D.to_csv(R + f'retest_{MODEL}_{SPLIT}_n{N}.csv', index=False); print(D.round(3).to_string(index=False))
