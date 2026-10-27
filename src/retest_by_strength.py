"""Human test-retest reliability on the CEK test half by partisan strength (independent, leaner, weak, strong partisan), as a
proxy for political engagement. Writes results/retest_by_strength.json. Usage: python retest_by_strength.py"""
import os, sys, json, numpy as np, pandas as pd
from paths import RES, HERE
os.environ['OTP'] = '1'; sys.argv = ['x', 'cg', 'qwen3.8-27b']
exec(open(os.path.join(HERE, 'final_eval.py')).read().split("sids = list(He.index)")[0])
sids = list(He.index); w1 = He.copy()
w2 = m.loc[sids, [COL[k].replace('_w1', '_w2') for k in KEYS]].astype(float); w2.columns = KEYS
for k in FLIP: w2[k] = 8 - w2[k]
st = np.abs(pe.to_numpy() - 3); lab = {0: 'independent', 1: 'leaner', 2: 'weak partisan', 3: 'strong partisan'}; out = {}
for s_ in range(4):
    idx = [sid for sid, s0 in zip(sids, st) if s0 == s_]
    out[lab[s_]] = float(np.nanmean([pd.concat([w1.loc[idx, k], w2.loc[idx, k]], axis=1).dropna().corr().iloc[0, 1] for k in KEYS]))
json.dump(out, open(os.path.join(RES, 'retest_by_strength.json'), 'w')); print(out)
