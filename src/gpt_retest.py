"""Empirical test-retest reliability for API-model personas: two independent joint-answer draws (variant V and V#2) for the same
persona; reliability = mean over items of corr(draw1, draw2) across personas (same estimator as the human w1/w2 panel).
Also reports the reliability after OT (OT applied independently to each draw as a stochastic map; expected correlation by simulation).
Usage: python gpt_retest.py MODEL-EFFORT SPLIT N V1,V2,..."""
import os, sys
from paths import DATA, RES, PAPER, HERE
import sys, json, numpy as np, pandas as pd, pyreadr
from variants import ITEMS
ME, SPLIT, N, VS = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].split(','); R = RES; KEYS = [k for k, *_ in ITEMS]
src = open(os.path.join(HERE, 'cek_items.py')).read(); g = {}; exec(src[src.index('COL = '):src.index('PID = ')], g); COL = g['COL']
FLIP = {'wall_main', 'flat_main', 'climate_3', 'climate_4', 'amtrak_3', 'vets_3', 'flat_3', 'flat_4'}
m = pyreadr.read_r(DATA + 'cg/study_1_mturk_cleaned.rds')[None].set_index('subject_id'); split = json.load(open(RES + 'exp2_split.json'))
ids = split[SPLIT]; HRT = np.mean([m.loc[ids, [COL[k], COL[k].replace('_w1', '_w2')]].astype(float).dropna().corr().iloc[0, 1] for k in KEYS])
Hc = m.loc[split['cal'], [COL[k] for k in KEYS]].astype(float); Hc.columns = KEYS
for k in FLIP: Hc[k] = 8 - Hc[k]
MC = np.stack([np.bincount(Hc[k].dropna().astype(int) - 1, minlength=7)[:7] / Hc[k].notna().sum() for k in KEYS])
rec = {}
for l in open(R + f'cvar_{ME}_{SPLIT}_n{N}.jsonl'):
    d = json.loads(l)
    if d.get('ans'): rec[(d['variant'], d['sid'])] = d['ans']
def ot_map(pL, pH):
    M = np.zeros((7, 7)); a = pL.copy(); b = pH.copy(); i = j = 0
    while i < 7 and j < 7:
        t = min(a[i], b[j]); M[i, j] += t; a[i] -= t; b[j] -= t
        if a[i] <= 1e-12: i += 1
        if j < 7 and b[j] <= 1e-12: j += 1
    rs = M.sum(1, keepdims=True); return np.where(rs > 0, M / np.maximum(rs, 1e-12), np.eye(7))
rng = np.random.default_rng(0); out = [dict(variant='HUMAN', retest=HRT, retest_ot=HRT)]
for v in VS:
    sids = sorted(s for (vv, s) in rec if vv == v and (v + '#2', s) in rec)
    A = np.array([rec[(v, s)] for s in sids]); B = np.array([rec[(v + '#2', s)] for s in sids])
    r = np.nanmean([np.corrcoef(A[:, j], B[:, j])[0, 1] for j in range(len(KEYS))])
    Ms = [ot_map(np.bincount(np.r_[A[:, j], B[:, j]] - 1, minlength=7) / (2 * len(A)), MC[j]) for j in range(len(KEYS))]
    def draw(X): return np.stack([np.array([rng.choice(7, p=Ms[j][x - 1]) + 1 for x in X[:, j]]) for j in range(len(KEYS))], 1)
    rot = np.mean([np.nanmean([np.corrcoef(a[:, j], b[:, j])[0, 1] for j in range(len(KEYS))]) for a, b in [(draw(A), draw(B)) for _ in range(20)]])
    out.append(dict(variant=v, n=len(sids), retest=r, retest_ot=rot))
D = pd.DataFrame(out); D['ratio'] = D.retest / HRT; D['ratio_ot'] = D.retest_ot / HRT; D.to_csv(R + f'retest_{ME}_{SPLIT}.csv', index=False); print(D.round(3).to_string(index=False))
