"""Spurious opinion churn: share of (respondent, item) pairs whose answer switches side of the scale midpoint (4) between two
measurements. Humans: CEK control respondents, wave 1 vs wave 2 (no treatment in between). Simulated: two independent draws from the
same persona's (post-processed) answer distribution (analytic). Usage: python churn.py MODEL"""
import os, sys
from paths import DATA, RES, PAPER, HERE
import sys, json, glob, numpy as np, pandas as pd, pyreadr
from variants import ITEMS
MODEL = sys.argv[1]; R = RES; KEYS = [k for k, *_ in ITEMS]
src = open(os.path.join(HERE, 'cek_items.py')).read(); g = {}; exec(src[src.index('COL = '):src.index('PID = ')], g); COL = g['COL']
FLIP = {'wall_main', 'flat_main', 'climate_3', 'climate_4', 'amtrak_3', 'vets_3', 'flat_3', 'flat_4'}
m = pyreadr.read_r(DATA + 'cg/study_1_mturk_cleaned.rds')[None].set_index('subject_id'); split = json.load(open(RES + 'exp2_split.json'))
ids = split['eval']; SEL = json.load(open(R + 'selected_cfg.json'))
def side(x): return np.sign(x - 4)
hf = []
for k in KEYS:
    d = m.loc[ids, [COL[k], COL[k].replace('_w1', '_w2')]].astype(float).dropna().to_numpy(); s1, s2 = side(d[:, 0]), side(d[:, 1]); ok = (s1 != 0) | (s2 != 0)
    hf.append(np.mean((s1 * s2 < 0)[ok]))
exec(open(os.path.join(HERE, 'final_eval.py')).read().split("sids = list(He.index)")[0].replace("DS, MODEL = sys.argv[1], sys.argv[2]", "DS, MODEL = 'cg', '" + MODEL + "'"))
sids = list(He.index)
def churn(P):  # P(two draws on opposite sides | not both at midpoint)
    lo, mid, hi = P[..., :3].sum(-1), P[..., 3], P[..., 4:].sum(-1); opp = 2 * lo * hi; both_mid = mid ** 2
    return float(np.mean(opp.mean(0) / (1 - both_mid.mean(0))))
TS = SEL['sharp_T']; rows = [dict(method='HUMAN (w1 vs w2)', churn=np.mean(hf))]
import re
def post(P, spec):  # '+OT', '+tempT+OT', '+sharpT+OT'
    m = re.match(r'^\+(?:(temp|sharp)([0-9.]+)\+)?OT$', spec); return ot(temp(P, float(m.group(2))) if m.group(1) else P)
for v in ['BASE', SEL['persona'], 'DIVERSE', 'ARGYLE', 'BACKSTORY', 'ANTHOLOGY']:
    f = R + f'var_{MODEL}_eval_n311_{v}.csv'
    if not glob.glob(f): continue
    P = loadP(f, sids); specs = {'+OT', f'+sharp{TS}+OT', '+sharp0.05+OT'} | {b[len(v):] for b in SEL['baselines'].values() if b.startswith(v + '+')}
    rows += [dict(method=v + sp, churn=churn(post(P, sp))) for sp in sorted(specs)]
D = pd.DataFrame(rows); D['ratio'] = D.churn / D.churn[0]; D.to_csv(R + f'churn_{MODEL}.csv', index=False); print(D.round(3).to_string(index=False))
