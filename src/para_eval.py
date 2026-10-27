"""Perturbed-prompt retest on the CEK test half: wave 1 = original question wording, wave 2 = reworded questions (paraphrase.py),
same persona text. Reliability per item = Cov(E1, E2) / sqrt(Var1 * Var2) of the two waves' post-processed answer distributions
(between-persona covariance of expected answers over total answer variance), averaged over items and divided by human retest.
With identical waves it reduces to the reliability used everywhere else. Each wave is calibrated separately. Every persona type
is evaluated over the post-processing grid; corr_F is the wave-1 value from final_eval. Usage: python para_eval.py MODEL"""
import os, sys
from paths import DATA, RES, PAPER, HERE
import os, sys, re, glob, json, numpy as np, pandas as pd
MODEL = sys.argv[1]; os.environ['OTP'] = '1'; sys.argv = ['x', 'cg', MODEL]
exec(open(os.path.join(HERE, 'final_eval.py')).read().split("sids = list(He.index)")[0])
sids = list(He.index); TS = SEL['sharp_T']; x = np.arange(1, 8)
HRT = np.mean([m.loc[sids, [COL[k], COL[k].replace('_w1', '_w2')]].astype(float).dropna().corr().iloc[0, 1] for k in KEYS])
def rel2(P1, P2):
    E1, E2 = P1 @ x, P2 @ x; V1 = P1 @ x ** 2 - E1 ** 2; V2 = P2 @ x ** 2 - E2 ** 2
    return np.mean([np.cov(E1[:, j], E2[:, j])[0, 1] / np.sqrt((E1[:, j].var(ddof=1) + V1[:, j].mean()) * (E2[:, j].var(ddof=1) + V2[:, j].mean())) for j in range(J)])
GRID = [('OT', None)] + [(f'temp{T}+OT', T) for T in (3.0, 2.0, 1.5)] + [(f'sharp{T}+OT', T) for T in (0.5, 0.35, 0.25, 0.15, 0.1, 0.05, 0.01)]
G = pd.read_csv(R + f'final_cg_{MODEL}_otp_reps.csv').groupby('method')[['corrF', 'retest_ratio']].mean(); out = []
for f2 in sorted(glob.glob(R + f'var_{MODEL}_eval_n311_*_para.csv')):
    v = f2.split('_n311_')[1][:-len('_para.csv')]; P1 = loadP(R + f'var_{MODEL}_eval_n311_{v}.csv', sids); P2 = loadP(f2, sids)
    for nm, T in GRID:
        k = f'{v}+{nm}'; a, b = (P1, P2) if T is None else (temp(P1, T), temp(P2, T))
        out.append(dict(method=k, persona=v, rel_para=rel2(ot(a), ot(b)) / HRT, rel_same=rel2(ot(a), ot(a)) / HRT, corrF=G.loc[k, 'corrF'] if k in G.index else np.nan))
D = pd.DataFrame(out); D.to_csv(R + f'para_rel_{MODEL}.csv', index=False); print(D.round(3).to_string())
